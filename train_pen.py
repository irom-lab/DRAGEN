import faulthandler; faulthandler.enable()
import os
import numpy as np
import torch
import random
import yaml
import shutil
from functools import reduce
from util.misc import *


def factors(n):    
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

from src import INIT_TYPE, TEST_TYPE, GEN_TYPE
from src.pendulum_env import PendulumEnv, PendulumVisionEnv
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.envs.gym import GymEnvWrapper

from rlpyt.algos.qpg.sac_new import SACNew
from rlpyt.agents.qpg.sac_new_agent import SacNewAgent


def f_pen(*args, **kwargs):
	return GymEnvWrapper(PendulumEnv(**kwargs))

def f_pen_vision(*args, **kwargs):
	return GymEnvWrapper(PendulumVisionEnv(**kwargs))

class TrainPendulum:

	def __init__(self, result_dir, 
						pi_args,
						q_args,
						obs_size,
              			policy_path=None, 
                 		num_cpus=10,
						cpu_offset=0,
                   		val_low_init=False,
						num_eval_per_env=8,
						num_frame_stack=3,
						reuse_replay_buffer=True,
						check_running=True,
						cuda_idx=0):
		save__init__args(locals())
		self.log_parent_dir = self.result_dir + 'train_logs/'
		ensure_directory(self.log_parent_dir)

		self.num_gpus = 1
		self.num_step_per_eval = 100
  
		# Copy policy into log_parent_dir
		if policy_path is not None:
			shutil.copyfile(policy_path, self.log_parent_dir+'initial_policy')
		self.policy_path = policy_path

		# Save replay buffer between retrains
		self.replay_buffer_dict = None

	def update_policy_path(self, policy_path):
		self.policy_path = policy_path

	def train(self, prefix='',
					train_img_path_all=None,
					num_itr=80,
					num_env_train=96,
					batch_T=32,
					batch_size=256,
					pretrain_std=0,
					min_steps_learn=int(1e4),
					replay_size=int(1e5),
					replay_fix_ratio=0.1,
					load_model_after_min_steps=False,
					replay_ratio=16,
					clip_grad_norm=5000,
					discount=0.99,
					learning_rate=1e-3,
					num_itr_actor_wait=0,
					linear_annealing=False,
					action_squash=2,
					reward_scale=1,
					fixed_alpha=True,
					initial_alpha=0.2,
					load_q_model=True,
					tie_weights=True,
					min_save_itr_ratio=0,
					min_save_pi_loss_ratio=10000,	# so no limit
					min_save_q_loss_ratio=10000,
					running_std_thres=40,
					num_itr_per_log=1,
     				log_tabular_only=True,
              		):
		"""
		One iteration takes batch_T * batch_B steps. So n_iter = n_steps / (batch_T * batch_B)
		During each iteration, after samples collected (batch_T, batch_B), PPO performs epoches(=4) of updates. During each update, the samples (num=batch_T * batch_B) is split into batches (num=minibatches=4). 
		So batch_T=5, batch_B=10, batch_size=5*10//4=12. And this repeats for 4 epochs. Number of gradient update = 4*4=16
		"""

		# Pick images to train, right now no weight for initial or gen imgs
		num_cpus = max(self.num_cpus, 1)	# usual setting

		# Debug
		print('Number of training envs: ', num_env_train)
		print('Number of cpu threads: ', num_cpus)

		# Eval using training envs, just for more trials
		num_env_eval = len(train_img_path_all)*self.num_eval_per_env

		# Configure steps
		n_steps = batch_T*num_env_train*num_itr
		log_interval_steps = batch_T*num_env_train*num_itr_per_log
		min_save_itr = int(num_itr*min_save_itr_ratio)

		# Load policy
		if self.policy_path is not None:
			trained_data = torch.load(self.policy_path)
			agent_state_dict = trained_data['agent_state_dict']
			optimizer_state_dict = trained_data['optimizer_state_dict']
			# replay_buffer_dict = trained_data['replay_buffer_dict']
		else:
			trained_data = None
			agent_state_dict = None
			optimizer_state_dict = None

		# Make a new folder for each retrain
		if prefix is not None:
			log_dir = self.log_parent_dir+prefix+'/'
		else:
			log_dir = self.log_parent_dir
		saliency_dir = None
		ensure_directory_hard(log_dir)

		# Copy over initial policy
		if self.policy_path is not None:
			shutil.copyfile(self.policy_path, log_dir+'initial_params.pkl')

		sampler = GpuSampler(
			EnvCls=f_pen_vision,
			env_kwargs=[dict(img_path_all=train_img_path_all, 
                    		num_frame_stack=self.num_frame_stack, 
                      		low_init=False,
                        	obs_size=self.obs_size) for _ in range(num_env_train)],
			batch_T=batch_T,
			batch_B=num_env_train,
			max_decorrelation_steps=0,
			eval_n_envs=num_env_eval,
			eval_env_kwargs=[dict(img_path_all=train_img_path_all, 
                         		num_frame_stack=self.num_frame_stack, 
                           		low_init=self.val_low_init,
                             	obs_size=self.obs_size) for _ in range(num_env_eval)],
			eval_max_steps=self.num_step_per_eval*num_env_eval
		)

		# SAC
		actor_detach_encoder = tie_weights	# if not tying weights, then not detach decoder and thus update encoder with actor loss
		algo = SACNew(discount=discount,
					learning_rate=learning_rate,
					num_itr_actor_wait=num_itr_actor_wait,
					actor_detach_encoder=actor_detach_encoder,
					linear_annealing=linear_annealing,
					batch_size=batch_size,
					replay_size=replay_size,
					min_steps_learn=min_steps_learn,
					replay_fix_ratio=replay_fix_ratio,
					load_model_after_min_steps=load_model_after_min_steps,
					replay_ratio=replay_ratio,
					clip_grad_norm=clip_grad_norm,
					fixed_alpha=fixed_alpha,
					initial_alpha=initial_alpha,
					reward_scale=reward_scale,
					initial_optim_state_dict=optimizer_state_dict,
					initial_replay_buffer_dict=self.replay_buffer_dict,
					)
		agent = SacNewAgent(initial_model_state_dict=agent_state_dict,
						load_model_after_min_steps=load_model_after_min_steps,
						load_q_model=load_q_model,
						tie_weights=tie_weights,
						model_kwargs=self.pi_args,
						q_model_kwargs=self.q_args,
						pretrain_std=pretrain_std,
						action_squash=action_squash,
						saliency_dir=saliency_dir)

		self.affinity = dict(cuda_idx=self.cuda_idx,
			workers_cpus=list(range(self.cpu_offset, num_cpus+self.cpu_offset)))
		runner = MinibatchRlEval(algo=algo,
								agent=agent,
								sampler=sampler,
								n_steps=n_steps,
								log_interval_steps=log_interval_steps,
								affinity=self.affinity,
								min_save_args=dict(min_save_itr=min_save_itr,
												min_save_pi_loss_ratio=min_save_pi_loss_ratio,
												min_save_q_loss_ratio=min_save_q_loss_ratio),
								running_std_thres=running_std_thres)

		run_ID = 0
		with logger_context(log_dir, run_ID, prefix, {}, override_prefix=True, log_tabular_only=log_tabular_only, snapshot_mode='last'):
			returns = runner.train(return_buffer=self.reuse_replay_buffer,
                          			check_running=self.check_running)

		# Save replay buffer
		if self.reuse_replay_buffer:
			best_itr, replay_buffer_dict = returns
			self.replay_buffer_dict = replay_buffer_dict
		else:
			best_itr = returns

		if best_itr > 0:
			# Remove initial_params
			if self.policy_path is not None:
				os.remove(log_dir+'initial_params.pkl') 
			return log_dir+'itr_'+str(best_itr)+'_best_params.pkl'
		else:
			return log_dir+'initial_params.pkl'	#! actually if we start training without initial policy, and retraining at first epoch is not good, initial_params does not exist - make sure num_itr_initial is enough to have a decent initial policy


if __name__ == '__main__':

	# Read config
	yaml_file_name = sys.argv[1]
	yaml_path = 'configs/'+yaml_file_name
	with open(yaml_path+'.yaml', 'r') as f:
		config = yaml.load(f, Loader=yaml.FullLoader)
	seed = config['seed']

	num_cpus = config['num_cpus']
	cpu_offset = config['cpu_offset']
	cuda_idx = config['cuda_idx']

	val_low_init = config['val_low_init']
	num_eval_per_env = config['num_eval_per_env']

	policy_path = config['policy_path']
	img_dir_list = config['img_dir_list']
	img_id_list = config['img_id_list']
	num_image_per_dir_list = config['num_image_per_dir_list']

	retrain_args = config['retrain_args']
	pi_args = config['pi_args']
	q_args = config['q_args']

	# Fix seeds
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	# Set up directory
	result_dir = os.getcwd()+"/result/"+yaml_file_name+'/'
	saliency_dir = result_dir + 'saliency/'
	ensure_directory_hard(result_dir)
	ensure_directory_hard(saliency_dir)

	# Save a copy of configuration
	with open(result_dir+'config.yaml', 'w') as f:
		yaml.dump(config, f, sort_keys=False)

	# Set up dir dict
	img_dir_dict = {}
	for dir_ind, img_dir in enumerate(img_dir_list):
		if img_id_list is not None:
			img_dir_dict[img_dir] = (img_id_list, INIT_TYPE)
		elif num_image_per_dir_list is not None:
			img_dir_dict[img_dir] = ([*range(num_image_per_dir_list[dir_ind])], INIT_TYPE)
		else:		
  			img_dir_dict[img_dir] = ([*range(retrain_args['num_env_train'])], INIT_TYPE)
	retrain_args['img_dir_dict'] = img_dir_dict

	# Train
	trainer = TrainPendulum(result_dir=result_dir,
                         	policy_path=policy_path,
							pi_args=pi_args,
							q_args=q_args,
                          	num_cpus=num_cpus,
							cpu_offset=cpu_offset,
                        	val_low_init=val_low_init,
                         	num_eval_per_env=num_eval_per_env,
                          	cuda_idx=cuda_idx)
	trainer.train(**retrain_args)

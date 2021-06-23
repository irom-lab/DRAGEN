import os
import numpy as np
from numpy import array
import torch
from util.misc import supress_stdout, save__init__args
from train_pen import f_pen_vision

from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.algos.qpg.sac_new import SACNew
from rlpyt.agents.qpg.sac_new_agent import SacNewAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEvalOnly


class EvaluatePendulum:
	def __init__(self, num_eval_per_env, 
						pi_args,
						q_args,
						obs_size,
						num_frame_stack=3,
						normalization=None,
                   		num_cpus=16, 
                     	num_gpus=1,
						cpu_offset=0,
                      	num_step_per_eval=100, 
                       	low_init=False,
						cuda_idx=0):
		save__init__args(locals())

		self.low_mass = False

		# Affinity is the always same
		self.affinity = make_affinity(run_slot=0,
									n_socket=1,
									n_cpu_core=num_cpus//2,
									n_gpu=num_gpus,
									cpu_per_run=num_cpus//2,
									gpu_per_run=num_gpus,
									alternating=True)

	def update_policy_path(self, policy_path):
		self.policy_path = policy_path

	def get_eval_img_path(self, img_dir, img_id_list):
		return sum([[img_dir+str(id)+'.png']*self.num_eval_per_env for id in img_id_list], [])

	def normalize_label(self, label_batch, num_img):
		label_batch = np.array_split(label_batch, num_img)
		label_batch = array([np.mean(l) for l in label_batch])
		label_batch = (label_batch - self.normalization[0])/(self.normalization[1]-self.normalization[0])
		return np.clip(label_batch, 0, 1)

	# @supress_stdout
	def evaluate(self, img_dir, img_id_list, return_reward_list=False):
		num_img = len(img_id_list)
		eval_img_path_all = self.get_eval_img_path(img_dir, img_id_list)
		num_env_eval = num_img*self.num_eval_per_env

		sampler = GpuSampler(
			EnvCls=f_pen_vision,
			env_kwargs=[dict(img_path=eval_img_path_all[0], 
						 			renders=False, discrete_actions=False, 
									num_frame_stack=self.num_frame_stack, 
							  		low_mass=self.low_mass, 
           							low_init=self.low_init,
                  					obs_size=self.obs_size)],
			batch_T=16,
			batch_B=1,	# should not matter
			max_decorrelation_steps=0,
			eval_n_envs=num_env_eval,
			eval_env_kwargs=[dict(img_path=img_path, 
						 			renders=False, discrete_actions=False, 
									num_frame_stack=self.num_frame_stack, 
							  		low_mass=self.low_mass, 
           							low_init=self.low_init,
                  					obs_size=self.obs_size) for img_path in eval_img_path_all],
			eval_max_steps=self.num_step_per_eval*num_env_eval
		)

		algo = SACNew(replay_size=int(1e2))	# not actually using if eval only, so default config
		if self.policy_path is not None:
			agent_state_dict = torch.load(self.policy_path, map_location='cuda:'+str(self.cuda_idx))['agent_state_dict']
		else:
			agent_state_dict = None
		agent = SacNewAgent(initial_model_state_dict=agent_state_dict,
							load_model_after_min_steps=False,
							load_q_model=False,
							tie_weights=False,# in case not tying when training
							model_kwargs=self.pi_args,
							q_model_kwargs=self.q_args,
							action_squash=2,
							saliency_dir=None)

		self.affinity = dict(cuda_idx=self.cuda_idx,
							workers_cpus=list(range(self.cpu_offset, self.num_cpus+self.cpu_offset)))
		runner = MinibatchRlEvalOnly(
			algo=algo,
			agent=agent,
			sampler=sampler,
			n_steps=int(1e2),	# does not matter
			log_interval_steps=int(1e1),	# does not matter
			affinity=self.affinity,
		)
		eval_reward_all = runner.eval()
		if self.normalization is not None:
			return self.normalize_label(eval_reward_all, num_img)
		elif return_reward_list:
			return eval_reward_all
		else:
			return np.mean(eval_reward_all)


if __name__ == '__main__':

	num_eval_per_env = 8	# different initial conditions
	low_init = True
	img_dir = '/'

	num_env_eval = 300
	img_id_list = range(num_env_eval)

	pi_args = {'channels': [16, 32],
				'kernel_sizes': [6, 4],
				'strides': [4, 2],
				'paddings': [0, 0],
				'hidden_sizes': [128, 128]}
	q_args = {'channels': [16, 32],
				'kernel_sizes': [6, 4],
				'strides': [4, 2],
				'paddings': [0, 0],
				'hidden_sizes': [256, 256]}
	normalization = [-900, -400]

	evaluator = EvaluatePendulum(num_cpus=16, 
                              	num_gpus=1, 
								obs_size=128,
                               	num_step_per_eval=100,
								pi_args=pi_args,
								q_args=q_args,
                               	num_eval_per_env=num_eval_per_env,
								policy_path="",
								normalization=normalization,
                                low_init=low_init)
	eval_reward_all = evaluator.evaluate(img_dir=img_dir,
                                    	img_id_list=img_id_list,
                                     	return_reward_list=False)
	print(eval_reward_all)
	print('Avg reward: ', np.mean(eval_reward_all))

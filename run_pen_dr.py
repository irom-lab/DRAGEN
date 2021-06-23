import os
import sys
import yaml
import logging
import logging.config
import time
import random

import numpy as np
from numpy import array
import torch

from src import INIT_TYPE, TEST_TYPE, GEN_TYPE
from train_pen import TrainPendulum
from eval_pen import EvaluatePendulum
from util.misc import *


def generate_by_gaussian(img_dir_dict, new_dir, num_new, std=0.05):
	"""
	Generate new images by randomly adding Gaussian noise to initial envs
	"""
	old_image_available_path_list = []
	for img_dir, (img_id_list, dir_type) in img_dir_dict.items():
		if dir_type == INIT_TYPE:
			old_image_available_path_list += [img_dir+str(id)+'.png' for id in img_id_list]
	old_image_path_list = random.sample(old_image_available_path_list, k=num_new)
	for img_ind, img_path in enumerate(old_image_path_list):
		old_image = array(Image.open(img_path))/255
		H, W, _ = old_image.shape
		noise = np.random.normal(0, std, size=old_image.shape)
		new_image = (old_image + noise).clip(min=0.0, max=1.0)
		new_image = Image.fromarray(np.uint8(new_image*255))
		new_image.save(new_dir+str(img_ind)+'.png')
		new_image.close()


def generate_by_cutout(img_dir_dict, new_dir, num_new, cutout_dim=8, top_limit=8, bottom_limit=40):
	"""
	Generate new images by randomly cutout a patch - set top and bottom limit to make sure cutout affects the digit
	"""
	old_image_available_path_list = []
	for img_dir, (img_id_list, dir_type) in img_dir_dict.items():
		if dir_type == INIT_TYPE:
			old_image_available_path_list += [img_dir+str(id)+'.png' for id in img_id_list]
	old_image_path_list = random.sample(old_image_available_path_list, k=num_new)
	for img_ind, img_path in enumerate(old_image_path_list):
		image = array(Image.open(img_path))/255
		H, W, _ = image.shape
		cutout_top_h = random.randint(top_limit, bottom_limit-cutout_dim)
		cutout_top_w = random.randint(top_limit, bottom_limit-cutout_dim)
		image[cutout_top_h:(cutout_top_h+cutout_dim), cutout_top_w:(cutout_top_w+cutout_dim)] = 0
		image = Image.fromarray(np.uint8(image*255))
		image.save(new_dir+str(img_ind)+'.png')
		image.close()


def generate_by_perlin(img_dir_dict, new_dir, num_new, octaves=10, noise_strength=0.5, rgb=True):
	"""
	Generate new images by randomly adding perlin noise to initial envs
	https://pypi.org/project/perlin-noise/
	# bigger octaves, smaller blobs
	"""
	old_image_available_path_list = []
	for img_dir, (img_id_list, dir_type) in img_dir_dict.items():
		if dir_type == INIT_TYPE:
			old_image_available_path_list += [img_dir+str(id)+'.png' for id in img_id_list]
	old_image_path_list = random.sample(old_image_available_path_list, k=num_new)
	for img_ind, img_path in enumerate(old_image_path_list):
		old_image = array(Image.open(img_path))/255
		H, W, _ = old_image.shape
		if not rgb:
			noise = PerlinNoise(octaves=octaves, seed=random.randint(0, 1e6))
			noise = array([[noise([i/H, j/W]) for j in range(H)] for i in range(W)])*noise_strength	# from [-1,1] to [-noise_strength,noise_strength]
			noise_full = np.repeat(noise[:,:,np.newaxis], 3, axis=2)
		else:
			noise_full = np.empty((H, W, 0))
			for _ in range(3):
				noise = PerlinNoise(octaves=octaves,seed=random.randint(0, 1e6))
				noise = array([[noise([i/H, j/W]) for j in range(H)] for i in range(W)])*noise_strength
				noise_full = np.concatenate((noise_full, noise[:,:,np.newaxis]), axis=2)
		new_image = (old_image + noise_full).clip(min=0.0, max=1.0)
		new_image = Image.fromarray(np.uint8(new_image*255))
		new_image.save(new_dir+str(img_ind)+'.png')
		new_image.close()


if __name__ == '__main__':
	# from IPython import embed; embed()

	# Read config
	yaml_file_name = sys.argv[1]
	yaml_path = 'configs/'+yaml_file_name
	with open(yaml_path+'.yaml', 'r') as f:
		config = yaml.load(f, Loader=yaml.FullLoader)

	# Fix seeds
	seed = config['seed']
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.backends.cudnn.benchmark = True	# may speed up

	num_cpus = config['num_cpus']
	cuda_idx = config['cuda_idx']
	device = 'cuda:'+str(cuda_idx)
	cpu_offset = config['cpu_offset']

	# Misc
	num_epoch = config['num_epoch']
	dr_method = config['dr_method']
	num_frame_stack = config['num_frame_stack']
	label_normalization = config['label_normalization']
	val_low_init = config['val_low_init']
	num_eval_per_env = config['num_eval_per_env']
	dim_obs = config['dim_obs']

	# Improving policy
	retrain_args = config['retrain_args']
	pi_args = config['pi_args']
	q_args = config['q_args']
	reuse_replay_buffer = config['reuse_replay_buffer']
	check_running = config['check_running']

	# Env params
	initial_env_dir_list = config['initial_env_dir_list']
	test_env_dir_list = config['test_env_dir_list']
	num_env_per_initial_dir = config['num_env_per_initial_dir']
	num_env_per_test_dir = config['num_env_per_test_dir']

	# Initialize folders
	result_dir = 'result/'+yaml_file_name+'/'
	if dr_method != 'none':
		data_dir = config['data_parent_dir']+yaml_file_name+'/'
		num_env_per_gen = config['num_env_per_gen']
		ensure_directory(data_dir)
		from PIL import Image
	ensure_directory(result_dir)

	# Initialize dir dict: key is dir_path, value is a tuple of (1) id list and (2) type (0 for initial, 1 for test, 2 for gen)
	env_dir_dict = {}
	for env_dir in initial_env_dir_list:
		env_dir_dict[env_dir] = ([*range(num_env_per_initial_dir)], INIT_TYPE)

	# Save a copy of configuration
	with open(result_dir+'config.yaml', 'w') as f:
		yaml.dump(config, f, sort_keys=False)

	# Initialize evaluating policy
	evaluator = EvaluatePendulum(num_cpus=num_cpus, 
							  	num_gpus=1,
								cpu_offset=cpu_offset,
							   	num_step_per_eval=100,
								obs_size=dim_obs,
								num_frame_stack=num_frame_stack,
								low_init=val_low_init,
        						normalization=label_normalization,
              					num_eval_per_env=num_eval_per_env,
                       		    pi_args=pi_args,
                       		    q_args=q_args,
								cuda_idx=cuda_idx)

	# Initialize training policy
	trainer = TrainPendulum(result_dir=result_dir,
                       		pi_args=pi_args,
                       		q_args=q_args,
							obs_size=dim_obs,				
                            num_cpus=num_cpus,
							cpu_offset=cpu_offset,
							num_frame_stack=num_frame_stack,
							val_low_init=val_low_init,
			  				num_eval_per_env=num_eval_per_env,
							reuse_replay_buffer=reuse_replay_buffer,
							check_running=check_running,
							cuda_idx=cuda_idx)

	# Training details to be recorded
	train_success_list = []	# using initial objects
	test_success_list = []

	# Add test dir to dict
	for env_dir in test_env_dir_list:
		env_dir_dict[env_dir] = ([*range(num_env_per_test_dir)], TEST_TYPE)

	# Initialize counter
	num_epoch_since_last_dr = 0
	num_dir_gen = 0

	# Name of saved training details
	train_details_path = None

	# Logging
	logging.config.dictConfig({
		'version': 1,
		'disable_existing_loggers': True,
	})
	logging.basicConfig(filename=result_dir+'log.txt',
					level=logging.NOTSET,
					format='%(process)d-%(levelname)s-%(asctime)s-%(message)s',
					datefmt='%m/%d/%Y %I:%M:%S')
	pil_logger = logging.getLogger('PIL')
	pil_logger.setLevel(logging.INFO)	# prevent spamming debug image by pil
	logging.info('start')

	# Run
	for epoch in range(num_epoch):

		# Record time for each epoch
		epoch_start_time = time.time()

		######################### Generate #########################
		if epoch > 0 and dr_method != '':
			# Declare new path
			new_gen_dir = data_dir + 'gen_' + str(num_dir_gen) + '/'
			ensure_directory(new_gen_dir)
			print('Generating new ', new_gen_dir)
			if 'perlin' in dr_method:
				from perlin_noise import PerlinNoise
				generate_by_perlin(img_dir_dict=env_dir_dict,
										new_dir=new_gen_dir,
										num_new=num_env_per_gen,
										rgb='rgb' in dr_method)
			elif 'gaussian' in dr_method:
				generate_by_gaussian(img_dir_dict=env_dir_dict,
										new_dir=new_gen_dir,
										num_new=num_env_per_gen)
			elif 'cutout' in dr_method:
				generate_by_cutout(img_dir_dict=env_dir_dict,
										new_dir=new_gen_dir,
										num_new=num_env_per_gen,
          								cutout_dim=config['cutout_dim'],
                  						top_limit=config['top_limit'],
                        				bottom_limit=config['bottom_limit'])
			new_env_id_list = np.arange(num_env_per_gen)

			# Add to dir dict
			env_dir_dict[new_gen_dir] = (new_env_id_list, GEN_TYPE)

			# Count
			num_dir_gen += 1

		######################### Retrain #########################

		# Pick which envs for training - #! choose all
		retrain_env_path_list = []
		for env_dir, (env_id_list, dir_type) in env_dir_dict.items():
			if dir_type != TEST_TYPE:
				retrain_env_path_list += [env_dir+str(id)+'.png' for id in env_id_list]

		# Use more itrs at 1st retrain
		retrain_args_copy = dict(retrain_args)	# make a copy
		retrain_args_copy.pop('num_itr_initial', None)
		retrain_args_copy.pop('num_itr_final', None)
		if epoch == 0:
			retrain_args_copy['num_itr'] = retrain_args['num_itr_initial']
		elif epoch == (num_epoch-1):
			retrain_args_copy['num_itr'] = retrain_args['num_itr_final']

		# Retrain
		new_policy_path = trainer.train(
								train_img_path_all=retrain_env_path_list,
								prefix='epoch_'+str(epoch),
								**retrain_args_copy)
		logging.info(f'Epoch {epoch} retrain, new policy {new_policy_path}')
		print('New policy, ', new_policy_path)

		# Update policy for evaluator and trainer
		trainer.update_policy_path(new_policy_path)
		evaluator.update_policy_path(new_policy_path)

		# Check if using initial policy
		flag_policy_improved = True
		if 'initial' in new_policy_path:
			flag_policy_improved = False

		#?
		torch.cuda.empty_cache()

		######################### Re-evaluate #########################
		print('Re-evaluating for all...')

		# Assume policy always improved after initial train
		if flag_policy_improved:
			train_success_batch = np.empty((0), dtype='float')
			test_success_batch = np.empty((0), dtype='float')
			for env_dir in initial_env_dir_list:
				label_batch = evaluator.evaluate(img_dir=env_dir,
								img_id_list=np.arange(num_env_per_initial_dir))
				train_success_batch = np.concatenate((train_success_batch, 
														label_batch))
			for env_dir in test_env_dir_list:
				label_batch = evaluator.evaluate(img_dir=env_dir,
									img_id_list=np.arange(num_env_per_test_dir))
				test_success_batch = np.concatenate((test_success_batch, 
														label_batch))
		train_success_list += [np.mean(train_success_batch)]
		logging.info(f'Epoch {epoch} retrain, train reward {train_success_list[-1]:.3f}')
		test_success_list += [np.mean(array(test_success_batch))]
		logging.info(f'Epoch {epoch} retrain, test reward {test_success_list[-1]:.3f}')

		# Debug
		epoch_duration = time.time() - epoch_start_time
		print("Epoch {:d}, Train success: {:.3f}, Test success: {:.3f} ".format(epoch, train_success_list[-1], test_success_list[-1]))

		# Save training details
		if train_details_path is not None:
			os.remove(train_details_path)
		train_details_path = result_dir+'train_details_'+str(epoch)
		torch.save({
			'epoch': epoch,
			'train_lists': [train_success_list, test_success_list],
			"seed_data": (seed, random.getstate(), np.random.get_state(), torch.get_rng_state()),
			}, train_details_path)

		# Check time
		print('This epoch took: %.2f\n' % (time.time()-epoch_start_time))

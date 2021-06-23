import os
import sys
import yaml
import logging
import logging.config
import time
import random

import math
import numpy as np
from numpy import array
import torch

from src import INIT_TYPE, TEST_TYPE, GEN_TYPE
from train_grasp import TrainGrasp
from eval_grasp import EvaluateGrasp
from util.misc import *
from util.mesh import *


def generate_by_chaining_2D(prim_dir_list, new_dir, num_new, keep_concave_part):
	"""
	Generate new objects by randomly chaining primitives
	"""
	num_obj_new = 0
	height_all = []
	prim_dir_obj_all = []
	for prim_dir in prim_dir_list:
		prim_dir_obj_all += [fnmatch.filter(os.listdir(prim_dir), '*.stl')]

	while 1:
		# Randomly choose one from each dir, and then chain them; or choose multiple ones from the same dir
		chosen_prim_path_list = []
		if len(prim_dir_list) > 1:
			for dir_ind, prim_dir in enumerate(prim_dir_list):
				chosen_prim = random.choice(prim_dir_obj_all[dir_ind])
				chosen_prim_path_list += [prim_dir+chosen_prim]
		else:
			chosen_prim = random.sample(prim_dir_obj_all[0], k=2)
			chosen_prim_path_list = [prim_dir_list[0]+prim for prim in chosen_prim]

		try:
			chained_mesh = chain_mesh_2D(prim_path_list=chosen_prim_path_list)
			processed_mesh = process_mesh(chained_mesh, 
                           			scale_down=False, 
                              		random_scale=True)
			ensure_directory_hard(new_dir + str(num_obj_new) + '/')
			convex_pieces = save_convex_urdf(processed_mesh, 
											new_dir, 
											num_obj_new, 
											mass=0.1,
											keep_concave_part=keep_concave_part)
			height_all +=[processed_mesh.bounds[1,2]-processed_mesh.bounds[0,2]]
		except:
			print('Cannot chain!')
			continue

		# Skip if too many convex pieces
		if len(convex_pieces) > 20:
			print('Too concave!')
			continue

		# Count
		num_obj_new += 1
		if num_obj_new == num_new:
			return height_all


def generate_by_chaining_3D(prim_dir_list, new_dir, num_new, keep_concave_part, target_xy_range, max_z):
	"""
	Generate new objects by randomly chaining primitives
	"""
	num_obj_new = 0
	height_all = []
	prim_dir_obj_all = []
	for prim_dir in prim_dir_list:
		prim_dir_obj_all += [fnmatch.filter(os.listdir(prim_dir), '*.stl')]

	max_num_attempt = num_new*5
	num_attempt = 0
	while 1:
		num_attempt += 1
		if num_attempt == max_num_attempt:
			raise ValueError('Chaining failed too often')

		# Randomly choose one from each dir, and then chain them; or choose multiple ones from the same dir
		chosen_prim_path_list = []
		if len(prim_dir_list) > 1:
			for dir_ind, prim_dir in enumerate(prim_dir_list):
				chosen_prim = random.choice(prim_dir_obj_all[dir_ind])
				chosen_prim_path_list += [prim_dir+chosen_prim]
		else:
			chosen_prim = random.sample(prim_dir_obj_all[0], k=2)	# chain 2 parts
			chosen_prim_path_list = [prim_dir_list[0]+prim for prim in chosen_prim]

		# try:
		chained_mesh = chain_mesh_3D(prim_path_list=chosen_prim_path_list)
		if chained_mesh is None:
			print('Cannot chain!')
			continue

		try:
			processed_mesh = process_mesh(chained_mesh,
										remove_body=False,	#!
										scale_down=False, 
										random_scale=False)
			processed_mesh = random_scale_down_mesh(processed_mesh,
                                			target_xy_range=target_xy_range,
                                        	max_z=max_z) 
			ensure_directory_hard(new_dir + str(num_obj_new) + '/')
			convex_pieces = save_convex_urdf(processed_mesh, 
											new_dir, 
											num_obj_new, 
											mass=0.1,
											keep_concave_part=keep_concave_part)
		except:
			print('Cannot process!')
			continue

		# Skip if too many convex pieces
		if len(convex_pieces) > 20:
			print('Too concave!')
			continue

		# Count
		height_all += [processed_mesh.bounds[1,2]-processed_mesh.bounds[0,2]]
		num_obj_new += 1
		if num_obj_new == num_new:
			return height_all


if __name__ == '__main__':
	# from IPython import embed; embed()

	if os.cpu_count() > 20:	# somehow on server, the default fork method does not work with pytorch, but works fine on desktop
		import multiprocessing
		multiprocessing.set_start_method('forkserver')

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
	torch.backends.cudnn.benchmark = True

	# Hardware
	cuda_idx = config['cuda_idx']
	device = 'cuda:'+str(cuda_idx)

	# Misc
	num_eval_per_env = config['num_eval_per_env']

	# Data
	initial_env_dir_list = config['initial_env_dir_list']
	num_env_per_initial_dir = config['num_env_per_initial_dir']
	test_env_dir_list = config['test_env_dir_list']
	num_env_per_test_dir = config['num_env_per_test_dir']

	# Domain Randomization (chaining primitives)
	dr_method = config['dr_method']
	target_xy_range = config['target_xy_range']
	max_z = config['max_z']
	keep_concave_part = config['keep_concave_part']
	num_retrain = config['num_retrain']
	num_env_per_gen = config['num_env_per_gen']
	num_env_per_retrain = config['num_env_per_retrain']
	retrain_sample_recency = config['retrain_sample_recency']
	mu_list = config['mu_list']
	mu = config['mu']
	sigma = config['sigma']
	retrain_args = config['retrain_args']
	eval_args = config['eval_args']

	# Initialize folders
	data_parent_dir = config['data_parent_dir']
	result_dir = 'result/'+yaml_file_name+'/'
	data_dir = data_parent_dir+yaml_file_name+'/'
	ensure_directory(result_dir)
	ensure_directory(data_dir)

	# Initialize dir dict: key is dir_path, value is a tuple of (1) id list and (2) type (0 for initial, 1 for test, 2 for gen)
	env_dir_dict = {}
	for env_dir in initial_env_dir_list:
		height_all =list(np.load(env_dir+'dim.npy')[:num_env_per_initial_dir,2])
		env_dir_dict[env_dir] = ([*range(num_env_per_initial_dir)], height_all, INIT_TYPE)

	# Save a copy of configuration
	with open(result_dir+'config.yaml', 'w') as f:
		yaml.dump(config, f, sort_keys=False)

	# Initialize evaluating policy (always cpu)
	evaluator = EvaluateGrasp(initial_policy_path=None,
							mu_list=mu_list, mu=mu, sigma=sigma, **eval_args)

	# Initialize training policy
	trainer = TrainGrasp(result_dir=result_dir, device=device,
					  	mu=mu, sigma=sigma, **retrain_args)

	# Training details to be recorded
	train_success_list = []	# using initial objects
	test_success_list = []

	# Add test dir to dict
	for env_dir in test_env_dir_list:
		height_all = list(np.load(env_dir+'dim.npy')[:num_env_per_test_dir,2])
		env_dir_dict[env_dir] = ([*range(num_env_per_test_dir)], height_all, TEST_TYPE)

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
	logging.info('start')

	# Run
	for epoch in range(num_retrain):

		# Record time for each epoch
		epoch_start_time = time.time()

		######################### New #########################

		if epoch > 0 and dr_method:	# no gen at the beginning
			# Declare new path
			new_gen_dir = data_dir + 'gen_' + str(epoch) + '/'
			ensure_directory(new_gen_dir)

			print('Generating new...')
			if dr_method == 'chain_3D':
				dr_func = generate_by_chaining_3D
			elif dr_method == 'chain':
				dr_func = generate_by_chaining_2D
			height_all = dr_func(prim_dir_list=initial_env_dir_list,
								new_dir=new_gen_dir,
								num_new=num_env_per_gen,
        						keep_concave_part=keep_concave_part,
              					target_xy_range=target_xy_range,
                   				max_z=max_z)
			new_env_id_list = np.arange(num_env_per_gen)

			# Evaluate label of new envs - use mu_list
			print('Evaluating newly generated...')
			mu_batch = array(evaluator.evaluate(obj_dir=new_gen_dir,
								obj_id_list=new_env_id_list,
								obj_height_list=height_all,
								num_eval=num_eval_per_env)[1], dtype='float')
			label_batch = get_label_from_mu(mu_batch, mu_list)
			print('Reward of newly generated: ', np.mean(label_batch))
			logging.info(f'Reward of newly generated: {np.mean(label_batch)}')

			# Add to dir dict
			env_dir_dict[new_gen_dir] = (new_env_id_list, height_all, GEN_TYPE)

		######################### Retrain #########################

		print(f'Retraining policy {epoch}...')
		logging.info(f'Retraining policy {epoch}...')

		# Pick which envs for training
		retrain_env_path_available_all = []
		retrain_env_height_available_all = []
		retrain_env_weight_all = []
		gen_dir_count = 0
		for env_dir, (env_id_list, height_list, dir_type) in env_dir_dict.items():
			if dir_type != TEST_TYPE:
				retrain_env_path_available_all += [env_dir+str(id)+'.urdf' for id in env_id_list]
				retrain_env_height_available_all += height_list
			if dir_type == INIT_TYPE:
				retrain_env_weight_all += [1]*len(env_id_list)
			elif dir_type == GEN_TYPE:
				retrain_env_weight_all += [math.exp(gen_dir_count*retrain_sample_recency)]*len(env_id_list)
				gen_dir_count += 1
		retrain_env_weight_all = array(retrain_env_weight_all)/np.sum(array(retrain_env_weight_all))
		retrain_env_path_list, chosen_id_list = weighted_sample_without_replacement(retrain_env_path_available_all, retrain_env_weight_all, k=min(num_env_per_retrain, len(retrain_env_path_available_all)))
		retrain_env_height_list = list(array(retrain_env_height_available_all)[chosen_id_list])

		# Use more itrs at 1st retrain
		retrain_args_copy = dict(retrain_args)	# make a copy
		retrain_args_copy.pop('num_step_initial', None)
		if epoch == 0:
			retrain_args_copy['num_step'] = retrain_args['num_step_initial']

		new_policy_path = trainer.run(obj_path_all=retrain_env_path_list,
										obj_height_all=retrain_env_height_list,
										prefix='epoch_'+str(epoch),
										**retrain_args_copy)
		logging.info(f'Epoch {epoch} retrain, new policy {new_policy_path}')

		# Update evaluator
		trainer.load_policy(new_policy_path)
		evaluator.load_policy(new_policy_path)

		######################### Re-evaluate #########################

		# INIT - eval for train_success and label
		train_success_batch = np.empty((0), dtype='float')
		train_success_dirs = []
		for env_dir, (env_id_list, height_list, dir_type) in env_dir_dict.items():
			if dir_type == INIT_TYPE:
				mu_batch = array(evaluator.evaluate(obj_dir=env_dir,
								obj_id_list=env_id_list,
								obj_height_list=height_list,
							num_eval=num_eval_per_env)[1], dtype='float')
				label_batch = get_label_from_mu(mu_batch, mu_list)
				train_success_batch = np.concatenate((train_success_batch, label_batch))
				train_success_dirs += [np.mean(label_batch)]

		# TEST - eval for test_succes
		test_success_batch = np.empty((0), dtype='float')
		test_success_dirs = []
		for env_dir, (env_id_list, height_list, dir_type) in env_dir_dict.items():
			if dir_type == TEST_TYPE:
				mu_batch = array(evaluator.evaluate(obj_dir=env_dir,
								obj_id_list=env_id_list,
								obj_height_list=height_list,
							num_eval=num_eval_per_env)[1], dtype='float')
				label_batch = get_label_from_mu(mu_batch, mu_list)
				test_success_batch = np.concatenate((test_success_batch,
													label_batch))
				test_success_dirs += [np.mean(label_batch)]

		train_success_list += [np.mean(train_success_batch)]
		logging.info(f'Epoch {epoch} retrain, train reward {train_success_dirs}, avg {train_success_list[-1]:.3f}')

		test_success_list += [np.mean(array(test_success_batch))]
		logging.info(f'Epoch {epoch} retrain, test reward {test_success_dirs}, avg {test_success_list[-1]:.3f}')

		# Reset
		num_retrain += 1

		########################  Debug  ########################

		epoch_duration = time.time() - epoch_start_time
		print("Epoch {:d}, Train success: {:.3f}, Test success: {:.3f} ".format(epoch, train_success_list[-1], test_success_list[-1]))

		# Save training details
		if train_details_path is not None:
			os.remove(train_details_path)
		train_details_path = result_dir+'train_details_'+str(epoch)
		torch.save({
			'epoch': epoch,
			'train_lists': [train_success_list, test_success_list],
			}, train_details_path)

		# Check time
		print('This epoch took: %.2f\n' % (time.time()-epoch_start_time))

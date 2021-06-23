import os
# If server, need to use osmesa for pyopengl/pyrender
if os.cpu_count() > 20:
	os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
		# https://github.com/marian42/mesh_to_sdf/issues/13
		# https://pyrender.readthedocs.io/en/latest/install/index.html?highlight=ssh#getting-pyrender-working-with-osmesa
else:
	os.environ['PYOPENGL_PLATFORM'] = 'egl'	# default one was pyglet, which hangs sometime for unknown reason: https://github.com/marian42/mesh_to_sdf/issues/19;
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
import matplotlib.pyplot as plt

from src import INIT_TYPE, TEST_TYPE, GEN_TYPE
from src.sample_sdf import PointSampler
from src.sdf_net import SDFDecoder
from src.pointnet_encoder import PointNetEncoder
from src.cost_predictor import CostPredictor
from train_grasp import TrainGrasp
from src.dataset_grasp import TrainDataset
from eval_grasp import EvaluateGrasp
from util.misc import *
from util.mesh import *


class Runner:

	def __init__(self, yaml_path, result_dir, device):
		save__init__args(locals())
		self.model_dir = result_dir + 'model/'
		self.latent_img_dir = result_dir + 'latent_img/'

		# Configure from yaml file
		with open(yaml_path+'.yaml', 'r') as f:
			config = yaml.load(f, Loader=yaml.FullLoader)
		self.config = config
		self.voxel_resolution = config['voxel_resolution']

		# always be one because of dataset design
		self.batch_size = config['batch_size']

		# NN params
		self.dim_latent = config['dim_latent']
		self.encoder_breadth = config['encoder_breadth']
		self.decoder_breadth = config['decoder_breadth']
		self.predictor_breadth = config['predictor_breadth']

		# Set up networks, calculate number of params
		self.encoder = PointNetEncoder(dim_latent=self.dim_latent,
                                	breadth=self.encoder_breadth).to(device)
		self.decoder = SDFDecoder(dim_latent=self.dim_latent,
						   			breadth=self.decoder_breadth,
									device=device).to(device)
		self.predictor = CostPredictor(dim_latent=self.dim_latent,
								dim_hidden=self.predictor_breadth).to(device)
		print('Num of encoder parameters: %d' % sum(p.numel() for p in self.encoder.parameters() if p.requires_grad))
		print('Num of decoder parameters: %d' % sum(p.numel() for p in self.decoder.parameters() if p.requires_grad))
		print('Num of cost predictor parameters: %d' % sum(p.numel() for p in self.predictor.parameters() if p.requires_grad))

		# Use one GPU
		self.decoder_accessor = self.decoder
		self.predictor_accessor = self.predictor

		# Set up optimizer
		self.optimizer = torch.optim.AdamW([
						{'params': self.encoder.parameters(),
							'lr': config['encoder_lr'],
	 						'weight_decay': config['encoder_weight_decay']},
						{'params': self.decoder.parameters(),
							'lr': config['decoder_lr'],
							'weight_decay': config['decoder_weight_decay']},
						{'params': self.predictor.parameters(),
							'lr': config['predictor_lr'],
							'weight_decay': config['predictor_weight_decay']},
						])
		if config['decayLR_use']:
			self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
										self.optimizer,
										milestones=config['decayLR_milestones'],
										gamma=config['decayLR_gamma'])
		else:
			self.scheduler = None


	def create_dataset(self, env_dir_dict, embed_id_dir_dict,
					num_sdf_available_per_obj, num_sdf_per_obj,
					num_surface_per_obj, **kwargs):
		'''
		Create dataholder, to be updated once new distribution generated
		# num_sdf_available_per_obj: number of sdf points for each object available before downsampled
		# num_sdf_per_obj: number of sdf points for each object - target!
		# num_surface_per_obj: number of surface points for each object (for pointnet encoder)
		'''
		self.train_data = TrainDataset(env_dir_dict,
								 		embed_id_dir_dict,
										num_sdf_available_per_obj,
										num_sdf_per_obj,
										num_surface_per_obj,
										device='cpu')
		self.train_dataloader = torch.utils.data.DataLoader(
										self.train_data,
										batch_size=self.batch_size,
										shuffle=True,
										drop_last=True,
										pin_memory=True,
										num_workers=4)


	def embed(self, epoch, norm_loss_ratio, latent_all, label_all, num_sdf_per_obj, clamp_lip):
		"""
		Resets latent
		"""
		epoch_loss = 0
		epoch_rec_loss = 0
		epoch_reg_loss = 0
		epoch_lip_loss = 0
		num_batch = 0

		# Switch NN mode
		self.encoder.train()
		self.decoder.train()
		self.predictor.train()
		l2 = torch.nn.MSELoss(reduction='none')

		# Save all the predictions for debugging
		pred_all = np.empty((0))

		# Run batches
		for batch_ind, data_batch in enumerate(self.train_dataloader):

			# Zero gradient
			self.optimizer.zero_grad(set_to_none=True)

			######################  Extract data  ######################
			batch_sdf, batch_surface, batch_obj_id_chosen = data_batch
			batch_sdf = batch_sdf.reshape(-1,4).to(self.device)
			batch_sdf_values = batch_sdf[:,-1]
			batch_sdf_points = batch_sdf[:,:3]
			batch_surface = batch_surface.to(self.device)
			batch_obj_id_chosen = batch_obj_id_chosen.squeeze(0)

			######################  Encode  ######################
			batch_latent = self.encoder.forward(batch_surface)	# batch x latent

			######################  Decode  ######################
			batch_latent_all = batch_latent.repeat_interleave(num_sdf_per_obj, dim=0) 	# Assign latent to each point of the object
			batch_sdf_pred = self.decoder.forward(batch_sdf_points, batch_latent_all) # Decode each latent/point to get sdf predictions

			######################  Rec loss  ######################
			rec_loss = torch.mean((batch_sdf_pred - batch_sdf_values)**2)

			######################  Reg loss  ######################
			batch_reward_pred = self.predictor.forward(batch_latent).flatten()
			batch_label = torch.from_numpy(label_all[batch_obj_id_chosen]).float().to(self.device)
			reg_loss = torch.mean(l2(batch_reward_pred, batch_label))

			######################  Lip loss  ######################
			if clamp_lip is None:
				lip_loss = torch.linalg.norm(self.predictor_accessor.linear_hidden[0].weight, ord=2)+torch.linalg.norm(self.predictor_accessor.linear_out[0].weight, ord=2)	# spectral norm
			else:
				lip_loss = (torch.linalg.norm(self.predictor_accessor.linear_hidden[0].weight, ord=2)+torch.linalg.norm(self.predictor_accessor.linear_out[0].weight, ord=2)-clamp_lip*16)**2	# clamping

			# Add reconstruction and regularization losses together
			batch_loss = rec_loss+\
						self.config['reg_loss_ratio']*reg_loss+\
						self.config['lip_loss_ratio']*lip_loss+\
						norm_loss_ratio*torch.mean(batch_latent**2)

			# Backward pass to get gradients
			batch_loss.backward()

			# Clip gradient if specified
			if self.config['gradientClip_use']:
				torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.config['gradientClip_thres'])
				torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.config['gradientClip_thres'])
				torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), self.config['gradientClip_thres'])

			# Update weights using gradient
			self.optimizer.step()

			# Store loss
			epoch_loss += batch_loss.item()
			epoch_rec_loss += rec_loss.item()
			epoch_reg_loss += reg_loss.item()
			epoch_lip_loss += lip_loss.item()
			num_batch += 1

			# Update latents for all distributions
			latent_all[batch_obj_id_chosen] =batch_latent.detach().cpu().numpy()
			pred_all = np.concatenate((pred_all, batch_reward_pred.detach().cpu().numpy()))

		# Decay learning rate if specified
		if self.scheduler is not None:
			self.scheduler.step()

		# Get batch average loss
		epoch_loss /= num_batch
		epoch_rec_loss /= num_batch
		epoch_reg_loss /= num_batch
		epoch_lip_loss /= num_batch

		return epoch_loss, epoch_rec_loss, epoch_reg_loss, epoch_lip_loss, latent_all, pred_all


	def get_predictor_lip(self):
		return self.predictor_accessor.get_lip()


	def encode_batch(self, surface_batch):
		"""
		Assume the shape as N x num_surface_per_obj x 3
		"""
		surface_test = torch.from_numpy(surface_batch).float().to(self.device)
		latent_test = self.encoder.forward(surface_test)	# num_test_obj x latent_dim
		return latent_test


	def predict(self, latent):
		"""
		Using the cost predictor
		"""
		if isinstance(latent, np.ndarray):
			latent = torch.from_numpy(latent).float().to(self.device)

		with torch.no_grad():
			pred = self.predictor.forward(latent).detach().cpu()
		return pred.squeeze(1).numpy()
			# return torch.where(pred > 0.5, 1., 0.).numpy()


	def adversarial(self, latent, eta=1.0, gamma=1.0, steps=10, target_drop=0.0):
		"""
		Adversarially perturb latent using the cost predictor and evaluated label/cost. Following https://github.com/duchi-lab/certifiable-distributional-robustness/blob/master/attacks_tf.py
		Also see https://github.com/ricvolpi/generalize-unseen-domains/blob/master/model.py
		Only takes a single datapoint for now; tricky to get batch to work
		"""
		l2 = torch.nn.MSELoss()
		latent = torch.from_numpy(latent).float().to(self.device).requires_grad_().reshape(1,-1)
		latent_detach = latent.detach()

		# Gradient ascent
		max_num_itr = 10
		for _ in range(max_num_itr):

			# make a copy
			eta_env = eta
			gamma_env = gamma
			latent_adv = latent.clone()
			ini_pred_reward = self.predictor.forward(latent_adv)

			latent_path_all = np.zeros((steps+1, latent.shape[1]))
			latent_path_all[0] = latent_adv.detach().cpu().numpy()

			for step in range(steps):
				pred_reward = self.predictor.forward(latent_adv)	# reward
				loss = -pred_reward - gamma_env*l2(latent_adv, latent_detach)
				grad = torch.autograd.grad(loss, latent_adv)[0] # returns a tuple of grads
				latent_adv += eta_env*grad
				# logging.info(f'step {step}, pred {pred_reward.item()}')

				latent_path_all[step+1] = latent_adv.detach().cpu().numpy()

			if (ini_pred_reward-pred_reward) > target_drop*1.5:
				eta *= 0.8	# too much perturbation
				gamma *= 2.0
			elif (ini_pred_reward-pred_reward) > target_drop:
				break  # good
			else:
				eta *= 1.2 # too little perturbation
				gamma *= 0.5
		return latent_adv.detach().cpu().numpy(), latent_path_all


	def generate(self, epoch, gen_dir, base_latent_all, eta, gamma, steps, target_drop=0.1, max_num_attempt=5):
		"""
		Generate new objects by adversarially perturbing existing latents using the cost predictor
		Sometimes some latent cannot generate new object, so we need to re-sample latent adversarially for the same new distribution
		"""
		num_new = len(base_latent_all)
		old_latent_all = base_latent_all
		new_latent_all = np.zeros((num_new, self.dim_latent))

		# Another attempt if not all objects processed
		flags = np.ones((num_new))
		height_all = np.zeros((num_new))
		keep_concave_part = config['keep_concave_part']
		for _ in range(max_num_attempt):
			for env_ind in range(num_new):

				# Skip if already generated
				if flags[env_ind] < 1:
					continue

				# Generate new
				old_latent = base_latent_all[env_ind]
				new_latent, latent_path_all = self.adversarial(
									latent=old_latent,
									eta=eta, gamma=gamma, steps=steps,
		 							target_drop=target_drop)

				# Get mesh using decoder, possibly corrupt
				old_mesh = self.decoder_accessor.get_mesh(torch.from_numpy(old_latent).float().to(self.device), voxel_resolution=self.voxel_resolution)
				new_mesh = self.decoder_accessor.get_mesh(torch.from_numpy(new_latent).float().to(self.device), voxel_resolution=self.voxel_resolution)
				if new_mesh is None or old_mesh is None:
					print('Cannot generate from latent!')
					continue

				# Try processing
				try:
					old_mesh = process_mesh(old_mesh, 
											scale_down=True, 
											smooth=False, #!
											random_scale=False)
					new_mesh = process_mesh(new_mesh, 
											scale_down=True, 
											smooth=False, #!
											random_scale=False)

					# Scale to original height
					new_mesh = match_mesh_height(new_mesh, old_mesh)

					# Export as decomposed stl and urdf - create new subdir for convex obj - for pybullet
					ensure_directory_hard(gen_dir + str(env_ind) + '/')
					convex_pieces = save_convex_urdf(new_mesh, 
											gen_dir, 
											env_ind,
											mass=0.1,
											keep_concave_part=keep_concave_part)
				except:
					print('Cannot process generated!')
					continue

				if len(convex_pieces) > 20:
					print('Too concave!')
					continue

				#? Use decompsoed parts as stl? avoid peculiarities when sampling sdf and causing reconstruction issue
				if keep_concave_part: # Export as (un-decomposed) stl - for sdf
					save_mesh = new_mesh
				else:
					save_mesh = create_mesh_from_pieces(convex_pieces)
				save_mesh.export(gen_dir+str(env_ind)+'.stl')

				# Add to all sampled dist; mark generated
				new_latent_all[env_ind] = new_latent
				flags[env_ind] = 0
				height_all[env_ind]=(save_mesh.bounds[1]-save_mesh.bounds[0])[2]

			# Quit if all objects perturbed
			if np.sum(flags) < 1e-3:
				break

			# Find closer latent
			eta /= 2
			gamma *= 2
			# steps = min(int(steps/2), 1)
			logging.info(f'Epoch {epoch} generate, double gamma locally')

		return old_latent_all, new_latent_all, flags, height_all


	def visualize(self, old_latent_all, new_latent_all, num_random_obj=20):
		"""
		Sample latent from all existing and visualize objects
		"""
		num_obj_generated = 0
		num_obj_attempt = 0
		obj_ind_all = random.sample(range(new_latent_all.shape[0]), k=num_random_obj)

		# Use subplots for all objects
		fig_obj, _ = plt.subplots(5, 4)	# assume 20 rn

		while num_obj_generated < num_random_obj:

			# Sample more if used up
			if num_obj_attempt >= num_random_obj:
				obj_ind_all = random.sample(range(new_latent_all.shape[0]), k=num_random_obj)
				num_obj_attempt = 0

			# Extract sample
			old_obj = old_latent_all[obj_ind_all[num_obj_attempt]]
			new_obj = new_latent_all[obj_ind_all[num_obj_attempt]]

			# Try
			num_obj_attempt += 1

			# Reconstruct mesh from latent
			old_mesh = self.decoder_accessor.get_mesh(torch.from_numpy(old_obj).float().to(self.device), voxel_resolution=self.voxel_resolution)
			new_mesh = self.decoder_accessor.get_mesh(torch.from_numpy(new_obj).float().to(self.device), voxel_resolution=self.voxel_resolution)
			if old_mesh is None or new_mesh is None:
				print('Cannot generate sample!')
				continue

			# Center, orient, scale
			try:
				old_mesh = process_mesh(old_mesh, 
                            			scale_down=True,
										smooth=False,
                               			random_scale=False)
				new_mesh = process_mesh(new_mesh, 
                            			scale_down=True,
										smooth=False,
                               			random_scale=False)
			except:
				print('Cannot process sampled!')
				continue

			# Save mesh for inspection - bot not decomposed
			if num_obj_generated < 5:
				old_mesh.export(self.latent_img_dir+str(epoch)+'_'+str(num_obj_generated)+'_old.stl')
				new_mesh.export(self.latent_img_dir+str(epoch)+'_'+str(num_obj_generated)+'_new.stl')

			# Predict
			old_reward =self.predict(latent=old_obj.reshape(1,-1))[0]
			new_reward =self.predict(latent=new_obj.reshape(1,-1))[0]

			# Save image of 2D cross section
			slice_2D_old, _ = old_mesh.section(plane_origin=old_mesh.centroid,
										plane_normal=[0,0,1]).to_planar()
			slice_2D_new, _ = new_mesh.section(plane_origin=new_mesh.centroid,
										plane_normal=[0,0,1]).to_planar()
			ax = fig_obj.axes[num_obj_generated]
			ax.set_aspect('equal')
			ax.scatter(slice_2D_old.vertices[:,0], slice_2D_old.vertices[:,1],
			  			s=1,color='lightgray')
			ax.scatter(slice_2D_new.vertices[:,0], slice_2D_new.vertices[:,1],
			  			s=2,color='gray')
			ax.text(x=0., y=0.01, s="{:.2f}".format(old_reward), fontsize=12, color='coral')
			ax.text(x=0., y=-0.01, s="{:.2f}".format(new_reward), fontsize=12, color='red')
			ax.axis('off')

			# Count
			num_obj_generated += 1

		plt.savefig(self.latent_img_dir+str(epoch)+'_random_obj.png')
		plt.close()


	def save_model(self, dir):
		torch.save(self.encoder.state_dict(), dir+'encoder.pt')
		torch.save(self.decoder.state_dict(), dir+'decoder.pt')
		torch.save(self.predictor.state_dict(), dir+'predictor.pt')


	def load_model(self, dir):
		self.encoder.load_state_dict(torch.load(dir+'encoder.pt', map_location=self.device))
		self.decoder.load_state_dict(torch.load(dir+'decoder.pt', map_location=self.device))
		self.predictor.load_state_dict(torch.load(dir+'predictor.pt', map_location=self.device))


def get_non_test_num_env_list(env_dict, dir_type_all=[INIT_TYPE, GEN_TYPE]):
	l = []
	for env_id_list, _, dir_type in env_dict.values():
		if dir_type in dir_type_all:
			l += [len(env_id_list)]
	return l


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
	torch.backends.cudnn.benchmark = True	# may speed up

	# Hardware
	cuda_idx = config['cuda_idx']
	device = 'cuda:'+str(cuda_idx)
	# Misc
	num_eval_per_env = config['num_eval_per_env']
	dim_latent = config['dim_latent']
	norm_loss_ratio = config['norm_loss_ratio']
	clamp_lip = config['clamp_lip']

	# Data
	initial_env_dir_list = config['initial_env_dir_list']
	num_env_per_initial_dir = config['num_env_per_initial_dir']
	test_env_dir_list = config['test_env_dir_list']
	num_env_per_test_dir = config['num_env_per_test_dir']

	# Generation (from latent)
	num_epoch_per_gen = config['num_epoch_per_gen']
	num_epoch_before_first_gen = config['num_epoch_before_first_gen']
	num_env_per_gen = config['num_env_per_gen']

	# Improving policy
	num_env_per_retrain = config['num_env_per_retrain']
	num_epoch_per_retrain = config['num_epoch_per_retrain']
	num_epoch_before_first_retrain = config['num_epoch_before_first_retrain']
	mu_list = config['mu_list']
	mu = config['mu']
	sigma = config['sigma']
	retrain_args = config['retrain_args']
	eval_args = config['eval_args']

	# Adversarial (gradient ascent)
	eta = config['eta']
	gamma = config['gamma']
	ga_steps = config['ga_steps']
	target_drop_percentage = config['target_drop_percentage']
	target_drop_percentage_rate = config['target_drop_percentage_rate']

	# Env params
	sdf_args = config['sdf_args']

	# Initialize folders
	data_parent_dir = config['data_parent_dir']
	result_dir = 'result/'+yaml_file_name+'/'
	model_dir = result_dir + 'runner_model/'
	latent_img_dir = result_dir + 'latent_img/'
	data_dir = data_parent_dir+yaml_file_name+'/'
	ensure_directory(result_dir)
	ensure_directory(model_dir)
	ensure_directory(latent_img_dir)
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

	# Initialize running env
	runner = Runner(yaml_path=yaml_path, result_dir=result_dir, device=device)

	# Initialize point sampler
	point_sampler = PointSampler(**sdf_args)

	# Training details to be recorded
	train_loss_list = []
	train_rec_loss_list = []
	train_reg_loss_list = []
	train_lip_loss_list = []
	train_success_list = []
	test_success_list = []
	train_lip_list = []

	# Save the latent and (groun-truth) label/reward of all images
	latent_all = np.zeros((num_env_per_initial_dir*len(initial_env_dir_list),
							dim_latent))

	# Add test dir to dict
	for env_dir in test_env_dir_list:
		height_all = list(np.load(env_dir+'dim.npy')[:num_env_per_test_dir,2])
		env_dir_dict[env_dir] = ([*range(num_env_per_test_dir)], height_all, TEST_TYPE)

	# Name of saved training details
	train_details_path = None

	# Initialize counter
	num_epoch_since_last_gen = 0
	num_epoch_since_last_retrain = 0
	num_env_gen = 0
	num_dir_gen = 0
	num_retrain = 0

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
	num_epoch = (config['num_retrain']-2)*num_epoch_per_retrain+num_epoch_before_first_retrain	# minus 2 to account for retrain at epoch 0
	epoch = 0
	while epoch <= num_epoch:

		# Record time for each epoch
		epoch_start_time = time.time()

		######################### New #########################

		# Generate a new distribution every some epochs
		if epoch >= num_epoch_before_first_gen and \
			num_epoch_since_last_gen >= num_epoch_per_gen:

			# Declare new path
			new_gen_dir = data_dir + 'gen_' + str(num_dir_gen) + '/'
			ensure_directory(new_gen_dir)

			# Adversarially generate and save new envs - Note that not all latent are updated during last embedding since only a set of envs are embedded now
			#? Favor sampling old envs with higher reward - prevent too difficult envs generated - not all envs re-evaluated in later stage of training, so less likely to sample them, but should be ok
			print('Generating new...')
			old_env_weights_all = np.exp(label_all*0)	# uniform weight
			old_env_weights_all /= np.sum(old_env_weights_all)
			adv_env_id_all, _ = weighted_sample_without_replacement([*range(len(label_all))], weights=old_env_weights_all, k=min(num_env_per_gen, len(label_all)))

			# Estimate the range of predictions
			pred_range = np.max(pred_all)-np.min(pred_all)
			target_drop = pred_range*target_drop_percentage

			# Save hist
			fig = plt.figure()
			plt.hist(pred_all, bins=np.linspace(0.0, 1.0, 20))
			plt.savefig(latent_img_dir+str(epoch)+'_pred_hist.png')
			plt.close(fig)

			# Perturb sampled latent adversarially
			old_latent, new_latent, flags, height_all = runner.generate(
       								epoch=epoch,
									gen_dir=new_gen_dir,
									base_latent_all=latent_all[adv_env_id_all],
									eta=eta, gamma=gamma, steps=ga_steps,
		 							target_drop=target_drop)

			# Filter ones actually generated
			new_env_id_list = np.where(flags<1)[0]
			old_latent_generated = old_latent[new_env_id_list]
			new_latent_generated = new_latent[new_env_id_list]

			# Sample surface points and sdf, 4 mins for each 100 objects
			print('Sampling surface points and sdf for newly generated...')
			point_sampler.reset_dir(directory=new_gen_dir)
			point_sampler.sample_new_surface_point_sdf(obj_id_list=new_env_id_list)

			# Evaluate label of new envs - use mu_list
			print('Evaluating newly generated...')
			mu_batch = array(evaluator.evaluate(obj_dir=new_gen_dir,
								obj_id_list=new_env_id_list,
								obj_height_list=height_all[new_env_id_list],
								num_eval=num_eval_per_env)[1], dtype='float')
			label_batch = get_label_from_mu(mu_batch, mu_list)
			print('Reward of newly generated: ', np.mean(label_batch))
			logging.info(f'Reward of newly generated: {np.mean(label_batch)}')

			# Add to latent - keep ungenerated ones - do not add to label here since labels reset after retraining
			latent_all = np.concatenate((latent_all, new_latent))

			# Add to dir dict - keep un-generated ones in height_all
			env_dir_dict[new_gen_dir] = (new_env_id_list, list(height_all), GEN_TYPE)
			np.save(new_gen_dir+'height.npy', height_all)

			# Visualize newly generated
			runner.visualize(old_latent_generated, new_latent_generated, num_random_obj=20)

			# Reset epoch count
			num_epoch_since_last_gen = 0
			num_env_gen += len(new_env_id_list)
			num_dir_gen += 1

		######################### Retrain #########################

		# Retrain using all existing images
		if epoch == 0 or (epoch >= num_epoch_before_first_retrain and \
			num_epoch_since_last_retrain >= num_epoch_per_retrain):

			print(f'Retraining policy {num_retrain}...')
			logging.info(f'Retraining policy {num_retrain}...')

			# Pick which envs for training
			retrain_env_path_available_all = []
			retrain_env_height_available_all = []
			retrain_env_weight_all = []
			gen_dir_count = 0
			for env_dir, (env_id_list, height_list, dir_type) in env_dir_dict.items():
				if dir_type != TEST_TYPE:
					retrain_env_path_available_all += [env_dir+str(id)+'.urdf' for id in env_id_list]
					retrain_env_height_available_all += list(array(height_list)[env_id_list])
				if dir_type == INIT_TYPE:
					retrain_env_weight_all += [1]*len(env_id_list)
				elif dir_type == GEN_TYPE:
					retrain_env_weight_all += [1]*len(env_id_list)
					gen_dir_count += 1
			retrain_env_weight_all = array(retrain_env_weight_all)/np.sum(array(retrain_env_weight_all))	# uniform weight for now
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

			# Sample envs to be embedded in the next iteration	#! use all!
			embed_id_dir_dict = {}
			for env_dir, (env_id_list, _, dir_type) in env_dir_dict.items():
				if dir_type == INIT_TYPE or dir_type == GEN_TYPE:
					embed_id_dir_dict[env_dir] = env_id_list

			label_all = np.empty((0), dtype='float')	# reset
			train_success_batch = np.empty((0), dtype='float')
			train_success_dirs = []
			print('Re-evaluating for all...')
			# INIT - eval for train_success and label
			for env_dir, (env_id_list, height_list, dir_type) in env_dir_dict.items():
				if dir_type == INIT_TYPE:
					mu_batch = array(evaluator.evaluate(obj_dir=env_dir,
									obj_id_list=env_id_list,
									obj_height_list=height_list,	# all
								num_eval=num_eval_per_env)[1], dtype='float')
					label_batch = get_label_from_mu(mu_batch, mu_list)
					train_success_batch = np.concatenate((train_success_batch, label_batch))
					train_success_dirs += [np.mean(label_batch)]
					label_all = np.concatenate((label_all, label_batch))

   			# GEN - eval for label for chosen ids
			for env_dir, (_, height_list, dir_type) in env_dir_dict.items():
				if dir_type == GEN_TYPE:
					chosen_id_list = embed_id_dir_dict[env_dir]
					label_batch = np.zeros(len(height_list))
					mu_batch_chosen = array(evaluator.evaluate(obj_dir=env_dir,
									obj_id_list=chosen_id_list,
									obj_height_list=list(array(height_list)[chosen_id_list]),	# chosen ones
								num_eval=num_eval_per_env)[1], dtype='float')
					label_batch_chosen = get_label_from_mu(mu_batch_chosen,
															mu_list)
					label_batch[chosen_id_list] = label_batch_chosen
					label_all = np.concatenate((label_all, label_batch))

			# TEST - eval for test_success
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

			# Reset epoch count
			num_epoch_since_last_retrain = 0
			num_retrain += 1

		######################### Embed #########################

		# Reset dataset and dataloader to add the new distribution
		if num_epoch_since_last_retrain == 0:
			runner.create_dataset(env_dir_dict, embed_id_dir_dict, **sdf_args)

		# Embed
		epoch_loss, epoch_rec_loss, epoch_reg_loss, epoch_lip_loss, latent_all, pred_all = runner.embed(epoch=epoch,
						norm_loss_ratio=norm_loss_ratio,
						latent_all=latent_all,
						label_all=label_all,
						num_sdf_per_obj=sdf_args['num_sdf_per_obj'],
						clamp_lip=clamp_lip)

		# Get Lipschitz constant of the cost predictor
		lip_predictor = runner.get_predictor_lip()

		########################  Record  ########################

		train_loss_list += [epoch_loss]
		train_rec_loss_list += [epoch_rec_loss]
		train_reg_loss_list += [epoch_reg_loss]
		train_lip_loss_list += [epoch_lip_loss]
		train_lip_list += [lip_predictor]

		# Debug
		epoch_duration = time.time() - epoch_start_time
		if epoch % config['num_epoch_per_loss_print'] == 0:
			print("Epoch {:d}, Loss: {:.8f}, Rec loss: {:.5f}, Reg loss: {:.5f}, Lip loss: {:.3f}; Time: {:.3f}".format(epoch, epoch_loss, epoch_rec_loss, epoch_reg_loss, epoch_lip_loss, epoch_duration))
		if epoch % config['num_epoch_per_loss_log'] == 0:
			logging.info(f'Epoch {epoch}, Rec loss: {epoch_rec_loss:.6f}, Reg loss: {epoch_reg_loss:.6f}, Lip loss: {epoch_lip_loss:.6f}')

		# Save model before advancing epoch count
		if (epoch % config['num_epoch_per_save_model'] == 0 or epoch==num_epoch) and epoch >= 0:
			runner.save_model(dir=model_dir)

			# Remove old
			if train_details_path is not None:
				os.remove(train_details_path)
			train_details_path = result_dir+'train_details_'+str(epoch)

			# Save training details
			torch.save({
				'epoch': epoch,
				'optimizer_state_dict': runner.optimizer.state_dict(),
				'train_lists': [train_loss_list, train_rec_loss_list, train_reg_loss_list, train_lip_loss_list, train_success_list, test_success_list, latent_all, label_all], # reward_all
				"seed_data": (seed, random.getstate(), np.random.get_state(), torch.get_rng_state()),
				"num_data": [num_epoch_since_last_gen, num_env_gen, num_dir_gen, env_dir_dict],
				}, train_details_path)

		# Count
		num_epoch_since_last_gen += 1
		num_epoch_since_last_retrain += 1
		epoch += 1

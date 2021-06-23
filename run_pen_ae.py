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
import matplotlib.pyplot as plt
from PIL import Image

from src import INIT_TYPE, TEST_TYPE, GEN_TYPE
from src.image_ae import Encoder, Decoder
from src.cost_predictor import CostPredictor
from src.dataset_pendulum import TrainDataset
from train_pen import TrainPendulum
from eval_pen import EvaluatePendulum
from util.misc import *


class Runner:

	def __init__(self, yaml_path, result_dir, device):
		save__init__args(locals())
		self.model_dir = result_dir + 'model/'
		self.latent_img_dir = result_dir + 'latent_img/'

		# Configure from yaml file
		with open(yaml_path+'.yaml', 'r') as f:
			config = yaml.load(f, Loader=yaml.FullLoader)
		self.config = config

		# always be one because of dataset design
		self.batch_size = config['batch_size']

		# NN params
		self.dim_img = config['dim_img']
		self.dim_latent = config['dim_latent']
		self.inner_channels = config['inner_channels']
		self.predictor_breadth = config['predictor_breadth']

		# Set up networks, calculate number of params
		self.encoder = Encoder(dim_latent=self.dim_latent, 
								dim_img=self.dim_img,
								variational=False,
								inner_channels=self.inner_channels,
        						mlp_down_factor=config['mlp_factor'],
              					num_layer=config['num_layer']).to(device)
		self.decoder = Decoder(dim_latent=self.dim_latent, 
								dim_img=self.dim_img,
								inner_channels=self.inner_channels,
        						use_upsampling=config['use_upsampling'],
								mlp_up_factor=config['mlp_factor'],
              					interp_mode=config['interp_mode'],
              					num_layer=config['num_layer']).to(device)
		# self.decoder = Decoder_MLP(dim_latent=self.dim_latent, 
								# dim_img=self.dim_img).to(device)
		self.predictor = CostPredictor(dim_latent=self.dim_latent,
								dim_hidden=self.predictor_breadth).to(device)
		print('Num of encoder parameters: %d' % sum(p.numel() for p in self.encoder.parameters() if p.requires_grad))
		print('Num of decoder parameters: %d' % sum(p.numel() for p in self.decoder.parameters() if p.requires_grad))
		print('Num of cost predictor parameters: %d' % sum(p.numel() for p in self.predictor.parameters() if p.requires_grad))

		# Use only use 1 GPU for now
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


	def create_dataset(self, env_dir_dict, embed_id_dir_dict):
		'''
		Create dataholder, to be updated once new distribution generated
		'''
		self.train_data = TrainDataset(env_dir_dict,
                                 		embed_id_dir_dict, 
                                   		device='cpu')
		self.train_dataloader = torch.utils.data.DataLoader(
										self.train_data,
										batch_size=self.batch_size,
										shuffle=True,
										drop_last=True,
										pin_memory=True,
										num_workers=4)


	def embed(self, epoch, norm_loss_ratio, latent_all, label_all, clamp_lip):
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

		# Save all the predictions for debugging
		pred_all = np.empty((0))

		# Run batches
		for _, data_batch in enumerate(self.train_dataloader):

			# Zero gradient, and apply AMP to forward pass
			self.optimizer.zero_grad(set_to_none=True)

			# Extract data
			batch_img, batch_img_id_chosen = data_batch
			batch_img = batch_img.to(self.device)
			batch_img_id_chosen = batch_img_id_chosen.squeeze(0)

			# Encode
			batch_latent = self.encoder.forward(batch_img)	# batch x latent

			# Decode
			batch_rec = self.decoder.forward(batch_latent)

			# Reconstruction loss
			rec_loss = torch.mean((batch_rec - batch_img)**2)

			# Regularization loss (label)
			batch_reward_pred = self.predictor.forward(batch_latent).flatten()
			batch_label = torch.from_numpy(label_all[batch_img_id_chosen]).float().to(self.device)
			reg_loss = torch.mean((batch_reward_pred-batch_label)**2)

			# Lipschitz loss
			if clamp_lip is None:
				lip_loss = torch.linalg.norm(self.predictor_accessor.linear_hidden[0].weight, ord=2)*torch.linalg.norm(self.predictor_accessor.linear_out[0].weight, ord=2)/16	# spectral norm and two sigmoids
			else:
				lip_loss = (torch.linalg.norm(self.predictor_accessor.linear_hidden[0].weight, ord=2)*torch.linalg.norm(self.predictor_accessor.linear_out[0].weight, ord=2)/16-clamp_lip)**2	# clamping

			# Norm loss - constraining latent to be close to zero
			norm_loss = torch.mean(batch_latent**2)

			# Add together
			batch_loss = rec_loss+\
						self.config['reg_loss_ratio']*reg_loss+\
						self.config['lip_loss_ratio']*lip_loss+\
						norm_loss_ratio*norm_loss

			# Backward pass to get gradients
			batch_loss.backward()

			# Update weights using gradient
			self.optimizer.step()

			# Store loss
			epoch_loss += batch_loss.item()
			epoch_rec_loss += rec_loss.item()
			epoch_reg_loss += reg_loss.item()
			epoch_lip_loss += lip_loss.item()
			num_batch += 1

			# Update latents for all distributions
			latent_all[batch_img_id_chosen]= batch_latent.detach().cpu().numpy()
			pred_all = np.concatenate((pred_all, batch_reward_pred.detach().cpu().numpy()))

		# Switch NN mode
		self.encoder.eval()
		self.decoder.eval()
		self.predictor.eval()

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


	def get_image_from_latent(self, latent):
		img_latent = torch.from_numpy(latent).float().to(self.device).view(1,-1)
		img_arr = self.decoder.forward(img_latent).squeeze(0).detach().cpu().numpy()
		return img_arr


	def encode_dir(self, img_dir, img_id_list):
		img_batch = []
		for id in img_id_list:
			img = Image.open(img_dir+str(id)+'.png')
			img_batch.append(array(img)/255)
		img_batch = np.moveaxis(array(img_batch), -1, 1)	# NHWC to NCHW
		img_batch = torch.from_numpy(img_batch).float().to(self.device)
		latent_batch = self.encoder.forward(img_batch)
		return latent_batch


	def predict(self, latent):
		if isinstance(latent, np.ndarray):
			latent = torch.from_numpy(latent).float().to(self.device)
		if latent.dim() == 1:
			latent = latent.reshape(1,-1)
		with torch.no_grad():
			pred = self.predictor.forward(latent).detach().cpu()
		return pred.squeeze(1).numpy()


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
		max_num_itr = 20
		for _ in range(max_num_itr):

			# make a copy
			eta_env = eta
			gamma_env = gamma
			latent_adv = latent.clone()
			ini_pred_reward = self.predictor.forward(latent_adv)

			# Modify target drop acccordining to initial prediction - drop less if smaller reward - not using now
			# target_drop_env = target_drop*math.exp(ini_pred_reward-1)
			target_drop_env = target_drop

			latent_path_all = np.zeros((steps+1, latent.shape[1]))
			latent_path_all[0] = latent_adv.detach().cpu().numpy()

			for step in range(steps):
				pred_reward = self.predictor.forward(latent_adv)	# reward
				loss = -pred_reward - gamma_env*l2(latent_adv, latent_detach)
				grad = torch.autograd.grad(loss, latent_adv)[0] # returns a tuple of grads
				latent_adv += eta_env*grad
				# logging.info(f'step {step}, pred {pred_reward.item()}')

				latent_path_all[step+1] = latent_adv.detach().cpu().numpy()

			if (ini_pred_reward-pred_reward) > target_drop_env*1.2:	# was 1.5
				eta *= 0.9	# too much perturbation
				gamma *= 1.25
			elif (ini_pred_reward-pred_reward) > target_drop_env:
				break  # right amount of drop in predicted reward
			else:
				eta *= 1.1 # too little perturbation
				gamma *= 0.75
		# logging.info(f'final eta {eta}, gamma {gamma}')
		return latent_adv.detach().cpu().numpy(), latent_path_all


	def generate(self, epoch, new_dir, base_latent_all, 
										eta, gamma, steps, target_drop=0.0):
		"""
		Generate new images by adversarially perturbing existing latents using the cost predictor
		Not parallelized
		"""
		num_new = len(base_latent_all)
		old_latent = base_latent_all
		new_latent = np.zeros((num_new, self.dim_latent))

		logging.info(f'Epoch {epoch} generate, eta {eta}, gamma {gamma}, steps {steps}, target drop: {target_drop}')
		for ind in range(num_new):
			olg_img_latent = base_latent_all[ind]

			# Generate new
			new_img_latent, latent_path_all = self.adversarial(latent=olg_img_latent, eta=eta, gamma=gamma, steps=steps, target_drop=target_drop)

			if ind < 10:
				fig_img, _ = plt.subplots(1, steps+1, figsize=(50,8))
				for sub_ind, latent in enumerate(latent_path_all):
					ax = fig_img.axes[sub_ind]
					img = self.get_image_from_latent(latent)
					reward = self.predict(latent=latent)[0]
					ax.set_aspect('equal')
					ax.imshow(np.moveaxis(img, 0, -1))
					ax.text(x=0.0, y=-5, s="{:.4f}".format(reward), fontsize=30, color='coral')
					ax.axis('off')
				fig_img.tight_layout()
				plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.15)
				plt.savefig(self.latent_img_dir+str(epoch)+'_'+str(ind)+'_path.png')
				plt.close()

			# Get image
			img_arr = self.get_image_from_latent(new_img_latent)
			img = Image.fromarray(np.moveaxis(np.uint8(img_arr*255), 0, -1))	# uint8, CHW to HWC
			img.save(new_dir+str(ind)+'.png')

			# Save old one too
			img_arr = self.get_image_from_latent(olg_img_latent)
			img = Image.fromarray(np.moveaxis(np.uint8(img_arr*255), 0, -1))	# uint8, CHW to HWC
			img.save(new_dir+str(ind)+'_old.png')

			# Add to all sampled dist; mark generated
			new_latent[ind] = new_img_latent

		return old_latent, new_latent


	def visualize(self, old_latent_all, new_latent_all, num_random_img):
		"""
		Sample latent from all existing and visualize images
		"""
		num_img_generated = 0
		num_img_attempt = 0
		img_ind_all = random.sample(range(new_latent_all.shape[0]), k=num_random_img)

		# Use subplots for all images
		fig_img, _ = plt.subplots(3, num_random_img, 
                            	figsize=(20*int(num_random_img/5), 20))

		while num_img_generated < num_random_img:

			# Sample more if used up
			if num_img_attempt >= num_random_img:
				img_ind_all = random.sample(range(new_latent_all.shape[0]), k=num_random_img)
				num_img_attempt = 0

			# Extract sample
			old_img_latent = old_latent_all[img_ind_all[num_img_attempt]]
			new_img_latent = new_latent_all[img_ind_all[num_img_attempt]]
			old_img_arr = self.get_image_from_latent(old_img_latent)
			new_img_arr = self.get_image_from_latent(new_img_latent)	
			diff = np.moveaxis(new_img_arr-old_img_arr, 0, -1)	# HWC
			def rgb2gray(rgb):
				return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
			diff_grey = rgb2gray(diff)
			diff_grey = (diff_grey-np.min(diff_grey))/(np.max(diff_grey)-np.min(diff_grey)) # normalize

			# Try
			num_img_attempt += 1

			# Predict
			old_reward = self.predict(latent=old_img_latent)[0]
			new_reward = self.predict(latent=new_img_latent)[0]

			# Show old and new image
			ax = fig_img.axes[num_img_generated]
			ax.set_aspect('equal')
			ax.imshow(np.moveaxis(old_img_arr, 0, -1))
			ax.text(x=0.0, y=-5, s="{:.3f}".format(old_reward), fontsize=50, color='coral')
			ax.axis('off')

			ax = fig_img.axes[num_img_generated+num_random_img]	# 2nd row
			ax.set_aspect('equal')
			ax.imshow(np.moveaxis(new_img_arr, 0, -1))
			ax.text(x=0.0, y=-5, s="{:.3f}".format(new_reward), fontsize=50, color='red')
			ax.axis('off')

			# Apply color map to diff
			ax = fig_img.axes[num_img_generated+2*num_random_img]	# 3nd row
			ax.set_aspect('equal')
			ax.imshow(diff_grey, cmap='jet')	# CHW to HWC
			ax.axis('off')

			# Count
			num_img_generated += 1

		fig_img.tight_layout()
		plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.15)
		plt.savefig(self.latent_img_dir+str(epoch)+'_random_img.png')
		plt.close()


	def save_model(self, dir):
		torch.save(self.encoder.state_dict(), dir+'encoder.pt')
		torch.save(self.decoder.state_dict(), dir+'decoder.pt')
		torch.save(self.predictor.state_dict(), dir+'predictor.pt')


	def load_model(self, dir):
		self.encoder.load_state_dict(torch.load(dir+'encoder.pt', map_location=self.device))
		self.decoder.load_state_dict(torch.load(dir+'decoder.pt', map_location=self.device))
		self.predictor.load_state_dict(torch.load(dir+'predictor.pt', map_location=self.device))


def choose_retrain_env(env_dir_dict):
	retrain_env_path_list = []
	for env_dir, (env_id_list, dir_type) in env_dir_dict.items():
		if dir_type != TEST_TYPE:
			retrain_env_path_list += [env_dir+str(id)+'.png' for id in env_id_list]
	return retrain_env_path_list	# choose all


def choose_embed_env(env_dir_dict):
	# use all initial dirs first
	embed_id_dir_dict = {}
	for env_dir, (env_id_list, dir_type) in env_dir_dict.items():
		if dir_type == INIT_TYPE or dir_type == GEN_TYPE:
			embed_id_dir_dict[env_dir] = env_id_list
	return embed_id_dir_dict


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
	torch.backends.cudnn.benchmark = True

	# Hardware
	num_cpus = config['num_cpus']
	cpu_offset = config['cpu_offset']
	cuda_idx = config['cuda_idx']
	device = 'cuda:'+str(cuda_idx)

	# Misc
	num_frame_stack = config['num_frame_stack']
	label_normalization = config['label_normalization']
	num_eval_per_env = config['num_eval_per_env']
	dim_obs = config['dim_obs']
	dim_latent = config['dim_latent']
	norm_loss_ratio = config['norm_loss_ratio']
	clamp_lip = config['clamp_lip']

	# Data
	initial_env_dir_list = config['initial_env_dir_list']
	num_env_per_initial_dir = config['num_env_per_initial_dir']
	test_env_dir_list = config['test_env_dir_list']
	num_env_per_test_dir = config['num_env_per_test_dir']
	num_env_per_gen = config['num_env_per_gen']

	# Improving policy
	num_epoch_per_retrain = config['num_epoch_per_retrain']
	num_epoch_before_first_retrain = config['num_epoch_before_first_retrain']
	retrain_args = config['retrain_args']
	pi_args = config['pi_args']
	q_args = config['q_args']
	reuse_replay_buffer = config['reuse_replay_buffer']

	# Adversarial (gradient ascent)
	eta = config['eta']
	gamma = config['gamma']
	ga_steps = config['ga_steps']
	target_drop_percentage = config['target_drop_percentage']
	target_drop_percentage_rate = config['target_drop_percentage_rate']

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
								low_init=1,
        						normalization=label_normalization,
								num_frame_stack=num_frame_stack,
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
							val_low_init=1,
							num_frame_stack=num_frame_stack,
							num_eval_per_env=num_eval_per_env,
							reuse_replay_buffer=reuse_replay_buffer,
							check_running=False,
							cuda_idx=cuda_idx)

	# Initialize running env
	runner = Runner(yaml_path=yaml_path, result_dir=result_dir, device=device)

	# Training details to be recorded
	train_loss_list = []
	train_rec_loss_list = []
	train_reg_loss_list = []
	train_lip_loss_list = []
	train_success_list = []
	aug_success_list = []
	test_success_list = []
	train_lip_list = []

	# Add test dir to dict
	for env_dir in test_env_dir_list:
		env_dir_dict[env_dir] = ([*range(num_env_per_test_dir)], TEST_TYPE)

	# Initialize latent and label
	num_initial_env = num_env_per_initial_dir*len(initial_env_dir_list)
	latent_all = np.zeros((num_initial_env, dim_latent))
	label_all = np.zeros((num_initial_env, dim_latent))

	# Name of saved training details
	train_details_path = None

	# Initialize counter
	num_epoch_since_last_gen = 0
	num_epoch_since_last_retrain = 0
	num_env_gen = 0
	num_dir_gen = 0
	num_retrain = 0
	epoch = 0

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
	num_epoch = (config['num_retrain']-2)*num_epoch_per_retrain+num_epoch_before_first_retrain	# minus 2 to account for retrain at epoch 0
	while epoch <= num_epoch:	# thus gen and retrain at the end

		# Record time for each epoch
		epoch_start_time = time.time()

		######################### Generate #########################
		if epoch >= num_epoch_before_first_retrain and \
			num_epoch_since_last_gen >= num_epoch_per_retrain:

			# Declare new path
			new_gen_dir = data_dir + 'gen_' + str(num_dir_gen) + '/'
			ensure_directory(new_gen_dir)

			# Adversarially generate and save new envs - always generate the number requested - not the case in grasping - Note that not all latent are updated during last embedding since only a set of envs are embedded now
			print('Generating new...')
			old_env_weights_all = np.exp(0)	# uniform weights
			old_env_weights_all /= np.sum(old_env_weights_all)
			adv_env_id_all, _ = weighted_sample_without_replacement([*range(len(label_all))], weights=old_env_weights_all, k=min(num_env_per_gen, len(label_all)))

			# Save hist
			fig = plt.figure()
			n, bins, patches = plt.hist(label_all[adv_env_id_all], bins=np.linspace(0.0, 1.0, 50))
			plt.xticks(bins)
			plt.savefig(latent_img_dir+str(epoch)+'_adv_label_hist.png')
			plt.close(fig)

			# Estimate the range of predictions
			pred_range = np.max(pred_all)-np.min(pred_all)
			target_drop = pred_range*target_drop_percentage

			# Generate
			old_latent, new_latent = runner.generate(epoch=epoch,
									new_dir=new_gen_dir,
									base_latent_all=latent_all[adv_env_id_all],
									eta=eta, 
									gamma=gamma, 
									steps=ga_steps,
									target_drop=target_drop)
			num_env_gen_batch = len(adv_env_id_all)
			new_env_id_list = np.arange(min(num_env_per_gen, num_env_gen_batch))

			# Evaluate policy reward
			print('Evaluating newly generated...')
			label_batch = evaluator.evaluate(img_dir=new_gen_dir,
											img_id_list=new_env_id_list)
			print('Reward of newly generated: ', np.mean(label_batch))
			logging.info(f'Reward of newly generated: {np.mean(label_batch)}')

			# Add to latent
			latent_all = np.concatenate((latent_all, new_latent))
			label_all = np.concatenate((label_all, label_batch))

			# Add to dir dict
			env_dir_dict[new_gen_dir] = (new_env_id_list, GEN_TYPE)

			# Visualize newly generated
			runner.visualize(old_latent, new_latent, 
								num_random_img=config['num_new_img_vis'])

			# Increase norm loss ratio each gen / clamp lip
			target_drop_percentage *= target_drop_percentage_rate

			# Reset epoch count
			num_epoch_since_last_gen = 0
			num_env_gen += num_env_gen_batch
			num_dir_gen += 1

		######################### Retrain #########################

		# Retrain using all existing images
		if epoch == 0 or (epoch >= num_epoch_before_first_retrain and \
			num_epoch_since_last_retrain >= num_epoch_per_retrain):

			print(f'Retraining policy {num_retrain}...')
			logging.info(f'Retraining policy {num_retrain}...')

			# Pick which envs for training - #! choose all
			retrain_env_path_list = choose_retrain_env(env_dir_dict)

			# Use more itrs at first/last retrain
			retrain_args_copy = dict(retrain_args)	# make a copy
			retrain_args_copy.pop('num_itr_initial', None)
			retrain_args_copy.pop('num_itr_final', None)
			if epoch == 0:
				retrain_args_copy['num_itr'] = retrain_args['num_itr_initial']
			elif epoch == num_epoch:
				retrain_args_copy['num_itr'] = retrain_args['num_itr_final']

			new_policy_path = trainer.train(
									train_img_path_all=retrain_env_path_list,
									prefix='epoch_'+str(epoch),
									**retrain_args_copy)
			logging.info(f'Epoch {epoch} retrain, new policy {new_policy_path}')

			# Update policy for evaluator and trainer
			trainer.update_policy_path(new_policy_path)
			evaluator.update_policy_path(new_policy_path)

			# Check if using initial policy
			flag_policy_improved = True
			if 'initial' in new_policy_path:
				flag_policy_improved = False

		######################### Re-evaluate #########################

			# Sample envs to be embedded in the next iteration - #! USE ALL!
			embed_id_dir_dict = choose_embed_env(env_dir_dict)

			print('Re-evaluating for all...')
			label_all_prev = label_all	# make a copy
			label_all = np.empty((0), dtype='float')	# reset
			train_success_batch = np.empty((0), dtype='float')
			aug_success_batch = np.empty((0), dtype='float')

			# INIT - eval for train_success and label
			for env_dir, (env_id_list, dir_type) in env_dir_dict.items():
				if dir_type == INIT_TYPE:
					if not flag_policy_improved:	# reuse prev labels
						label_batch = label_all_prev[:len(env_id_list)]
					else:
						label_batch = evaluator.evaluate(img_dir=env_dir,
														img_id_list=env_id_list)
					train_success_batch = np.concatenate((train_success_batch, label_batch))
					aug_success_batch = np.concatenate((aug_success_batch, label_batch))
					label_all = np.concatenate((label_all, label_batch))
					label_all_prev = label_all_prev[len(env_id_list):]	# remove some - assume in order of INIT/GEN/TEST

   			# GEN - eval for label for chosen ids
			for env_dir, (env_id_list, dir_type) in env_dir_dict.items():       
				if dir_type == GEN_TYPE:
					if not flag_policy_improved:	# reuse prev labels
						label_batch = label_all_prev[:len(env_id_list)]
					else:
						chosen_id_list = embed_id_dir_dict[env_dir]
						label_batch = np.zeros(len(env_id_list))
						label_batch_chosen = evaluator.evaluate(img_dir=env_dir,
													img_id_list=chosen_id_list)
						label_batch[chosen_id_list] = label_batch_chosen
					label_all = np.concatenate((label_all, label_batch))
					label_all_prev = label_all_prev[len(env_id_list):]	# remove some - assume in order of INIT/GEN/TEST
					aug_success_batch = np.concatenate((aug_success_batch, label_batch))

			# TEST - eval for test_success - assume always improved after 1st train
			if flag_policy_improved:
				test_success_batch = np.empty((0), dtype='float')
				for env_dir, (env_id_list, dir_type) in env_dir_dict.items():       
					if dir_type == TEST_TYPE:
						label_batch = evaluator.evaluate(img_dir=env_dir,
														img_id_list=env_id_list)
						test_success_batch = np.concatenate((test_success_batch, label_batch))

			# Save train and test reward
			train_success_list += [np.mean(train_success_batch)]
			logging.info(f'Epoch {epoch} retrain, train reward {train_success_list[-1]:.3f}')
			aug_success_list += [np.mean(aug_success_batch)]
			logging.info(f'Epoch {epoch} retrain, aug reward {aug_success_list[-1]:.3f}')
			test_success_list += [np.mean(array(test_success_batch))]
			logging.info(f'Epoch {epoch} retrain, test reward {test_success_list[-1]:.3f}')
			print(f'Epoch {epoch} retrain, train reward {train_success_list[-1]:.3f}, aug reward {aug_success_list[-1]:.3f}, test reward {test_success_list[-1]:.3f}')

			# Reset epoch count
			num_epoch_since_last_retrain = 0
			num_retrain += 1

		######################### Embed #########################

		# Reset dataset and dataloader to add the new distribution - assume retrain more often than gen
		if num_epoch_since_last_retrain == 0: 
			runner.create_dataset(env_dir_dict, embed_id_dir_dict)

		# Embed
		epoch_loss, epoch_rec_loss, epoch_reg_loss, epoch_lip_loss, latent_all, pred_all = runner.embed(epoch=epoch,
							norm_loss_ratio=norm_loss_ratio,
							latent_all=latent_all,
							label_all=label_all,
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
				'train_lists': [train_loss_list, train_rec_loss_list, train_reg_loss_list, train_lip_loss_list, train_success_list, test_success_list, aug_success_list, latent_all, label_all], # reward_all
				"seed_data": (seed, random.getstate(), np.random.get_state(), torch.get_rng_state()),
				"num_data": [num_epoch_since_last_gen, num_env_gen, num_dir_gen, env_dir_dict],
				}, train_details_path)

		# Count
		num_epoch_since_last_gen += 1
		num_epoch_since_last_retrain += 1
		epoch += 1

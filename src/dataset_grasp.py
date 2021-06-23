from src import *
from torch.utils.data.dataset import Dataset
import random


class TrainDataset(Dataset):
	'''
	This class reads data of the raw files reducing the RAM requirement;
	tradeoff is slow speed.
	'''
	def __init__(self, env_dir_dict,
						embed_id_dir_dict,
						num_sdf_available_per_obj=20000,	# not used
						num_sdf_per_obj=500,
						num_surface_per_obj=1024,
						device='cpu'):
     
		self.device = device
		self.env_dir_dict = env_dir_dict
		self.embed_id_dir_dict = embed_id_dir_dict

		self.num_list = [len(env_id_list) for _, env_id_list in embed_id_dir_dict.items()]
		self.dir_list = [env_dir for env_dir, _ in embed_id_dir_dict.items()]
		self.len = sum(self.num_list)

		# SDF and surface details
		self.num_sdf_available_per_obj = num_sdf_available_per_obj
		self.num_sdf_per_obj = num_sdf_per_obj
		self.SDF_CUTOFF = 0.1
		self.num_surface_per_obj = num_surface_per_obj


	def __getitem__(self, index):
		# Find which dir
		env_id_global = 0
		for dir_num_env, dir_name in zip(self.num_list, self.dir_list):
			if index < dir_num_env:
				break
			else:
				index -= dir_num_env
				env_id_global += len(self.env_dir_dict[dir_name][0])	# count in global
		# now index is local in embed_id_dir_dict
		env_id_local = self.embed_id_dir_dict[dir_name][index]
		# env_id_local is local in env_dir_dict
		env_id_global += env_id_local
		# env_id_global is global in env_dir_dict

		# Load sdf
		# sdf_obj = np.load(dir+str(obj_id)+'-sdf.npy')
		sdf_obj = np.load(dir_name+'sdf/'+str(env_id_local)+'-sdf.npy')

		# Balance positive and negative sdf of an object
		signs = sdf_obj[:,-1] > 0
		indices_positive = list(np.nonzero(signs)[0])
		indices_negative = list(np.nonzero(~signs)[0])
		if len(indices_positive) < self.num_sdf_per_obj//2:
			indices_positive_chosen = indices_positive
			indices_negative_chosen = random.sample(indices_negative, k=self.num_sdf_per_obj-len(indices_positive))
		elif len(indices_negative) < self.num_sdf_per_obj//2:
			indices_negative_chosen = indices_negative
			indices_positive_chosen = random.sample(indices_positive, k=self.num_sdf_per_obj-len(indices_negative))
		else:
			indices_positive_chosen = random.sample(indices_positive, k=self.num_sdf_per_obj//2)
			indices_negative_chosen = random.sample(indices_negative, k=self.num_sdf_per_obj//2)

		# Clamp sdf
		sdf_obj[:,-1] = np.clip(sdf_obj[:,-1], -self.SDF_CUTOFF, self.SDF_CUTOFF)
		sdf_obj_chosen = np.concatenate((sdf_obj[indices_positive_chosen]*10, 
										sdf_obj[indices_negative_chosen]*10))
		#* Make sdf points and sdf 10 times bigger, need to make generated objects 10 times smaller to fit gripper

		# Load surface
		surface_obj = np.load(dir_name+'sdf/'+str(env_id_local)+'-surface.npy')

		return (torch.from_numpy(sdf_obj_chosen).float().to(self.device),
				torch.from_numpy(surface_obj).float().to(self.device),
				torch.tensor(env_id_global, device=self.device).long()
				)

	def __len__(self):
		return self.len

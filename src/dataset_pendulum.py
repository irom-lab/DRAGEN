from src import *
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision import transforms


class TrainDataset(Dataset):
	'''
	This class reads data of the raw files reducing the RAM requirement;
	tradeoff is slow speed.
	'''
	def __init__(self, env_dir_dict,
						embed_id_dir_dict,
						device='cpu'):
		self.device = device

		# Box, ellipse, triangle
		self.env_dir_dict = env_dir_dict
		self.embed_id_dir_dict = embed_id_dir_dict

		self.num_list = [len(env_id_list) for _, env_id_list in embed_id_dir_dict.items()]
		self.dir_list = [env_dir for env_dir, _ in embed_id_dir_dict.items()]
		self.len = sum(self.num_list)

		self.preprocess = transforms.Compose([
			transforms.ToTensor(),
		])


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

		# Load image
		img = Image.open(dir_name+str(env_id_local)+'.png')
		img = self.preprocess(img)

		return (img.float().to(self.device),
				torch.tensor(env_id_global).long().to(self.device)
				)

	def __len__(self):
		return self.len

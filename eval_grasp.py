import numpy as np
from numpy import array
import torch
import pybullet as p
import concurrent.futures
import time
import random
import psutil

from src.fcn import FCN
from util.depth import getParameters
from util.grasp import rotate_tensor, euler2quat, quatMult
from util.misc import save__init__args
from panda.panda_env import PandaEnv


class EvaluateGrasp:

	def __init__(self, initial_policy_path=None,
              			num_cpus=16,
						cpu_offset=0, 
						mu_list=None,
                 		mu=0.3,
                   		sigma=0.03,
                     	img_size=96,
                      	inner_channels=24,
                       	num_theta=6,
						long_finger=True,	# default
						delta_z=0.03,
						max_obj_height=0.05,
                        gui=0):
		save__init__args(locals())
		self.thetas = np.linspace(0, 1, num=num_theta, endpoint=False)*np.pi
		self.num_attempt_per_step = 1 # Not doing multi object
		self.device = 'cpu'

		# Height of the EE before and after reaching down
		self.min_ee_z = 0.15 # EE height when fingers contact the table

		# Load model and freeze all parameters		
		self.fcn = FCN(inner_channels=inner_channels, 
                 		out_channels=1,
                        img_size=img_size).to(self.device)
		if initial_policy_path is not None:
			self.load_policy(initial_policy_path)
		for _, param in self.fcn.named_parameters():
			param.requires_grad = False
		self.fcn.eval()

		# Pixel to xy
		pixel_xy_path = 'data/pixel2xy'+str(img_size)+'.npz'
		self.pixel2xy_mat = np.load(pixel_xy_path)['pixel2xy']  # HxWx2

		# Initialize panda env
		self.panda_env = PandaEnv(mu=mu, sigma=sigma, long_finger=long_finger)
		self.camera_params = getParameters()


	def load_policy(self, policy_path):
		model_dict = torch.load(policy_path+'.pt', map_location=self.device)
		# filter out bias in conv layer except for output
		filtered_dict = {k: v for k, v in model_dict.items() if not('output' not in k and '.0.bias' in k)}
		self.fcn.load_state_dict(filtered_dict)


	def load_obj(self, obj_path_list, 
              			obj_height_list, 
                 		mu=None, 
                   		sigma=None, 
                    	scale=1):
		"""
		If mu/sigma not specified, use the declared ones
		"""
		if mu is None:
			mu = self.mu
		if sigma is None:
			sigma = self.sigma

		obj_id_list = []
		obj_initial_height_list = {}
		env_x = [0.5, 0.5] 
		env_y = [0.0, 0.0]
		env_yaw = [0.0, 0.0]  # [-np.pi, np.pi]
		num_obj = len(obj_path_list)

		obj_x_initial = np.random.uniform(low=env_x[0], high=env_x[1], 
                                    size=(num_obj, ))
		obj_y_initial = np.random.uniform(low=env_y[0], high=env_y[1],
                                    size=(num_obj, ))
		obj_orn_initial_all = np.random.uniform(low=env_yaw[0], high=env_yaw[1],
                                    			size=(num_obj,3))
		obj_orn_initial_all[:,:-1] = 0

		for obj_ind in range(num_obj):
			pos = [obj_x_initial[obj_ind],
          			obj_y_initial[obj_ind],
             		obj_height_list[obj_ind]/2+0.001]
			obj_id = p.loadURDF(obj_path_list[obj_ind],
								basePosition=pos, 
								baseOrientation=p.getQuaternionFromEuler(obj_orn_initial_all[obj_ind]),
        						globalScaling=scale)
			obj_id_list += [obj_id]

			# Infer number of links - change dynamics for each
			num_joint = p.getNumJoints(obj_id)
			link_all = [-1] + [*range(num_joint)]
			for link_id in link_all:
				p.changeDynamics(obj_id, link_id,
					lateralFriction=mu,
					spinningFriction=sigma,
					frictionAnchor=1,
				)

		# Let objects settle (actually do not need since we know the height of object and can make sure it spawns very close to table level)
		for _ in range(10):
			p.stepSimulation()

		# Record object initial height (for comparing with final height when checking if lifted). Note that obj_initial_height_list is a dict
		for obj_id in obj_id_list:
			pos, _ = p.getBasePositionAndOrientation(obj_id)
			obj_initial_height_list[obj_id] = pos[2]
		return obj_id_list, obj_initial_height_list


	def evaluate(self, obj_dir, obj_id_list, obj_height_list, obj_scale_list=None, num_eval=1):
		num_trial = len(obj_id_list)

		# Evaluate all urdf if obj_id_list not provided
		# obj_path_list = fnmatch.filter(os.listdir(obj_dir), '*.urdf')
		obj_path_list = [str(obj_id)+'.urdf' for obj_id in obj_id_list]

		# Split for each worker
		trial_ind_batch_all = np.array_split(np.arange(num_trial), self.num_cpus)

		# Do not scale if not specified
		if obj_scale_list is None:
			obj_scale_list = [1]*num_trial

		# Construct args - one cpu per worker
		args = (([obj_dir+obj_path_list[id] for id in trial_ind_batch], 
                 [obj_height_list[id] for id in trial_ind_batch],
                 [obj_scale_list[id] for id in trial_ind_batch],
              	 self.mu_list,
                 self.cpu_offset+batch_ind) for batch_ind, trial_ind_batch in enumerate(trial_ind_batch_all))

		with torch.no_grad():
			success_list = []
			mu_used_list = []
			with concurrent.futures.ProcessPoolExecutor(self.num_cpus) as executor:
				res_batch_all = list(executor.map(self.grasp_helper, args))
				for res_batch in res_batch_all:
					success_list += res_batch[0]
					mu_used_list += res_batch[1]
				executor.shutdown()
		return success_list, mu_used_list


	def grasp_helper(self, args):
		return self.grasp(args[0], args[1], args[2], args[3], args[4])


	def grasp(self, obj_path_list, 
					obj_height_list,
					obj_scale_list,
              		mu_list, 
                	cpu_id=0, 
                 	gui=False):

		# Assign CPU - somehow PyBullet very slow if assigning cpu in GUI mode
		if not gui:
			ps = psutil.Process()
			ps.cpu_affinity([cpu_id])
			torch.set_num_threads(1)

		# Params
		initial_ee_pos_before_img = array([0.3, -0.5, 0.25])
		initial_ee_orn = array([1.0, 0.0, 0.0, 0.0])  # straight down

		# Initialize pybullet
		if gui:
			p.connect(p.GUI, options="--width=2600 --height=1800")
			p.resetDebugVisualizerCamera(0.8, 135, -30, [0.5, 0, 0])
		else:
			p.connect(p.DIRECT)

		##################### Reset table, arm #######################
		self.panda_env.reset_env()

		########################
		success_rate_all = []
		mu_all = []
		obj_id_list = []
		for obj_path, obj_height, obj_scale in zip(obj_path_list, obj_height_list, obj_scale_list):

			success = 0
			mu_used = 1 # if failure, use mu=1

			for mu in mu_list: 

				# Reset arm
				self.panda_env.reset_arm_joints_ik(initial_ee_pos_before_img, initial_ee_orn)
				self.panda_env.grasp(targetVel=0.10)  # open gripper

				# Clear all objects and load new
				for obj_id in obj_id_list:
					p.removeBody(obj_id)
				obj_id_list, obj_initial_height_list = self.load_obj([obj_path], [obj_height], mu=mu, scale=obj_scale)	# one object
				if gui:
					time.sleep(2)

				# Infer
				depth_orig = torch.from_numpy(self.get_depth()[np.newaxis,np.newaxis]).to('cpu')
				depth_rot_all = torch.empty((0,1,self.img_size,self.img_size))
				for theta in self.thetas:
					depth_rotated = rotate_tensor(depth_orig, theta=theta)
					depth_rot_all = torch.cat((depth_rot_all,depth_rotated))
				pred_infer = self.fcn(depth_rot_all).squeeze(1).detach().numpy()

				# Apply spatial (3D) argmax to pick pixel and theta
				(theta_ind, px, py) = np.unravel_index(np.argmax(pred_infer), pred_infer.shape)
				x, y = self.pixel2xy_mat[py, px]  # actual pos, a bug
				theta = self.thetas[theta_ind]
				if gui:
					print(theta_ind, x, y)
					time.sleep(2)

				# Find the target z height
				z = depth_rot_all[theta_ind, 0, px, py]*self.max_obj_height
				z_target = max(0, z - self.delta_z) # clip
				z_target_ee = z_target + self.min_ee_z

				# Rotate into local frame
				xy_orig = array([[np.cos(theta), -np.sin(theta)],
				[np.sin(theta),np.cos(theta)]]).dot(array([[x-0.5],[y]]))
				xy_orig[0] += 0.5

				# Execute, reset ik literally on top of object, reach down, grasp, lift, check success
				ee_pos_before = np.append(xy_orig, z_target_ee+0.10)
				ee_pos_after = np.append(xy_orig, z_target_ee+0.05)
				ee_orn = quatMult(euler2quat([theta,0.,0.]), initial_ee_orn)
				for _ in range(3):
					self.panda_env.reset_arm_joints_ik(ee_pos_before, ee_orn)
					p.stepSimulation()
				if gui:
					time.sleep(2)

				ee_pos = np.append(xy_orig, z_target_ee)
				self.panda_env.move_pos(ee_pos, absolute_global_quat=ee_orn, numSteps=300)
				if gui:
					time.sleep(2)
				self.panda_env.grasp(targetVel=-0.10)  # always close gripper
				self.panda_env.move_pos(ee_pos, absolute_global_quat=ee_orn, numSteps=100)	# keep pose until gripper closes
				if gui:
					time.sleep(2)
				self.panda_env.move_pos(ee_pos_after, absolute_global_quat=ee_orn, numSteps=150)	# lift

				# check if lifted
				self.check_success(obj_id_list, obj_initial_height_list)
				if len(obj_id_list) == 0:
					if gui:
						time.sleep(5)
					success = 1
					mu_used = mu
					break

			success_rate_all += [success]
			mu_all += [mu_used]
		return success_rate_all, mu_all


	def check_success(self, obj_id_list, obj_initial_height_list):
		height = []
		obj_to_be_removed = []

		# Determine which object is lifted
		for obj_id in obj_id_list:
			pos, _ = p.getBasePositionAndOrientation(obj_id)
			height += [pos[2]]
			if pos[2] - obj_initial_height_list[obj_id] > 0.03:
				obj_to_be_removed += [obj_id]

		# Remove lifted objects
		for obj_id in obj_to_be_removed:
			p.removeBody(obj_id)
			obj_id_list.remove(obj_id)

		# Determine success
		if len(obj_to_be_removed) > 0:
			return 1
		else:
			return 0


	def get_depth(self):
		viewMat = self.camera_params['viewMatPanda']
		projMat = self.camera_params['projMatPanda']
		width = self.camera_params['imgW']
		height = self.camera_params['imgH']
		near = self.camera_params['near']
		far = self.camera_params['far']

		img_arr = p.getCameraImage(width=width, 
                             	   height=height, 
                                   viewMatrix=viewMat,
                                   projectionMatrix=projMat, 
                                   flags=p.ER_NO_SEGMENTATION_MASK)
		orig_dim = 512
		center = orig_dim//2
		crop_dim = self.img_size	# 128: 15cm square; 96: 9cm square

		depth = np.reshape(img_arr[3], (width,height))[center-crop_dim//2:center+crop_dim//2,center-crop_dim//2:center+crop_dim//2]
		depth = far*near/(far - (far - near)*depth)
		depth = (0.3 - depth)/self.max_obj_height  # set table zero, and normalize
		depth = depth.clip(min=0., max=1.)
		return depth


if __name__ == '__main__':
	# Fix seeds
	seed = 0
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	# Configure objects
	obj_dir_list = ['']
	initial_policy_path = ''

	# Initialize trianing env
	evaluator = EvaluateGrasp(initial_policy_path,
								num_cpus=1,
								cpu_offset=19,
								mu=0.3,
								sigma=0.03,
								mu_list=[0.3],
        						long_finger=True)
	obj_ind = 0
	# dim_all = np.load(obj_dir_list[0]+'dim.npy')
	# height_list = dim_all[:, 2]
	height_list = [0.025]
	# height_list = [dim_all[obj_ind, 2]]
	success, mu = evaluator.grasp([obj_dir_list[0]+str(obj_ind)+'.urdf'], 
									mu_list=np.arange(0.10,0.51,0.05), 
									obj_height_list=height_list,
									gui=True)
	print(success, mu)

	# success_list, mu_list = evaluator.evaluate(obj_dir_list[0], 
    #                                         obj_id_list=np.arange(60),
    #                                         obj_height_list=height_list)
	# print(success_list, mu_list)
	# for ind, success in enumerate(success_list):
	# 	if not success:
	# 		print(ind)

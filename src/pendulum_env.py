"""
Modified from https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/cartpole_bullet.py

URDF: https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_data/cartpole.urdf
"""
from src import *

import logging
import time
from PIL import Image
import gym
import matplotlib.pyplot as plt
from collections import deque
import random

logger = logging.getLogger(__name__)

def angle_normalize(x):
	return (((x+np.pi) % (2*np.pi)) - np.pi)


class PendulumVisionEnv(gym.Env):
	"""
	https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py#L58
	The official OpenAI Gym version of PendulumEnv has the initial position of the pendulum to be between -pi and pi; and velocity between -1 and 1.
	https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
	The official version of CartpoleEnv has the initial position of cart and pole close to origin (center and upright).
	"""
	metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

	def __init__(self,	obs_size=32,
						img_path=None,
						img_path_all=None, 
              			renders=False, 
                 		discrete_actions=False, 
                   		low_mass=False, 
                     	low_init=False, 
                      	num_frame_stack=3):
		# start the bullet physics server
		self._renders = renders
		self._discrete_actions = discrete_actions
		self._render_height = 200
		self._render_width = 200
		self._physics_client_id = -1

		self.max_speed = 8.
		self.max_torque = 2.
		self.dt = .05
		if low_mass:
			self.m = 0.1
		else:
			self.m = 1.
		self.low_init = low_init

		# Open image
		self.sample_img_flag = False
		if img_path_all is not None: # randomly sample one
			self.img_path_all = img_path_all
			img_path = random.sample(img_path_all, k=1)[0]
			self.sample_img_flag = True
		self.img_path = img_path
		self.im = Image.open(img_path)
		self.crop_top_left = int(self.im.size[0]//2-obs_size//2)
		self.crop_bottom_right = int(self.im.size[0]//2+obs_size//2)
		self.obs_size = obs_size

		self.action_space = gym.spaces.Box(
			low=-self.max_torque,
			high=self.max_torque, shape=(1,),
			dtype=np.float32
		)
		self.num_frame_stack = num_frame_stack
		self._frames = deque([], maxlen=num_frame_stack)
		num_obs_channel = num_frame_stack*3

		self.observation_space = gym.spaces.Box(
			low=0., high=1., shape=(num_obs_channel, obs_size, obs_size), dtype=np.float32)
		self.last_img = None

		self.initial_state = None

		self.seed()
		self.viewer = None
		self._configure()

	def _configure(self, display=None):
		self.display = display

	def seed(self, seed=None):
		self.np_random, seed = gym.utils.seeding.np_random(seed)
		return [seed]

	def _get_obs(self, debug=False):
		"""
		Right now let the virtual camera at the joint.
		"""
		return np.concatenate(list(self._frames), axis=0)

	def step(self, u):
		th, thdot = self.state  # th := theta

		g = 9.81
		m = self.m
		l = 1.
		dt = self.dt

		u = np.clip(u, -self.max_torque, self.max_torque)[0]
		self.last_u = u  # for rendering
		costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

		newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
		newth = th + newthdot * dt
		newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
		self.state = np.array([newth, newthdot])

		# Process obs here
		cur_img = self.get_image()
		self._frames.appendleft(cur_img)
		return self._get_obs(), -costs, False, {}

	def get_image(self):
		th, _ = self.state
		im_rot = self.im.rotate(-th*180/np.pi)
		img_rot_crop = im_rot.crop(	# center
	  				[self.crop_top_left, self.crop_top_left,
					self.crop_bottom_right, self.crop_bottom_right])
		cur_img = np.moveaxis(np.asarray(img_rot_crop, dtype='float32')/256, -1, 0)	# 3xHxW
		return cur_img

	def reset(self):
		if self.low_init:
			theta_std = 0.2
			thetadot_std = 0.2
			theta = self.np_random.normal(loc=-np.pi, scale=theta_std)
			thetadot = self.np_random.normal(loc=0, scale=thetadot_std)
			self.state = np.array([theta, thetadot])
		else:
			low = np.array([-np.pi, -1])
			high = np.array([np.pi, 1])
			self.state = self.np_random.uniform(low=low, high=high)
		self.initial_state = np.copy(self.state)

		self.last_u = None

		# Resample
		if self.sample_img_flag:
			self.img_path = random.sample(self.img_path_all, k=1)[0]

		for _ in range(self.num_frame_stack):
			self._frames.appendleft(self.get_image())
		return self._get_obs()

	def render(self, mode='human'):
		from util import gym_rendering	# overrides gym one
		import pyglet
		if self.viewer is None:
			self.viewer = gym_rendering.Viewer(500, 500)
			self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)

			rod = gym_rendering.make_capsule(1, .2)
			rod.set_color(.8, .3, .3)
			self.pole_transform = gym_rendering.Transform()
			rod.add_attr(self.pole_transform)
			self.viewer.add_geom(rod)
			axle = gym_rendering.make_circle(.05)
			axle.set_color(0, 0, 0)
			self.viewer.add_geom(axle)

			fname = os.path.join(parentdir, "data/clockwise.png")
			self.arrow_img = gym_rendering.Image(fname, 1., 1., loc_x=-0.5, loc_y=-0.5)
			self.imgtrans = gym_rendering.Transform()
			self.arrow_img.add_attr(self.imgtrans)

			self.background = gym_rendering.Image(self.img_path, 4., 4., alpha=0.5)	# make background transparent

		# Add robot view to the top right corner of the viewer
		rot_bg = np.ascontiguousarray(np.flip(np.moveaxis(np.uint8(self._frames[0]*255), 0, -1), axis=0))	# HWC, pyglet somehows flips img vertically
		rot_bg = (pyglet.gl.GLubyte*rot_bg.size).from_buffer(rot_bg)	# pyglet is very annoying to use
		self.rot_bg = gym_rendering.Image(rot_bg, 1, 1, loc_x=0.7, loc_y=0.7, alpha=1.0, fromarray=True, array_dim=self.obs_size)
		self.viewer.add_onetime(self.rot_bg)
		self.viewer.add_onetime(self.background)
		self.viewer.add_onetime(self.arrow_img)
		self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
		if self.last_u:
			self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)
		return self.viewer.render(return_rgb_array=mode == 'rgb_array')

	def configure(self, args):
		pass
	
	def close(self):
		if self._physics_client_id >= 0:
			self._p.disconnect()
		self._physics_client_id = -1


if __name__ == '__main__':

	# Test single environment in GUI
	env = PendulumVisionEnv(img_path='/home/allen/data/wasserstein/landmark_v1/0.png', renders=True, discrete_actions=False)
	env.reset()

	for t in range(100):
		action = 1.0
		obs, _, _, _ = env.step(action)
		plt.imshow(np.moveaxis(np.asarray(obs),0,-1))
		plt.show()    # Default is a blocking call
		time.sleep(1)

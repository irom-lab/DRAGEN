from PIL import Image, ImageDraw
import numpy as np

from util.misc import *
from util.mesh import *



if __name__ == '__main__':
	# 0: centered; 1: closer in radial direction; 2: farther in radial direction; 3: smaller; 4: larger
	landmark_shape = 'ellipse'
	version = '0'
	img_dir = ''	# TODO
	if version == '0':
		landmark_center_offset_range = [0.0, 0.5]
		landmark_size_scale_range =	[0.5, 0.75]
		landmark_center_offset_range = [0.50, 0.75]
		landmark_size_scale_range =	[0.5, 0.75]
	if version == '1':
		landmark_center_offset_range = [0.65, 0.75]
		landmark_size_scale_range =	[0.5, 0.75]
	if version == '2':
		landmark_center_offset_range = [-0.25, -0.15]
		landmark_size_scale_range =	[0.5, 0.75]
	if version == '3':
		landmark_center_offset_range = [0.0, 0.5]
		landmark_size_scale_range =	[0.30, 0.35]
	if version == '4':
		landmark_center_offset_range = [0.0, 0.5]
		landmark_size_scale_range =	[0.90, 0.95]
	ensure_directory_hard(img_dir)
	num_img = 200
	dim_img = 96
	dim_obs = 64
	obs_offset = dim_img//2-dim_obs//2

	for img_ind in range(num_img):
		print(img_ind)

		# Background
		rgb_bg = tuple(np.random.randint(0, 256, size=(3,))) # random bg color
		image = Image.new('RGB', (dim_img, dim_img), color = rgb_bg)
		draw = ImageDraw.Draw(image)

		# Landmark color
		rgb_landmark = tuple(np.random.randint(0, 256, size=(3,)))
		fill ='rgb('+str(rgb_landmark[0])+','+str(rgb_landmark[1])+','+str(rgb_landmark[2])+')'

		# Landmark size
		landmark_scale = np.random.uniform(low=landmark_size_scale_range[0], 
                                			high=landmark_size_scale_range[1], 
                                   			size=(2,))
		landmark_center_offset = np.random.uniform(low=landmark_center_offset_range[0], high=landmark_center_offset_range[1])
		landmark_center = int(obs_offset+dim_obs/4+dim_obs/4*landmark_center_offset)

		# Draw ellipse
		landmark_top_x = int(landmark_center-dim_obs/4*landmark_scale[0])
		landmark_top_y = int(landmark_center-dim_obs/4*landmark_scale[1])
		landmark_bottom_x = int(landmark_center+dim_obs/4*landmark_scale[0])
		landmark_bottom_y = int(landmark_center+dim_obs/4*landmark_scale[1])
		draw.ellipse([landmark_top_x,landmark_top_y,
						landmark_bottom_x,landmark_bottom_y], fill=fill)

		# Observation
		# draw.rectangle([obs_offset,obs_offset,obs_offset+dim_obs,obs_offset+dim_obs], width=1)

		# Debug
		# plt.imshow(processed)
		# plt.show()

		# Save image
		image.save(img_dir+str(img_ind)+'.png')


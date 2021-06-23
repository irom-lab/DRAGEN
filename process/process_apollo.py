import os
import PIL
from PIL import Image, ImageDraw
import numpy as np
from util.misc import *
from util.mesh import *


if __name__ == '__main__':
	"https://stackoverflow.com/questions/51486297/cropping-an-image-in-a-circular-way-using-python"

	# 1920x1080
	data_dir = ''	# TODO: specify raw data path
	category_list = ['Residential', 'Downtown_filtered', 'Urban_Intersection', 'Urban_Straight_Road']	# categories of locations used in the paper
	save_dir =''	# TODO: specify save path
	ensure_directory_hard(save_dir)
	category_img_limit = 1000	# 420 images

	obs_size = 64
	img_size = 92
	padded_img_size = 96
	print('Img size: ', img_size)
	print('Padded img size: ', padded_img_size)
	img_ind = 0
	for category in category_list:
		category_img_count = 0

		root = data_dir + category
		for path, subdirs, files in os.walk(root):
			for name in files:
				if category_img_count == category_img_limit:
					continue

				img_path = os.path.join(path, name)

				# Open original image
				orig_image = Image.open(img_path)

				# Downsize
				orig_fit_image = orig_image.resize([img_size, img_size], resample=PIL.Image.LANCZOS)
				orig_fit_image_np = np.array(orig_fit_image)

				# Create same size alpha layer with circle
				alpha = Image.new('L', orig_fit_image.size,0)
				draw = ImageDraw.Draw(alpha)
				draw.pieslice([0,0,img_size,img_size],0,360,fill=255)
				alpha_np = np.array(alpha)

				# Add alpha layer to RGB
				overlay_image_rgba_np = np.dstack((orig_fit_image_np, alpha_np))
				overlay_image_rgba = Image.fromarray(overlay_image_rgba_np)

				# Convert RGBA to RGB
				background = Image.new('RGBA', overlay_image_rgba.size, (255,255,255))
				overlay_image_rgb = Image.alpha_composite(background, overlay_image_rgba).convert('RGB')

				# Add pad
				padded_background = Image.new('RGB', (padded_img_size, padded_img_size), (255,255,255))
				padded_background.paste(overlay_image_rgb, (int(padded_img_size//2-img_size//2), int(padded_img_size//2-img_size//2)))

				# Debug
				# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
				# ax1.imshow(orig_image)
				# ax2.imshow(overlay_image_rgb)
				# ax3.imshow(padded_background)
				# plt.show()

				# Save with alpha
				padded_background.save(save_dir+str(img_ind)+'.png')
				img_ind += 1
				category_img_count += 1
		category_img_count = 0

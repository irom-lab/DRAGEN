import random
import PIL
from PIL import Image
import numpy as np
from util.misc import *
from util.mesh import *


if __name__ == '__main__':

	# MNIST: 28x28, and height of each digit is about 20
	# scaled to 32, then about 24
	from mnist import MNIST
	mndata = MNIST('')	# TODO: specify MNIST data path
	images, labels = mndata.load_training()	# 60000 available
	process_digit_all = [9]	# TODO: specify which digit to process
	save_dir = ''	# TODO: specify save directory
	ensure_directory_hard(save_dir)

	# MNIST
	dim_obs = 32
	bg_size = 48
	resize_range = [32, 32]
	height_offset = -1 # 2 for six; -2 for seven; 0 for five; 0 for four; 0 for three; 0 for two; -1 for 9 - this helps digits to align with those from USPS dataset
	image_ind = 0
	for image, label in zip(images, labels):

		if label not in process_digit_all:
			continue
		print(image_ind)

		# Background
		bg = Image.new('RGB', (bg_size, bg_size), color=(0,0,0))

		# Open image
		digit = np.uint8(array(image).reshape(28,28))

		digit = Image.fromarray(digit)

		# Crop into square?
		resize_dim = random.randint(resize_range[0], resize_range[1])
		digit = digit.resize([resize_dim, resize_dim], resample=PIL.Image.LANCZOS)

		# Attach MNIST to the center
		top_left_corner = bg_size//2-resize_dim//2
		bg.paste(digit, [top_left_corner,top_left_corner+height_offset])	# shift down a bit

		# Observation
		# draw = ImageDraw.Draw(bg)
		# draw.rectangle([bg_size//2-dim_obs//2,bg_size//2-dim_obs//2,bg_size//2+dim_obs//2,bg_size//2+dim_obs//2], width=1)

		# # Debug
		# plt.imshow(bg)
		# plt.show()

		# Save image
		bg.save(save_dir+str(image_ind)+'.png')
		image_ind += 1

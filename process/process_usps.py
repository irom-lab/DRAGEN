import PIL
from PIL import Image
import numpy as np
from util.misc import *
from util.mesh import *


if __name__ == '__main__':


	# USPS: 16x16, and height of each digit is 16
	# scaled to 24
	import h5py
	file = ''	# TODO: specify USPS data path
	save_dir = ''	# TODO: specify save path
	with h5py.File(file, 'r') as hf:
		train = hf.get('train')
		X_tr = train.get('data')[:]
		y_tr = train.get('target')[:]
		test = hf.get('test')
		X_te = test.get('data')[:]
		y_te = test.get('target')[:]
	ensure_directory_hard(save_dir)
	process_digit_all = [9]	# TODO: specify which digit to process

	# USPS
	dim_obs = 32
	bg_size = 48
	resize_dim = 24
	image_ind = 0
	for digit, label in zip(X_tr, y_tr):
		if label not in process_digit_all:
			continue
		print(image_ind)

		# Background
		bg = Image.new('RGB', (bg_size, bg_size), color=(0,0,0))

		digit = np.uint8(array(digit*255).reshape(16,16))
		digit = Image.fromarray(digit)

		# Crop into square?
		digit = digit.resize([resize_dim, resize_dim], resample=PIL.Image.LANCZOS)

		# Attach MNIST to the center
		top_left_corner = bg_size//2-resize_dim//2
		bg.paste(digit, [top_left_corner,top_left_corner])

		# Observation
		# draw = ImageDraw.Draw(bg)
		# draw.rectangle([bg_size//2-dim_obs//2,bg_size//2-dim_obs//2,bg_size//2+dim_obs//2,bg_size//2+dim_obs//2], width=1)

		# Debug
		# plt.imshow(bg)
		# plt.show()

		# Save image
		bg.save(save_dir+str(image_ind)+'.png')
		image_ind += 1

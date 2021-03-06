from src import *


class FCN(nn.Module):

	def __init__(self, inner_channels=64, out_channels=1, img_size=96):
		super(FCN, self).__init__()
		bias = False

		# Downsample
		self.down_layer_1 = nn.Sequential(  # Nx1x96x96
								nn.Conv2d(in_channels=1,	# depth only
				  						  out_channels=inner_channels//4, 
				  						  kernel_size=6, stride=2, padding=2,
              								bias=bias),
								nn.BatchNorm2d(inner_channels//4),
								nn.ReLU(),
								)    # Nx(inner_channels//4)x48x48

		self.down_layer_2 = nn.Sequential(
								nn.Conv2d(in_channels=inner_channels//4, 
				  						  out_channels=inner_channels//2, 
						  				  kernel_size=4, stride=2, padding=1,
              								bias=bias),
								nn.BatchNorm2d(inner_channels//2),
								nn.ReLU(),
								)    # Nx(inner_channels//2)x24x24

		self.down_layer_3 = nn.Sequential(
								nn.Conv2d(in_channels=inner_channels//2, 
				  						  out_channels=inner_channels, 
						  				  kernel_size=4, stride=2, padding=1,
              								bias=bias),
								nn.BatchNorm2d(inner_channels),
								nn.ReLU(),
								)    # Nx(inner_channels)x12x12

		# Upsample
		self.up_layer_1 = nn.Sequential(
								nn.Conv2d(in_channels=inner_channels,
				  						  out_channels=inner_channels//2,
						  				  kernel_size=3, stride=1, padding=1,
              								bias=bias),
								nn.BatchNorm2d(inner_channels//2),
								nn.ReLU(),
								Interpolate(size=[img_size//4,img_size//4], mode='bilinear'),  # Nx(inner_channels//2)x24x24
								)

		self.up_layer_2 = nn.Sequential(
								nn.Conv2d(in_channels=inner_channels//2,
				  						  out_channels=inner_channels//4,
						  				  kernel_size=3, stride=1, padding=1,
              								bias=bias),
								nn.BatchNorm2d(inner_channels//4),
								nn.ReLU(),
								Interpolate(size=[img_size//2,img_size//2], mode='bilinear'), # Nx(inner_channels//4)x48x48
								)

		self.up_layer_3 = nn.Sequential(
								nn.Conv2d(in_channels=inner_channels//4,
				  						  out_channels=inner_channels//8,
						  				  kernel_size=3, stride=1, padding=1,
              								bias=bias),
								nn.BatchNorm2d(inner_channels//8),
								nn.ReLU(),
								Interpolate(size=[img_size,img_size], mode='bilinear'), # Nx(inner_channels//8)x96x96
		   						)

		self.output_layer = nn.Sequential(
							nn.Conv2d(in_channels=inner_channels//8, 
				 					  out_channels=out_channels, 
									  kernel_size=1, stride=1, padding=0),	# 1x1 convolution
							)

	def forward(self, x):
		x = self.down_layer_1(x)
		x = self.down_layer_2(x)
		x = self.down_layer_3(x)
  
		x = self.up_layer_1(x)
		x = self.up_layer_2(x)
		x = self.up_layer_3(x)
		return self.output_layer(x)

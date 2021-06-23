from src import *


class Encoder(torch.nn.Module):
	def __init__(self, dim_latent,
						dim_img,
						variational=False,
			  			inner_channels=16,
        				mlp_down_factor=8,
            			num_layer=5):
		super(Encoder, self).__init__()
		self.variational = variational
		self.num_layer = num_layer
		dim_conv_output = (dim_img//(2**num_layer))**2*inner_channels	# assume last conv layer outputs inner_channels
		print('Encoder dim conv output: ', dim_conv_output)

		# Downsample
		sequence = list()
		num_channel_init = inner_channels//(2**num_layer)	# assume at least 2 layers
		for layer_ind in range(num_layer):
			if layer_ind == 0:
				channel_in = 3
			else:
				channel_in = num_channel_init*(2**layer_ind)
			channel_out = num_channel_init*(2**(layer_ind+1))
			sequence.extend([nn.Conv2d(in_channels=channel_in,	# RGB
				  						  out_channels=channel_out, 
				  						  kernel_size=4, stride=2, padding=1,
              								bias=True), 
							# nn.BatchNorm2d(channel_out),
							nn.ReLU()])
		self.conv = torch.nn.Sequential(*sequence)

		if variational:
			self.down_mlp = nn.Sequential(
				nn.Linear(dim_conv_output, dim_conv_output//mlp_down_factor),
				nn.ReLU(),
				)
			self.mlp_mu = nn.Linear(dim_conv_output//mlp_down_factor, dim_latent)
			self.mlp_logvar = nn.Linear(dim_conv_output//mlp_down_factor, dim_latent)
		else:
			self.down_mlp = nn.Sequential(
				nn.Linear(dim_conv_output, dim_conv_output//mlp_down_factor),
				nn.ReLU(),
				nn.Linear(dim_conv_output//mlp_down_factor, dim_latent),
				)

	def forward(self, x):
		B = x.shape[0]
		x = self.conv(x)
		x = x.reshape(B, -1)
		x = self.down_mlp(x)
		if self.variational:
			mu = self.mlp_mu(x)
			logvar = self.mlp_logvar(x)
			return mu, logvar
		else:
			return x


class Decoder(torch.nn.Module):
	def __init__(self, dim_latent,
						dim_img,
			  			inner_channels=16,
        				mlp_up_factor=8,
						use_upsampling=True,
            			interp_mode='bilinear',
               			num_layer=5):	# nearest
		super(Decoder, self).__init__()
		self.use_upsampling = use_upsampling
		self.num_layer = num_layer

		self.dim_img_latent_channel = inner_channels
		self.dim_img_latent = dim_img//(2**num_layer)
		dim_conv_input = (self.dim_img_latent)**2*inner_channels
		print('Decoder dim conv input: ', dim_conv_input)

		# Upsample
		self.up_mlp = nn.Sequential(
					nn.Linear(dim_latent, dim_conv_input//mlp_up_factor),
					nn.ReLU(),
					nn.Linear(dim_conv_input//mlp_up_factor, dim_conv_input),
					)

		sequence = list()
		for layer_ind in range(self.num_layer):
			num_channel = inner_channels//(2**layer_ind)
			sequence.extend([nn.Upsample(scale_factor=2, mode=interp_mode),
							slice(),
							nn.ReflectionPad2d(1),
							nn.Conv2d(in_channels=num_channel,
									out_channels=num_channel//2, 
									kernel_size=3, stride=1, padding=1,
										bias=True),
                    		nn.ReLU()
                      		])
		self.conv = torch.nn.Sequential(*sequence)

		self.output_layer = nn.Sequential(
							nn.Conv2d(in_channels=num_channel//2, 
				 					  out_channels=3, 
									  kernel_size=1, stride=1, padding=0, bias=True),	# 1x1 convolution
							nn.Sigmoid()	# back to [0,1] range
							)

	def forward(self, x):
		B = x.shape[0]
		x = self.up_mlp(x)
		x = x.view(B, self.dim_img_latent_channel, self.dim_img_latent, self.dim_img_latent)
		x = self.conv(x)
		x = self.output_layer(x)
		return x


if __name__ == '__main__':
	pass

from src import *


class PointNetEncoder(nn.Module):
	def __init__(self, dim_latent, breadth=128, is_variational = False):
		super(PointNetEncoder, self).__init__()

		self.is_variational = is_variational
		if is_variational:
			self.filename = 'variational-' + self.filename

		self.encoder = PN1_Module(input_num_chann=3,
									num_pn_output=breadth,
							  		num_mlp_output=dim_latent)

		
	def forward(self, x):
		x = x.permute(0,2,1)
		return self.encoder(x)


class PN1_Module(nn.Module):
	def __init__(self, 
				 input_num_chann=3, 
				 num_pn_output=128, 
				 dim_mlp_append=0, 
				 num_mlp_output=16):
		"""
		* Use spatial softmax instead of max pool by default
		"""
		
		super(PN1_Module, self).__init__()

		self.dim_mlp_append = dim_mlp_append
		self.num_mlp_output = num_mlp_output

		# CNN
		self.conv_1 = nn.Sequential(
								nn.Conv1d(input_num_chann, num_pn_output//4, 1),
								nn.ReLU(),
								)
		self.conv_2 = nn.Sequential(
								nn.Conv1d(num_pn_output//4, num_pn_output//2,1),
								nn.ReLU(),
								)
		self.conv_3 = nn.Sequential(
								nn.Conv1d(num_pn_output//2, num_pn_output, 1),
								)
  
		# Spatial softmax
		self.sm = nn.Softmax(dim=2)

		# MLP
		num_linear_1_input = num_pn_output*3 + dim_mlp_append
		self.linear_1 = nn.Sequential(
								nn.Linear(num_linear_1_input, num_pn_output),
								nn.ReLU(),
								)

		self.linear_2 = nn.Sequential(
								nn.Linear(num_pn_output, num_pn_output//2),
								nn.ReLU(),
								)

		# Output layers
		self.linear_out = nn.Linear(num_pn_output//2, num_mlp_output)


	def forward(self, x):

		B = x.shape[0]

		# CNN
		out = self.conv_1(x)
		out = self.conv_2(out)
		out = self.conv_3(out)

		# Spatial softmax
		s = self.sm(out)  # B x 128 x 1536, normalized among points
		xyz = x[:,:3,:]
		out_pn = torch.bmm(s, xyz.permute(0,2,1)).view(B, -1)

		# MLP
		x = self.linear_1(out_pn)
		x = self.linear_2(x)
		out_mlp = self.linear_out(x).squeeze(1)
		return out_mlp

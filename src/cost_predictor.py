from src import *


class CostPredictor(nn.Module):
	def __init__(self, dim_latent, dim_hidden=20):
		super(CostPredictor, self).__init__()

		self.linear_hidden = nn.Sequential(
								nn.Linear(dim_latent, dim_hidden, bias=True),
								nn.Sigmoid(),
								)

		self.linear_out = nn.Sequential(
								nn.Linear(dim_hidden, 1, bias=True),
								nn.Sigmoid(),
								)

	def get_lip(self):
		return float(1/16*torch.linalg.norm(self.linear_hidden[0].weight.data, ord=2)*torch.linalg.norm(self.linear_out[0].weight.data, ord=2))	# each sigmoid is 1/4


	def forward(self, latent):
		out = self.linear_hidden(latent)
		return self.linear_out(out)

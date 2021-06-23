from torch import nn
import torch
import numpy as np
from numpy import array

import warnings
warnings.filterwarnings('ignore')

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

INIT_TYPE = 0
TEST_TYPE = 1
GEN_TYPE = 2


class Interpolate(nn.Module):
	def __init__(self, size, mode):
		super(Interpolate, self).__init__()
		self.interp = nn.functional.interpolate
		self.size = size
		self.mode = mode
		
	def forward(self, x):
		x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
		return x


def reparameterize(mu, logvar):
	std = torch.exp(0.5*logvar)
	eps = torch.randn_like(std)
	return mu + eps*std


class slice(torch.nn.Module):
	def __init__(self):
		super(slice, self).__init__()
	def forward(self, x):
		return x[:,:, 1:-1, 1:-1]	# for reflection

import numpy as np
import torch

from scipy.spatial import ConvexHull, Delaunay
from numpy.linalg import det
from scipy.stats import dirichlet


def sample_from_nsphere(radius, num_sample, dim):
	raw_points = np.random.normal(0, 1, size=(num_sample, dim))
	norm_points = raw_points / np.linalg.norm(raw_points, axis=1, keepdims=True) * radius
	return norm_points


def sample_from_convex_hull(latent_all, num_sample):
	"""
	uniform sampling within convex hull: https://stackoverflow.com/questions/59073952/how-to-get-uniformly-distributed-points-in-convex-hull
	"""
	dim_latent = latent_all.shape[1]

	hull_v = latent_all[ConvexHull(latent_all).vertices]
	deln = Delaunay(hull_v)
	deln_sim = hull_v[deln.simplices]

	# Sample simplices based on their area
	vols = np.abs(det(deln_sim[:, :dim_latent, :] - deln_sim[:, dim_latent:, :])) / np.math.factorial(dim_latent)
	sample_simplice = np.random.choice(len(vols), size=num_sample, p=vols / vols.sum())
	samples = np.einsum('ijk, ij -> ik', deln_sim[sample_simplice], dirichlet.rvs([1]*(dim_latent + 1), size = num_sample))
	return samples, hull_v


def sample_from_ball(center=np.array([0.,0.]), num=100, radius=1):
	dim = len(center)
	"""
	For any dimension. https://stackoverflow.com/questions/54544971/how-to-generate-uniform-random-points-inside-d-dimension-ball-sphere
	"""
	# First generate random directions by normalizing the length of a
	# vector of random-normal values (these distribute evenly on ball).
	random_directions = np.random.normal(size=(dim, num))
	random_directions /= np.linalg.norm(random_directions, axis=0)
	# Second generate a random radius with probability proportional to
	# the surface area of a ball with a given radius.
	random_radii = np.random.random(num) ** (1/dim)
	# Return the list of random (direction & length) points.
	samples = radius * (random_directions * random_radii).T
	return samples + center

def get_points_in_unit_sphere(n, device):
	x = torch.rand(int(n * 2.5), 3, device=device) * 2 - 1
	mask = (torch.norm(x, dim=1) < 1).nonzero().squeeze()
	mask = mask[:n]
	x = x[mask, :]
	if x.shape[0] < n:
		print("Warning: Did not find enough points.")
	return x


def get_voxel_coordinates(resolution=64, size=0.1, center=0, return_torch_tensor=False):	# size was 2
	if type(center) == int:
		center = (center, center, center)
	points = np.meshgrid(
		np.linspace(center[0] - size/2, center[0] + size/2, resolution),
		np.linspace(center[1] - size/2, center[1] + size/2, resolution),
		np.linspace(center[2] - size/2, center[2] + size/2, resolution)
	)
	points = np.stack(points)
	points = np.swapaxes(points, 1, 2)
	points = points.reshape(3, -1).transpose()
	if return_torch_tensor:
		return torch.tensor(points, dtype=torch.float32, device=device)
	else:
		return points.astype(np.float32)


def show_sdf_point_cloud(points, sdf):
	import pyrender
	colors = np.zeros(points.shape)
	colors[sdf < 0, 2] = 1
	colors[sdf > 0, 0] = 1
	cloud = pyrender.Mesh.from_points(points, colors=colors)

	scene = pyrender.Scene()
	scene.add(cloud)
	viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)

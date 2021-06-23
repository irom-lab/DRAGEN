"""
MIT License

Copyright (c) 2020 Marian Kleineberg
Copyright (c) 2021 Intelligent Robot Motion Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from src import *

import trimesh
import skimage.measure
from util.geom import get_points_in_unit_sphere, get_voxel_coordinates


class SDFVoxelizationHelperData():
	def __init__(self, device, voxel_resolution, size, sphere_only=True):
		sample_points = get_voxel_coordinates(voxel_resolution, size=size)

		if sphere_only:
			unit_sphere_mask = np.linalg.norm(sample_points, axis=1) < 1.1
			sample_points = sample_points[unit_sphere_mask, :]
			self.unit_sphere_mask = unit_sphere_mask.reshape(voxel_resolution, voxel_resolution, voxel_resolution)

		self.sample_points = torch.tensor(sample_points, device=device)
		self.point_count = self.sample_points.shape[0]

sdf_voxelization_helper = dict()


class SDFDecoder(nn.Module):
	def __init__(self, dim_latent, breadth, device):
		super(SDFDecoder, self).__init__()
		self.device = device
		self.layers1 = nn.Sequential(
			nn.Linear(3 + dim_latent, breadth),
			nn.ReLU(),
			nn.Linear(breadth, breadth),
			nn.ReLU(),
			nn.Linear(breadth, breadth),
			nn.ReLU(),
		)

		self.layers2 = nn.Sequential(
			nn.Linear(breadth + dim_latent + 3, breadth),
			nn.ReLU(),
			nn.Linear(breadth, breadth),
			nn.ReLU(),
			nn.Linear(breadth, 1),
			nn.Tanh()
		)

	def forward(self, points, latent_codes):
		input = torch.cat((points, latent_codes), dim=1)
		x = self.layers1(input)
		x = torch.cat((x, input), dim=1)
		x = self.layers2(x)*0.2
		return x.squeeze()

	def evaluate_in_batches(self, points, latent_code, batch_size=100000, return_cpu_tensor=True):
		latent_codes = latent_code.repeat(batch_size, 1)
		with torch.no_grad():
			batch_count = points.shape[0] // batch_size
			if return_cpu_tensor:
				result = torch.zeros((points.shape[0]))
			else:
				result = torch.zeros((points.shape[0]), device=points.device)
			for i in range(batch_count):
				result[batch_size * i:batch_size * (i+1)] = self(points[batch_size * i:batch_size * (i+1), :], latent_codes)
			remainder = points.shape[0] - batch_size * batch_count
			result[batch_size * batch_count:] = self(points[batch_size * batch_count:, :], latent_codes[:remainder, :])
		return result

	def get_voxels(self, latent_code, voxel_resolution, size, sphere_only=False, pad=False):	# sphere_only and pad was true
		if not (voxel_resolution, sphere_only) in sdf_voxelization_helper:
			helper_data = SDFVoxelizationHelperData(self.device, voxel_resolution, size, sphere_only)
			sdf_voxelization_helper[(voxel_resolution, sphere_only)] = helper_data
		else:
			helper_data = sdf_voxelization_helper[(voxel_resolution, sphere_only)]

		with torch.no_grad():
			distances = self.evaluate_in_batches(helper_data.sample_points, latent_code).numpy()

		if sphere_only:
			voxels = np.ones((voxel_resolution, voxel_resolution, voxel_resolution), dtype=np.float32)
			voxels[helper_data.unit_sphere_mask] = distances
		else:
			voxels = distances.reshape(voxel_resolution, voxel_resolution, voxel_resolution)
			if pad:
				voxels = np.pad(voxels, 1, mode='constant', constant_values=1)

		return voxels

	def get_mesh(self, latent_code, voxel_resolution=64, sphere_only=True, raise_on_empty=False, size=2., level=0):

		voxels = self.get_voxels(latent_code, voxel_resolution=voxel_resolution, size=size,sphere_only=sphere_only)

		try:
			vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(voxels, level=level, spacing=(size / voxel_resolution, size / voxel_resolution, size / voxel_resolution))
		except ValueError as value_error:
			if raise_on_empty:
				raise value_error
			else:
				return None

		# Shift coordinate of vertices to the center of the object
		vertices -= size / 2	

		# Generate mesh
		try:
			mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
			return mesh
		except:
			return None


	def get_uniform_surface_points(self, latent_code, point_count=1000, voxel_resolution=64, sphere_only=True, level=0):
		mesh = self.get_mesh(latent_code, voxel_resolution=voxel_resolution, sphere_only=sphere_only, level=level)
		return mesh.sample(point_count)

	def get_normals(self, latent_code, points):
		if latent_code.requires_grad or points.requires_grad:
			raise Exception('get_normals may only be called with tensors that don\'t require grad.')
		
		points.requires_grad = True
		latent_codes = latent_code.repeat(points.shape[0], 1)
		sdf = self(points, latent_codes)
		sdf.backward(torch.ones(sdf.shape[0], device=self.device))
		normals = points.grad
		normals /= torch.norm(normals, dim=1).unsqueeze(dim=1)
		return normals

	def get_surface_points(self, latent_code, sample_size=100000, sdf_cutoff=0.1, return_normals=False, use_unit_sphere=True):
		if use_unit_sphere:
			points = get_points_in_unit_sphere(n=sample_size, device=self.device) * 1.1
		else:
			points = torch.rand((sample_size, 3), device=self.device) * 2.2 - 1
		points.requires_grad = True
		latent_codes = latent_code.repeat(points.shape[0], 1)
	
		sdf = self(points, latent_codes)

		sdf.backward(torch.ones((sdf.shape[0]), device=self.device))
		normals = points.grad
		normals /= torch.norm(normals, dim=1).unsqueeze(dim=1)
		points.requires_grad = False

		# Move points towards surface by the amount given by the signed distance
		points -= normals * sdf.unsqueeze(dim=1)

		# Discard points with truncated SDF values
		mask = (torch.abs(sdf) < sdf_cutoff) & torch.all(torch.isfinite(points), dim=1)
		points = points[mask, :]
		normals = normals[mask, :]
		
		if return_normals:
			return points, normals
		else:
			return points

	def get_surface_points_in_batches(self, latent_code, amount = 1000):
		result = torch.zeros((amount, 3), device=self.device)
		position = 0
		iteration_limit = 20
		while position < amount and iteration_limit > 0:
			points = self.get_surface_points(latent_code, sample_size=amount * 6)
			amount_used = min(amount - position, points.shape[0])
			result[position:position+amount_used, :] = points[:amount_used, :]
			position += amount_used
			iteration_limit -= 1
		return result

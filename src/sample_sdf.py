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


import os
# If server, need to use osmesa for pyopengl/pyrender
if os.cpu_count() > 20:
	os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
		# https://github.com/marian42/mesh_to_sdf/issues/13
		# https://pyrender.readthedocs.io/en/latest/install/index.html?highlight=ssh#getting-pyrender-working-with-osmesa
else:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'	# default one was pyglet, which hangs sometime for unknown reason: https://github.com/marian42/mesh_to_sdf/issues/19;
import numpy as np
import psutil
import concurrent.futures
import trimesh
from mesh_to_sdf import get_surface_point_cloud, BadMeshException
from util.misc import ensure_directory


class PointSampler:

	def __init__(self, num_cpus=16, 
						cpu_offset=0,
						num_sdf_available_per_obj=20000,	# 200000, was 1000!
						num_surface_per_obj=2048,
						model_extension='.stl',
    					**kwargs):
		self.MODEL_EXTENSION = model_extension
		self.SDF_CLOUD_SAMPLE_SIZE = num_sdf_available_per_obj
		self.SURFACE_CLOUD_SAMPLE_SIZE = num_surface_per_obj
		self.num_cpus = num_cpus
		self.cpu_offset = cpu_offset

	def reset_dir(self, directory):
		self.DIRECTORY_MODELS = directory
		self.DIRECTORY_SDF = directory + 'sdf/'

	def get_npy_filename(self, model_filename, qualifier=''):
		return self.DIRECTORY_SDF + model_filename[len(self.DIRECTORY_MODELS):-len(self.MODEL_EXTENSION)] + qualifier + '.npy'

############################################################################

	def get_bad_mesh_filename(self, model_filename):
		return self.DIRECTORY_SDF + model_filename[len(self.DIRECTORY_MODELS):-len(self.MODEL_EXTENSION)] + '.badmesh'

	def mark_bad_mesh(self, model_filename):
		filename = self.get_bad_mesh_filename(model_filename)
		ensure_directory(os.path.dirname(filename))            
		open(filename, 'w').close()

	def is_bad_mesh(self, model_filename):
		return os.path.exists(self.get_bad_mesh_filename(model_filename))

############################################################################

	def process_model_file(self, file_list, cpu_id):

		# Assign CPU
		ps = psutil.Process()
		ps.cpu_affinity([cpu_id])

		for filename in file_list:
			sdf_cloud_filename = self.get_npy_filename(filename, '-sdf')
			surface_cloud_filename = self.get_npy_filename(filename, '-surface')

			if self.is_bad_mesh(filename):
				continue

			# Load mesh
			mesh = trimesh.load(filename)
			# mesh = scale_to_unit_sphere(mesh)	# do not scale!
			mesh_bounds = np.maximum(abs(mesh.bounds[0]), mesh.bounds[1])	# half, use the larger side for all 3 dims, but already centered

			# Sample point cloud (surface) of the object
			pcl = trimesh.sample.sample_surface(mesh, self.SURFACE_CLOUD_SAMPLE_SIZE)[0]	# points and indices
			np.save(surface_cloud_filename, pcl)

			# Sample sdf (heavy computations)
			surface_point_cloud = get_surface_point_cloud(mesh, surface_point_method='scan', scan_count=100, scan_resolution=400, sample_point_count=1000000)	# default uses 10m for sample method, scan method creates about 4m points
			# surface_point_method: The method to generate a surface point cloud. Either 'scan' or 'sample'. The scanning method creates virtual scans while the sampling method uses the triangles to sample surface points. The sampling method only works with watertight meshes with correct face normals, but avoids some of the artifacts that the scanning method creates.
			try:
				points, sdf, model_size = surface_point_cloud.sample_sdf_near_surface_with_bounds(mesh_bounds=mesh_bounds, number_of_points=self.SDF_CLOUD_SAMPLE_SIZE, sign_method='depth', min_size=0.015,)
				# sign_method: The method to determine the signs of the SDF values. Either 'normal' or 'depth'. The normal method uses normals of the point cloud. It works better for meshes with holes, but sometimes results in "bubble" artifacts. The depth method avoids the bubble artifacts but is less accurate.
				combined = np.concatenate((points, sdf[:, np.newaxis]), axis=1)
				ensure_directory(os.path.dirname(sdf_cloud_filename))
				np.save(sdf_cloud_filename, combined)

				# Debug
				if model_size < 0.1:
					print(model_size, filename)
			except BadMeshException:
				# tqdm.write("Skipping bad mesh. ({:s})".format(filename))
				self.mark_bad_mesh(filename)
				continue
		return 1

	def process_model_file_helper(self, args):
		return self.process_model_file(args[0], args[1])

	def sample_new_surface_point_sdf(self, obj_id_list):
		"""
		Path already resetted to the target one; do not combine
		"""
		ensure_directory(self.DIRECTORY_SDF)
		files = [self.DIRECTORY_MODELS + str(id) + '.stl' for id in obj_id_list]

		# Split into batches
		num_trial = len(files)
		file_id_batch_all = np.array_split(np.arange(num_trial), self.num_cpus)
		args = (([files[id] for id in file_id_batch], 		
				self.cpu_offset+batch_ind) for batch_ind, file_id_batch in enumerate(file_id_batch_all))

		# Order does not matter
		with concurrent.futures.ProcessPoolExecutor(self.num_cpus) as executor:
			res_batch_all = list(executor.map(self.process_model_file_helper, args))
			executor.shutdown()
		return 1



if __name__ == '__main__':
	sampler = PointSampler(num_cpus=16,
							cpu_offset=0,
							num_sdf_available_per_obj=20000,
							num_surface_per_obj=2048)
	sampler.reset_dir(directory='/home/allen/data/wasserstein/grasp/3dnet_train/')
	sampler.sample_new_surface_point_sdf(obj_id_list=np.arange(255))

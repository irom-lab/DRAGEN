import os
import random
from copy import deepcopy
import fnmatch
import numpy as np
from numpy import array
import matplotlib.pyplot as plt

import trimesh
from scipy import interpolate
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely import affinity


def create_mesh_from_pieces(pieces):
	return trimesh.util.concatenate(pieces)


def save_convex_urdf(mesh, dir_path, ind, mass=0.1, keep_concave_part=False):
	return trimesh.exchange.urdf.export_urdf(mesh, 
										dir_path, 
										ind=ind,
										mass=mass,
										keep_concave_part=keep_concave_part)


def chain_mesh_3D(prim_path_list, translate_bound=0.04, translate_bound_rate=0.5, translate_bound_min=0.01):
	"""
	Translate in x/y only, no translation in z nor rotation
	"""
	# Try until meshes intersect
	num_prim = len(prim_path_list)
	while 1:
		prim_list = []
		# Sort such that primitives can be separated as much as possible
		offset_list = np.sort(np.random.uniform(-translate_bound, translate_bound, size=(num_prim*2,))).reshape((num_prim, 2))
		for offset, prim_path in zip(offset_list, prim_path_list):
			prim = trimesh.load(prim_path)
			# print(prim_path, prim.body_count, offset)

			# Translate
			# angle = random.choice(range(180))
			align_matrix = np.array([[1, 0, 0, offset[0]],
									[0, 1, 0, offset[1]],
									[0, 0, 1, 0],
									[0, 0, 0, 1]])
			prim.apply_transform(align_matrix)

			# Add to list
			prim_list += [prim]

		# Check intersection - assume chaining 2 meshes - blender is faster than scad, but might be less robust
		sec = prim_list[0].intersection(prim_list[1], engine='blender')
		if isinstance(sec, trimesh.Scene):	# if intersects, return trimesh instead
			translate_bound *= translate_bound_rate	# was 0.8, faster now
			if translate_bound < translate_bound_min:
				# raise ValueError('Translate bound too small')
				return None
			continue

		# Chain
		try:
			chained_mesh = trimesh.util.concatenate(prim_list)
			break
		except:
			return None
	return chained_mesh


def chain_mesh_2D(prim_path_list, translate_bound=0.02, height=0.05):
	"""
	Combine using shapely 2D
	"""
	polygon_list = []
	for prim_path in prim_path_list:
		prim = trimesh.load(prim_path)
		section = section_mesh(prim)
		
		# Convert trimesh.Path2D to shapely.geometry.polygon.Polygon
		polygon = section.polygons_closed[0]
		vs = list(polygon.exterior.coords)
		polygon = Polygon(vs)

		# Rotate, centroid or center
		angle = random.choice(range(180))
		polygon = affinity.rotate(polygon, angle=angle, origin='centroid')

		# Translate
		offset = np.random.uniform(low=-translate_bound, high=translate_bound, size=(2,))
		polygon = affinity.translate(polygon, xoff=offset[0], yoff=offset[1], zoff=0)

		# Add to list
		polygon_list += [polygon]

	# Chain
	chained_polygon = unary_union(polygon_list)

	# Convert back to 3D mesh
	chained_mesh = trimesh.creation.extrude_polygon(chained_polygon, height=height)
	return chained_mesh


def stabilize_mesh(mesh):
	# Find stable pose and then re-orient
	transforms, probs = trimesh.poses.compute_stable_poses(mesh)	# transforms in (n,4,4), ordered using probability
	mesh.apply_transform(transforms[0])
	return mesh


def match_one_dim(new_mesh, old_mesh):
	# Ignore z for now
	old_dim = old_mesh.bounds[:,:2]
	new_dim = new_mesh.bounds[:,:2]
	# Pick either x or y to rescale
	flag = random.randint(0,1)
	if flag:
		x_ratio = (new_dim[1,0]-new_dim[0,0])/(old_dim[1,0]-old_dim[0,0])
		new_mesh.apply_scale([1/x_ratio, 1, 1])
	else:
		y_ratio = (new_dim[1,1]-new_dim[0,1])/(old_dim[1,1]-old_dim[0,1])
		new_mesh.apply_scale([1, 1/y_ratio, 1])
	return new_mesh

def match_mesh_height(new_mesh, old_mesh):
	old_height = old_mesh.bounds[1,2]-old_mesh.bounds[0,2]
	new_height = new_mesh.bounds[1,2]-new_mesh.bounds[0,2]
	new_mesh.apply_scale([1, 1, old_height/new_height])
	return new_mesh


def process_mesh(mesh, remove_body=True,
                 		scale_down=True, 
                 		smooth=False, 
                   		random_scale=False, 
                    	scale_max_dim_range=[0.03,0.07]):

	# Remove extra vodies
	if remove_body:
		mesh = remove_extra_bodies_from_mesh(mesh)

	# Remove NaN and Inf values; merging duplicate vertices
	# If validate: Remove triangles which have one edge of their rectangular 2D oriented bounding box shorter than tol.merge; remove duplicated triangles; ensure triangles are consistently wound and normals face outwards
	mesh.process(validate=True)

	# Scale down (had to apply factor of 10 when sampling sdf)
	if scale_down:
		mesh.apply_scale(0.1)

	# Repair - does not work for big holes
	trimesh.repair.fill_holes(mesh)
	trimesh.repair.broken_faces(mesh)
	trimesh.repair.fix_inversion(mesh)

	# Smoothing	in place
	if smooth:
		trimesh.smoothing.filter_laplacian(mesh)
		# trimesh.smoothing.filter_humphrey(mesh)

	# Scale down the object randomly to fit the gripper
	if random_scale:
		# Take cross section, choose the largest in area if multiple 
		section = section_mesh(mesh)
		if section is None:
			return None

		# Center 2D path
		xy_center = (section.bounds[1] - section.bounds[0])/2
		offset = -(section.bounds[0]+xy_center)
		align_matrix = np.array([[1, 0, offset[0]],
								[0, 1, offset[1]],
								[0, 0, 1]])
		section.apply_transform(align_matrix)

		# Orient such that x dim is longer
		# if upper_bound[0] < upper_bound[1]:
		# 	align_matrix = np.array([[0, -1, 0,],
		# 							[1, 0, 0,],
		# 							[0, 0, 1]])	# 90 degrees
		# 	section.apply_transform(align_matrix)

		# Scale to fit the gripper
		upper_bound = section.bounds[1] # get upper bound (half) of 2D path
		max_dim_2d = np.random.uniform(scale_max_dim_range[0], 
                                 		scale_max_dim_range[1], 
                                   		size=(2,))/2	# divide by 2 since bound is half the length
		section.apply_scale([max_dim_2d[0]/upper_bound[0], 
							max_dim_2d[1]/upper_bound[1]])		

		# Extrude back into 3D
		mesh = section.extrude(height=0.05)

	# Center 3D mesh
	return center_mesh(mesh)    


def random_scale_down_mesh(mesh, target_xy_range=[0.04, 0.07], max_z=0.1):
	# Scale down in target_xy_range and clip z
	x_dim = mesh.bounds[1,0] - mesh.bounds[0,0]
	y_dim = mesh.bounds[1,1] - mesh.bounds[0,1]

	# Both x/y too large - apply same scale to x/y/z
	target_xy = np.random.uniform(target_xy_range[0], target_xy_range[1], size=(1,))[0]
	scale = target_xy/max([x_dim, y_dim])
	mesh.apply_scale([scale, scale, scale])

	# Check z
	z_dim = mesh.bounds[1,-1] - mesh.bounds[0,-1]
	if z_dim > max_z:
		z_scale = max_z/z_dim
		mesh.apply_scale([1, 1, z_scale])
	return mesh


def remove_extra_bodies_from_mesh(mesh):
	# Take the body with the largest volume
	mesh_all = mesh.split()
	if len(mesh_all) == 0:
		return mesh
	volume_all = [m.volume for m in mesh_all]
	return mesh_all[np.argmax(volume_all)]


def center_mesh(mesh):
	xyz_center = (mesh.bounds[1] - mesh.bounds[0])/2
	offset = -(mesh.bounds[0]+xyz_center)
	align_matrix = np.array([[1, 0, 0, offset[0]],
							[0, 1, 0, offset[1]],
							[0, 0, 1, offset[2]],
							[0, 0, 0, 1]])
	mesh.apply_transform(align_matrix)
	return mesh


def section_mesh(mesh):
	section = mesh.section(plane_origin=mesh.centroid, 
							plane_normal=[0,0,1])
	section, _ = section.to_planar()
	split_sections = section.split()
	if len(split_sections) == 0:	# broken mesh
		return None
	areas = [s.area for s in split_sections]
	section = split_sections[np.argmax(areas)]
	return section


def perturb_mesh(mesh_path):
	mesh = trimesh.load(mesh_path)

	# Take cross-sectional area, assume always possible
	section = section_mesh(mesh)

	# Convert to Shapely Polygon, merge vertices if too close, and then interpolate between vertices
	polygon = section.polygons_closed[0]
	vs = list(polygon.exterior.coords)	# extract ordered vertices
	vs_new = []
	for ind, v in enumerate(vs):
		if ind == len(vs)-1:
			continue

		# get neighboring points
		v = array(v)
		v_next = array(vs[ind+1])

		# interpolate
		seg_len = np.linalg.norm(v_next-v)
		num_sub_seg = max(1, int(round(seg_len/0.001)))	# make vertices 1mm apart
		v_interp_all = np.linspace(v, v_next, num_sub_seg)

		for v_interp in v_interp_all:
			vs_new += [tuple(v_interp)]
	if np.linalg.norm(array(vs[-1])-array(vs_new[-1])) > 1e-5:
		vs_new += [vs[-1]]	# add last point if not already included
	num_vert = len(vs_new)

	# Somtimes stuck; quit if too many attempts
	num_attempt = 0
	while 1:
		# Make a copy of all vertices
		vs_perturb = deepcopy(vs_new)

		#* (to be tuned) Choose number of vertices in a segment
		#! This can fail when num_vert_per_seg = 0
		num_vert_per_seg = random.randint(num_vert//10, num_vert//5)
		num_seg = num_vert // num_vert_per_seg	# round down
		# print('number of seg: ', num_seg)

		#* (to be tuned) Choose number of segments to perturb
		num_perturb_seg = max(1,random.randint(num_seg//2, 3*num_seg//4))	# number of segments perturbed
		# print('number of seg perturbed: ', num_perturb_seg)
		perturb_seg_inds = random.sample(range(num_seg), k=num_perturb_seg)	# choose which segments to be perturbed

		# Perturb each segment
		for seg_ind in perturb_seg_inds:
			start_perturb_vert_ind = num_vert_per_seg*seg_ind

			# Choose which two indices to spline
			# TODO: other shape of spline? 1/3 points instead of 2?
			vs_perturb_seg = np.empty((0,2))
			spline_top_ind_left = random.randint(num_vert_per_seg // 4, 
												num_vert_per_seg // 2)
			spline_top_ind_right = random.randint(num_vert_per_seg // 2, 
												3*num_vert_per_seg // 4)
			if spline_top_ind_left == spline_top_ind_right:
				spline_top_ind_right = spline_top_ind_left + 1

			#* (to be tuned) Choose how much to perturb for each vertice, same for now
			perturb_amount = random.randint(1,2)*0.005

			# Find the vertices for the spline
			for perturb_vert_ind in range(num_vert_per_seg):
				
				# Global index
				curr_vert_ind = start_perturb_vert_ind + perturb_vert_ind
				v = array(vs_perturb[curr_vert_ind])
				
				if perturb_vert_ind in [spline_top_ind_left, spline_top_ind_right]:
		
					#* QKFIX: Right now discard object if spline inds out of bound; for some reason this happens sometimes
					if curr_vert_ind == 0 or (curr_vert_ind+1) >= num_vert:
						break	# error will be caught when interpolating
  
					# Estimate normal using neighboring points
					line = array(vs_perturb[curr_vert_ind+1]) - \
		 					array(vs_perturb[curr_vert_ind-1])
					normal = array([line[1], -line[0]])
					normal /= np.linalg.norm(normal)

					v += normal*perturb_amount
				vs_perturb_seg = np.vstack((vs_perturb_seg, v))
			spline_inds = [0,spline_top_ind_left,spline_top_ind_right,-1]

			# 2D spline
			try:
				tck, u = interpolate.splprep([vs_perturb_seg[spline_inds,0], 
											vs_perturb_seg[spline_inds,1]],
											s=0, k=3)
				out = interpolate.splev(np.linspace(0, 1.01, num_vert_per_seg), tck, der=0)
			except:
				continue

			# Modify each perturbed vert
			for perturb_vert_ind in range(num_vert_per_seg):
				v = [out[0][perturb_vert_ind], out[1][perturb_vert_ind]]
				vs_perturb[start_perturb_vert_ind+perturb_vert_ind] = tuple(v)

		# TODO: simplify again (simplify_spline)?
		# Extrude back into 3D
		perturb_polygon = Polygon(vs_perturb)
		try:  # Sometimes perturbed polygon self-intersects and is invalid
			mesh = trimesh.creation.extrude_polygon(perturb_polygon, height=0.05)
			return mesh
		except:
			num_attempt += 1
			if num_attempt >= 3:
				return None


def save_urdf(dir, urdfName, 
              		meshName, 
                	objMass=0.1, 
                 	x_scale=1, y_scale=1, z_scale=1):
	"""
	#* Save URDF file at the specified path with the name. Assume 0.1kg mass and random inertia. Single base link.
	"""

	# Write to an URDF file
	f = open(dir + urdfName + '.urdf', "w+")

	f.write("<?xml version=\"1.0\" ?>\n")
	f.write("<robot name=\"%s.urdf\">\n" % urdfName)

	f.write("\t<link name=\"baseLink\">\n")
	f.write("\t\t<inertial>\n")
	f.write("\t\t\t<origin rpy=\"0 0 0\" xyz=\"0 0 0\"/>\n")
	f.write("\t\t\t\t<mass value=\"%.1f\"/>\n" % objMass)
	f.write("\t\t\t\t<inertia ixx=\"6e-5\" ixy=\"0\" ixz=\"0\" iyy=\"6e-5\" iyz=\"0\" izz=\"6e-5\"/>\n")
	f.write("\t\t</inertial>\n")

	f.write("\t\t<visual>\n")
	f.write("\t\t\t<origin rpy=\"0 0 0\" xyz=\"0 0 0\"/>\n")
	f.write("\t\t\t<geometry>\n")
	f.write("\t\t\t\t<mesh filename=\"%s\" scale=\"%.2f %.2f %.2f\"/>\n" % (meshName, x_scale, y_scale, z_scale))
	f.write("\t\t\t</geometry>\n")
	f.write("\t\t\t<material name=\"yellow\">\n")
	f.write("\t\t\t\t<color rgba=\"0.98 0.84 0.35 1\"/>\n")
	f.write("\t\t\t</material>\n")
	f.write("\t\t</visual>\n")

	f.write("\t\t<collision>\n")
	f.write("\t\t\t<origin rpy=\"0 0 0\" xyz=\"0 0 0\"/>\n")
	f.write("\t\t\t<geometry>\n")
	f.write("\t\t\t\t<mesh filename=\"%s\" scale=\"%.2f %.2f %.2f\"/>\n" % (meshName, x_scale, y_scale, z_scale))
	f.write("\t\t\t</geometry>\n")
	f.write("\t\t</collision>\n")
	f.write("\t</link>\n")
	f.write("</robot>\n")

	f.close()


if __name__ == '__main__':
	pass

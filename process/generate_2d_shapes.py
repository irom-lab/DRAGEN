import numpy as np
from glob import glob
import trimesh
from shapely.geometry import Polygon
from util.mesh import saveURDF


def main():

	num_object = 100
	HEIGHT = 0.05	# fixed height for all 2D objects
	generate_train = True	# generate test data
	generate_shape = 0	# 0 for ractangle; 1 for ellipse; 2 for triangle
	save_folder_name = ''	# TODO: specify save path

	# Generate rectangles
	if generate_shape == 0:
		if generate_train:
			side_range = [0.03,0.06]
		else:
			side_range = [0.02,0.07]
		length_all = np.random.uniform(low=side_range[0], high=side_range[1], size=(num_object,))
		width_all = np.random.uniform(low=side_range[0], high=side_range[1], size=(num_object,))
		for obj_ind in range(num_object):
			length = length_all[obj_ind]
			width = width_all[obj_ind]
			x0 = (-length/2.,-width/2.)	# left bottom
			x1 = (-length/2., width/2)
			x2 = (length/2, width/2)
			x3 = (length/2, -width/2)
			verts = [x0, x1, x2, x3]
			box_polygon = Polygon(verts)
			box_mesh = trimesh.creation.extrude_polygon(box_polygon, height=HEIGHT)

			# Make centroid as origin
			align_matrix = np.array([[1, 0, 0, 0],
									[0, 1, 0, 0],
									[0, 0, 1, -HEIGHT/2],
									[0, 0, 0, 1]])
			box_mesh.apply_transform(align_matrix)

			# Save as stl
			box_mesh.export(save_folder_name + str(obj_ind) + '.stl')

			# Save URDF
			saveURDF(path=save_folder_name, urdfName=str(obj_ind), meshName=str(obj_ind)+'.stl', objMass=0.1, x_scale=1, y_scale=1, z_scale=1)

	# Generate ellipses
	elif generate_shape == 1:
		if generate_train:
			x_radius_range = [0.01, 0.05]
			y_radius_range = [0.01, 0.03]
		else:
			x_radius_range = [0.03, 0.06]
			y_radius_range = [0.02, 0.035]
		x_radius_all = np.random.uniform(low=x_radius_range[0], high=x_radius_range[1], size=(num_object,))
		y_radius_all = np.random.uniform(low=y_radius_range[0], high=y_radius_range[1], size=(num_object,))
		for obj_ind in range(num_object):
			x_radius = x_radius_all[obj_ind]
			y_radius = y_radius_all[obj_ind]
			angles = np.linspace(start=-np.pi, stop=np.pi, num=300)
			x_y = np.vstack((x_radius*np.cos(angles), y_radius*np.sin(angles))).T

			ellipse_polygon = Polygon([tuple(point) for point in x_y])
			ellipse_mesh = trimesh.creation.extrude_polygon(ellipse_polygon, height=HEIGHT)

			# Center
			align_matrix = np.array([[1, 0, 0, 0],
									[0, 1, 0, 0],
									[0, 0, 1, -HEIGHT/2],
									[0, 0, 0, 1]])
			ellipse_mesh.apply_transform(align_matrix)

			# Save as stl
			ellipse_mesh.export(save_folder_name + str(obj_ind) + '.stl')

			# Save URDF
			saveURDF(path=save_folder_name, urdfName=str(obj_ind), meshName=str(obj_ind)+'.stl', objMass=0.1, x_scale=1, y_scale=1, z_scale=1)

	# Generate triangles
	elif generate_shape == 2:
		if generate_train:
			a1_range = [45,90]
			a2_range = [45,60]
			l1_range = [0.03, 0.07] # oppsite to a1 angle
		else:
			a1_range = [60,90]
			a2_range = [20,45]
			l1_range = [0.04, 0.08] 
		a1_all = np.random.uniform(low=a1_range[0], high=a1_range[1], size=(num_object,))*np.pi/180
		a2_all = np.random.uniform(low=a2_range[0], high=a2_range[1], size=(num_object,))*np.pi/180
		a3_all = 2*np.pi-a1_all-a2_all
		l1_all = np.random.uniform(low=l1_range[0], high=l1_range[1], size=(num_object,))
		l2_all = l1_all*np.sin(a1_all)/(np.sin(a1_all)*np.cos(a3_all)+np.sin(a3_all)*np.cos(a1_all))
		l3_all = l1_all*np.sin(a3_all)/(np.sin(a1_all)*np.cos(a3_all)+np.sin(a3_all)*np.cos(a1_all))

		for obj_ind in range(num_object):
			l1 = l1_all[obj_ind]
			l2 = l2_all[obj_ind]
			l3 = l3_all[obj_ind]
			a1 = a1_all[obj_ind]
			a2 = a2_all[obj_ind]

			x0 = (-np.tan(a2/2)*l1/(np.tan(a1/2)+np.tan(a2/2)),-np.tan(a1/2)*np.tan(a2/2)*l1/(np.tan(a1/2)+np.tan(a2/2)))	# left bottom
			x1 = (x0[0]+l3*np.cos(a1), abs(x0[1]+l3*np.sin(a1)))
			x2 = (l1+x0[0], x0[1])
			verts = [x0, x1, x2]
			triangle_polygon = Polygon(verts)
			triangle_mesh = trimesh.creation.extrude_polygon(triangle_polygon, height=HEIGHT)

			# Center
			xy_center = (triangle_mesh.bounds[1,:2] - triangle_mesh.bounds[0,:2])/2
			offset = -(triangle_mesh.bounds[0,:2]+xy_center)
			align_matrix = np.array([[1, 0, 0, offset[0]],
									[0, 1, 0, offset[1]],
									[0, 0, 1, -HEIGHT/2],
									[0, 0, 0, 1]])
			triangle_mesh.apply_transform(align_matrix)

			# Save as stl
			triangle_mesh.export(save_folder_name + str(obj_ind) + '.stl')

			# Save URDF
			saveURDF(path=save_folder_name, urdfName=str(obj_ind), meshName=str(obj_ind)+'.stl', objMass=0.1, x_scale=1, y_scale=1, z_scale=1)


if __name__ == '__main__':
	main()


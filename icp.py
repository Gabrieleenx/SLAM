import numpy as np 
import random

class ICP(object):
    def __init__(self):
        self.max_layers = 15
        self.sphere_layers = return_sphere(self.max_layers)
        self.occ_t = 150
        self.num_points = 400
        


    def icp(self, scan, map_, pos_xyz, rot_zyx):
        # in
        # map, 3d grid 
        # scan point cloud
        # pos_xyz = size(3,1)
        # rot_zyx = size(3,1)
        # out 
        # pos_xyz = size(3,1)
        # rot_zyx = size(3,1)  
        # guess work at its best
        var_tol = 1 
        est_var = 100
        max_iter = 5
        itr = 0
        while est_var > var_tol and itr <= max_iter:
            itr += 1
            scan_samp, map_samp = self.find_closest_points(scan, map_, pos_xyz, rot_zyx)
            Rot, Tran = find_rigid_alignment(map_samp, scan_samp)
            #print(Rot, Tran)
            rot_zyx[0,0] += np.arctan2(Rot[1,0], Rot[0,0])
            pos_xyz = Rot.dot(pos_xyz) + Tran.reshape((3,1))
            # to be continiued
        return pos_xyz, rot_zyx




    def find_closest_points(self, scan, map_, pos_xyz, rot_zyx):
        # in
        # map, 3d grid 
        # scan point cloud
        # pos_xyz = size(3,1)
        # rot_zyx = size(3,1)
        # out
        # cor_points = pointcloud with size(scan) 
        map_resolution_m = 0.05
        map_y = 16
        map_x = 16
        map_z = 6
        reshape_index_m = np.array([int(map_y//map_resolution_m) * int(map_z//map_resolution_m), 
                                    int(map_z//map_resolution_m), 1])
        # occupied treshold
        occ_t = 150
        rand_index = random.sample(range(0, scan.shape[0]), self.num_points)
        scan_cloud = scan[rand_index]
        

        scan_rot = Rotate_zyx(rot_zyx[0,0], rot_zyx[1,0], rot_zyx[2,0], scan_cloud)
        scan_rot_T = Translate_points(pos_xyz[0,0], pos_xyz[1,0], pos_xyz[2,0], scan_rot)

        map_point_index = scan_rot_T/map_resolution_m

        cor_points = np.zeros(scan_rot_T.shape)
        #print(cor_points)
        #print(scan.shape)
        j = -1
        for point in map_point_index:
            j += 1
            i = 0
            if point[0] < 20 or point[1] < 20 or point[2] < 20: 
                cor_points[j] = point
                #print('no point')
                continue
            if point[0] > 300 or point[1] > 300 or point[2] > 100: 
                cor_points[j] = point
                #print('no point')
                continue

            while True:
                point_ = np.rint(point)
                indices = self.sphere_layers[i] + point_.reshape(3,1).astype(int)
                values = map_[indices[0], indices[1], indices[2]]

                #print(values)
                val_g = np.greater(values, self.occ_t)
                if any(val_g):
                    #print('----------point--------')
                    idex = np.argwhere(values > self.occ_t)
                    cor_points[j] = np.array([indices[0, idex[0,0]], indices[1, idex[0,0]], indices[2, idex[0,0]]])
                    break

                i += 1
                if i == self.max_layers:
                    cor_points[j] = point
                    #print('no point')
                    break

        return scan_rot_T, cor_points*map_resolution_m




def Rotate_zyx(Rz, Ry, Rx, cloudpoints):
    RotM_z = np.array([[np.cos(Rz), -np.sin(Rz), 0],
                    [np.sin(Rz), np.cos(Rz), 0],
                    [0, 0, 1]]) 
    
    RotM_Y = np.array([[np.cos(Ry), 0, np.sin(Ry)],
                    [0, 1, 0],
                    [-np.sin(Ry), 0, np.cos(Ry)]]) 

    RotM_X = np.array([[1, 0, 0],
                    [0, np.cos(Rx), -np.sin(Rx)],
                    [0, np.sin(Rx), np.cos(Rx)]]) 

    R_zyx = RotM_z.dot(RotM_Y).dot(RotM_X)

    point_size = cloudpoints.shape[0]

    new_points = np.einsum('ij,nj->ni', R_zyx, cloudpoints)

    return new_points

def Translate_points(Tx, Ty, Tz, cloudpoints):
    cloudpoints_ = cloudpoints + np.array([Tx, Ty, Tz])
    return cloudpoints_

def return_sphere(layers=15):
    sphere = []
    sphere.append(np.array([[0],[0],[0]]))
    for i in range(layers):
        side = 2*i + 3 # side lenght 
        x_list = []
        y_list = []
        z_list = []
        for jx in range(side):
            x = jx - (i+1)
            for jy in range(side):
                y = jy - (i+1)
                for jz in range(side):
                    z = jz - (i+1)

                    if np.linalg.norm([x, y, z]) <= (i+1.05):
                        if np.linalg.norm([x, y, z]) <= (i+0.05):
                            continue
                        else:
                            x_list.append(x)
                            y_list.append(y)
                            z_list.append(z)
        sphere.append(np.array([x_list, y_list, z_list]))

    return sphere


def find_rigid_alignment(A, B):
    # code from John Lambert
	"""
	2-D or 3-D registration with known correspondences.
	Registration occurs in the zero centered coordinate system, and then
	must be transported back.

		Args:
		-	A: Numpy array of shape (N,D) -- Reference Point Cloud (target)
		-	B: Numpy array of shape (N,D) -- Point Cloud to Align (source)

		Returns:
		-	R: optimal rotation
		-	t: optimal translation
	"""
	num_pts = A.shape[0]
	dim = A.shape[1]

	a_mean = np.mean(A, axis=0)
	b_mean = np.mean(B, axis=0)

	# Zero-center the point clouds
	A -= a_mean
	B -= b_mean

	N = np.zeros((dim, dim))
	for i in range(num_pts):
		N += A[i].reshape(dim,1).dot( B[i].reshape(1,dim) )
	N = A.T.dot(B)

	U, D, V_T = np.linalg.svd(N)
	S = np.eye(dim)
	det = np.linalg.det(U) * np.linalg.det(V_T.T)
	
	# Check for reflection case
	if not np.isclose(det,1.):
		S[dim-1,dim-1] = -1

	R = U.dot(S).dot(V_T)
	t = R.dot( b_mean.reshape(dim,1) ) - a_mean.reshape(dim,1)
	return R, -t.squeeze()

'''                        
indeces = loc_cloud_rot.astype(int).dot(self.reshape_index_m)
indeces = indeces[indeces>=0]
indeces = indeces[indeces <= self.map_max_index]
sum_ = np.sum(self.map_xyz.reshape(-1)[indeces])
'''
if __name__ == '__main__':
    #print(return_sphere())
    ICP()






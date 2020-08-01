import numpy as np
import cupy as cu
import cupyx


map_x = 30
map_y = 30
map_z = 6
map_resolution_m = 0.05 # 0.05 m resolution
map_xyz = np.zeros((int(map_x//map_resolution_m), int(map_y//map_resolution_m), int(map_z//map_resolution_m)), dtype=np.int32)

reshape_index_m = cu.array(map_xyz)
indeces = cu.array([0, 1.1,2, 5,8])
indeces = indeces.astype(np.int32)

cupyx.scatter_add(reshape_index_m.reshape(-1), indeces, 9)
indeces.astype(np.int32)
print(indeces)
#indeces = cu.asnumpy(indeces) 
Rz =  cu.asnumpy(indeces[1]) 
#Rz = 2
RotM_z = np.array([[np.cos(Rz), -np.sin(Rz), 0],
                        [np.sin(Rz), np.cos(Rz), 0],
                        [0, 0, 1]]) 
cu.array(RotM_z)
print('hello',cu.zeros((4,4)), Rz,)

# should work :)

# also fix rounding approx for hopeful better performance
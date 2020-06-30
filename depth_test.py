#!/home/gabriel/anaconda3/bin/python

import rospy
from sensor_msgs.msg import Image, PointCloud, Imu
from geometry_msgs.msg import Point32, Quaternion
import std_msgs.msg
import message_filters
import numpy as np
from struct import *
import sys


#np.set_printoptions(threshold=sys.maxsize)
class SLAM(object):
    def __init__(self, height, width):
        # camera
        self.dist_y_img = np.zeros((height, width), dtype=np.uint16) # y = deapth, x = horizontal, z = vertical...
        self.dist_x_img = np.zeros((height, width), dtype=int)
        self.dist_z_img = np.zeros((height, width), dtype=int)
        self.height = height
        self.Rz = 0
        self.width = width
        self.focal_point = 385.0
        self.horizontal_distance = np.reshape(np.arange(640)-320, (1, 640)) / self.focal_point
        self.vertical_distance = np.reshape(np.arange(480)-240, (480, 1)) / self.focal_point
        #map
        self.map_x = 10
        self.map_y = 10
        self.map_z = 6
        self.map_resolution_m = 0.05 # 0.05 m resolution
        self.map_xyz = np.zeros((int(self.map_x//self.map_resolution_m), int(self.map_y//self.map_resolution_m), int(self.map_z//self.map_resolution_m)))

        self.reshape_index_m = np.array([int(self.map_y//self.map_resolution_m) * int(self.map_z//self.map_resolution_m), 
                                        int(self.map_z//self.map_resolution_m), 1])
        self.map_max_index =int(self.map_x//self.map_resolution_m) * int(self.map_y//self.map_resolution_m) * int(self.map_z//self.map_resolution_m) -1
        # rotation and position
        self.euler_zyx = np.array([[0.0], [0.0], [0.0]])
        self.pos_xyz = np.array([[self.map_x], [self.map_y], [self.map_z]])/(2)

        # publish
        self.pub = rospy.Publisher('/PointCloud', PointCloud, queue_size=1)


    def callback(self, data):
        print('thats up')
        # convert byte data to numpy 2d array
        k = np.frombuffer(data.data, dtype=np.uint16)
        self.dist_y_img = k.reshape((self.height, self.width))
        self.stamp = data.header.stamp
        self.frame_id = data.header.frame_id
        self.seq = data.header.seq 
        self.dist_x_img = np.multiply(self.dist_y_img, self.horizontal_distance)
        self.dist_z_img = np.multiply(self.dist_y_img, self.vertical_distance)
        self.cloudpoints = self.img_to_cloudpoints()

        T_cloud = self.Rotate_zyx_Translate_points(Rz=0, Ry=0, Rx=0, Tx=self.pos_xyz[0, 0], Ty=self.pos_xyz[1, 0] , Tz=self.pos_xyz[2, 0])
        self.add_points_to_map(T_cloud)
        # will be on its own therad in future...
        self.Publich_PointCloud(T_cloud)

    def img_to_cloudpoints(self):
        cloudpoints = np.zeros((self.height*self.width, 3))
        delete_points = np.array([]) 
        xx = self.dist_x_img / 1000
        yy = self.dist_y_img / 1000
        zz = self.dist_z_img / 1000
        img = np.array([xx, yy, zz])
        cloudpoints = img.transpose().reshape(self.height*self.width,3)

        return cloudpoints[~np.all(cloudpoints == 0, axis=1)]

    def Publich_PointCloud(self, cloudpoints):
        cloud_msg = PointCloud()
        cloud_msg.header.stamp = self.stamp
        cloud_msg.header.frame_id = self.frame_id
        cloud_msg.header.seq = self.seq
        # want a faster method for convertion to array of Point32. 
        for i in range(cloudpoints.shape[0]//10):
            j = i*10
            cloud_msg.points.append(Point32(-cloudpoints[j, 1], cloudpoints[j, 2], cloudpoints[j, 0])) 
            # Rviz vizualization is retarded, -y,z,x instead of xyz!! might be bug in Rviz for pointcloud. 
        self.pub.publish(cloud_msg)

    def Rotate_zyx_Translate_points(self, Rz, Ry, Rx, Tx, Ty, Tz):
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

        point_size = self.cloudpoints.shape[0]

        # So fast for rotation of a large pointcloud
        new_points = np.einsum('ij,nj->ni', R_zyx, self.cloudpoints)

        # Too slow ... 
        '''
        new_points = np.zeros((point_size, 3))
        for i in range(point_size):
            new_points[i] = R_zyx.dot(self.cloudpoints[i])
        '''
        new_points += np.array([Tx, Ty, Tz])

        return new_points

    def add_points_to_map(self, T_cloud):
        #faster
        map_point_index = T_cloud/self.map_resolution_m
        self.map_xyz += -0.05        
        indeces = map_point_index.astype(int).dot(self.reshape_index_m)
        indeces = indeces[indeces>=0]
        indeces = indeces[indeces<= self.map_max_index]
        np.add.at(self.map_xyz.reshape(-1), indeces, 0.1)
        self.map_xyz = np.clip(self.map_xyz, 0, 1)

        # too slow, need a faster method...      
        #for i in range(indeces.shape[1]):
        #    self.map_xyz[indeces[0, i],indeces[1, i],indeces[2, i]] += 0.1
        #self.map_xyz[np.where(self.map_xyz < 0)] = 0
        #self.map_xyz[np.where(self.map_xyz > 1)] = 1

        

def listener():
    print('Hi')
    slam = SLAM(480, 640)
  
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/camera/depth/image_rect_raw", Image, slam.callback)
    rospy.spin()


if __name__ == '__main__':

    listener()

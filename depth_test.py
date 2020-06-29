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
        self.dist_y_img = np.zeros((height, width), dtype=np.uint16) # y = deapth, x = horizontal, z = vertical...
        self.dist_x_img = np.zeros((height, width), dtype=int)
        self.dist_z_img = np.zeros((height, width), dtype=int)
        self.height = height
        self.Rz = 0
        self.width = width
        self.focal_point = 385.0
        self.horizontal_distance = np.reshape(np.arange(640)-320, (1, 640)) / self.focal_point
        self.vertical_distance = np.reshape(np.arange(480)-240, (480, 1)) / self.focal_point
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

        T_cloud = self.Rotate_zyx_Translate_points(Rz=0, Ry=0, Rx=0, Tx=0, Ty=0, Tz=0)

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



def listener():
    print('Hi')
    slam = SLAM(480, 640)
  
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/camera/depth/image_rect_raw", Image, slam.callback)
    rospy.spin()


if __name__ == '__main__':

    listener()

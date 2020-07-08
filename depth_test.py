#!/home/gabriel/anaconda3/bin/python

import rospy
import cv2
from sensor_msgs.msg import Image, PointCloud, Imu
from geometry_msgs.msg import Point32, Quaternion, Pose, PoseStamped
import std_msgs.msg
import message_filters
import numpy as np
from struct import *
import sys
import random
from orientation_estimate import Orientation_estimate

#np.set_printoptions(threshold=sys.maxsize)
class SLAM(object):
    def __init__(self, height, width, downsample):
        # camera
        self.dist_y_img = np.zeros((height, width), dtype=np.uint16) # y = deapth, x = horizontal, z = vertical...
        self.dist_x_img = np.zeros((height, width), dtype=int)
        self.dist_z_img = np.zeros((height, width), dtype=int)
        self.height = height
        self.Rz = 0
        self.width = width
        self.downsample = downsample
        self.d_height = int(self.height/self.downsample)
        self.d_width = int(self.width/self.downsample)
        self.focal_point = 385.0
        self.horizontal_distance = np.reshape(np.arange(self.d_width)-int(self.d_width/2), (1, self.d_width)) / (self.focal_point/self.downsample)
        self.vertical_distance = -1*np.reshape(np.arange(self.d_height)-int(self.d_height/2), (self.d_height, 1)) / (self.focal_point/self.downsample)

        #map
        self.map_x = 16
        self.map_y = 16
        self.map_z = 6
        self.map_resolution_m = 0.05 # 0.05 m resolution
        self.map_xyz = np.zeros((int(self.map_x//self.map_resolution_m), int(self.map_y//self.map_resolution_m), int(self.map_z//self.map_resolution_m)))
        self.reshape_index_m = np.array([int(self.map_y//self.map_resolution_m) * int(self.map_z//self.map_resolution_m), 
                                        int(self.map_z//self.map_resolution_m), 1])
        self.map_max_index =int(self.map_x//self.map_resolution_m) * int(self.map_y//self.map_resolution_m) * int(self.map_z//self.map_resolution_m) -1
        self.map_threshold = 0.4
        self.n_remove = 1000
        self.no_map = 1
        self.uncertain = 0
        # rotation and position
        self.euler_zyx = np.array([[0.0], [0.0], [0.0]])
        self.pos_xyz = np.array([[self.map_x], [self.map_y], [self.map_z]])/(2)
        self.oritentaion = Orientation_estimate()
        self.loc_certainty = 0.0
        # publish
        self.pub = rospy.Publisher('/PointCloud', PointCloud, queue_size=3)
        self.pub_pose = rospy.Publisher('/Pose', PoseStamped, queue_size=3)

    def callback(self, data, gyr_data, acc_data):
        
        # convert byte data to numpy 2d array
        k = np.frombuffer(data.data, dtype=np.uint16)
        dist_y_img = k.reshape((self.height, self.width))

        self.dist_y_img = cv2.resize(dist_y_img, dsize=(self.d_width, self.d_height), interpolation=cv2.INTER_CUBIC)
       
        self.stamp = data.header.stamp
        self.frame_id = data.header.frame_id
        self.seq = data.header.seq 
        self.dist_x_img = np.multiply(self.dist_y_img, self.horizontal_distance)
        self.dist_z_img = np.multiply(self.dist_y_img, self.vertical_distance)
        self.cloudpoints = self.img_to_cloudpoints()
        self.euler_zyx = self.oritentaion.callback(gyr_data, acc_data)
        self.localization()
        #print(self.euler_zyx)
        print('FPS = ', 1/self.oritentaion.delta_T)
        T_cloud = self.Rotate_zyx_Translate_points(Rz=self.euler_zyx[0,0], Ry=self.euler_zyx[1,0],
            Rx=self.euler_zyx[2,0], Tx=self.pos_xyz[0, 0], Ty=self.pos_xyz[1, 0],
            Tz=self.pos_xyz[2, 0], cloudpoints=self.cloudpoints)
        self.add_points_to_map(T_cloud)
        # will be on its own therad in future...
        #Map_point_cloud = self.Map_to_point_cloud()
        #self.Publich_PointCloud(Map_point_cloud)
        self.Publish_Pose()

    def img_to_cloudpoints(self):
        cloudpoints = np.zeros((self.d_height*self.d_width, 3))
        delete_points = np.array([]) 
        xx = self.dist_x_img / 1000
        yy = self.dist_y_img / 1000
        zz = self.dist_z_img / 1000
        img = np.array([xx, yy, zz])
        cloudpoints = img.transpose().reshape(self.d_height*self.d_width,3)

        return cloudpoints[~np.all(cloudpoints == 0, axis=1)]

    def Publich_PointCloud(self):
        cloudpoints = self.Map_to_point_cloud()
        cloud_msg = PointCloud()
        cloud_msg.header.stamp = self.stamp
        cloud_msg.header.frame_id = self.frame_id
        cloud_msg.header.seq = self.seq
        # want a faster method for convertion to array of Point32. 
        for i in range(cloudpoints.shape[0]//10):
            j = i*10
            cloud_msg.points.append(Point32(cloudpoints[j, 0], -cloudpoints[j, 2], cloudpoints[j, 1])) 
            # Change to camera frame
        self.pub.publish(cloud_msg)

    def Rotate_zyx_Translate_points(self, Rz, Ry, Rx, Tx, Ty, Tz, cloudpoints):
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

        # So fast for rotation of a large pointcloud
        new_points = np.einsum('ij,nj->ni', R_zyx, cloudpoints)

        # Too slow ... 
        '''
        new_points = np.zeros((point_size, 3))
        for i in range(point_size):
            new_points[i] = R_zyx.dot(self.cloudpoints[i])
        '''
        new_points += np.array([Tx, Ty, Tz])

        return new_points

    def Rotate_zyx(self, Rz, Ry, Rx, cloudpoints):
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

        # So fast for rotation of a large pointcloud
        new_points = np.einsum('ij,nj->ni', R_zyx, cloudpoints)

        return new_points

    def Translate_points(self, Tx, Ty, Tz, cloudpoints):
        cloudpoints_ = cloudpoints + np.array([Tx, Ty, Tz])
        return cloudpoints_


    def add_points_to_map(self, T_cloud):
        if self.loc_certainty >= 0.8:
            self.no_map = 0
        if self.loc_certainty >= 0.6 or self.no_map:
            map_point_index = T_cloud/self.map_resolution_m
            remove_index = self.remove_old_points(T_cloud).astype(int).dot(self.reshape_index_m)
            remove_index = remove_index[remove_index>=0]
            remove_index = remove_index[remove_index <= self.map_max_index]
            np.add.at(self.map_xyz.reshape(-1), remove_index, -0.03*(self.downsample))
            indeces = map_point_index.astype(int).dot(self.reshape_index_m)
            indeces = indeces[indeces>=0] # might exist faster methods
            indeces = indeces[indeces <= self.map_max_index]
            np.add.at(self.map_xyz.reshape(-1), indeces, 0.005*(self.downsample**2))
            self.map_xyz.clip(0, 1, out= self.map_xyz) 
        else:
            self.uncertain = 1
            print('uncertain location')

        # too slow, need a faster method...      
        #for i in range(indeces.shape[1]):
        #    self.map_xyz[indeces[0, i],indeces[1, i],indeces[2, i]] += 0.1
        #self.map_xyz[np.where(self.map_xyz < 0)] = 0
        #self.map_xyz[np.where(self.map_xyz > 1)] = 1

    def remove_old_points(self, T_cloud):
        rand_index = random.sample(range(0, T_cloud.shape[0]), self.n_remove)
        rand_map_pos = T_cloud[rand_index] 
        delta = rand_map_pos - self.pos_xyz.transpose()[0]
        dist = np.linalg.norm(delta, axis=1)
        n_points = dist/self.map_resolution_m
        n_points = n_points.astype(int)
        total_sum_points = n_points.sum() # + 26*self.n_remove # too be added
        remove_indeces = np.zeros((total_sum_points, 3))
        pos = self.pos_xyz.transpose()[0]
        last_point = 0
        for i in range(self.n_remove):
            remove_indeces[last_point:last_point+n_points[i],:] = np.linspace(pos, rand_map_pos[i], n_points[i])
            last_point += n_points[i]
        return remove_indeces/self.map_resolution_m

    def localization(self):
        num_points = 300
        sum_prob = 0

        rand_index = random.sample(range(0, self.cloudpoints.shape[0]), num_points)
        loc_cloud = self.cloudpoints[rand_index]
        rot_z_loc = 0
        Tx_loc = 8
        Ty_loc = 8
        Tz_loc = 3
        length = 0.25
        rotation_view = 40
        map_conv = 1/self.map_resolution_m
        if self.uncertain: # not working too well 
            rotation_view = 120

        n_steps = int(length / self.map_resolution_m)
        for i in range(int(rotation_view/3)):
            rot_z = self.euler_zyx[0,0] + (-rotation_view/2 + 3*i)*np.pi/180
            cloud_rot = self.Rotate_zyx(Rz=rot_z, Ry=self.euler_zyx[1,0],
                        Rx=self.euler_zyx[2,0], cloudpoints=loc_cloud)
            '''
            loc_cloud_rot = self.Translate_points(Tx=8, Ty=8, Tz=3, cloudpoints=cloud_rot)
            loc_cloud_rot = loc_cloud_rot/self.map_resolution_m
            indeces = loc_cloud_rot.astype(int).dot(self.reshape_index_m)
            indeces = indeces[indeces>=0]
            indeces = indeces[indeces <= self.map_max_index]
            if np.sum(self.map_xyz.reshape(-1)[indeces]) > sum_prob:
                sum_prob = np.sum(self.map_xyz.reshape(-1)[indeces])
                rot_z_loc = rot_z
            '''                 
            for j in range(n_steps):
                T_x = self.pos_xyz[0, 0] + j*self.map_resolution_m - length/2
                for k in range(n_steps):  
                    T_y = self.pos_xyz[1, 0] + k*self.map_resolution_m - length/2
                    for z in range(n_steps): 
                        T_z = self.pos_xyz[2, 0] + z*self.map_resolution_m - length/2
                        #print(T_x, T_y, T_z)
                        loc_cloud_rot = self.Translate_points(Tx=T_x, Ty=T_y, Tz=T_z, cloudpoints=cloud_rot)
                        loc_cloud_rot = loc_cloud_rot*map_conv
                        indeces = loc_cloud_rot.astype(int).dot(self.reshape_index_m)
                        indeces = indeces[indeces>=0]
                        indeces = indeces[indeces <= self.map_max_index]
                        if np.sum(self.map_xyz.reshape(-1)[indeces]) > sum_prob:
                            sum_prob = np.sum(self.map_xyz.reshape(-1)[indeces])
                            rot_z_loc = rot_z
                            #print('in', rot_z, T_x, T_y, T_z)
                            Tx_loc = T_x
                            Ty_loc = T_y
                            Tz_loc = T_z
        
        self.loc_certainty = sum_prob/num_points
        if self.loc_certainty >= 0.6 or self.no_map:
            self.euler_zyx[0,0] = rot_z_loc
            self.pos_xyz[0, 0] = Tx_loc
            self.pos_xyz[1, 0] = Ty_loc
            self.pos_xyz[2, 0] = Tz_loc
            if self.loc_certainty >= 0.75:
                self.oritentaion.quaternions = self.zyx_to_quat(self.euler_zyx[0,0], self.euler_zyx[1,0],self.euler_zyx[2,0])
        print('rot z', rot_z_loc, Tx_loc, Ty_loc, Tz_loc, self.loc_certainty)
    
    def zyx_to_quat(self, z, y, x):
        quat = np.array([[np.cos(x/2) * np.cos(y/2) * np.cos(z/2) + np.sin(x/2) * np.sin(y/2) * np.sin(z/2)],
        [np.sin(x/2) * np.cos(y/2) * np.cos(z/2) - np.cos(x/2) * np.sin(y/2) * np.sin(z/2)],
        [np.cos(x/2) * np.sin(y/2) * np.cos(z/2) + np.sin(x/2) * np.cos(y/2) * np.sin(z/2)],
        [np.cos(x/2) * np.cos(y/2) * np.sin(z/2) - np.sin(x/2) * np.sin(y/2) * np.cos(z/2)]])
        return quat
        
    def Map_to_point_cloud(self):
        index = np.argwhere(self.map_xyz[:,:,:] > self.map_threshold)
        #int(0.4*self.map_z//self.map_resolution_m):int(0.6*self.map_z//self.map_resolution_m)
        return index * self.map_resolution_m

    def Publish_Pose(self):
        quat_np = self.zyx_to_quat(0, -self.euler_zyx[0,0]-np.pi/2, 0) # only z rotation :(
        P = Pose()
        #P.header.frame_id = self.frame_id
        P.orientation.w = quat_np[0,0]
        P.orientation.x = quat_np[1,0]
        P.orientation.y = quat_np[2,0]
        P.orientation.z = quat_np[3,0]

        P.position.x = self.pos_xyz[0, 0]
        P.position.y = -self.pos_xyz[2, 0]
        P.position.z = self.pos_xyz[1, 0]

        msg = PoseStamped()
        msg.header.frame_id = self.frame_id
        msg.pose = P
        self.pub_pose.publish(msg)
        

    
      
'''
def listener():
    print('Hi')
    slam = SLAM(480, 640)
  
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/camera/depth/image_rect_raw", Image, slam.callback)
    rospy.spin()
'''
def listener():
    slam = SLAM(480, 640, 2) # height, widht, downsample
    rospy.init_node('listener', anonymous=True)
    depth_camera = message_filters.Subscriber("/camera/depth/image_rect_raw", Image)
    gyr_sub = message_filters.Subscriber("/camera/gyro/sample", Imu)
    acc_sub = message_filters.Subscriber("/camera/accel/sample", Imu)

    # using approximate synchronizer. 
    ts = message_filters.ApproximateTimeSynchronizer([depth_camera, gyr_sub, acc_sub], 1, 0.2, allow_headerless=False)
    ts.registerCallback(slam.callback)
    while not rospy.is_shutdown():
        slam.Publich_PointCloud()
        rospy.sleep(2)

    #rospy.spin()

if __name__ == '__main__':

    listener()

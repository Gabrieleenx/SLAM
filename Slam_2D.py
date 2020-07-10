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

# work in progress... #notworking:) # some rotaion of points not following :(
#np.set_printoptions(threshold=sys.maxsize)
class SLAM(object):
    def __init__(self, height, width, downsample):
        # camera
        self.dist_y_img = np.zeros((height, width), dtype=np.uint16) # y = deapth, x = horizontal, z = vertical...
        self.dist_x_img = np.zeros((height, width), dtype=int)
        self.height = height
        self.Rz = 0
        self.width = width
        self.downsample = downsample
        self.d_height = int(self.height/self.downsample)
        self.d_width = int(self.width/self.downsample)
        self.focal_point = 385.0
        self.horizontal_distance = np.reshape(np.arange(self.d_width)-int(self.d_width/2), (1, self.d_width)) / (self.focal_point/self.downsample)

        #map
        self.map_x = 16
        self.map_y = 16
        self.map_resolution_m = 0.04 # 0.05 m resolution
        self.map_xyz = np.zeros((int(self.map_x//self.map_resolution_m), int(self.map_y//self.map_resolution_m)), dtype=np.int16)
        self.reshape_index_m = np.array([int(self.map_y//self.map_resolution_m), 1])
        self.map_max_index =int(self.map_x//self.map_resolution_m) * int(self.map_y//self.map_resolution_m) -1
        self.map_threshold = 100 #0.4
        #self.n_remove = self.width
        self.no_map = 1
        self.uncertain = 0
        # rotation and position
        self.euler_zyx = np.array([[0.0], [0.0], [0.0]])
        self.pos_xyz = np.array([[self.map_x], [self.map_y]])/(2)
        self.oritentaion = Orientation_estimate()
        self.loc_certainty = 0.0
        # publish
        self.pub = rospy.Publisher('/PointCloud', PointCloud, queue_size=3)
        self.pub_pose = rospy.Publisher('/Pose', PoseStamped, queue_size=3)
        
        self.fps_list = np.array([])

    def callback(self, data, gyr_data, acc_data):
        # convert byte data to numpy 2d array
        k = np.frombuffer(data.data, dtype=np.uint16)
        dist_y_img = k.reshape((self.height, self.width))
        min_h = int(self.width/2) - 10
        max_h = int(self.width/2) + 10
        self.dist_y_img = np.mean(dist_y_img[min_h:max_h,:], axis=0)
        self.stamp = data.header.stamp
        self.frame_id = data.header.frame_id
        self.seq = data.header.seq 
        self.dist_x_img = np.multiply(self.dist_y_img, self.horizontal_distance)
        self.cloudpoints = self.img_to_cloudpoints()
        self.euler_zyx = self.oritentaion.callback(gyr_data, acc_data)
        self.localization()
        
        self.fps_list = np.append(self.fps_list, [self.oritentaion.delta_T])
        print('FPS = ', 1/self.oritentaion.delta_T, 1/np.mean(self.fps_list))
        T_cloud = self.Rotate_zyx(Rz=self.euler_zyx[0,0], cloudpoints=self.cloudpoints)
        T_cloud = self.Translate_points(Tx=self.pos_xyz[0, 0], Ty=self.pos_xyz[1, 0], cloudpoints=T_cloud)
       
        self.add_points_to_map(T_cloud)
        #self.Publich_PointCloud()

        self.Publish_Pose()
    
    def img_to_cloudpoints(self):
        cloudpoints = np.zeros((self.d_height*self.d_width, 3))
        delete_points = np.array([]) 
        xx = self.dist_x_img[0] / 1000
        yy = self.dist_y_img / 1000
        img = np.array([xx, yy])
        cloudpoints = img.transpose().reshape(self.d_width,2)

        return cloudpoints[~np.all(cloudpoints == 0, axis=1)]
    
    def Publich_PointCloud(self):
        cloudpoints = self.Map_to_point_cloud()
        cloud_msg = PointCloud()
        cloud_msg.header.stamp = self.stamp
        cloud_msg.header.frame_id = self.frame_id
        cloud_msg.header.seq = self.seq
        # want a faster method for convertion to array of Point32. 
        for i in range(cloudpoints.shape[0]):
            j = i
            cloud_msg.points.append(Point32(cloudpoints[j, 0], 0, cloudpoints[j, 1])) 
            # Change to camera frame
        self.pub.publish(cloud_msg)

    def Publich_PointCloud_2(self, cloudpoints):
        cloud_msg = PointCloud()
        cloud_msg.header.stamp = self.stamp
        cloud_msg.header.frame_id = self.frame_id
        cloud_msg.header.seq = self.seq
        # want a faster method for convertion to array of Point32. 
        for i in range(cloudpoints.shape[0]):
            j = i
            cloud_msg.points.append(Point32(cloudpoints[j, 0], 0, cloudpoints[j, 1])) 
            # Change to camera frame
        self.pub.publish(cloud_msg)
    def Rotate_zyx(self, Rz, cloudpoints):
        RotM_z = np.array([[np.cos(Rz), -np.sin(Rz)],
                        [np.sin(Rz), np.cos(Rz)]]) 
    
        # So fast for rotation of a large pointcloud
        new_points = np.einsum('ij,nj->ni', RotM_z, cloudpoints)

        return new_points

    def Translate_points(self, Tx, Ty, cloudpoints):
        cloudpoints_ = cloudpoints + np.array([Tx, Ty])
        return cloudpoints_


    def add_points_to_map(self, T_cloud):
        if self.loc_certainty >= 0.7:
            self.no_map = 0
        if self.loc_certainty >= 0.6 or self.no_map:
            self.uncertain = 0
            map_point_index = T_cloud/self.map_resolution_m
            remove_index = self.remove_old_points(T_cloud).astype(int).dot(self.reshape_index_m)
            remove_index = remove_index[remove_index>=0]
            remove_index = remove_index[remove_index <= self.map_max_index]
            np.add.at(self.map_xyz.reshape(-1), remove_index, int(-1))
            indeces = map_point_index.astype(int).dot(self.reshape_index_m)
            indeces = indeces[indeces>=0] # might exist faster methods
            indeces = indeces[indeces <= self.map_max_index]
            np.add.at(self.map_xyz.reshape(-1), indeces, int(20))
            self.map_xyz.clip(0, 300, out= self.map_xyz) 
            #self.Publich_PointCloud_2(cloudpoints=T_cloud)
            #self.map_xyz.reshape(-1)[indeces] = self.map_xyz.reshape(-1)[indeces].clip(0,300)
        else:
            self.uncertain = 1
            print('uncertain location')

    def remove_old_points(self, T_cloud):
        #rand_index = random.sample(range(0, T_cloud.shape[0]), self.n_remove)
        n_remove = T_cloud.shape[0]
        
        rand_map_pos = T_cloud
        delta = T_cloud - self.pos_xyz.transpose()[0]
        delta = delta*1.08
        rand_map_pos = delta + self.pos_xyz.transpose()[0]
        dist = np.linalg.norm(delta, axis=1)
        n_points = dist/self.map_resolution_m
        n_points = n_points.astype(int)
        total_sum_points = n_points.sum() # + 26*self.n_remove # too be added
        remove_indeces = np.zeros((total_sum_points, 2))
        pos = self.pos_xyz.transpose()[0]
        last_point = 0
        for i in range(n_remove):
            remove_indeces[last_point:last_point+n_points[i],:] = np.linspace(pos, rand_map_pos[i], n_points[i])
            last_point += n_points[i]
        return remove_indeces/self.map_resolution_m

    def localization(self):
        
        sum_prob = 0
        #rand_index = random.sample(range(0, self.cloudpoints.shape[0]), num_points)
        loc_cloud = np.copy(self.cloudpoints)
        num_points = self.cloudpoints.shape[0]
        rot_z_loc = 0
        Tx_loc = 8
        Ty_loc = 8
        length = 0.25
        rotation_view = 40
        map_conv = 1/self.map_resolution_m
        if self.uncertain: # not working too well 
            rotation_view = 120
            length = 0.5

        n_steps = int(length / self.map_resolution_m)
        for i in range(int(rotation_view/3)):
            rot_z = self.euler_zyx[0,0] + (-rotation_view/2 + 3*i)*np.pi/180
            cloud_rot = self.Rotate_zyx(Rz=rot_z, cloudpoints=loc_cloud)
                        
            for j in range(n_steps):
                T_x = self.pos_xyz[0, 0] + j*self.map_resolution_m - length/2
                for k in range(n_steps):  
                    T_y = self.pos_xyz[1, 0] + k*self.map_resolution_m - length/2

                    #print(T_x, T_y, T_z)
                    loc_cloud_rot = self.Translate_points(Tx=T_x, Ty=T_y, cloudpoints=cloud_rot)
                    loc_cloud_rot = loc_cloud_rot*map_conv
                    indeces = loc_cloud_rot.astype(int).dot(self.reshape_index_m)
                    indeces = indeces[indeces>=0]
                    indeces = indeces[indeces <= self.map_max_index]
                    sum_ = np.sum(self.map_xyz.reshape(-1)[indeces])
                    if sum_ > sum_prob:
                        sum_prob = sum_
                        rot_z_loc = rot_z
                        #print('in', rot_z, T_x, T_y, T_z)
                        Tx_loc = T_x
                        Ty_loc = T_y
                        
                        
        
        self.loc_certainty = (sum_prob/300)/num_points
        if self.loc_certainty >= 0.6 or self.no_map:
            self.euler_zyx[0,0] = rot_z_loc
            self.pos_xyz[0, 0] = Tx_loc
            self.pos_xyz[1, 0] = Ty_loc
            if self.loc_certainty >= 0.7:
                self.oritentaion.quaternions = self.zyx_to_quat(self.euler_zyx[0,0], self.euler_zyx[1,0],self.euler_zyx[2,0])
        print('rot z', rot_z_loc, Tx_loc, Ty_loc, self.loc_certainty)
    
    def zyx_to_quat(self, z, y, x):
        quat = np.array([[np.cos(x/2) * np.cos(y/2) * np.cos(z/2) + np.sin(x/2) * np.sin(y/2) * np.sin(z/2)],
        [np.sin(x/2) * np.cos(y/2) * np.cos(z/2) - np.cos(x/2) * np.sin(y/2) * np.sin(z/2)],
        [np.cos(x/2) * np.sin(y/2) * np.cos(z/2) + np.sin(x/2) * np.cos(y/2) * np.sin(z/2)],
        [np.cos(x/2) * np.cos(y/2) * np.sin(z/2) - np.sin(x/2) * np.sin(y/2) * np.cos(z/2)]])
        return quat
        
    def Map_to_point_cloud(self):
        index = np.argwhere(self.map_xyz[:,:] > self.map_threshold)
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
        P.position.y = 0
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
    slam = SLAM(480, 640, 1) # height, widht, downsample
    rospy.init_node('listener', anonymous=True)
    depth_camera = message_filters.Subscriber("/camera/depth/image_rect_raw", Image)
    gyr_sub = message_filters.Subscriber("/camera/gyro/sample", Imu)
    acc_sub = message_filters.Subscriber("/camera/accel/sample", Imu)

    # using approximate synchronizer. 
    ts = message_filters.ApproximateTimeSynchronizer([depth_camera, gyr_sub, acc_sub], 1, 0.2, allow_headerless=False)
    ts.registerCallback(slam.callback)
    rospy.sleep(1)
    while not rospy.is_shutdown():
        slam.Publich_PointCloud()
        rospy.sleep(2)

    #rospy.spin()

if __name__ == '__main__':

    listener()
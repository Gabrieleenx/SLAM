#!/home/gabriel/anaconda3/bin/python

import rospy
from sensor_msgs.msg import PointCloud2
import numpy as np
import ros_numpy

def callback(data):
	print('thats up')
	print(pointcloud2_to_array(data.data))
	# rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)



def pointcloud2_to_array(cloud_msg, squeeze=True):
    ''' Converts a rospy PointCloud2 message to a numpy recordarray

    Reshapes the returned array to have shape (height, width), even if the height is 1.

    The reason for using np.fromstring rather than struct.unpack is speed... especially
    for large point clouds, this will be <much> faster.
    '''
    # construct a numpy record type equivalent to the point type of this cloud
    dtype_list =  ros_numpy.point_cloud2.fields_to_dtype(cloud_msg.fields, cloud_msg.point_step)

    # parse the cloud into an array
    cloud_arr = np.fromstring(cloud_msg.data, dtype_list)

    # remove the dummy fields that were added
    cloud_arr = cloud_arr[
        [fname for fname, _type in dtype_list if not (fname[:len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)]]

    if squeeze and cloud_msg.height == 1:
        return np.reshape(cloud_arr, (cloud_msg.width,))
    else:
        return np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))

def listener():
	print('Hi')
	rospy.init_node('listener', anonymous=True)

	rospy.Subscriber("/camera/depth/color/points", PointCloud2, callback)

	rospy.spin()




if __name__ == '__main__':
    listener()



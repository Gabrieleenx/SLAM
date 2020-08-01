#!/home/gabriel/anaconda3/bin/python

import rospy
import sys
#sys.path.append('/home/gabriel/catkin_ws/src/darknet_ros/darknet_ros_msgs')
from darknet_ros_msgs.msg import BoundingBox, BoundingBoxes

def callback(data):
    print('hi')
    for box in data.bounding_boxes:
        print('x min', box.xmin, 'y min', box.ymin, 'x max', box.xmax, 'y max', box.ymax)

def listener():
    rospy.init_node('listener darknet', anonymous=True)
    rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, callback)
    rospy.spin()

if __name__ == '__main__':

    listener()
#!/home/gabriel/anaconda3/bin/python

import rospy
from sensor_msgs.msg import Imu
import numpy as np
import message_filters
#import threading
from geometry_msgs.msg import Quaternion

# todo, orientation imu estimation
# calibration imu, both gyro and acc
# figure out why EKF in measurement update has a unstable error covariance estimation 
# worst case use complementary filter instead... 

#np.set_printoptions(precision=None, suppress=None)

class Orientation_estimate(object):
	def __init__(self):
		self.last_time = 0.0
		self.euler_xyz = np.array([[0.0], [0.0], [0.0]])
		self.quaternions = np.array([[1.0], [0.0], [0.0], [0.0]])
		self.P = np.eye(4, dtype = float) # error covariance matrix
		self.correction_gyro = np.array([[0.000508907], [0.00069779], [0.002485042]])
		self.cov_gyro = (0.0001) *np.array([[0.6, -0.15, -0.03],[-0.15, 0.6, 0.05],[-0.03, 0.05, 0.6]])
		self.grav = np.array([[0],[0],[9.81]])
		self.cov_acc = np.array([[0.0001, 0.0, 0.0],[0.0, 0.0001, -0.001],[0.0, -0.0001, 0.0015]])
		self.delta_T = 0.1

	def callback(self, gyr_data, acc_data):

		gyro = np.array([[gyr_data.angular_velocity.x],[gyr_data.angular_velocity.z],[-gyr_data.angular_velocity.y]])
		acc = np.array([[acc_data.linear_acceleration.x], [acc_data.linear_acceleration.z], [-acc_data.linear_acceleration.y]])

		time = acc_data.header.stamp.secs + acc_data.header.stamp.nsecs / 1e9
		if self.last_time == 0:
			self.last_time = time
			delta_time = 0.0
		else:
			delta_time = time - self.last_time
			self.last_time = time
			self.delta_T = delta_time
		# something wrong in the acc update :(
		self.quaternions, self.P = orientation_prediction(self.quaternions, self.P, delta_time, gyro, self.correction_gyro, self.cov_gyro)
		self.quaternions = normalize_quat(self.quaternions)

		self.quaternions, self.P = orientation_acc_update(self.quaternions, self.P, acc, self.cov_acc, self.grav)
		self.quaternions = normalize_quat(self.quaternions)
		
		return quaternions_to_euler_zyx(self.quaternions)
		
		
def orientation_prediction(q, P, delta_time, gyro, correction_gyro, cov_gyro):
	F =  np.eye(4, dtype = float) + delta_time * 0.5 * Somega(gyro)
	dgw = delta_time * 0.5 * np.array([[-q[1,0], -q[2,0], -q[3,0]],
	[q[0,0], -q[3,0], q[2,0]],
	[q[3,0], q[0,0], -q[1,0]],
	[-q[2,0], q[1,0], q[0,0]]])

	P = np.dot(F, np.dot(P, F.transpose())) + np.dot(dgw, np.dot(cov_gyro, dgw.transpose()))
	q = np.dot(F, q)
	return q, P

def orientation_acc_update(q, P, acc, cov_acc, grav):
	if abs(np.linalg.norm(acc) - 9.7) >= 0.3: # some truble with calibration of imu. 
		return q, P
	Q0, Q1, Q2, Q3 = dQqdq(q)
	Q0_t = np.dot(Q0.transpose(), grav)
	Q1_t = np.dot(Q1.transpose(), grav)
	Q2_t = np.dot(Q2.transpose(), grav)
	Q3_t = np.dot(Q3.transpose(), grav)
	h = np.concatenate((Q0_t, Q1_t, Q2_t, Q3_t), axis=1)
	S = h.dot(P).dot(h.transpose()) + cov_acc
	K = P.dot(h.transpose()).dot(np.linalg.pinv(S))
	q += K.dot(acc - np.dot(Qq(q).transpose(), grav))
	P += -0.00001* K.dot(S).dot(K.transpose()) # something wrong. creates instabilites, should not be scaled

	return q, P

def orientation_map_update(euler_xyz, P):
	return euler_xyz

def Somega(omega):
	wx = omega[0,0]
	wy = omega[1,0]
	wz = omega[2,0]
	return np.array([[0., -wx, -wy, -wz],
					[wx, 0., wz, -wy],
					[wy, -wz, 0., wx],
					[wz, wy, -wx, 0.]])

def Qq(q):
	q0 = q[0,0]
	q1 = q[1,0]
	q2 = q[2,0]
	q3 = q[3,0]

	Q = np.array([[2*(q0**2 + q1**2)-1, 2*(q1*q2 - q0*q3), 2*(q1*q3+q0*q2)],
	[2*(q1*q2 + q0*q3), 2*(q0**2 + q2**2)-1, 2*(q2*q3 - q0*q1)],
	[2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 2*(q0**2 + q3**2)-1]])
	return Q

def dQqdq(q):
	# derivative of Q(q) with respect to qi for i =0,1,2,3
	q0 = q[0,0]
	q1 = q[1,0]
	q2 = q[2,0]
	q3 = q[3,0]

	Q0 = 2 * np.array([[2*q0, -q3, q2],
	[q3, 2*q0, -q1],
	[-q2, q1, 2*q0]])

	Q1 = 2 * np.array([[2*q1, q2, q3],
	[q2, 0., -q0],
	[q3, q0, 0]])

	Q2 = 2 * np.array([[0., q1, q0],
	[q1, 2*q2, q3],
	[-q0, q3, 0.]])

	Q3 = 2 * np.array([[0., -q0, q1],
	[q0, 0., q2],
	[q1, q2, 2*q3]])
	
	return Q0, Q1, Q2, Q3

def quaternions_to_euler_xyz(q):
	x = np.arctan2(2*(q[0,0] * q[1,0] + q[2,0] * q[3,0]), 1 - 2 * (q[1,0]**2 + q[2,0]**2))
	y = np.arcsin(2*(q[0,0] * q[2,0] - q[3,0] * q[1,0]))
	z = np.arctan2(2*(q[0,0] * q[3,0] + q[1,0] * q[2,0]), 1 - 2 * (q[2,0]**2 + q[3,0]**2))
	return np.array([[x],[y],[z]])

def quaternions_to_euler_zyx(q):
	w = q[0,0]
	x = q[1,0]
	y = q[2,0]
	z = q[3,0]

	t0 = x*x-z*z
	t1 = w*w - y*y
	xx = 0.5*(t0+t1)
	xy = x*y+w*z
	xz = w*y-x*z
	t = xx*xx+xy*xy
	yz = 2*(y*z+w*x)

	z_ = np.arctan2(xy, xx)
	y_ = np.arctan(xz/np.sqrt(t))
	
	if abs(t) >= 0.05:
		x_ = np.arctan2(yz, t1-t0)
	else:
		x_ = 2*np.arctan2(x,w) - np.sign(xz)*z_
	return np.array([[z_], [y_], [x_]])


def normalize_quat(q):
	return q / np.linalg.norm(q)


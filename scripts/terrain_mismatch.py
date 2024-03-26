#!/usr/bin/env python3
import roslib
# roslib.load_manifest('learning_tf')
import rospy
import numpy as np
import math
# import tf
from std_msgs.msg import Float32
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import Odometry
# import skimage
import time
import cv2
import yaml
from yaml.loader import SafeLoader
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image


import math
def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z # in radians



class OffsetEstimator:
    def __init__(self, maps_topic,odometry_topic):
        rospy.Subscriber(odometry_topic, Odometry, self.handle_odometry, queue_size=2)
        rospy.Subscriber(maps_topic, GridMap, self.handle_map, queue_size=8)

        self.publisher = rospy.Publisher('/terrain_mismatch', Float32, queue_size=3)

        self.pose = None
        self.pose_se3 = None
        self.header = None
        self.p = None
        self.o = None

        self.R,self.P,self.Y = None, None, None

        self.fill_ground = -100
        self.fill_semantic = 9

        self.ground = None
        self.offset = None
        self.info = None

        self.xgrid = None
        self.ygrid = None

        #velodyne - [2.055,2.138,,0.49842]
        #front_left - [2.64,1.271,-1.3]
        #front_right - [1.15166,1.705,-1.3]
        #rear_left - [3.4334,4.206,-1.3]
        #rear_right - [1.864,4.566,-1.3]

        height_diff = -1.85

        transform_front_left = np.identity(4)
        transform_front_left[0:3,-1] = np.array([-0.91-(-1.5), 0-(-0.78), height_diff])
        transform_front_right = np.identity(4)
        transform_front_right[0:3,-1] = np.array([-0.91-(-1.5), 0-(0.78), height_diff])
        transform_rear_left = np.identity(4)
        transform_rear_left[0:3,-1] = np.array([-0.91-(1.5), 0-(-0.78), height_diff])
        transform_rear_right = np.identity(4)
        transform_rear_right[0:3,-1] = np.array([-0.91-(1.5), 0-(0.78), height_diff])

        self.transforms = [transform_front_left, transform_front_right, transform_rear_left, transform_rear_right]

        self.tire_colors = [
                            np.array([1, 0, 0, 1]),
                            np.array([0.5, 0, 0, 1]),
                            np.array([0, 0, 1, 1]),
                            np.array([0, 0, 0.5, 1]),
                           ]

    def handle_odometry(self, msg):
        self.pose = msg.pose.pose
        self.pose_se3 = self.pose_msg_to_se3(msg)
        self.header = msg.header
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        self.p = [position.x,position.y,position.z]
        # print(self.p[2])
        self.o = [orientation.x, orientation.y,orientation.z, orientation.w]
        self.R,self.P,self.Y = euler_from_quaternion(orientation.x, orientation.y,orientation.z, orientation.w)
        # self.R*= 180/(np.pi)
        # self.P*= 180/(np.pi)
        # self.Y*= 180/(np.pi)
        # print(self.P) #down positive, up negative?
        # if self.pub_tires:
        #     self.publish_tire_points()

    def handle_map(self, msg):
        now = time.perf_counter()

        lx = msg.info.length_x
        ly = msg.info.length_y
        res = msg.info.resolution
        origin = msg.info.pose

        idx = msg.layers.index('terrain')
        data = msg.data[idx]

        nx = data.layout.dim[0].size
        ny = data.layout.dim[1].size

        map_data = np.copy(np.array(data.data).reshape(nx, ny)[::-1, ::-1])

        P = [origin.position.x - msg.info.length_x/2, origin.position.y - msg.info.length_y/2]

        tp = self.estimate_tire_points()
        tx = np.floor((tp[:,0]-P[0])/res)
        ty = np.floor((tp[:,1]-P[1])/res)
        xloc,yloc = ty.astype(int), tx.astype(int)

        h = map_data[xloc,yloc]
        diff = np.abs(h-tp[:,2])

        # print(np.mean(diff[:2]), np.mean(diff[2:]))

        meandiff = np.mean(diff)
        self.publisher.publish(meandiff)


        # print(time.perf_counter() - now)

        # viz = map_data.copy()
        # viz[xloc,yloc] = np.max(viz)
        # viz = viz - np.min(viz)
        # viz /= np.max(viz)
        # # print(np.min(viz),np.max(viz))
        # # viz *= 255
        # cv2.imshow('test',viz)
        # cv2.waitKey(1)

    def publish_tire_points(self):

        if self.pose is None:
            # rospy.loginfo_throttle(0.1, "Have not received vehicle odometry yet")
            return

        # rospy.loginfo_throttle(0.1,"Estimating tire poestimate_tireints!")
        estimated_points = self.estimate_tire_points()
        self.temp = estimated_points


        pcl_msg = self.np_points_to_pointcloud2(estimated_points, self.header)

        self.tire_pub.publish(pcl_msg)

    def estimate_tire_points(self):
        tire_points = np.ones((4, 7))

        for i, transform in enumerate(self.transforms):
            # import pdb;pdb.set_trace()
            tire_points[i, 0:3] = (self.pose_se3 @ transform)[0:3, -1]
            tire_points[i, 3:] = self.tire_colors[i]
            # tire_points[i] = (transform @ self.pose_se3)[0:3, -1]

        return tire_points

    def pose_msg_to_se3(self, msg):
        quaternion_msg = msg.pose.pose.orientation
        Q = np.array([quaternion_msg.w, quaternion_msg.x, quaternion_msg.y, quaternion_msg.z])
        rot_mat = self.quaternion_rotation_matrix(Q)

        se3 = np.zeros((4, 4))
        se3[:3, :3] = rot_mat
        se3[0, 3] = msg.pose.pose.position.x
        se3[1, 3] = msg.pose.pose.position.y
        se3[2, 3] = msg.pose.pose.position.z
        se3[3, 3] = 1

        return se3

    def quaternion_rotation_matrix(self, Q):
        """
        Covert a quaternion into a full three-dimensional rotation matrix. Copied from https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/

        Input
        :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix.
                This rotation matrix converts a point in the local reference
                frame to a point in the global reference frame.
        """
        # Extract the values from Q
        q0 = Q[0]
        q1 = Q[1]
        q2 = Q[2]
        q3 = Q[3]

        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)

        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)

        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1

        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])

        return rot_matrix

if __name__ == "__main__":
    rospy.init_node("offset_estimator_node")
    maps_topic = "/local_gridmap"
    odometry_topic="/integrated_to_init"


    ghe = OffsetEstimator(maps_topic,odometry_topic)
    # plt.show()
    rospy.spin()
    # r = rospy.Rate(10)
    # while not rospy.is_shutdown():
        # r.sleep()

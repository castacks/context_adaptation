#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
# from learned_cost_map.msg import FloatStamped
import numpy as np
import rospkg
from threading import Lock

import scipy
import scipy.signal
from scipy.signal import welch
from scipy.integrate import simps
from cv_bridge import CvBridge
import os
import yaml

import roslib
# roslib.load_manifest('learning_tf')
import rospy
import numpy as np
import math
# import tf

from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Int32, Float32, Float32MultiArray
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Float32

from grid_map_msgs.msg import GridMap

# import skimage
import time
import cv2
import yaml
from yaml.loader import SafeLoader
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import signal, stats
# from PIL import Image
# from sklearn.cluster import KMeans

import dynamic_reconfigure.client

import pickle
from context_adaptation_common.links_cluster import LinksCluster, Subcluster


import torch.nn as nn
import torch.nn.functional as F

from matplotlib.animation import FuncAnimation, ArtistAnimation
from rosbag_to_dataset.dtypes.gridmap import GridMapConvert



class Context_Clusterer(object):
    # def __init__(self,cost_topic, odom_topic, im_topic, output_topic, filter_size=15, buffer_size = 100, speed_init = 3.0, max_cost = .34, velocity_margin = 1.5, speed_cap = 5):
    # def __init__(self,cost_topic, odom_topic, im_topic, output_topic, filter_size=10, buffer_size = 10, speed_init = 3.0, max_cost = .32, velocity_margin = 1.5, speed_cap = 5):
    def __init__(self,cost_topic, odom_topic , gridmap_topic, costmap_topic, vel_pub_topic, config):

        self._lock = Lock()

        rp = rospkg.RosPack()
        assets_dir = os.path.join(rp.get_path("context_adaptation"), "assets") + '/'


        self.odom_msg = None

        self.hz_counter = 0

        rospy.Subscriber(gridmap_topic, GridMap, self.handle_map, queue_size=5)

        rospy.Subscriber(cost_topic, Float32, self.handle_cost, queue_size=1)
        rospy.Subscriber(odom_topic, Odometry, self.handle_odom, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(1.0/10.0), self.run_map)
        self.image = None
        self.new_msg = False
        self.cost = 0.0

        self.costmap_pub = rospy.Publisher(costmap_topic,GridMap,queue_size=2)
        self.max_vel_pub = rospy.Publisher(vel_pub_topic,Float32,queue_size=2)

        # self.cost_map = np.zeros((20,50))
        #
        # self.cost_model = nn.Sequential(
        #     nn.Linear(768, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 16),
        #     nn.ReLU(),
        #     nn.Linear(16,1),
        #     # nn.Sigmoid()
        # )
        # self.cost_model.cuda()

        # self.loss = nn.MSELoss()
        # self.opt = torch.optim.SGD(self.cost_model.parameters(),lr = .001)
        # self.train_in_buffer = torch.zeros((500,768)).cuda()
        # self.train_label_buffer = torch.zeros((500,1)).cuda()
        # self.buffer_idx = 0
        # self.buffer_full = False

        height_diff = -1.85

        transform_front_left = np.identity(4)
        transform_front_left[0:3,-1] = np.array([-0.91-(-1.5), 0-(-0.78), height_diff])
        transform_front_right = np.identity(4)
        transform_front_right[0:3,-1] = np.array([-0.91-(-1.5), 0-(0.78), height_diff])
        transform_rear_left = np.identity(4)
        transform_rear_left[0:3,-1] = np.array([-0.91-(1.5), 0-(-0.78), height_diff])
        transform_rear_right = np.identity(4)
        transform_rear_right[0:3,-1] = np.array([-0.91-(1.5), 0-(0.78), height_diff])

        self.tire_transforms = [transform_front_left, transform_front_right, transform_rear_left, transform_rear_right]


        self.channels = []
        self.grid_map_cvt = GridMapConvert(channels=self.channels, size=[1, 1])

        # self.cluster_mapping = np.zeros((20,50))
        self.cluster_means = np.ones(20)
        self.cluster_counts = np.ones(20)

        print('DONE WITH INIT')


    def update_plot(self, frame):
        # self.ln.set_data(self.x_data, self.y_data)
        # return self.ln
        return

    def handle_odom(self, msg):
        self.velocity = np.linalg.norm([msg.twist.twist.linear.x,msg.twist.twist.linear.y])
        self.odom_msg = msg
        self.pose_se3 = self.pose_msg_to_se3(msg)

    def handle_cost(self, msg):
        self.cost = msg.data
        # print('cost', self.cost)

    def handle_map(self,msg):
        with self._lock:
            self.dino_map = msg
            self.new_msg = True

            # print(msg.layers)
            if len(self.channels) == 0:
                for layer in msg.layers:
                    if 'dino' in layer:
                        self.channels.append(layer)

                self.grid_map_cvt.channels = self.channels

    def update_train_buffer(self,input,label):
        # print("BUFFER: ", input.shape, self.train_in_buffer.shape)
        self.train_in_buffer[self.buffer_idx] = input
        self.train_label_buffer[self.buffer_idx] = label
        self.buffer_idx += 1
        # print("Buffer: " , self.buffer_idx, label)
        if self.buffer_idx == self.train_in_buffer.shape[0]:
            self.buffer_full = True
            self.buffer_idx = 0

    def estimate_tire_points(self):
        tire_points = np.ones((4, 3))

        for i, transform in enumerate(self.tire_transforms):
            # import pdb;pdb.set_trace()
            tire_points[i, 0:3] = (self.pose_se3 @ transform)[0:3, -1]
            # tire_points[i, 3:] = self.tire_colors[i]
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

    def run_map(self, event):
        now = time.perf_counter()
        print('----')

        if not self.new_msg:
            print('no new map')
            return

        with self._lock:
            if self.dino_map is not None:
                # lx = self.dino_map.info.length_x
                # ly = self.dino_map.info.length_y
                # res = self.dino_map.info.resolution
                # origin = self.dino_map.info.pose
                #
                # # idx = self.dino_map.layers.index('terrain')
                # data = self.dino_map.data
                #
                # nx = data.layout.dim[0].size
                # ny = data.layout.dim[1].size
                #
                # map_data = np.copy(np.array(data.data).reshape(nx, ny)[::-1, ::-1])
                #

                info = self.dino_map.info
                nx = int(info.length_x / info.resolution)
                ny = int(info.length_y / info.resolution)
                self.grid_map_cvt.size = [nx, ny]

                gridmap = self.grid_map_cvt.ros_to_numpy(self.dino_map)

                msg_header = self.dino_map.info.header
                self.new_msg = False
            else:
                gridmap = None

        if gridmap is None:
            print("NO MAP")
            return

        da = gridmap['data'].argmin(axis=0)
        unc_map = gridmap['data'].min(axis=0)

        unc_map /= 26
        unc_map[unc_map < .85] = 0

        P = gridmap['origin']
        res = gridmap['resolution']
        tp = self.estimate_tire_points()
        tx = np.floor((tp[:,0]-P[0])/res)
        ty = np.floor((tp[:,1]-P[1])/res)
        xloc,yloc = ty.astype(int), tx.astype(int)

        classes = da[xloc,yloc]
        spot = stats.mode(classes)[0]

        # self.cluster_mapping[spot,1:] = self.cluster_mapping[spot,:-1]
        # self.cluster_mapping[spot,0] = self.cost
        # means = np.mean(self.cluster_mapping,axis=1)
        # means[means == 0] = 1

        self.cluster_counts[spot] += 1
        alpha = 1./(1+self.cluster_counts[spot])
        self.cluster_means[spot] = alpha * self.cost + (1.-alpha)*self.cluster_means[spot]

        roughness_map = self.cluster_means[da]
        # print(self.cluster_means)

        costmap = (unc_map + roughness_map)/2.0

        # costmap[xloc,yloc] = 1

        costmap_msg = self.costmap_to_gridmap(costmap, info)
        self.costmap_pub.publish(costmap_msg)




        if self.hz_counter == 50000:
            self.hz_counter = 0
        self.hz_counter += 1
        # if self.hz_counter % 4 != 0:
        #     return


        print(time.perf_counter() - now, 'time')

    def costmap_to_gridmap(self, costmap, info, costmap_layer='costmap'):
        """
        convert costmap into gridmap msg

        Args:
            costmap: The data to load into the gridmap
            msg: The input msg to extrach metadata from
            costmap: The name of the layer to get costmap from
        """
        costmap_msg = GridMap()
        costmap_msg.info = info
        # print("FRAME _ ", costmap_msg.info.header.frame_id)
        costmap_msg.layers = [costmap_layer]

        costmap_layer_msg = Float32MultiArray()
        costmap_layer_msg.layout.dim.append(
            MultiArrayDimension(
                label="column_index",
                size=costmap.shape[0],
                stride=costmap.shape[0]
            )
        )
        costmap_layer_msg.layout.dim.append(
            MultiArrayDimension(
                label="row_index",
                size=costmap.shape[0],
                stride=costmap.shape[0] * costmap.shape[1]
            )
        )

        costmap_layer_msg.data = costmap[::-1, ::-1].flatten()
        costmap_msg.data.append(costmap_layer_msg)

        #add dummy elevation
        costmap_msg.layers.append('elevation')
        layer_data = np.zeros_like(costmap) + self.odom_msg.pose.pose.position.z - 1.73 #+ costmap
        gridmap_layer_msg = Float32MultiArray()
        gridmap_layer_msg.layout.dim.append(
            MultiArrayDimension(
                label="column_index",
                size=layer_data.shape[0],
                stride=layer_data.shape[0]
            )
        )
        gridmap_layer_msg.layout.dim.append(
            MultiArrayDimension(
                label="row_index",
                size=layer_data.shape[0],
                stride=layer_data.shape[0] * layer_data.shape[1]
            )
        )

        gridmap_layer_msg.data = layer_data.flatten()
        costmap_msg.data.append(gridmap_layer_msg)

        return costmap_msg

def main():
    rospy.init_node("context_publisher", log_level=rospy.INFO)
    rospy.loginfo("Initialized context_publisher node")
    cost_topic = rospy.get_param("~cost_topic")
    odom_topic = rospy.get_param("~odom_topic")
    gridmap_topic = rospy.get_param("~gridmap_topic")
    viz = rospy.get_param("~viz")
    pub_anchors = rospy.get_param("~pub_anchors")
    pub_stats = rospy.get_param("~pub_stats")
    config_file = rospy.get_param("~config_file")
    costmap_topic = rospy.get_param("~costmap_topic")
    vel_pub_topic = rospy.get_param("~vel_pub_topic")

    rp = rospkg.RosPack()
    # config_file = os.path.join(rp.get_path("context_adaptation"), "assets","context_configs") + '/' + config_file
    config_dict = yaml.safe_load(open(config_file, 'r'))
    node = Context_Clusterer(cost_topic, odom_topic , gridmap_topic, costmap_topic, vel_pub_topic, config_dict)
    rate = rospy.Rate(10)

    if viz:
        print("+++++++++++++++++++++++++++++++")
        ani = FuncAnimation(node.fig, node.update_plot)
        # ani = ArtistAnimation(node.fig, node.ax)
        plt.show(block=True)

    while not rospy.is_shutdown():
        rate.sleep()


if __name__ == "__main__":
    main()

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
import matplotlib.pyplot as plt
from scipy import signal, stats

import dynamic_reconfigure.client

import pickle
from context_adaptation_common.utils import pose_msg_to_se3

import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib.animation import FuncAnimation, ArtistAnimation
from rosbag_to_dataset.dtypes.gridmap import GridMapConvert

import gpytorch

import matplotlib
# CMAP = matplotlib.cm.get_cmap('plasma')
CMAP = matplotlib.cm.get_cmap('magma')


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1]))
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # self.covar_module.base_kernel.lengthscale = 7.
        self.covar_module.base_kernel.lengthscale = 7

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4)
        self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class SpectralDeltaGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, num_deltas, noise_init=None):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-11))
        likelihood.register_prior("noise_prior", gpytorch.priors.HorseshoePrior(0.1), "noise")
        likelihood.noise = 1e-2

        super(SpectralDeltaGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base_covar_module = gpytorch.kernels.SpectralDeltaKernel(
            num_dims=train_x.size(-1),
            num_deltas=num_deltas,
        )
        base_covar_module.initialize_from_data(train_x[0], train_y[0])
        self.covar_module = gpytorch.kernels.ScaleKernel(base_covar_module)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class Context_Clusterer(object):
    def __init__(self,cost_topic, odom_topic , gridmap_topic, costmap_topic, vel_pub_topic, config):

        self._lock = Lock()

        rp = rospkg.RosPack()
        assets_dir = os.path.join(rp.get_path("context_adaptation"), "assets") + '/'


        self.odom_msg = None

        self.hz_counter = 0



        self.timer = rospy.Timer(rospy.Duration(1.0/10.0), self.run_map)
        self.image = None
        self.new_msg = False
        self.cost = 0.0

        self.speed_mismatch = 0.0
        self.terrain_mismatch = 0.0

        self.costmap_pub = rospy.Publisher(costmap_topic,GridMap,queue_size=2)
        self.max_vel_pub = rospy.Publisher(vel_pub_topic,Float32,queue_size=2)

        # self.cost_map = np.zeros((20,50))
        #


        height_diff = -1*config['ROBOT']['frame_height']
        front_x = config['ROBOT']['front_x']
        back_x = config['ROBOT']['back_x']
        width = config['ROBOT']['width']

        transform_front_left = np.identity(4)
        transform_front_left[0:3,-1] = np.array([front_x, width, height_diff])
        transform_front_right = np.identity(4)
        transform_front_right[0:3,-1] = np.array([front_x, -width, height_diff])
        transform_rear_left = np.identity(4)
        transform_rear_left[0:3,-1] = np.array([back_x, width, height_diff])
        transform_rear_right = np.identity(4)
        transform_rear_right[0:3,-1] = np.array([back_x, -width, height_diff])

        self.tire_transforms = [transform_front_left, transform_front_right, transform_rear_left, transform_rear_right]


        self.channels = []
        self.grid_map_cvt = GridMapConvert(channels=self.channels, size=[1, 1])


        #TODO Grab this dynamically as well
        self.VLAD_CLUSTS = config['VLAD']['n_clusters']
        self.gridmap_size = None

        self.residual_max = config['BEHAVIOR']['residual_max']
        self.uc_thresh = config['BEHAVIOR']['uncertainty_threshold']

        # self.loss = nn.MSELoss()
        # self.opt = torch.optim.SGD(self.cost_model.parameters(),lr = .00000001)
        buffer_len = config['BEHAVIOR']['buffer_size']
        self.buffer_update_freq = config['BEHAVIOR']['buffer_update_freq']
        self.train_in_buffer = torch.zeros((buffer_len,self.VLAD_CLUSTS)).cuda()
        self.train_label_buffer = torch.zeros((buffer_len,1)).cuda()
        self.train_buffer_classes = np.zeros((buffer_len))
        self.buffer_idx = 0
        self.buffer_full = False
        self.in_std = 0.0
        self.in_mean = 0.0

        self.cost_offset = .0

        # for i in range(self.VLAD_CLUSTS):
        #     tinput = torch.ones(self.VLAD_CLUSTS).float().cuda()
        #     tinput[i] = 0.5
        #     self.update_train_buffer(tinput, 1.0, i)

        # for i in range(self.VLAD_CLUSTS):
        #     tinput = torch.ones(self.VLAD_CLUSTS).float().cuda() - .8
        #     tinput[i] = 0.0
        #     self.update_train_buffer(tinput, 1.0, i)

        self.gp = None
        self.likelihood = None
        self.gp_params = None

        # self.gp_params = torch.load('gp_params')
        self.gp_params = torch.load(os.path.join(assets_dir, 'gp_params', 'gp_params_ks_7.pt'))
        # print(self.gp_params)
        # print(os.getcwd())

        rospy.Subscriber(gridmap_topic, GridMap, self.handle_map, queue_size=5)

        rospy.Subscriber(cost_topic, Float32, self.handle_cost, queue_size=1)
        rospy.Subscriber(odom_topic, Odometry, self.handle_odom, queue_size=1)
        print('DONE WITH INIT')


    def update_plot(self, frame):
        # self.ln.set_data(self.x_data, self.y_data)
        # return self.ln
        return

    def handle_odom(self, msg):
        self.velocity = np.linalg.norm([msg.twist.twist.linear.x,msg.twist.twist.linear.y])
        self.odom_msg = msg
        self.pose_se3 = pose_msg_to_se3(msg)

    def handle_cost(self, msg):
        self.cost = msg.data
        self.cost -= self.cost_offset
        # print('cost', self.cost)

    def handle_map(self,msg):
        with self._lock:
            self.dino_map = msg
            self.new_msg = True

            # print(msg.layers)
            if len(self.channels) == 0:
                for layer in msg.layers:
                    if 'VLAD' in layer:
                        self.channels.append(layer)

                self.grid_map_cvt.channels = self.channels

    def update_train_buffer(self,input,label,class_id):
        # print("BUFFER: ", input.shape, self.train_in_buffer.shape)
        self.train_in_buffer[self.buffer_idx] = input
        self.train_label_buffer[self.buffer_idx] = label
        self.train_buffer_classes[self.buffer_idx] = class_id
        self.buffer_idx = min(self.buffer_idx+1, self.train_in_buffer.shape[0])
        # print(self.buffer_idx)
        # print("Buffer: " , self.buffer_idx, label)
        if self.buffer_idx == self.train_in_buffer.shape[0]:
            self.buffer_full = True
            # self.buffer_idx = 0

    def insert_train_buffer(self,input,label, class_id, idx):
        # print("BUFFER: ", input.shape, self.train_in_buffer.shape)
        self.train_in_buffer[idx] = input
        self.train_label_buffer[idx] = label
        self.train_buffer_classes[idx] = class_id

    def estimate_tire_points(self):
        tire_points = np.ones((4, 3))

        for i, transform in enumerate(self.tire_transforms):
            tire_points[i, 0:3] = (self.pose_se3 @ transform)[0:3, -1]

        return tire_points

    def run_map(self, event):
        now = time.perf_counter()
        print('----')
        if self.hz_counter == 50000:
            self.hz_counter = 0
        self.hz_counter += 1

        if not self.new_msg:
            print('no new map')
            return

        with self._lock:
            if self.dino_map is not None:
                info = self.dino_map.info
                nx = int(info.length_x / info.resolution)
                ny = int(info.length_y / info.resolution)
                self.grid_map_cvt.size = [nx, ny]

                #HACK assumes square for now
                self.gridmap_size = nx
                # print("$$$$$$$$$$$$$$$$$", nx)

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

        # print(gridmap['data'].shape)

        # print(gridmap['data'][:,120,120]/26.)
        # print(np.max(unc_map))
        unc_map /= self.residual_max
        unc_map[unc_map < self.uc_thresh] = 0

        P = gridmap['origin']
        res = gridmap['resolution']
        tp = self.estimate_tire_points()
        tx = np.floor((tp[:,0]-P[0])/res)
        ty = np.floor((tp[:,1]-P[1])/res)
        xloc,yloc = ty.astype(int), tx.astype(int)

        classes = da[xloc,yloc]
        spot = stats.mode(classes)[0]

        input = torch.tensor(gridmap['data']/self.residual_max).cuda()
        # tinput = torch.tensor(gridmap['data']).cuda()
        # input = (tinput - tinput.mean())/tinput.std()
        # print(input.shape)
        # print(input[:,xloc,yloc])

        if self.hz_counter % self.buffer_update_freq == 0:
            input_sample = input[:,xloc[0],yloc[0]]
            if torch.count_nonzero(input_sample) == 0:
                print("IN UNKOWN SPACE")
            else:
                if self.buffer_full:
                    print("))))))))))))))))))))))))))))))))))))))))))))))))))")
                    most_class = stats.mode(self.train_buffer_classes)[0]
                    # print(np.where(self.train_buffer_classes == most_class)[0])
                    # print(np.histogram(self.train_buffer_classes))
                    insert_idx = np.random.choice(np.where(self.train_buffer_classes == most_class)[0])
                    # insert_idx = torch.argmin(torch.abs(self.train_label_buffer - self.cost))
                    # print(input_sample)
                    self.insert_train_buffer(input_sample, self.cost, spot, insert_idx)
                    # self.update_train_buffer(input_sample, self.cost, spot)
                else:
                    self.update_train_buffer(input_sample, self.cost, spot)

        costmap = np.zeros((self.gridmap_size,self.gridmap_size))

        # if self.buffer_full:
        if self.hz_counter % 10 == 0 or self.gp is None:
            print('****************************************************')
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
            # print(self.train_in_buffer, self.train_label)
            if not self.buffer_full:
                train_in_buffer = self.train_in_buffer[:self.buffer_idx]
                train_label_buffer = self.train_label_buffer[:self.buffer_idx,0]
                train_buffer_classes = self.train_buffer_classes[:self.buffer_idx]
            else:
                train_in_buffer = self.train_in_buffer
                train_label_buffer = self.train_label_buffer[:,0]
                train_buffer_classes = self.train_buffer_classes
            # print(np.histogram(train_buffer_classes))
            # print(train_in_buffer, train_in_buffer.shape)
            self.in_mean = torch.mean(train_in_buffer, dim = 0)
            self.in_std = torch.std(train_in_buffer, dim = 0)
            train_in_buffer = (train_in_buffer - self.in_mean)/self.in_std
            self.gp = ExactGPModel(train_in_buffer, train_label_buffer, self.likelihood).cuda()
            if self.gp_params is not None:
                self.gp.load_state_dict(self.gp_params)
            # print(self.gp.parameters())
        # else:
        #     if not self.buffer_full:
        #         train_in_buffer = self.train_in_buffer[:self.buffer_idx]
        #         train_label_buffer = self.train_label_buffer[:self.buffer_idx,0]
        #     else:
        #         train_in_buffer = self.train_in_buffer
        #         train_label_buffer = self.train_label_buffer[:,0]
        #
        #     # print(train_in_buffer.shape)
        #
        #     if train_in_buffer.shape[0] < 3: #self.buffer_idx > 10:
        #         return
        #
        #     # print(train_in_buffer, train_label_buffer)
        #
        #     optimizer = torch.optim.SGD(self.gp.parameters(), lr=0.01)
        #     mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp)
        #     optimizer.zero_grad()
        #     output = self.gp(train_in_buffer)
        #     loss = -mll(output, train_label_buffer)
        #     print("LOSS - ", loss.item())
        #     # print("PARAMS ", self.gp.covar_module)
        #     if not torch.isnan(loss):
        #         loss.backward()
        #         optimizer.step()
        #
        #     self.gp_params = self.gp.state_dict()
        # #
        # if self.hz_counter % 20 == 0 and self.gp_params is not None:
        #     print(self.gp_params)
        #     torch.save(self.gp_params, 'gp_params')
        # print("LOSS: ", loss.item())
        with torch.no_grad():
            self.gp.eval()
            self.likelihood.eval()
            input = input.permute(1,2,0).view(-1,self.VLAD_CLUSTS)
            input = (input - self.in_mean)/self.in_std
            observed_pred = self.likelihood(self.gp(input))
            # observed_pred = self.gp.likelihood(self.gp(input.permute(1,2,0).view(-1,8)))
            # observed_pred += .5
            costmap = observed_pred.mean.cpu().numpy().reshape(self.gridmap_size,self.gridmap_size)
            costmap_var = observed_pred.variance.cpu().numpy().reshape(self.gridmap_size,self.gridmap_size)
            # costmap += observed_pred.variance.cpu().numpy().reshape(200,200)
            costmap += self.cost_offset
            # costmap = costmap_var
            # print(costmap.min(), costmap.max(), '+++++')
            # costmap *= .5
                # print(observed_pred.shape)


        # costmap = (unc_map + costmap)/2.0
        costmap /= 2.0
        ids = np.where(unc_map != 0)
        costmap[ids] = unc_map[ids]
        # costmap = (costmap - .65)/.2

        var_min = costmap_var.min()
        var_max = costmap_var.max()
        # var_min = .2
        # var_max = .3
        print('###', var_min, var_max)

        var_viz = (costmap_var - var_min)/(var_max-var_min)
        var_viz = np.clip(var_viz,0,1)
        # var_viz[xloc,yloc] = 1

        costmap_msg = self.costmap_to_gridmap(var_viz, info)
        self.costmap_pub.publish(costmap_msg)





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

        # vcostmap = np.clip(costmap*2.,0,1)
        vcostmap = np.clip(costmap*2.,0,.6)/.6
        # vcostmap = costmap

        gridmap_cs = (CMAP(vcostmap) * 255).astype(np.int32)
        gridmap_color = gridmap_cs[..., 0] * (2**16) + gridmap_cs[..., 1] * (2**8) + gridmap_cs[..., 2]
        gridmap_color = gridmap_color.view(dtype=np.float32)

        costmap_msg.layers.append('rgb_viz')
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

        gridmap_layer_msg.data = gridmap_color[::-1, ::-1].flatten()
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

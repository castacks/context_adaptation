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
gpytorch.settings.debug(False)

import matplotlib
# CMAP = matplotlib.cm.get_cmap('plasma')
CMAP = matplotlib.cm.get_cmap('magma')


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,lengthscale):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1]))
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module.base_kernel.lengthscale = lengthscale
        # self.covar_module.base_kernel.lengthscale = 3.
        # self.covar_module.base_kernel.lengthscale = 10


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class Context_Clusterer(object):
    def __init__(self,cost_topic, odom_topic , gridmap_topic, costmap_topic, vel_pub_topic, config):

        # print(os.getcwd())

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
        self.speedmap_pub = rospy.Publisher('/shortrange_speedmap',GridMap,queue_size=2)
        self.max_vel_pub = rospy.Publisher(vel_pub_topic,Float32,queue_size=2)
        self.cvar_pub = rospy.Publisher('/hdif_speedmap_cvar',Float32,queue_size=2)

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

        self.cost_lengthscale = config['BEHAVIOR']['cost_lengthscale']
        self.speed_lengthscale = config['BEHAVIOR']['speed_lengthscale']
        self.train_kernel = config['BEHAVIOR']['train_kernel']

        # self.loss = nn.MSELoss()
        # self.opt = torch.optim.SGD(self.cost_model.parameters(),lr = .00000001)
        buffer_len = config['BEHAVIOR']['buffer_size']
        self.buffer_update_freq = config['BEHAVIOR']['buffer_update_freq']
        self.train_in_buffer = torch.zeros((buffer_len,self.VLAD_CLUSTS + 2)).cuda()
        self.train_label_buffer = torch.zeros((buffer_len,1)).cuda()
        self.train_buffer_classes = np.zeros((buffer_len))
        self.toi = np.zeros((buffer_len))
        self.buffer_idx = 0
        self.buffer_full = False
        self.in_std = 0.0
        self.in_mean = 0.0

        # buffer_checkpoint = torch.load('buffer_checkpoint_turnpike.pt')
        # buffer_checkpoint = torch.load(os.path.join(assets_dir, 'gp_params', 'buffer_checkpoint.pt'))
        # buffer_checkpoint = torch.load(os.path.join(assets_dir, 'gp_params', 'buffer_checkpoint_turnpike.pt'))
        # buffer_checkpoint = torch.load('buffer_checkpoint_debug.pt')
        buffer_checkpoint = torch.load('buffer_checkpoint_fig8.pt')
        # print("INIT SIZE", init_size)
        # self.train_in_buffer = buffer_checkpoint['train_buffer'].cuda()
        # self.train_label_buffer = buffer_checkpoint['train_labels'].cuda()
        # self.train_buffer_classes = buffer_checkpoint['train_classes']
        # self.buffer_idx = buffer_checkpoint['buffer_idx']
        # self.buffer_full = buffer_checkpoint['full']

        self.cost_offset = .0
        self.velocity = 0.0
        self.fake_velocity = 0.0


        #probably would be smarter to use the actual update buffer method oops
        avoid_class = 3 #looks like 3 or 5|6
        # num_insert = 10
        # spread = .9
        num_insert = 8
        spread = 1.2
        avoid_data = torch.ones(num_insert,self.VLAD_CLUSTS + 2).cuda() * .9
        avoid_classes = np.zeros((num_insert)) + avoid_class
        avoid_labels = torch.ones((num_insert,1)).cuda()
        # avoid_data[:,avoid_class] *= 0
        # avoid_data[:,6] *= .2
        avoid_data[:,-1] = 1.0
        for i in range(num_insert):
            avoid_data[i,-2] = i*spread
            avoid_data[i,:self.VLAD_CLUSTS] = torch.Tensor([19.777752, 23.18438 , 21.908875, 18.12688 , 25.28553 , 23.31651 ,
           21.984331, 24.273615]).cuda() / self.residual_max
            # avoid_data[i,-1] += torch.randint(-1000,1001,[1])[0] * .0001
            # avoid_data[i,:self.VLAD_CLUSTS] += torch.randint(-1000,1001,[self.VLAD_CLUSTS]).cuda() * .0001

        self.train_in_buffer = torch.vstack((avoid_data,self.train_in_buffer))
        self.train_label_buffer = torch.vstack(( avoid_labels, self.train_label_buffer))
        self.train_buffer_classes = np.concatenate((avoid_classes, self.train_buffer_classes))

        self.num_insert = num_insert
        self.buffer_idx += self.num_insert

        self.gp = None
        self.likelihood = None
        self.gp_params = None

        # self.gp_params = torch.load('gp_params_speed_input')
        self.gp_params = torch.load(os.path.join(assets_dir, 'gp_params', 'gp_params_speed_input'))

        self.speed_gp = None
        self.speed_likelihood = None
        self.speed_gp_params = None
        self.speed_likelihood = None
        # self.speed_gp_params = torch.load(os.path.join(assets_dir, 'gp_params', 'speed_gp_params'))
        self.speed_gp_params = torch.load(os.path.join(assets_dir, 'gp_params', 'speed_gp_params_new'))

        # print(self.speed_gp_params)
        self.speed_gp_indices = [0,1,2,3,4,5,6,7,9]

        self.max_roughness = config['BEHAVIOR']['max_cost']
        self.max_velocity = config['BEHAVIOR']['max_velocity']
        self.velocity_margin = config['BEHAVIOR']['velocity_margin']
        self.cvar_alpha = .0
        self.rough_history = 0.2
        self.vel_history = 0.0
        self.costmap_cvar = config['BEHAVIOR']['costmap_cvar']
        self.unknown_cost = config['BEHAVIOR']['unknown_cost']
        self.unknown_speed = config['BEHAVIOR']['unknown_speed']
        self.maxpool_speedmap = config['BEHAVIOR']['maxpool_speedmap']

        self.support_data = []
        self.errors = []

        rospy.Subscriber(gridmap_topic, GridMap, self.handle_map, queue_size=5)

        rospy.Subscriber(cost_topic, Float32, self.handle_cost, queue_size=1)
        rospy.Subscriber(odom_topic, Odometry, self.handle_odom, queue_size=1)
        rospy.Subscriber('/hdif_max_roughness', Float32, self.handle_max_roughness, queue_size=1)
        rospy.Subscriber('/hdif_cvar', Float32, self.handle_cvar, queue_size=1)
        print('DONE WITH INIT')


    def update_plot(self, frame):
        # self.ln.set_data(self.x_data, self.y_data)
        # return self.ln
        return

    def handle_odom(self, msg):
        vel = np.linalg.norm([msg.twist.twist.linear.x,msg.twist.twist.linear.y])
        self.velocity = vel
        # print("VELOCITY - ", self.velocity)
        # self.actual_vel = vel
        self.odom_msg = msg
        self.pose_se3 = pose_msg_to_se3(msg)
        self.vel_history = self.vel_history + (self.velocity - self.vel_history) * .1

    def handle_cost(self, msg):
        self.cost = msg.data
        self.cost -= self.cost_offset
        self.rough_history = self.rough_history + (msg.data - self.rough_history) * .2
        # print('cost', self.cost)

    def handle_max_roughness(self, msg):
        self.max_roughness = msg.data
        self.fake_velocity = msg.data

    def handle_cvar(self, msg):
        self.cvar_alpha = msg.data

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
        # if self.train_in_buffer[0][:-1].sum() == 0.0:
        #     self.buffer_idx = 0
        self.train_in_buffer[self.buffer_idx] = input
        self.train_label_buffer[self.buffer_idx] = label
        self.train_buffer_classes[self.buffer_idx] = class_id
        # self.toi[self.buffer_idx] = self.hz_counter
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
        # self.toi[idx] = self.hz_counter

    def insert_train_buffer_FIFO(self,input,label, class_id):
        # print("BUFFER: ", input.shape, self.train_in_buffer.shape)
        idx = np.argmin(self.toi)
        self.train_in_buffer[idx] = input
        self.train_label_buffer[idx] = label
        self.train_buffer_classes[idx] = class_id
        self.toi[idx] = self.hz_counter

    def estimate_tire_points(self):
        tire_points = np.ones((4, 3))

        for i, transform in enumerate(self.tire_transforms):
            tire_points[i, 0:3] = (self.pose_se3 @ transform)[0:3, -1]

        return tire_points

    def run_map(self, event):
        now = time.perf_counter()
        print('----')
        if self.hz_counter == 5000000:
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

        # print(gridmap['data'].shape)

        da = gridmap['data'].argmin(axis=0)
        unc_map = gridmap['data'].min(axis=0)

        # print(gridmap['data'].shape)

        # print(gridmap['data'][:,120,120]/26.)
        # print(np.max(unc_map))
        # print(unc_map.min(), unc_map.max())
        unc_map /= self.residual_max

        unc_map[unc_map < self.uc_thresh] = 0

        e_kernel = np.ones((2, 2), np.float32)
        unc_map = cv2.erode(unc_map, e_kernel, iterations=1)
        unc_map = cv2.dilate(unc_map, e_kernel, iterations=1)
        #
        # unc_map[unc_map < .78] = 0


        P = gridmap['metadata']['origin']
        res = gridmap['metadata']['resolution']
        tp = self.estimate_tire_points()
        tx = np.floor((tp[:,0]-P[0])/res)
        ty = np.floor((tp[:,1]-P[1])/res)
        xloc,yloc = ty.astype(int), tx.astype(int)

        if np.any(xloc > self.gridmap_size) or np.any(yloc > self.gridmap_size): #Super Odometry probably borked
            print("WARNING: OUT OF BOUNDS! Not computing costmap")
            return

        classes = da[xloc,yloc]
        spot = stats.mode(classes)[0]

        # np.save('/home/physics_atv/physics_atv_ws/gridmap_data', gridmap['data'])

        input = torch.from_numpy(gridmap['data']/self.residual_max)
        # print(input[:,xloc,yloc])
        # print("VEL MAX ", self.train_in_buffer[:,-2].max())
        zero_mask = (input == 0).all(dim=0).cpu().numpy()
        known_mask = ~zero_mask

        t1 = time.perf_counter()
        # print("T1 ", t1  - now)
        # if True:
        #     if self.gp is not None:
        #         with torch.no_grad():
        #             self.gp.eval()
        #             self.likelihood.eval()
        #             test_sample = input[:,xloc,yloc].mean(dim=1).clone()
        #             test_sample = torch.cat((test_sample, torch.Tensor([self.velocity]))).cuda()
        #             test_label = self.rough_history
        #             test_sample = test_sample.unsqueeze(0)
        #             test_sample = (test_sample - self.in_mean[:-1])/self.in_std[:-1]
        #             rough_pred = self.likelihood(self.gp(test_sample[:])).mean[0]
        #             rough_pred = (rough_pred * self.in_std[-1]) + self.in_mean[-1]
        #             error = np.abs((rough_pred - test_label).cpu().numpy())
        #             print("PRED VS SAMPLE", error, len(self.errors))
        #             self.errors.append(error)

        if (self.velocity > .3) and self.hz_counter % self.buffer_update_freq == 0:
            # print("UPDATING+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            # input_sample = input[:,xloc[0],yloc[0]]
            input_sample = input[:,xloc,yloc].mean(dim=1)
            # print(n_input_sample.shape)
            # support_add = input[:,np.random.choice(np.arange(75,175)),np.random.choice(np.arange(75,175))].cpu().numpy()
            # if np.sum(np.abs(support_add)) > .1:
            #     self.support_data.append(support_add)
            # input_sample = torch.cat((input_sample, torch.Tensor([self.velocity]).cuda(),torch.Tensor([self.cost]).cuda()))
            input_sample = torch.cat((input_sample, torch.Tensor([self.velocity]),torch.Tensor([self.cost]))).cuda()
            # self.support_data.append(input_sample.cpu().numpy())

            if torch.count_nonzero(input_sample[:-2]) == 0:
                print("IN UNKNOWN SPACE")
            else:
                # self.support_data.append(input_sample.cpu().numpy())
                if self.buffer_full:
                    print("))))))))))))))))))))))))))))))))))))))))))))))))))")
                    most_class = stats.mode(self.train_buffer_classes)[0]
                    vel_hist = torch.histc(self.train_in_buffer[self.train_buffer_classes == most_class,-2], bins=10, min=0,max=10)
                    most_vel = torch.argmax(vel_hist)
                    edges = np.arange(0,11)
                    sel_min = edges[most_vel]
                    sel_max = edges[most_vel+1]
                    train_vels = self.train_in_buffer[:,-2].cpu().numpy()
                    insert_idx = np.random.choice(np.where((self.train_buffer_classes == most_class) & (train_vels > sel_min) & (train_vels < sel_max))[0])
                    if insert_idx > self.num_insert: #don't remove human labels
                        self.insert_train_buffer(input_sample, self.cost, spot, insert_idx)
                    # self.insert_train_buffer_FIFO(input_sample, self.cost, spot)
                else:
                    self.update_train_buffer(input_sample, self.cost, spot)

        # costmap = np.zeros((self.gridmap_size,self.gridmap_size))
        # speedmap = np.zeros((self.gridmap_size,self.gridmap_size)) + 4.5

        costmap = np.zeros(self.gridmap_size*self.gridmap_size)
        speedmap = np.zeros(self.gridmap_size*self.gridmap_size) + 4.5

        t2 = time.perf_counter()
        # print("T2 ", t2  - t1)

        # if self.buffer_full:
        if self.hz_counter % 10 == 0 or self.gp is None:
            print('****************************************************')
            form_now = time.perf_counter()
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
            self.speed_likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
            # print(self.train_in_buffer, self.train_label)
            if not self.buffer_full:
                train_in_buffer = self.train_in_buffer[:self.buffer_idx]
                train_label_buffer = self.train_label_buffer[:self.buffer_idx,0]
                train_buffer_classes = self.train_buffer_classes[:self.buffer_idx]
            else:
                train_in_buffer = self.train_in_buffer
                train_label_buffer = self.train_label_buffer[:,0]
                train_buffer_classes = self.train_buffer_classes
            # print(train_in_buffer.shape, '-- train buffer shape')
            # print(np.histogram(train_buffer_classes))
            # print(train_in_buffer[:10], train_in_buffer.shape)
            self.in_mean = torch.mean(train_in_buffer, dim = 0)
            self.in_std = torch.std(train_in_buffer, dim = 0)
            # print(self.in_mean)
            train_in_buffer = (train_in_buffer - self.in_mean)/self.in_std
            # self.gp = ExactGPModel(train_in_buffer[:,:-1], train_label_buffer, self.likelihood, self.cost_lengthscale).cuda()
            self.gp = ExactGPModel(train_in_buffer[:,:-1], train_in_buffer[:,-1], self.likelihood, self.cost_lengthscale).cuda()
            # print(train_in_buffer[:,-2])
            self.speed_gp = ExactGPModel(train_in_buffer[:,self.speed_gp_indices], train_in_buffer[:,-2], self.speed_likelihood, self.speed_lengthscale).cuda()
            if self.gp_params is not None:
                self.gp.load_state_dict(self.gp_params)
            if self.speed_gp_params is not None:
                self.speed_gp.load_state_dict(self.speed_gp_params)
            # print('init_gp_time', time.perf_counter() - form_now)
            # print(self.gp.parameters())
        elif self.train_kernel:
            if not self.buffer_full:
                train_in_buffer = self.train_in_buffer[:self.buffer_idx]
                train_label_buffer = self.train_label_buffer[:self.buffer_idx,0]
            else:
                train_in_buffer = self.train_in_buffer
                train_label_buffer = self.train_label_buffer[:,0]

            if train_in_buffer.shape[0] < 13: #self.buffer_idx > 10:
                return

            # optimizer = torch.optim.SGD(self.gp.parameters(), lr=0.01)
            # with gpytorch.settings.debug(False):
            #     self.gp.train()
            #     self.likelihood.train()
            #     mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp)
            #     optimizer.zero_grad()
            #     output = self.gp(train_in_buffer[:,:-1])
            #     # print(output)
            #     loss = -mll(output, train_in_buffer[:,-1])
            #     print("LOSS - ", loss.item())
            #     # print("PARAMS ", self.gp.covar_module)
            #     if not torch.isnan(loss):
            #         loss.backward()
            #         optimizer.step()
            #
            #     self.gp_params = self.gp.state_dict()

            optimizer = torch.optim.SGD(self.speed_gp.parameters(), lr=0.01)
            with gpytorch.settings.debug(False):
                self.speed_gp.train()
                # self.likelihood.train()
                self.speed_likelihood.train()
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.speed_likelihood, self.speed_gp)
                optimizer.zero_grad()
                output = self.speed_gp(train_in_buffer[:,self.speed_gp_indices])
                # print(output)
                loss = -mll(output, train_in_buffer[:,-2])
                print("LOSS - ", loss.item())
                # print("PARAMS ", self.gp.covar_module)
                if not torch.isnan(loss):
                    loss.backward()
                    optimizer.step()

                self.speed_gp_params = self.speed_gp.state_dict()

        if self.hz_counter % 20 == 0:# and self.gp_params is not None:
            # print(self.gp_params)
            # print(self.gp.covar_module.base_kernel.lengthscale)
            # print(self.speed_gp_params)
            torch.save(self.speed_gp_params, 'speed_gp_params_new')
            # savedict = {'train_buffer': self.train_in_buffer, 'train_labels': self.train_label_buffer, 'train_classes': self.train_buffer_classes,
            #             'buffer_idx': self.buffer_idx, 'full' :self.buffer_full, 'toi':self.toi}
            # torch.save(savedict, 'buffer_checkpoint_FIFO.pt')
            # print("shape ", .shape)
            # np.save("support_data", np.array(self.support_data))
            print("SAVED _ _ _ _ _ _ _ __ _ _ _ _ _ _ _ _ __ _ _ __", self.hz_counter)
            # np.save("ERROR_LIST", np.array(self.errors))
        # print("LOSS: ", loss.item())

        t3 = time.perf_counter()
        # print("T3 ", t3  - t2)

        infer_now = time.perf_counter()
        with torch.no_grad():
            if self.buffer_idx < 5 + self.num_insert: #we need at least a couple samples apart from single label
                costmap = np.zeros((self.gridmap_size,self.gridmap_size))
                speedmap = np.zeros((self.gridmap_size,self.gridmap_size)) + 4.5
                costmap_var = np.zeros_like(costmap)
                speedmap_var = np.zeros_like(speedmap)
            else:
                self.gp.eval()
                self.speed_gp.eval()
                self.likelihood.eval()
                self.speed_likelihood.eval()
                input = input.permute(1,2,0).view(-1,self.VLAD_CLUSTS)

                zero_mask_f = zero_mask.flatten()
                keep_mask = ~zero_mask_f
                input = input[keep_mask]

                vel_append = torch.ones(input.shape[0],1) * self.velocity
                # vel_append = torch.ones(input.shape[0],1) * self.fake_velocity
                rough_append = torch.ones(input.shape[0],1) * self.max_roughness
                # input = torch.hstack((input,vel_append.cuda(), rough_append.cuda()))
                input = torch.hstack((input,vel_append, rough_append)).cuda()
                input = (input - self.in_mean)/self.in_std

                t41 = time.perf_counter()
                print("T41 ", t41  - t3)


                observed_pred = self.likelihood(self.gp(input[:,:-1]))
                cost_pred = observed_pred.mean
                cost_var = observed_pred.variance
                # costmap_var = observed_pred.variance.cpu().numpy().reshape(self.gridmap_size,self.gridmap_size)
                # costmap += self.cost_offset

                observed_pred = self.speed_likelihood(self.speed_gp(input[:,self.speed_gp_indices]))
                speed_pred = observed_pred.mean
                # speedmap = (speedmap * self.in_std[-2]) + self.in_mean[-2]

                speed_var = observed_pred.variance#.cpu().numpy().reshape(self.gridmap_size,self.gridmap_size)

                t42 = time.perf_counter()
                print("T42 ", t42  - t41)

                print("CVAR - ", self.cvar_alpha)

                phi = stats.norm.pdf(stats.norm.ppf(self.cvar_alpha))
                cvar = speed_pred + (speed_var * phi)/(1.0-self.cvar_alpha)
                speed_pred = cvar
                # speedmap = speedmap + self.cvar_alpha * speedmap_var * 10
                speed_pred = (speed_pred * self.in_std[-2]) + self.in_mean[-2]


                cmap_cvar = self.costmap_cvar
                phi = stats.norm.pdf(stats.norm.ppf(cmap_cvar))
                cvar = cost_pred + (cost_var * phi)/(1.0-cmap_cvar)
                cost_pred = cvar

                cost_pred = (cost_pred * self.in_std[-1]) + self.in_mean[-1]

                t43 = time.perf_counter()
                print("T43 ", t43  - t42)

                costmap[keep_mask] = cost_pred.cpu().numpy()
                speedmap[keep_mask] = speed_pred.cpu().numpy()

                costmap = costmap.reshape(self.gridmap_size,self.gridmap_size)
                speedmap = speedmap.reshape(self.gridmap_size,self.gridmap_size)

                if self.maxpool_speedmap:
                    speedmap = torch.from_numpy(speedmap).cuda()
                    ksize = 3
                    padding = (ksize - 1)//2
                    m = torch.nn.MaxPool2d(ksize,stride=1, padding = padding)
                    speedmap = m(speedmap.unsqueeze(0))[0].cpu().numpy()

                if (self.rough_history < self.max_roughness) and np.abs(self.vel_history - speedmap[xloc[0],yloc[0]]) < self.velocity_margin:
                    self.cvar_alpha += 0.005
                elif (self.rough_history > self.max_roughness):# and np.abs(self.vel_history - speedmap[xloc[0],yloc[0]]) < self.velocity_margin:
                    self.cvar_alpha -= 0.02
                self.cvar_alpha = np.clip(self.cvar_alpha, 0,.99)

                #if we're not tracking well at low speeds then increase cvar
                if (np.abs(self.vel_history - speedmap[xloc[0],yloc[0]]) > .4) and speedmap[xloc[0],yloc[0]] < 2.8:
                    self.cvar_alpha += .01

                t44 = time.perf_counter()
                print("T44 ", t44  - t43)


            if self.buffer_idx < 30 + self.num_insert: #let's give the speedmap a bit more time
                speedmap = np.zeros((self.gridmap_size,self.gridmap_size)) + 4.5
                self.cvar_alpha = 0.0
                # speedmap_var = np.zeros_like(speedmap)
            # costmap = costmap_var
            # print(costmap.min(), costmap.max(), '+++++')
            # costmap *= .5
                # print(observed_pred.shape)
        # print("infer", time.perf_counter() - infer_now)
        t4 = time.perf_counter()
        print("T4 ", t4  - t3)

        # costmap = (unc_map + costmap)/2.0



        # self.velocity = speedmap[xloc[0],yloc[0]] + np.random.normal(scale = 0.5)
        # self.velocity = np.minimum(self.velocity, self.actual_vel)
        # print('velocity - ', self.velocity)

        print("COSTMAP - ", costmap.min(), costmap.max())

        costmap /= 2.0
        # costmap *= 0.0
        ids = np.where(unc_map != 0)
        costmap[ids] = unc_map[ids]
        # costmap = (costmap - .65)/.2
        costmap[~np.isfinite(costmap)] = 0.0
        costmap[zero_mask] = self.unknown_cost
        speedmap[zero_mask] = self.unknown_speed

        # var_min = costmap_var.min()
        # var_max = costmap_var.max()
        # var_min = .2
        # var_max = .3
        # print('###', var_min, var_max)
        print("Speedmap Min/Max ", speedmap.min(), speedmap.max(), speedmap[xloc[0],yloc[0]])
        # print("Speedmap VAR Min/Max ", speedmap_var.min(), speedmap_var.max())

        # var_viz = (costmap_var - var_min)/(var_max-var_min)
        # var_viz = np.clip(var_viz,0,1)
        # var_viz[~np.isfinite(var_viz)] = 0.0
        # var_viz[xloc,yloc] = 1

        speedmap = np.clip(speedmap,0,self.max_velocity)


        # tpublish = time.perf_counter()
        t5 = time.perf_counter()
        print("T5 ", t5  - t4)
        # costmap_msg = self.costmap_to_gridmap(costmap, info, variance = var_viz)
        costmap_msg = self.costmap_to_gridmap(costmap, info)
        self.costmap_pub.publish(costmap_msg)


        speedmap_msg = self.costmap_to_gridmap(speedmap, info, costmap_layer = 'speedmap', norm_factor = self.max_velocity)
        self.speedmap_pub.publish(speedmap_msg)

        # print("PUBLISH ", time.perf_counter() - tpublish)
        t6 = time.perf_counter()
        print("T6 ", t6  - t5)


        cvar_msg = Float32()
        cvar_msg.data = self.cvar_alpha
        self.cvar_pub.publish(cvar_msg)
        # self.costmap_msg = costmap_msg
        # self.speedmap_msg = speedmap_msg

        # if self.hz_counter % 4 != 0:
        #     return


        print(time.perf_counter() - now, 'time', self.buffer_idx, 'buffer_idx')

    def costmap_to_gridmap(self, costmap, info, costmap_layer='costmap', variance = None, norm_factor = None):
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
        # layer_data = np.zeros_like(costmap) + self.odom_msg.pose.pose.position.z - 1.73 #+ costmap

        if variance is not None:
            layer_data = np.zeros_like(costmap) + self.odom_msg.pose.pose.position.z - 1.73 - variance*5
        else:
            layer_data = np.zeros_like(costmap) + self.odom_msg.pose.pose.position.z - 1.73 #+ costmap
        # print(variance.max(), variance.min())
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

        gridmap_layer_msg.data = layer_data[::-1, ::-1].flatten()
        costmap_msg.data.append(gridmap_layer_msg)

        # vcostmap = np.clip(costmap,0,1)
        if norm_factor is None:
            # vcostmap = np.clip(costmap*2.,0,.6)/.6
            vcostmap = np.clip(costmap*2.,0,1)
        else:
            vcostmap = np.clip(costmap,0,norm_factor)/norm_factor
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
    # rospy.spin()


if __name__ == "__main__":
    main()

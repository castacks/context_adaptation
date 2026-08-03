#!/usr/bin/env python3
import numpy as np
import os
import yaml

import math

import time
import cv2
import yaml
from yaml.loader import SafeLoader
import matplotlib.pyplot as plt
from scipy import signal, stats

import pickle
from context_adaptation.common.utils import pose_msg_to_se3

import torch
import torch.nn.functional as F

from matplotlib.animation import FuncAnimation, ArtistAnimation
from context_adaptation.common.gridmap_converter import GridMapConvert
from context_adaptation.models.ExactGP import GPTraversability
from context_adaptation.models.TravMLP import MLPTravRegress

from tartandriver_utils.os_utils import load_yaml

MODEL_MAP = {'GP': GPTraversability, 'MLPr': MLPTravRegress}
import matplotlib
try :
    CMAP = matplotlib.cm.get_cmap('magma')
    SMAP = matplotlib.cm.get_cmap('jet')
except:
    CMAP = matplotlib.pyplot.get_cmap('magma')
    SMAP = matplotlib.pyplot.get_cmap('jet')    

#TODO
#filepaths cleanup
#split into collect/infer
#inference class
#time-based stuff

class SalonCostmap():
    def __init__(self, config_file='costmap_configs/GP_base.yaml'):

        #TODO
        self.config_file = config_file
        
        self.assets_dir = '/home/tartandriver/tartandriver_ws/src/planning/context_adaptation/assets'

        config = load_yaml(os.path.join(self.assets_dir, self.config_file))

        self.rotate_odom = config['ROBOT']['rotate_odom'] #True
        self.modulate_speed_cvar = config['BEHAVIOR']['modulate_speed_cvar'] #False

        self.hz_counter = 0

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

        feats_conf = config['BEHAVIOR']['feats']
        self.VLAD_CLUSTS = feats_conf['n_clusters']

        self.residual_max = feats_conf['residual_max']
        self.uc_thresh = feats_conf['uncertainty_threshold']

        # self.cost_lengthscale = config['BEHAVIOR']['cost_lengthscale']
        # self.speed_lengthscale = config['BEHAVIOR']['speed_lengthscale']
        # self.train_kernel = config['BEHAVIOR']['train_kernel']

        buffer_len = config['BEHAVIOR']['buffer_size']
        self.buffer_update_freq = config['BEHAVIOR']['buffer_update_freq']
        self.train_in_buffer = torch.zeros((buffer_len,self.VLAD_CLUSTS + 2)).cuda()
        # self.train_label_buffer = torch.zeros((buffer_len,1)).cuda()
        self.train_buffer_classes = np.zeros((buffer_len))
        self.toi = np.zeros((buffer_len))
        self.buffer_idx = 0
        self.buffer_full = False

        self.cost_offset = .0
        self.velocity = 0.0
        self.fake_velocity = 0.0

        self.num_insert = 0
        for label in feats_conf['labels']:
            feat = torch.Tensor(label['feat']).cuda()/self.residual_max
            cost = label['cost']
            self.insert_label(feat, cost, feats_conf['num_insert'], feats_conf['spread'])

        self.buffer_idx += self.num_insert

        train_in_buffer = self.train_in_buffer[:self.buffer_idx]
        
        #TODO parameterize and account for geom features
        # self.trav_model_indices = [0,1,2,3,4,5,6,7,8]
        # self.speed_model_indices = [0,1,2,3,4,5,6,7,9]
        base_indices = np.arange(self.VLAD_CLUSTS + 2)
        self.rough_idx = self.VLAD_CLUSTS + 1
        self.speed_idx = self.VLAD_CLUSTS
        self.trav_model_indices = [idx for idx in base_indices if idx != self.rough_idx]
        self.speed_model_indices = [idx for idx in base_indices if idx != self.speed_idx]

        self.check_ready()

        trav_config = config['BEHAVIOR']['trav_model']
        speed_config = config['BEHAVIOR']['speed_model']

        if 'weights' in trav_config:
            trav_params_dir = os.path.join(self.assets_dir, 'gp_params', trav_config['weights'])
        else:
            trav_params_dir = None

        if 'weights' in speed_config:
            speed_params_dir = os.path.join(self.assets_dir, 'gp_params', speed_config['weights'])
        else:
            speed_params_dir = None


        self.trav_model = MODEL_MAP[trav_config['model_type']](train_in_buffer[:,self.trav_model_indices], 
                                           train_in_buffer[:,-1],
                                           trav_config,
                                           params_dir = trav_params_dir)
        self.speed_model = MODEL_MAP[speed_config['model_type']](train_in_buffer[:,self.speed_model_indices], 
                                           train_in_buffer[:,-2],
                                           speed_config,
                                           params_dir = speed_params_dir)
        

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


    def handle_odom(self, odom, vel):
        self.velocity = vel
        self.pose_se3 = odom.cpu().numpy()

        if self.rotate_odom:
            #TODO TEMP fix
            self.pose_se3[:3,:3] = self.pose_se3[:3,:3].T
            theta = np.radians(90)
            R_adj = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [0,              0,             1]
            ])
            self.pose_se3[:3,:3] = self.pose_se3[:3,:3] @ R_adj

        self.vel_history = self.vel_history + (self.velocity - self.vel_history) * .1


    def handle_cost(self, cost):
        self.cost = cost
        self.cost -= self.cost_offset
        self.rough_history = self.rough_history + (cost - self.rough_history) * .2


    def handle_max_roughness(self, msg):
        self.max_roughness = msg.data
        self.fake_velocity = msg.data


    def handle_cvar(self, msg):
        self.cvar_alpha = msg.data


    def handle_map(self,msg):
        with self._lock:
            self.dino_map = msg
            self.new_msg = True

            if len(self.channels) == 0:
                # print("MSG layers = ", msg.layers);
                for layer in msg.layers:
                    # if 'VLAD' in layer:
                    if 'dino' in layer :
                        self.channels.append(layer)

                self.grid_map_cvt.channels = self.channels

    def insert_label(self, feat, cost, num_insert, spread):
        avoid_class = feat.argmin()
        avoid_data = torch.zeros(num_insert, self.VLAD_CLUSTS + 2).cuda()
        avoid_classes = np.zeros((num_insert)) + avoid_class.item()
        avoid_data[:,-1] = cost
        # if cost >= 1.:
            # avoid_data[0,-1] = .1
        for i in range(num_insert):
            avoid_data[i,-2] = i*spread
            avoid_data[i,:self.VLAD_CLUSTS] = feat

        self.num_insert += num_insert
        self.train_in_buffer = torch.vstack((avoid_data,self.train_in_buffer))
        self.train_buffer_classes = np.concatenate((avoid_classes, self.train_buffer_classes))

    def update_train_buffer(self,input,label,class_id):
        self.train_in_buffer[self.buffer_idx] = input
        # self.train_label_buffer[self.buffer_idx] = label
        self.train_buffer_classes[self.buffer_idx] = class_id
        self.buffer_idx = min(self.buffer_idx+1, self.train_in_buffer.shape[0])

        if self.buffer_idx == self.train_in_buffer.shape[0]:
            self.buffer_full = True
            # self.buffer_idx = 0

        if not self.is_ready:
            self.check_ready()

    def check_ready(self):
        self.is_ready = (len(self.train_in_buffer[:self.buffer_idx,self.rough_idx].unique() > 1) &
                      len(self.train_in_buffer[:self.buffer_idx,self.speed_idx].unique() > 1))

    def insert_train_buffer(self,input,label, class_id, idx):
        self.train_in_buffer[idx] = input
        # self.train_label_buffer[idx] = label
        self.train_buffer_classes[idx] = class_id

        if not self.is_ready:
            self.check_ready()

    def insert_train_buffer_FIFO(self,input,label, class_id):
        idx = np.argmin(self.toi)
        self.train_in_buffer[idx] = input
        # self.train_label_buffer[idx] = label
        self.train_buffer_classes[idx] = class_id
        self.toi[idx] = self.hz_counter

    def load_buffer(self, fpath):
        state_dict = torch.load(fpath, weights_only=False)
        for key, val in state_dict.items():
            # print(attr, key)
            # val = state_dict[key]
            val = val.cuda() if hasattr(val, 'cuda') else val
            setattr(self, key, val)

        if not self.buffer_full:
                train_in_buffer = self.train_in_buffer[:self.buffer_idx]
        else:
            train_in_buffer = self.train_in_buffer

        self.trav_model.train(train_in_buffer[:,:-1], train_in_buffer[:,-1])
        self.speed_model.train(train_in_buffer[:,self.speed_model_indices], train_in_buffer[:,-2])

    def save_buffer(self, fpath):
        save_props = ['train_in_buffer', 
                       'train_buffer_classes', 
                       'buffer_idx', 'num_insert', 
                       'buffer_full']
        
        state_dict = {}
        for key in save_props:
            val = getattr(self, key)
            state_dict[key] = val.cpu() if hasattr(val, 'cpu') else val

        torch.save(state_dict, fpath)

    def estimate_tire_points(self):
        tire_points = np.ones((4, 3))

        for i, transform in enumerate(self.tire_transforms):
            tire_points[i, 0:3] = (self.pose_se3 @ transform)[0:3, -1]

        return tire_points

    def prep_inputs(self, feats):
        vel_append = torch.ones(feats.shape[0],1).cuda() * self.velocity
        rough_append = torch.ones(feats.shape[0],1) * self.max_roughness
        input = torch.hstack((feats,vel_append.cuda(), rough_append.cuda())).cuda()
        # input = (input - self.in_mean)/self.in_std

        return input

    def predict_img(self, feat_img):
        min_result = feat_img.min(axis=2)
        unc_img = min_result.values
        unc_img /= self.residual_max
        unc_img[unc_img < self.uc_thresh] = 0

        # cost_img = torch.zeros((feat_img.shape[0],feat_img.shape[1]))

        feat_flat = feat_img.reshape(-1,self.VLAD_CLUSTS)/self.residual_max
        feat_flat = self.prep_inputs(feat_flat)

        img_pred, cost_var = self.trav_model.forward(feat_flat[:,:-1], self.costmap_cvar)

        # observed_pred = self.likelihood(self.gp(feat_flat[:,:-1]))
        # img_pred = observed_pred.mean
        # img_var = observed_pred.variance
        # img_pred = (img_pred * self.in_std[-1]) + self.in_mean[-1]
        cost_img = img_pred.reshape(feat_img.shape[0],feat_img.shape[1])

        cost_img = torch.clip(cost_img, 0,1)
        cost_img *= .5
        ids = torch.where(unc_img != 0)
        cost_img[ids] = unc_img[ids]

        return cost_img
        # return cost_var.reshape(feat_img.shape[0],feat_img.shape[1])

    def run(self, featmap, metadata, odom, vel, cost, feat_img = None):
        now = time.perf_counter()
        if self.hz_counter == 5000000:
            self.hz_counter = 0
        self.hz_counter += 1

        # np.save('/home/tartandriver/tartandriver_ws/featmap_img', feat_img.cpu().numpy())

        # res_out = {'cost_image': None}
        res_out = {}

        self.handle_odom(odom, vel)
        self.handle_cost(cost)

        # np.save('/home/tartandriver/cat_tartandriver_ws/featmap', featmap.cpu().numpy())

        og_device = featmap.device

        min_result = featmap.min(axis=0)
        da = min_result.indices
        unc_map = min_result.values

        unc_map /= self.residual_max
        unc_map[unc_map < self.uc_thresh] = 0

        e_kernel_size = 2
        eroded = -F.max_pool2d(-unc_map.view(1,1,*unc_map.shape), kernel_size=e_kernel_size, stride=1, padding=0)
        dilated = F.max_pool2d(eroded, kernel_size=e_kernel_size, stride=1, padding=0)
        unc_map = dilated[0,0]

        P = metadata.origin.cpu().numpy()
        res = metadata.resolution.cpu().numpy()
        tp = self.estimate_tire_points()
        # print(tp.shape, P.shape, res.shape)
        tx = np.floor((tp[:,0]-P[0])/res[0])
        ty = np.floor((tp[:,1]-P[1])/res[1])
        xloc,yloc = ty.astype(int), tx.astype(int)

        gridmap_size = featmap.shape[1:]
        gridmap_len = np.prod(gridmap_size)

        if np.any(xloc > gridmap_size[0]) or np.any(yloc > gridmap_size[1]): #Super Odometry probably borked, wait till back online
            print("WARNING: OUT OF BOUNDS! Not computing costmap")
            # return None
            costmap = torch.zeros(gridmap_size)
            speedmap = torch.zeros(gridmap_size)
            res_out['costmap'] = costmap
            res_out['speedmap'] = speedmap
            return res_out

        # self.get_logger().info("++++++++++++++++++ L349 after np.any +++++++++++++ ");

        classes = da[xloc,yloc]
        spot,_ = torch.mode(classes)

        # np.save('/home/physics_atv/physics_atv_ws/gridmap_data', gridmap['data'])

        # input = torch.from_numpy(featmap/self.residual_max)
        input = featmap/self.residual_max
        zero_mask = (input == 0).all(dim=0).cpu().numpy()
        known_mask = ~zero_mask

        if (self.velocity > .3) and self.hz_counter % self.buffer_update_freq == 0:
            # input_sample = input[:,xloc[0],yloc[0]]
            input_sample = input[:,xloc,yloc]

            in_observed = torch.count_nonzero(input_sample[:,:-2], dim=1) > 0

            if not in_observed.any():
                print("IN UNKNOWN SPACE")
            elif cost < 0.0:
                print("INVALID/EMPTY COST")
            else:
                input_sample = input_sample[in_observed].mean(dim=1)
                input_sample = torch.cat((input_sample, torch.Tensor([self.velocity]).cuda(),torch.Tensor([self.cost]).cuda()))
                if self.buffer_full:
                    # print("))))))))))))))))))))))))))))))))))))))))))))))))))")
                    most_class = stats.mode(self.train_buffer_classes)[0]
                    vel_hist = torch.histc(self.train_in_buffer[self.train_buffer_classes == most_class,-2], bins=10, min=0,max=10)
                    most_vel = torch.argmax(vel_hist)
                    edges = np.arange(0,11)
                    sel_min = edges[most_vel]
                    sel_max = edges[most_vel+1]
                    train_vels = self.train_in_buffer[:,-2].cpu().numpy()
                    insert_idx = np.random.choice(np.where((self.train_buffer_classes == most_class) & (train_vels >= sel_min) & (train_vels <= sel_max))[0])
                    if insert_idx > self.num_insert: #don't remove human labels
                        self.insert_train_buffer(input_sample, self.cost, spot, insert_idx)
                    # self.insert_train_buffer_FIFO(input_sample, self.cost, spot)
                else:
                    self.update_train_buffer(input_sample, self.cost, spot)

        costmap = torch.zeros(gridmap_len).to(og_device)
        speedmap = torch.zeros(gridmap_len).to(og_device) + self.max_velocity*.75

        if not self.buffer_full:
                train_in_buffer = self.train_in_buffer[:self.buffer_idx]
        else:
            train_in_buffer = self.train_in_buffer

        speed_train_buffer = train_in_buffer.clone()
        # keep_idxs = speed_train_buffer[:,-1] <= self.max_roughness
        # speed_train_buffer = speed_train_buffer[keep_idxs]
        self.trav_model.train(train_in_buffer[:,:-1], train_in_buffer[:,-1])
        self.speed_model.train(speed_train_buffer[:,self.speed_model_indices], speed_train_buffer[:,-2])

        with torch.no_grad():
            # if self.buffer_idx < 0 + self.num_insert: #we need at least a couple samples apart from single label
            if not self.is_ready:
                costmap = torch.zeros(gridmap_size)
                speedmap = torch.zeros(gridmap_size) + self.max_velocity*.75
                costmap_var = torch.zeros_like(costmap)
                speedmap_var = torch.zeros_like(speedmap)

                if feat_img is not None:
                    cost_img = torch.zeros(feat_img.shape[:2]).float()
                    res_out['cost_image'] = cost_img
            else:
                input = input.permute(1,2,0).view(-1,self.VLAD_CLUSTS)

                zero_mask_f = zero_mask.flatten()
                keep_mask = ~zero_mask_f
                input = input[keep_mask]

                input = self.prep_inputs(input)

                cost_pred, cost_var = self.trav_model.forward(input[:,:-1], self.costmap_cvar)
                speed_pred, speed_var = self.speed_model.forward(input[:,self.speed_model_indices], self.cvar_alpha)

                costmap[keep_mask] = cost_pred
                speedmap[keep_mask] = speed_pred

                costmap = costmap.reshape(gridmap_size)
                speedmap = speedmap.reshape(gridmap_size)

                if self.maxpool_speedmap:
                    ksize = 3
                    padding = (ksize - 1)//2
                    m = torch.nn.MaxPool2d(ksize,stride=1, padding = padding)
                    speedmap = m(speedmap.unsqueeze(0))[0]

                if self.modulate_speed_cvar:
                    if (self.rough_history < self.max_roughness) and torch.abs(self.vel_history - speedmap[xloc[0],yloc[0]]) < self.velocity_margin:
                        self.cvar_alpha += 0.0005
                    elif (self.rough_history > self.max_roughness):# and np.abs(self.vel_history - speedmap[xloc[0],yloc[0]]) < self.velocity_margin:
                        self.cvar_alpha -= 0.02
                    self.cvar_alpha = np.clip(self.cvar_alpha, 0,.99)

                    #atv can't track well at low speeds then increase cvar
                    if (torch.abs(self.vel_history - speedmap[xloc[0],yloc[0]]) > .4) and speedmap[xloc[0],yloc[0]] < 2.8:
                        self.cvar_alpha += .01

                if feat_img is not None:
                    cost_img = self.predict_img(feat_img)
                    res_out['cost_image'] = cost_img

            if self.buffer_idx < 15 + self.num_insert: #let's give the speedmap a bit more time
                speedmap = torch.zeros(gridmap_size) + self.max_velocity*.75
                self.cvar_alpha = 0.0

        
        costmap /= 2.0
        # costmap[xloc,yloc] = 1. #only turn on to debug tire position queries
        # costmap *= 0.0
        ids = torch.where(unc_map != 0)

        costmap = costmap.to(og_device)
        speedmap = speedmap.to(og_device)

        costmap[ids] = unc_map[ids]
        costmap[~torch.isfinite(costmap)] = 0.0       
        costmap[zero_mask] = self.unknown_cost
        speedmap[zero_mask] = self.unknown_speed

        # print("Speedmap Min/Max ", speedmap.min(), speedmap.max(), speedmap[xloc[0],yloc[0]])

        speedmap = torch.clip(speedmap,0,self.max_velocity)
        # costmap = costmap.cpu()
        # speedmap = speedmap.cpu()
        # print(costmap.shape, speedmap.shape)

        

        res_out['costmap'] = costmap
        res_out['speedmap'] = speedmap
        
        return res_out
#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
# from learned_cost_map.msg import FloatStamped
import numpy as np
import rospkg

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
from std_msgs.msg import Int32, Float32
import skimage
import time
import cv2
import yaml
from yaml.loader import SafeLoader
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import signal, stats
# from PIL import Image
from sklearn.cluster import KMeans

import torchvision.models as models
import torch
from torch import nn
from torchvision import transforms

import dynamic_reconfigure.client

import pickle
from context_adaptation_common.links_cluster import LinksCluster, Subcluster

from ultralytics import FastSAM

import torch.nn as nn
import torch.nn.functional as F

from context_adaptation_common.anyloc_utils import DinoV2ExtractFeatures
from context_adaptation_common.anyloc_utils import VLAD
from typing import Literal, Union
import distinctipy as COLORS
from matplotlib.animation import FuncAnimation, ArtistAnimation

# Csub = np.array(COLORS.get_colors(32))
#Csub = np.load('colormap.npy')


class Context_Clusterer(object):
    # def __init__(self,cost_topic, odom_topic, im_topic, output_topic, filter_size=15, buffer_size = 100, speed_init = 3.0, max_cost = .34, velocity_margin = 1.5, speed_cap = 5):
    def __init__(self,cost_topic, odom_topic, im_topic, output_topic, filter_size=10, buffer_size = 50, speed_init = 3.0, max_cost = .3, velocity_margin = .8, speed_cap = 5.5):

        self.viz = False
        # Set up subscribers
        # rospy.Subscriber('/lester/imu/data', Imu, self.handle_odom, queue_size=1)
        # rospy.Subscriber('/novatel/imu/data', Imu, self.handle_imu, queue_size=1)
        rospy.Subscriber(im_topic, Image, self.handle_image, queue_size=5)
        # rospy.Subscriber(im_topic, CompressedImage, self.handle_image, queue_size=1)
        rospy.Subscriber(cost_topic, Float32, self.handle_cost, queue_size=1)
        rospy.Subscriber(odom_topic, Odometry, self.handle_odom, queue_size=1)

        self.context_pub = rospy.Publisher(output_topic,Int32,queue_size=2)
        self.max_vel_pub = rospy.Publisher('/controller/target_input',Float32,queue_size=2)



        NCLUSTERS = 3
        SNAME = 'DINO_IM224_ln31_CROP_20_feats_holdout_2_not_turnpike_reduced'

        desc_layer: int = 31
        desc_facet: Literal["query", "key", "value", "token"] = "value"
        num_c: int = 32
        encoder = DinoV2ExtractFeatures("dinov2_vitg14", desc_layer,
            desc_facet, device="cuda")
        # dim = 1536
        resnet152 = encoder

        vlad = VLAD(8, desc_dim=None,cache_dir='./dino_clusters/not_turnpike_8')
        vlad.fit(None)


        # for p in resnet152.parameters():
        #     p.requires_grad = False
        #
        # resnet152.eval()
        # resnet152.cuda()
        self.model = resnet152
        self.vlad = vlad
        # links = LinksCluster(.6,.4,1.0,store_vectors=True)

        #20,give,.8,.7,1.0

        with open(SNAME + '_' + str(NCLUSTERS) + "_clusters.data", "rb") as input_file:
            kmeans = pickle.load(input_file)

        # s=r
        #TODO, resolve image size diff

        # self.links = LinksCluster(.4,.3,1.0,store_vectors=True,initial_centroids = kmeans.centroids.cpu().numpy(),dist_metric='cosine',n_vectors=1000)
        # self.links = LinksCluster(.4,.3,1.0,store_vectors=True,dist_metric='cosine')

        # self.links = LinksCluster(.2,.1,1.0,store_vectors=True,initial_centroids = kmeans.centroids.cpu().numpy(),dist_metric='cosine',n_vectors=500)
        self.links = LinksCluster(.5,.4,1.0,store_vectors=True,dist_metric='cosine')



        #ARL
        # self.links = LinksCluster(.78,.75,1.0,store_vectors=True,initial_centroids = kmeans.cluster_centers_,dist_metric='cosine',n_vectors=5000)
        # self.links = LinksCluster(.8,.7,1.0,store_vectors=True,dist_metric='cosine')



        print("INITIALIZED WITH " + str(len(self.links.clusters)) + " Clusters")

        self.buff = np.zeros(filter_size)#-1
        self.tsu = time.perf_counter()
        self.update_freq = 5

        self.image_size = 224
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size,self.image_size)),
            # transforms.CenterCrop(self.image_size),
            # transforms.ToTensor(),
            # transforms.Lambda(expand_greyscale)
        ])

        # self.preprocess = transforms.Compose([
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])

        self.bridge = CvBridge()

        self.speed_map = {}
        self.speed_init = speed_init
        self.buffer_size = buffer_size
        self.max_cost = max_cost
        self.speed_cap = speed_cap
        self.velocity_margin = velocity_margin
        for i in range(len(self.links.clusters)):
            self.speed_map[i] = [speed_init, np.zeros(self.buffer_size) + self.max_cost/2]


        self.velocity = 0
        self.cost = 0


        if self.viz:
            self.fig, self.ax = plt.subplots(1,3)
            self.ln, = self.ax[0].plot([], [], 'ro')
            self.x_data, self.y_data = [] , []
            self.viz_skip = 5
            self.skip_counter = 0

            self.anchor_dict = {}
            # self.anchor_feats = np.zeros()

        self.hz_counter = 0

        print('DONE WITH INIT')


    def update_plot(self, frame):
        # self.ln.set_data(self.x_data, self.y_data)
        # return self.ln
        return

    def handle_odom(self, msg):
        self.velocity = np.linalg.norm([msg.twist.twist.linear.x,msg.twist.twist.linear.y])

    def handle_cost(self, msg):
        self.cost = msg.data
        # print(self.cost)

    def handle_image(self, msg):
        now = time.perf_counter()

        # self.hz_counter += 1
        # if self.hz_counter % 4 != 0:
        #     return
        # np_arr = np.frombuffer(msg.data, np.uint8)
        # img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # print(img.shape)

        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')


        # print(np.mean(img))
        img = img.astype(float)/255.0
        img = img[:,:,::-1].copy()



        input = torch.Tensor(img)#.unsqeeze(0).cuda()
        input = input.permute(2,0,1).unsqueeze(0).cuda()
        # tim = input.permute(0,2,3,1)
        # tim = tim[0].cpu().numpy()
        # plt.imshow(tim)
        # plt.show()
        # print(input.shape)
        # cv2.imshow('input',tim)
        # cv2.waitKey(0)
        # input = self.preprocess(input)
        # print(input.shape)
        input = self.transform(input)
        # print(input.shape)
        # print(torch.mean(input))
        # print(input.shape)
        feat_vec = self.model(input)
        out = self.vlad.generate(feat_vec[0].cpu(),reduce=True)
        print(out.shape)

        if self.viz:
            # res = self.vlad.generate_res_vec(feat_vec[0].cpu())
            # # print(res.shape, v.shape)
            # da = res.abs().sum(dim=2).argmin(dim=1).reshape(16, 16)
            # da = F.interpolate(da[None, None, ...].to(float),
            # (self.image_size, self.image_size), mode='nearest')[0, 0].to(da.dtype)
            #
            # da = Csub[da.numpy().astype(int)]
            #
            # # da = np.transpose(da,(2,0,1))
            # # print(da.shape,da.mean())
            # # cv2.imshow((da*255).astype(np.uint8))
            # cv2.imshow('clusters',da)
            # cv2.waitKey(1)
            ...

        # print('infered')
        # print(out.shape)
        if len(out.shape) > 2:
            vec = torch.mean(out,dim=[2,3])[0].cpu().numpy()
        else:
            vec = out.numpy()
        # vec = torch.mean(out,dim=[2,3])[0].cpu().numpy()
        #TODO, should already be normalized right?

        # vec = vec/np.linalg.norm(vec)

        pred,probs_list = self.links.predict(vec,return_probs=True)


        if self.viz:
            self.skip_counter += 1
            if self.skip_counter % self.viz_skip == 0:
                self.skip_counter = 0
                self.ax[0].cla()
                self.ax[1].cla()
                # self.ax[2].cla()
                self.ax[0].bar(np.arange(len(probs_list)), probs_list)
                self.ax[0].set_ylim(0, 1)

                res = self.vlad.generate_res_vec(feat_vec[0].cpu())
                # print(res.shape, v.shape)
                da = res.abs().sum(dim=2).argmin(dim=1).reshape(16, 16)
                da = F.interpolate(da[None, None, ...].to(float),
                (self.image_size, self.image_size), mode='nearest')[0, 0].to(da.dtype)

                da = Csub[da.numpy().astype(int)]
                viz_img = .5*cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA) + .5*da
                # da = np.transpose(da,(2,0,1))
                # print(da.shape,da.mean())
                # print(da.shape)
                self.ax[1].imshow(viz_img)


            # self.ax[2].imshow(viz_img)


        # print(pred)
        self.buff[1:] = self.buff[:-1]
        self.buff[0] = pred

        outpred = stats.mode(self.buff)[0]
        cur_context = int(outpred)

        if cur_context in self.speed_map:
            # print(self.velocity)
            if np.abs(self.velocity - self.speed_map[cur_context][0]) < self.velocity_margin:
                self.speed_map[cur_context][1][1:] = self.speed_map[cur_context][1][:-1]
                self.speed_map[cur_context][1][0] = self.cost

                if np.mean(self.speed_map[cur_context][1]) > self.max_cost:
                    self.speed_map[cur_context][0] *= .999
                elif np.mean(self.speed_map[cur_context][1]) < self.max_cost*.8:
                    self.speed_map[cur_context][0] *= 1.001
                    self.speed_map[cur_context][0] = np.min([self.speed_map[cur_context][0], self.speed_cap])
        else:
            self.speed_map[cur_context] = [self.speed_init, np.zeros(self.buffer_size) + self.cost]

        if self.viz:
            if cur_context in self.anchor_dict:
                cossim = probs_list[cur_context]
                best = self.anchor_dict[cur_context][1]
                if cossim > best:
                    self.anchor_dict[cur_context] = (cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA),cossim)
                    self.ax[2].cla()
                    self.ax[2].imshow(self.anchor_dict[cur_context][0])
            else:
                cossim = probs_list[cur_context]
                self.anchor_dict[cur_context] = (cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA),cossim)
                self.ax[2].cla()
                self.ax[2].imshow(self.anchor_dict[cur_context][0])

            self.ax[2].set_title("CONTEXT: " + str(cur_context))

        print(cur_context)
        # print(self.speed_map)
        # for i in range(len(self.speed_map)):
            # print(i, self.speed_map[i][0])
        # print('---')
        out_context = Int32()
        out_context.data = cur_context
        self.context_pub.publish(out_context)

        out_speed = Float32()
        out_speed.data = self.speed_map[cur_context][0]
        self.max_vel_pub.publish(out_speed)
        print(time.perf_counter() - now, 'time')

if __name__ == "__main__":
    # print(os.listdir('dino_clusters/not_turnpike_8'))
    # print(os.listdir('assets'))
    rp = rospkg.RosPack()
    script_path = os.path.join(rp.get_path("context_adaptation"), "assets")
    
    rospy.init_node("context_publisher", log_level=rospy.INFO)
    rospy.loginfo("Initialized context_publisher node")
    cost_topic = '/traversability_cost'
    odom_topic = '/odometry/filtered_odom'
    im_topic = '/multisense/left/image_rect_color'
    # cost_topic = '/traversability_cost'
    # odom_topic = '/lester/odom'
    # im_topic = '/lester/stereo_left/image_rect_color/compressed'
    output_topic="/context_id"
    node = Context_Clusterer(cost_topic, odom_topic , im_topic, output_topic)
    rate = rospy.Rate(10)

    #ani = FuncAnimation(node.fig, node.update_plot)
    # ani = ArtistAnimation(node.fig, node.ax)
    #plt.show(block=True)

    while not rospy.is_shutdown():
        rate.sleep()

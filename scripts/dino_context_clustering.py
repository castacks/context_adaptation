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

Csub = np.array(COLORS.get_colors(32))
# Csub = np.load('colormap.npy')


class Context_Clusterer(object):
    # def __init__(self,cost_topic, odom_topic, im_topic, output_topic, filter_size=15, buffer_size = 100, speed_init = 3.0, max_cost = .34, velocity_margin = 1.5, speed_cap = 5):
    # def __init__(self,cost_topic, odom_topic, im_topic, output_topic, filter_size=10, buffer_size = 10, speed_init = 3.0, max_cost = .32, velocity_margin = 1.5, speed_cap = 5):
    def __init__(self,cost_topic, odom_topic , im_topic, context_topic, vel_pub_topic, config, viz=True, pub_anchors = True, pub_stats=True):

        self.viz = viz
        self.pub_anchors = pub_anchors
        self.pub_stats = pub_stats

        rp = rospkg.RosPack()
        assets_dir = os.path.join(rp.get_path("context_adaptation"), "assets") + '/'


        init_centroid = False
        if 'sname' in config['ONLINE_CLUSTERING']:
            NCLUSTERS = config['ONLINE_CLUSTERING']['n_clusters']
            SNAME = assets_dir + config['ONLINE_CLUSTERING']['sname']

            with open(SNAME + '_' + str(NCLUSTERS) + "_clusters.data", "rb") as input_file:
                kmeans = pickle.load(input_file)
            init_centroid = True

        desc_layer: int = config['VLAD']['desc_layer']
        desc_facet: Literal["query", "key", "value", "token"] = "value"
        encoder = DinoV2ExtractFeatures(config['VLAD']['model'], desc_layer,
            desc_facet, device="cuda")

        vlad = VLAD(config['VLAD']['n_clusters'], desc_dim=None,cache_dir=assets_dir + config['VLAD']['cache_dir'])
        vlad.fit(None)

        self.model = encoder
        self.vlad = vlad


        # self.links = LinksCluster(.4,.3,1.0,store_vectors=True,initial_centroids = kmeans.centroids.cpu().numpy(),dist_metric='cosine',n_vectors=1000)
        # self.links = LinksCluster(.4,.3,1.0,store_vectors=True,dist_metric='cosine')

        # self.links = LinksCluster(.2,.1,1.0,store_vectors=True,initial_centroids = kmeans.centroids.cpu().numpy(),dist_metric='cosine',n_vectors=500)
        # self.links = LinksCluster(.55,.45,1.0,store_vectors=True,dist_metric='cosine')

        cst = config['ONLINE_CLUSTERING']['cluster_similarity_threshold']
        scst = config['ONLINE_CLUSTERING']['subcluster_similarity_threshold']
        psm = config['ONLINE_CLUSTERING']['pair_similarity_maximum']

        if init_centroid:
            nv = config['ONLINE_CLUSTERING']['n_vectors']
            self.links = LinksCluster(cst, scst, psm, store_vectors=True,initial_centroids = kmeans.centroids.cpu().numpy(),dist_metric='cosine',n_vectors=nv)
            self.num_contexts = len(self.links.clusters)
        else:
            self.links = LinksCluster(cst, scst, psm, store_vectors=True,dist_metric='cosine')
            self.num_contexts = 0



        #ARL
        # self.links = LinksCluster(.78,.75,1.0,store_vectors=True,initial_centroids = kmeans.cluster_centers_,dist_metric='cosine',n_vectors=5000)
        # self.links = LinksCluster(.8,.7,1.0,store_vectors=True,dist_metric='cosine')



        print("INITIALIZED WITH " + str(len(self.links.clusters)) + " Clusters")

        self.buff = np.zeros(config['BEHAVIOR']['filter_size'])#-1
        self.tsu = time.perf_counter()
        self.update_freq = config['MISC']['update_freq']

        self.image_size = config['MISC']['image_size']
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
        self.speed_init = config['BEHAVIOR']['speed_init']
        self.buffer_size = config['BEHAVIOR']['buffer_size']
        self.max_cost = config['BEHAVIOR']['max_cost']
        self.speed_cap = config['BEHAVIOR']['speed_cap']
        self.velocity_margin = config['BEHAVIOR']['velocity_margin']
        self.adaptation_factor = config['BEHAVIOR']['adaptation_factor']

        for i in range(len(self.links.clusters)):
            self.speed_map[i] = [self.speed_init , np.zeros(self.buffer_size) + self.max_cost/2]

        self.velocity = 0
        self.cost = 0
        self.weighted_avg_speed = config['BEHAVIOR']['weighted_avg_speed']


        if self.viz or self.pub_anchors:
            self.fig, self.ax = plt.subplots(1,3)
            self.ln, = self.ax[0].plot([], [], 'ro')
            self.x_data, self.y_data = [] , []
            self.viz_skip = 5
            self.skip_counter = 0

            self.anchor_dict = {}
            # self.anchor_feats = np.zeros()

        self.hz_counter = 0


        # Set up subscribers
        # rospy.Subscriber('/lester/imu/data', Imu, self.handle_odom, queue_size=1)
        # rospy.Subscriber('/novatel/imu/data', Imu, self.handle_imu, queue_size=1)
        rospy.Subscriber(im_topic, Image, self.handle_image, queue_size=5)
        # rospy.Subscriber(im_topic, CompressedImage, self.handle_image, queue_size=1)
        rospy.Subscriber(cost_topic, Float32, self.handle_cost, queue_size=1)
        rospy.Subscriber(odom_topic, Odometry, self.handle_odom, queue_size=1)

        self.context_pub = rospy.Publisher(context_topic,Int32,queue_size=2)
        self.max_vel_pub = rospy.Publisher(vel_pub_topic,Float32,queue_size=2)

        if self.pub_anchors:
            self.anchor_pub = rospy.Publisher('/context/current_anchor',Image,queue_size=2)

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
        print('----')
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
        # print(out.shape)

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
        print(probs_list)

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
                    self.speed_map[cur_context][0] *= 1.0-self.adaptation_factor
                elif np.mean(self.speed_map[cur_context][1]) < self.max_cost*.8:
                    self.speed_map[cur_context][0] *= 1.0+self.adaptation_factor
                    self.speed_map[cur_context][0] = np.min([self.speed_map[cur_context][0], self.speed_cap])
        else:
            self.speed_map[cur_context] = [self.speed_init, np.zeros(self.buffer_size) + self.cost]
            self.num_contexts += 1

        if self.viz or self.pub_anchors:
            if cur_context in self.anchor_dict:
                cossim = probs_list[cur_context]
                best = self.anchor_dict[cur_context][1]
                if cossim > best:
                    self.anchor_dict[cur_context] = (cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA),cossim)
                    if self.viz:
                        self.ax[2].cla()
                        self.ax[2].imshow(self.anchor_dict[cur_context][0])
                    # if self.pub_anchors:
                    #     img_msg = self.bridge.cv2_to_imgmsg(self.anchor_dict[cur_context][0], "passthrough")
                    #     self.anchor_pub.publish(img_msg)
                    #     print('here')
            else:
                cossim = probs_list[cur_context]
                self.anchor_dict[cur_context] = (cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA),cossim)
                if self.viz:
                    self.ax[2].cla()
                    self.ax[2].imshow(self.anchor_dict[cur_context][0])
                # if self.pub_anchors:
                #     # vimg =
                #     img_msg = self.bridge.cv2_to_imgmsg(self.anchor_dict[cur_context][0], "passthrough")
                #     self.anchor_pub.publish(img_msg)
                #     print('here')


            if self.viz:
                self.ax[2].set_title("CONTEXT: " + str(cur_context))

            if self.pub_anchors:
                # vimg =
                vimg = (self.anchor_dict[cur_context][0]*255).astype(np.uint8)
                img_msg = self.bridge.cv2_to_imgmsg(vimg, "rgb8")
                self.anchor_pub.publish(img_msg)
                # print('here')

        print(cur_context)
        # print(self.speed_map)
        # for i in range(len(self.speed_map)):
            # print(i, self.speed_map[i][0])
        # print('---')
        out_context = Int32()
        out_context.data = cur_context
        self.context_pub.publish(out_context)


        if self.weighted_avg_speed:
            # print(probs_list)
            probs_list = np.array(probs_list) + 1
            speed_list = np.array([self.speed_map[k][0] for k in range(self.num_contexts)])
            # print(speed_list, probs_list)
            cmd_speed = np.average(speed_list,weights=probs_list)
        else:
            cmd_speed = self.speed_map[cur_context][0]

        out_speed = Float32()
        out_speed.data = cmd_speed
        self.max_vel_pub.publish(out_speed)
        print(time.perf_counter() - now, 'time')

if __name__ == "__main__":
    rospy.init_node("context_publisher", log_level=rospy.INFO)
    rospy.loginfo("Initialized context_publisher node")
    cost_topic = rospy.get_param("~cost_topic")
    odom_topic = rospy.get_param("~odom_topic")
    im_topic = rospy.get_param("~im_topic")
    viz = rospy.get_param("~viz")
    pub_anchors = rospy.get_param("~pub_anchors")
    pub_stats = rospy.get_param("~pub_stats")
    config_file = rospy.get_param("~config_file")
    context_topic = rospy.get_param("~context_topic")
    vel_pub_topic = rospy.get_param("~vel_pub_topic")

    rp = rospkg.RosPack()
    config_file = os.path.join(rp.get_path("context_adaptation"), "assets","context_configs") + '/' + config_file
    config_dict = yaml.safe_load(open(config_file, 'r'))

    node = Context_Clusterer(cost_topic, odom_topic , im_topic, context_topic, vel_pub_topic, config_dict, viz=viz, pub_anchors = pub_anchors, pub_stats=pub_stats)
    rate = rospy.Rate(10)

    if viz:
        ani = FuncAnimation(node.fig, node.update_plot)
        # ani = ArtistAnimation(node.fig, node.ax)
        plt.show(block=True)

    while not rospy.is_shutdown():
        rate.sleep()

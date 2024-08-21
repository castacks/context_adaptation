import os
import rosbag
import numpy as np
from natsort import natsorted
from tqdm import tqdm
import matplotlib.pyplot as plt

bag_dir= '/media/matthew/Extreme SSD/20240726/2024-07-26-14-06-36_gupta_first_exp'
# #32/40
# # laps = np.array([[40,152], [152.5,297], [299,448]])
# laps = np.array([[32,152], [152.5,297], [299,448]])
laps = np.array([[32,152], [153.5,296], [301,447]])


# bag_dir= '/media/matthew/Extreme Pro/2024726/gupta_irl_take25lcb_data'
# laps = np.array([[13,158.0], [162,293], [296,435]])


exp_name = bag_dir.split('/')[-1]


rough = np.load(os.path.join(exp_name, 'roughness.npy'))
vels = np.load(os.path.join(exp_name, 'vels.npy'))
gps = np.load(os.path.join(exp_name, 'gps.npy'))
odom = np.load(os.path.join(exp_name, 'odom.npy'))

print(rough[0,0], vels[0,0])

laps += vels[0,0]

fig, ax = plt.subplots(2,3)
# ax.flatten()
vel_thresh = 1.0
r_super_thresh = .20

for i,lap in enumerate(laps):
    print(lap)
    # print(rough[:,0])
    start_id_r = np.argmin(np.abs(rough[:,0] - lap[0]))
    end_id_r = np.argmin(np.abs(rough[:,0] - lap[1]))
    # print(start_id_r, end_id_r)

    start_id_v = np.argmin(np.abs(vels[:,0] - lap[0]))
    end_id_v = np.argmin(np.abs(vels[:,0] - lap[1]))

    cur_rough = rough[start_id_r:end_id_r]
    cur_vels = vels[start_id_v:end_id_v]
    ax[0,i].plot(cur_rough[:,0], cur_rough[:,1])
    ax[1,i].plot(cur_vels[:,0], cur_vels[:,1])


    print("LAP " + str(i+1))
    rough_eval = cur_rough[cur_rough[:,2] > vel_thresh]
    vel_eval = cur_vels[cur_vels[:,1] > vel_thresh]

    # print(rough_eval.shape)
    rough_beyond = rough_eval[rough_eval[:,1]>r_super_thresh]
    print(rough_beyond.shape)
    rough_beyond_proportion = rough_beyond.shape[0] / rough_eval.shape[0]


    print("MEAN ROUGHNESS - ", rough_eval[:,1].mean(), "+/-", rough_eval[:,1].std())
    print("MEAN VEL - ", vel_eval[:,1].mean(), "+/-", vel_eval[:,1].std())
    print("BEYOND ROUGHNESS - ", rough_beyond_proportion)

    #DONT FORGET TO FILTER OUT LOW VELS

plt.show()

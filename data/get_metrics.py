import os
import rosbag
import numpy as np
from natsort import natsorted
from tqdm import tqdm
import matplotlib.pyplot as plt

# bag_dir= '/media/matthew/Extreme SSD/20240726/2024-07-26-14-06-36_gupta_first_exp'
# #32/40
# # laps = np.array([[40,152], [152.5,297], [299,448]])
# laps = np.array([[32,152], [152.5,297], [299,448]])
# laps = np.array([[32,152], [153.5,296], [301,447]])

#geometric?
# bag_dir= '/media/matthew/Extreme SSD/20240815/2024-08-15-15-07-21_semantic_lcb2'
# laps = np.array([[30, 180.],[181,362], [365,548], [550,705]])

#semantic
# bag_dir= '/media/matthew/Extreme SSD/20240815/2024-08-15-15-21-51_semantic_lcb2'
# laps = np.array([[28, 131.],[133,256], [258,381], [382,496],[498,612]])

# #IRL
# bag_dir= '/media/matthew/Extreme SSD/20240815/2024-08-15-14-52-11_lcb2'
# laps = np.array([[35., 139],[140,254], [256,366] ,[368,478],[479,590]])

# #C2
# bag_dir= '/media/matthew/Extreme SSD/20240823/2024-08-23-c2'
# laps = np.array([[21., 210],[212,360], [365,520],[524,665], [771,909]])
#mean vel goes up while max goes down
# mean rough approaches bound


#C4
# bag_dir= '/media/matthew/Extreme SSD/20240823/2024-08-23-c4'
# laps = np.array([[12., 172],[180,295], [308,417],[422,534], [598,707]])
# laps = np.array([[12., 172],[180,295], [308,417],[422,534], [592,713]])

#normal
bag_dir= '/media/matthew/Extreme SSD/20240823/2024-08-23-normal_hdif'
laps = np.array([[21, 149.],[153,330], [337,490], [496,640],[647,801]])


#max vel stays about the same, mean vel increases then stagnates , max vel around 6.2
#mean roughness increases

# bag_dir= '/media/matthew/Extreme Pro/2024726/gupta_irl_take25lcb_data'
# laps = np.array([[13,158.0], [162,293], [296,435]])


exp_name = bag_dir.split('/')[-1]


rough = np.load(os.path.join(exp_name, 'roughness.npy'))
vels = np.load(os.path.join(exp_name, 'vels.npy'))
gps = np.load(os.path.join(exp_name, 'gps.npy'))
odom = np.load(os.path.join(exp_name, 'odom.npy'))
scvar = np.load(os.path.join(exp_name, 'scvar.npy'))

# plt.plot(scvar)
# plt.show()

# print("GPS MAX, ", gps.max(axis=0))

print(rough[0,0], vels[0,0])

laps += vels[0,0]

fig, ax = plt.subplots(2,5)
# ax.flatten()
vel_thresh = .5
r_super_thresh = .4
vel_max = 8.5

MEANR = []
RB = []
MEANV = []
MAXV = []

STDR = []
STDV = []

for i,lap in enumerate(laps):
    print(lap)
    # print(rough[:,0])
    start_id_r = np.argmin(np.abs(rough[:,0] - lap[0]))
    end_id_r = np.argmin(np.abs(rough[:,0] - lap[1]))
    # print(start_id_r, end_id_r)

    # start_id_v = np.argmin(np.abs(vels[:,0] - lap[0]))
    # end_id_v = np.argmin(np.abs(vels[:,0] - lap[1]))

    start_id_v = np.argmin(np.abs(gps[:,-1] - lap[0]))
    end_id_v = np.argmin(np.abs(gps[:,-1] - lap[1]))

    cur_rough = rough[start_id_r:end_id_r]
    # cur_vels = vels[start_id_v:end_id_v]
    cur_vels = gps[start_id_v:end_id_v:,[-1,3]]
    cur_vels[cur_vels > vel_max] = vel_max
    ax[0,i].plot(cur_rough[:,0], cur_rough[:,1])
    ax[1,i].plot(cur_vels[:,0], cur_vels[:,1])


    print("LAP " + str(i+1))
    rough_eval = cur_rough[cur_rough[:,2] > vel_thresh]
    vel_eval = cur_vels[cur_vels[:,1] > vel_thresh]

    # print(rough_eval.shape)
    rough_beyond = rough_eval[rough_eval[:,1]>r_super_thresh]
    print(rough_beyond.shape)
    rough_beyond_proportion = rough_beyond.shape[0] / rough_eval.shape[0]


    # print("MEAN ROUGHNESS - ", rough_eval[:,1].mean(), "+/-", rough_eval[:,1].std())
    # print("MEAN VEL - ", vel_eval[:,1].mean(), "+/-", vel_eval[:,1].std())
    # print("BEYOND ROUGHNESS - ", rough_beyond_proportion)

    print(vel_eval.shape)

    MEANR.append(rough_eval[:,1].mean())
    STDR.append(rough_eval[:,1].std())

    MEANV.append(vel_eval[:,1].mean())
    STDV.append(vel_eval[:,1].std())
    MAXV.append(vel_eval[:,1].max())

    RB.append(rough_beyond_proportion)

    #DONT FORGET TO FILTER OUT LOW VELS

print("MEAN ROUGHNESS - ", MEANR)
print("BEYOND ROUGHNESS - ", RB)
print("MEAN VEL - ", MEANV)
print("MAX VEL - ", MAXV)

plt.show()

import os
import rosbag
import numpy as np
from natsort import natsorted
from tqdm import tqdm

bag_dir= '/media/matthew/Extreme Pro/20240815/2024-08-15-15-21-51_semantic_lcb2'
# bag_dir= '/media/matthew/Extreme Pro/2024726/gupta_irl_take25lcb_data'
# bag_dir= '/media/matthew/Extreme Pro/2024726/try2'

exp_name = bag_dir.split('/')[-1]
bag_names = os.listdir(bag_dir)
bag_names = natsorted(bag_names)
# Topic you want to extract messages from
topic_list = ['/odometry/filtered_odom','/integrated_to_init','/traversability_cost','/hdif_speedmap_cvar']

odom = []
gps = []
gps_t = []
vels = []
vt = []
roughness = []
rt = []
scvar = []

cur_vel = 0
cur_rough = 0

for bag_path in tqdm(bag_names):
    if '.bag' not in bag_path:
        continue
    if 'elev' in bag_path:
        continue
    # Open the bag file
    fname = os.path.join(bag_dir, bag_path)
    with rosbag.Bag(fname) as bag:
        # Iterate over messages in the bag
        for topic, msg, t in bag.read_messages(topics=topic_list):
            # Check if the message is of type Odometry
            ts = t.to_sec()
            # print(ts, topic)
            if topic == '/odometry/filtered_odom':
                x = msg.pose.pose.position.x
                y = msg.pose.pose.position.y
                z = msg.pose.pose.position.z

                vx = msg.twist.twist.linear.x
                vy = msg.twist.twist.linear.y
                vz = msg.twist.twist.linear.z

                gps.append([x,y,z,np.linalg.norm([vx,vy,vz]),cur_rough, ts])

            if topic == '/integrated_to_init':
                vx = msg.twist.twist.linear.x
                vy = msg.twist.twist.linear.y
                vz = msg.twist.twist.linear.z
                x = msg.pose.pose.position.x
                y = msg.pose.pose.position.y
                z = msg.pose.pose.position.z

                odom.append([x,y,z,vx,vy,vz])
                vels.append(np.linalg.norm([vx,vy,vz]))
                vt.append(ts)

                cur_vel = np.linalg.norm([vx,vy,vz])

            if topic == '/traversability_cost':
                roughness.append([msg.data,cur_vel])
                cur_rough = msg.data
                rt.append(ts)

            if topic == '/hdif_speedmap_cvar':
                scvar.append(msg.data)
        # break


os.mkdir(exp_name)
rt = np.array(rt)
print(rt)
roughness = np.array(roughness)
roughness = np.hstack([rt.reshape(-1,1),roughness.reshape(-1,2)])
odom = np.array(odom)
vels = np.array(vels)
vt = np.array(vt)
vels = np.hstack([vt.reshape(-1,1), vels.reshape(-1,1)])
gps = np.array(gps)
scvar = np.array(scvar)

print(roughness.shape)
print(vels.shape)
print(gps.shape)

np.save(os.path.join(exp_name, 'roughness'), roughness)
np.save(os.path.join(exp_name, 'vels'), vels)
np.save(os.path.join(exp_name, 'gps'), gps)
np.save(os.path.join(exp_name, 'odom'), odom)
np.save(os.path.join(exp_name, 'scvar'), scvar)

import os
import rosbag
import numpy as np
from natsort import natsorted
from tqdm import tqdm
import matplotlib.pyplot as plt
import rasterio
from rasterio.enums import Resampling
import yaml
CMAP = plt.cm.get_cmap('jet')


DAT = rasterio.open(r"/home/matthew/SARA/gps_filter/gascola.tif")
print(DAT.meta)
print(DAT.crs)
print(DAT.units)

# map = dat.read()[:3]/255.0
SCALE = .5

def grab_points(file, skip=2):
    out = []
    for wp in file['waypoints']:
        x = wp['pose']['x']
        y = wp['pose']['y']
        out.append([x,y])

    out = np.array(out)
    out = out[::skip]

    return out

def _to_map(points):
    row,col = DAT.index(-points[:,1], points[:,0])
    # print(row[:5])
    row = np.array(row).astype(float)
    # print(row[:5])
    col = np.array(col).astype(float)
    row *= SCALE
    col *= SCALE

    return row,col

map = DAT.read(out_shape=(DAT.count,
    int(DAT.height * SCALE),
    int(DAT.width * SCALE)
),
resampling=Resampling.bilinear)[:3]/255.0


fig8 = yaml.safe_load(open('/home/matthew/Downloads/warehouse_fig8.yaml', 'r'))
fig8_points = grab_points(fig8)
fig8_r, fig8_c = _to_map(fig8_points)

#unc -
# 92061
# 0.20931906032986775

# alter
# 87844
# 0.21654128073526885

#semantic
# 98176
# 0.18718885777405486

#velociraptor
# 79843
# 0.23839697316588476


bag_dir= '/media/matthew/Extreme Pro/20240531/2024-05-01-20-05-38_dino_irl_uncertainty'
sem_times = np.array([14.0,1220])
alter_times = np.array([19.0,935])
unc_times = np.array([11.0,990])
velo_times = np.array([25.0,935])


if 'sem' in bag_dir:
    exp_times = sem_times
elif 'alter' in bag_dir:
    exp_times = alter_times
elif 'unc' in bag_dir:
    exp_times = unc_times
else:
    exp_times = velo_times

exp_name = bag_dir.split('/')[-1]


rough = np.load(os.path.join(exp_name, 'roughness.npy'))
vels = np.load(os.path.join(exp_name, 'vels.npy'))


exp_times += vels[0,0]

vel_thresh = .5
r_super_thresh = .25

filtered = []

for r in tqdm(rough):
    t = r[0]
    vel_id = np.argmin(np.abs(vels[:,0] - t))
    vel = vels[vel_id,1]
    # print(t, vels[vel_id,0])
    if vel > .3:
        filtered.append(r[1])

filtered = np.array(filtered)
print(len(filtered))
print(np.mean(filtered))

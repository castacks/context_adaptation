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



# bag_dir= '/media/matthew/Extreme SSD/20240726/2024-07-26-14-06-36_gupta_first_exp'
# # #32/40
# # # laps = np.array([[40,152], [152.5,297], [299,448]])
# laps = np.array([[32,152], [152.5,297], [299,448]])
# laps = np.array([[32,152], [153.5,296], [301,447]])


# bag_dir= '/media/matthew/Extreme Pro/2024726/gupta_irl_take25lcb_data'
# laps = np.array([[13,158.0], [162,293], [296,435]])

# bag_dir = '/media/matthew/Extreme Pro/20240823/online_hdif/2024-08-23-c4'
# laps = np.array([[12., 172],[180,295], [308,417],[422,534], [592,713]])

bag_dir= '/media/matthew/Extreme SSD/20240823/2024-08-23-c2'
laps = np.array([[21., 210],[212,360], [365,520],[524,665], [771,909]])

exp_name = bag_dir.split('/')[-1]


rough = np.load(os.path.join(exp_name, 'roughness.npy'))
vels = np.load(os.path.join(exp_name, 'vels.npy'))
gps = np.load(os.path.join(exp_name, 'gps.npy'))
odom = np.load(os.path.join(exp_name, 'odom.npy'))
scvar = np.load(os.path.join(exp_name, 'scvar.npy'))

# plt.plot(scvar)
# plt.show()

print(rough[0,0], vels[0,0])
print(gps.shape)

laps += vels[0,0]

fig, ax = plt.subplots(1,5)
ax.flatten()
vel_thresh = 1.0
r_super_thresh = .25

for i,lap in enumerate(laps):

    start_id = np.argmin(np.abs(gps[:,-1] - lap[0]))
    end_id = np.argmin(np.abs(gps[:,-1] - lap[1]))

    cur_gps = gps[start_id:end_id]

    cur_gps = cur_gps[cur_gps[:,3] > 1.]

    row, col = _to_map(cur_gps[:,:2])

    # print(row.min(), row.max(),col.min(), col.max())
    gps_vels = cur_gps[:,3]
    gps_rough = cur_gps[:,4]
    # print(gps_vels.max())
    rough_score = gps_rough.copy()
    rough_score[rough_score < .2] = 0.0
    print(rough_score.mean())
    # rough_score[rough_score != 0.0] = 1.0
    gps_score = np.clip((gps_vels/5.0),0,1) - 5*rough_score
    gps_score = np.clip(gps_score,0,1)
    # print(gps_rough.mean())
    print("SCORE - ", gps_score.mean())
    # colors = CMAP(vels/7.0)
    # print(vels.min(), vels.max())
    # print(gps_vels.shape)
    # print(colors)

    ax[i].imshow(np.transpose(map,(1,2,0)))
    ax[i].scatter(fig8_c,fig8_r,s=40,c='w')
    im = ax[i].scatter(col,row,c=gps_vels,cmap='jet',vmin=0.0,vmax=6.0,s=1.6)
    # im = ax[i].scatter(col,row,c=gps_rough,cmap='plasma',vmin=0.0,vmax=1.0,s=1.6)
    # im = ax[i].scatter(col,row,c=gps_score,cmap='RdYlGn',vmin=0.0,vmax=1.0,s=1.6)


    ax[i].set_xlim([SCALE*200/.2, SCALE*250/.2])
    ax[i].set_ylim([SCALE*180/.2, SCALE*90/.2])

    ax[i].set_title("Lap " + str(i+1))
    ax[i].axis("off")


fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.show()

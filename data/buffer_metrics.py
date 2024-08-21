import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as stats

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
def mean_sum_l2_distance(data):
    """
    Calculate the sum of the L2 distances between each sample and all other samples, then take the mean.

    :param data: A numpy array of shape (n_samples, n_features)
    :return: The mean of the sum of L2 distances for each sample
    """
    # Compute the pairwise L2 distances
    # print(data[:10])
    print(data.min(), data.max())
    distances = pairwise_distances(data, metric='euclidean')
    # print(distances[:10])
    # print(distances.shape)

    # Sum the distances for each sample (excluding self-distances which are zero)
    sum_distances = np.sum(distances, axis=1)/data.shape[0]
    # print(sum_distances.shape)

    # Compute the mean of the sum of distances
    mean_distance = np.mean(sum_distances)

    return mean_distance

def pca_variance(data, variance_threshold = .9):
    """
    Perform PCA on the given data and determine the number of components
    required to explain the given percentage of variance.

    :param data: A numpy array of shape (n_samples, n_features)
    :param variance_threshold: The percentage of variance to be explained (0 < variance_threshold <= 100)
    :return: The number of components required to explain the given percentage of variance
    """

    # Standardize the data
    # scaler = StandardScaler()
    # data_normalized = scaler.fit_transform(data)

    # Fit PCA to the normalized data
    pca = PCA()
    pca.fit(data)

    # Calculate the cumulative sum of the explained variance ratio
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    print(cumulative_variance)

    # Determine the number of components required to explain the desired variance
    num_components = np.argmax(cumulative_variance >= variance_threshold) + 1

    return num_components


buffer_intelligent = torch.load(os.path.join('/home/matthew/physics_atv_ws/src/learning/context_adaptation/assets', 'gp_params', 'buffer_checkpoint_intelligent' + '.pt'))
train_in = buffer_intelligent['train_buffer'].cpu()
train_intelligent = train_in.numpy()
toi_intelligent = buffer_intelligent['toi']

buffer_FIFO = torch.load(os.path.join('/home/matthew/physics_atv_ws/src/learning/context_adaptation/assets', 'gp_params', 'buffer_checkpoint_FIFO' + '.pt'))
train_in = buffer_FIFO['train_buffer'].cpu()
train_FIFO = train_in.numpy()
toi_FIFO = buffer_FIFO['toi']

train_intelligent = train_intelligent[:,:8]
train_FIFO = train_FIFO[:,:8]

# support_data = np.load(os.path.join('/home/matthew/physics_atv_ws/src/learning/context_adaptation/assets', 'gp_params', 'support_data.npy'))
# support_data = np.load(os.path.join('/home/matthew/physics_atv_ws/src/learning/context_adaptation', 'data', 'support_data_gupta.npy'))
support_data = np.load(os.path.join('/home/matthew/physics_atv_ws/src/learning/context_adaptation', 'data', 'support_data_turnpike.npy'))

# print(train_FIFO[:10])
# print(train_intelligent[:10])
# plt.plot(toi_FIFO)
# plt.plot(toi_intelligent)
# plt.show()
# s=r

from sklearn.manifold import TSNE

all_data = support_data

print(all_data.shape)

all_tsne = TSNE(n_components=2, perplexity=50, random_state=0).fit_transform(all_data[:,:8])

plt.scatter(all_tsne[:,0], all_tsne[:,1])
plt.show()


buff_size = 50
t = buff_size
buff_intelligent = all_data[:buff_size].copy()
intelligent_tsne = all_tsne[:buff_size].copy()
buff_classes = np.argmin(buff_intelligent[:,:8],axis=1)

buff_FIFO = all_data[:buff_size].copy()
FIFO_tsne = all_tsne[:buff_size].copy()
buff_toi = np.arange(buff_FIFO.shape[0])

from tqdm import tqdm
for i in tqdm(range(buff_size,support_data.shape[0])):
    sample = support_data[i]

    #FIFO UPDATE
    idx = np.argmin(buff_toi)
    buff_FIFO[idx] = sample
    buff_toi[idx] = t
    FIFO_tsne[idx] = all_tsne[i]

    #Intelligent UPDATE
    most_class = stats.mode(buff_classes)[0]
    # print(most_class)
    vel_hist = torch.histc(torch.from_numpy(buff_intelligent)[buff_classes == most_class,-2], bins=10, min=0,max=10)
    most_vel = torch.argmax(vel_hist)
    edges = np.arange(0,11)
    sel_min = edges[most_vel]
    sel_max = edges[most_vel+1]
    train_vels = buff_intelligent[:,-2]
    insert_idx = np.random.choice(np.where((buff_classes == most_class) & (train_vels > sel_min) & (train_vels < sel_max))[0])
    buff_intelligent[insert_idx] = sample
    buff_classes[insert_idx] = np.argmin(sample[:8])
    # print(np.argmin(sample[:8]))
    # print(insert_idx)
    intelligent_tsne[insert_idx] = all_tsne[i]

    # print(idx, insert_idx)


    # if i % 50 == 0:
    #     # print(buff_FIFO.shape)
    #     # plt.scatter(all_tsne[:,0], all_tsne[:,1])
    #     # plt.scatter(intelligent_tsne[:,0],intelligent_tsne[:,1])
    #     # plt.scatter(FIFO_tsne[:,0],FIFO_tsne[:,1] + .1)
    #     #
    #     # plt.show()
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection='3d')
    #     ax.scatter(intelligent_tsne[:,0] ,intelligent_tsne[:,1] , buff_intelligent[:,-2])#, c=buff_intelligent[:,-1])
    #     ax.scatter(FIFO_tsne[:,0] + .2,FIFO_tsne[:,1] + .2, buff_FIFO[:,-2])#, c=buff_FIFO[:,-1])
    #     ax.set_xlabel('Fx')
    #     ax.set_ylabel('Fy')
    #     ax.set_zlabel('Velocity')
    #     ax.set_xlim3d(left=all_tsne[:,0].min(),right=all_tsne[:,0].max())
    #     ax.set_ylim3d(bottom=all_tsne[:,1].min(),top=all_tsne[:,1].max())
    #     ax.set_zlim3d(bottom=0,top=8)
    #     ax.xaxis.set_ticks([])
    #     ax.yaxis.set_ticks([])
    #     ax.zaxis.set_ticks([])
    #     ax.view_init(elev=15,azim=-64)
    #     plt.show()

    t += 1

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(intelligent_tsne[:,0] ,intelligent_tsne[:,1] , buff_intelligent[:,-2])#, c=buff_intelligent[:,-1])
# ax.scatter(FIFO_tsne[:,0] + .4,FIFO_tsne[:,1] + .4, buff_FIFO[:,-2])#, c=buff_FIFO[:,-1])
# ax.set_xlabel('Fx')
# ax.set_ylabel('Fy')
# ax.set_zlabel('Velocity')
# ax.set_xlim3d(left=all_tsne[:,0].min(),right=all_tsne[:,0].max())
# ax.set_ylim3d(bottom=all_tsne[:,1].min(),top=all_tsne[:,1].max())
# ax.set_zlim3d(bottom=0,top=8)
# ax.xaxis.set_ticks([])
# ax.yaxis.set_ticks([])
# ax.zaxis.set_ticks([])
# ax.view_init(elev=15,azim=-64)
# plt.show()

mean = np.mean(all_data,axis = 0).reshape(1,-1)
std = np.std(all_data,axis = 0).reshape(1,-1)
buff_intelligent = (buff_intelligent.copy() - mean)/std
buff_FIFO = (buff_FIFO.copy() - mean)/std

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(intelligent_tsne[:,0] ,intelligent_tsne[:,1] , buff_intelligent[:,-2])#, c=buff_intelligent[:,-1])
ax.scatter(FIFO_tsne[:,0] + .4,FIFO_tsne[:,1] + .4, buff_FIFO[:,-2])#, c=buff_FIFO[:,-1])
ax.set_xlabel('Fx')
ax.set_ylabel('Fy')
ax.set_zlabel('Velocity')
ax.set_xlim3d(left=all_tsne[:,0].min(),right=all_tsne[:,0].max())
ax.set_ylim3d(bottom=all_tsne[:,1].min(),top=all_tsne[:,1].max())
ax.set_zlim3d(bottom=0,top=8)
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.zaxis.set_ticks([])
ax.view_init(elev=15,azim=-64)
plt.show()

num_intelligent = pca_variance(buff_intelligent)
num_FIFO = pca_variance(buff_FIFO)

sum_dists_intelligent = mean_sum_l2_distance(buff_intelligent)
sum_dists_FIFO = mean_sum_l2_distance(buff_FIFO)

print(num_intelligent, num_FIFO)
print(sum_dists_intelligent, sum_dists_FIFO)

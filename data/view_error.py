import numpy as np
import matplotlib.pyplot as plt

data_dir = '/home/matthew/physics_atv_ws/src/learning/context_adaptation/data/'
fname = 'ERROR_LIST_turnpike.npy'
errors = np.load(data_dir + fname)
# print(errors)
# print(np.isnan(errors))
errors = errors[~np.isnan(errors)]
# errors = errors[5:]
plt.plot(errors)


fname = 'ERROR_LIST_turnpike_nn.npy'
errors = np.load(data_dir + fname)
plt.plot(errors)

plt.show()

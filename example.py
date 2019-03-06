from DL_functions import *
import numpy as np

# load data
Z = np.loadtxt("realZ_small.txt")
R1 = np.loadtxt("realstock_return.txt")
R2 = np.loadtxt("realportfolio_return.txt")
M = np.loadtxt("realMKT.txt")
T = M.shape[0] # number of periods
data_input = dict(characteristics=Z, stock_return=R1, target_return=R2, factor=M[:, 0:3])

# set parameters
training_para = dict(epoch=50, train_ratio=0.6, split="future", activation=tf.nn.tanh, train_algo=tf.train.AdamOptimizer, learning_rate=0.005, batch_size=120)

# design network layers
layer_size = [64, 32, 16, 8, 4]

# construct deep factors
f, f_oos, alpha, loss = dl_alpha(data_input, layer_size, training_para, value_index=0, ens=100)

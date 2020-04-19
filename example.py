from DL_functions import *
import numpy as np

# load data
Z = np.loadtxt("realZ_sample.txt")
R1 = np.loadtxt("realstock_return.txt")
R2 = np.loadtxt("realportfolio_return.txt")
M = np.loadtxt("realMKT.txt")
T = M.shape[0] # number of periods
data_input = dict(characteristics=Z, stock_return=R1, target_return=R2, factor=M[:, 0:3])

# set parameters
training_para = dict(epoch=50, train_ratio=1, train_algo=tf.compat.v1.train.AdamOptimizer,
                     split="future", activation=tf.nn.tanh, start=1, batch_size=120, learning_rate=0.005,
                     Lambda=1)
# design network layers
layer_size = [64, 32, 16, 8, 4]

# construct deep factors
f, f_oos, loss = dl_alpha(data_input, layer_size, training_para)

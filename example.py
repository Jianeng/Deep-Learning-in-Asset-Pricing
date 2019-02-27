from DL_functions import *
import numpy as np

# load data
Z = np.loadtxt("realZ_small.txt")
R1 = np.loadtxt("realstock_return.txt")
R2 = np.loadtxt("realportfolio_return.txt")
M = np.loadtxt("realMKT.txt")
T = M.shape[0] # number of periods

# 
training_para = dict(epoch=50, train_ratio=0.6, split="future", activation=tf.nn.tanh)

data_input = dict(characteristics=Z, stock_return=R1, target_return=R2, factor=M[:, 0:3])

layer_size_tanh = [64, 32, 16, 8, 4]

dl_alpha(data_input, layer_size_tanh, training_para, value_index=0, ens=100)

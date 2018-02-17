import numpy as np
import scipy.io
import tensorflow as tf

mat_data1 = scipy.io.loadmat('train_x.mat')
Features_colum1 = mat_data1['train_x']
train_x = np.array(Features_colum1, dtype=np.float32)

print('train_x')
print(train_x)
print('train_x_part')
print(train_x[1][:])



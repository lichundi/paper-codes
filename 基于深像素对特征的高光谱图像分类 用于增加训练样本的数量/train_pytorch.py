'''
@Author: your name
@Date: 2020-07-09 11:07:13
@LastEditTime: 2020-07-09 11:07:14
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \undefinedc:\Users\李春娣\Desktop\学校\实验室\代码学习\PPFs_keras_pytorch\PPFs\train_pytorch.py
'''
import scipy.io as sio
import 
import numpy as np
import math

fea1 = 10
fea2 = fea1
fea3 = int(2*fea2)
fea4 = fea3
fea5 = int(2*fea4)
fea6 = fea5
fea7 = int(2*fea6)
fea8 = fea7

batchsize = 256

NUM_EPOCHES = 3
NUM_EPOCHS_PER_DECAY = 300
INITIAL_LEARNING_RATE = 0.01
LEARNING_RATE_DECAY_FACTOR = 0.1
MOVING_AVERAGE_DECAY = 0.9999

train_dataf = 'traindata.mat'
train_labelf = 'trainlabel.mat'

train_data = sio.loadmat(train_dataf)['traindata']

train_labels = sio.loadmat(train_labelf)['trainlabel']


classes = np.max(train_labels)   # max() 方法返回给定参数的最大值,
deepth = train_data.shape[3] # train_data.shape[3]第三维的长度

train_images = train_data.transpose((0,2,3,1))

train_labels -= 1






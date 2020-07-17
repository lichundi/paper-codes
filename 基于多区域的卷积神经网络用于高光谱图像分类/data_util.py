#coding: utf-8

import os
import cv2
import tifffile as tiff
import numpy as np
import scipy.io as sio
import keras as K
from sklearn.preprocessing import MultiLabelBinarizer
# from construct_multi_mat import args


BATCH_SIZE = 450
TRAIN_H = 'train_data_H.mat'

# os即operating system（操作系统），Python 的 os 模块封装了常见的文件和目录操作。
# os.path模块主要用于文件的属性获取  os.path.exists()就是判断括号里的文件是否存在的意思，括号内的可以是文件路径。

weights_path = os.path.join('weights/univ/diverse/new_net/')# os.path.join()函数：连接两个或更多的路径名组件
if not os.path.exists(weights_path):
    os.mkdir(weights_path)  # os.mkdir() 方法用于以数字权限模式创建目录
PATH = '../file/univ/diverse/200/augmentation/'
TEST_PATH = '../file/univ/diverse/'
NUM_CLASS = 9
HEIGHT = 610
WIDTH = 340
NUM_CHN = 103


# weights_path=os.path.join('weights/salinas/diverse/new_net/')
# if not os.path.exists(weights_path):
#     os.mkdir(weights_path)
# PATH='../file/salinas/diverse/200/1/'
# TEST_PATH='../file/salinas/diverse/'
# NUM_CLASS=16
# HEIGHT=512
# WIDTH= 217
# NUM_CHN= 224

# weights_path=os.path.join('weights/indian/diverse/new_net/')
# if not os.path.exists(weights_path):
#     os.mkdir(weights_path)
# PATH='../file/indian/diverse/200/1/'
# TEST_PATH='../file/indian/diverse/'
# NUM_CLASS=8
# HEIGHT=145
# WIDTH= 145
# NUM_CHN= 220


class DataSet(object):
    def __init__(self, hsi, labels):
        self._hsi = hsi
        self._labels = labels
    #读取方法装饰
    @property  # Python内置的@property装饰器就是负责把一个方法变成属性调用的：
    def hsi(self):
        return self._hsi

    @property
    def labels(self):
        return self._labels


def read_data(path, filename_H, data_style, key):
    if data_style == 'train':
        # 使用 import scipy.io as sio  所有的元素都封装在ndarray 中。
        train_data = sio.loadmat(os.path.join(path, filename_H))  # 读取 mat 文件
        hsi = np.array(train_data[key])
        # hsi=sample_wise_standardization(hsi)
        train_labl = np.array(train_data['label'])
        return DataSet(hsi, train_labl)
    else:
        test_data = sio.loadmat(os.path.join(path, filename_H))
        hsi = test_data[key]
        test_labl = test_data['label']
        # np.reshape这个方法是在不改变数据内容的情况下，改变一个数组的格式，参数及返回值
        test_labl = np.reshape(test_labl.T, (test_labl.shape[1]))
        idx = test_data['idx']
        idy = test_data['idy']
        idx = np.reshape(idx.T, (idx.shape[1]))  # shape[0]和shape[1]分别代表行和列的长度
        # np.reshape不改变数据内容的情况下，改变一个数据的格式、参数及返回值
        idy = np.reshape(idy.T, (idy.shape[1]))
        # hsi=sample_wise_standardization(hsi)
        return DataSet(hsi, test_labl), idx, idy


def sample_wise_standardization(data):
    import math
    _mean = np.mean(data)
    _std = np.std(data)
    npixel = np.size(data) * 1.0         # *乘法  **乘方
    min_stddev = 1.0 / math.sqrt(npixel)
    return (data - _mean) / max(_std, min_stddev)


def eval(predication, labels):
    """
    evaluate test score
    """
    num = labels.shape[0]#num 取labels行的长度
    count = 0
    for i in xrange(num):
        if(np.argmax(predication[i]) == labels[i]):  # argmax返回的是最大数的索引
            count += 1
    return 100.0*count/num


def generate_map(predication, idx, idy):
    maps = np.zeros([HEIGHT, WIDTH])
    for i in xrange(len(idx)):
        maps[idx[i], idy[i]] = np.argmax(predication[i])+1  # =赋值 ==判断是否相等
    return maps

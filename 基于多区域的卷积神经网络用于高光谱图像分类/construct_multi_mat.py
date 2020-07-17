# coding:utf-8
import os
import argparse  # 存储参数库
import numpy as np
import os
import time
import cv2
import numpy as np
import scipy.io as sio
import keras as K
import keras.layers as L
import tensorflow as tf
import tifffile as tiff


parser = argparse.ArgumentParser()  # 创建对象
parser.add_argument('--train_label_name',  # parser.add_argument添加参数  给一个 ArgumentParser 添加程序参数信息是通过调用 add_argument() 方法完成的
                    type=str,
                    default='mask_train_200_10',
                    help='random sample sets')
parser.add_argument('--ksize',  # name or flags - 一个命名或者一个选项字符串的列表，例如 foo 或 -f, --foo。
                    type=int,  # type - 命令行参数应当被转换成的类型。
                    default=11,  # default - 当参数未在命令行中出现时使用的值。
                    help='window size')  # help - 一个此选项作用的简单描述。
args = parser.parse_args()  # 使参数创建并生效  ArgumentParser 通过 parse_args() 方法解析参数。
r = args.ksize//2  # //整除2
CON_type = 'diverse'
NUM_train = 'augmentation'


NUM_CLASS = 9
mdata_name = 'pavia_univ.mat'
data_type = 'mat'
dataset = 'univ'


# NUM_CLASS=8
# mdata_name='indian.mat'
# data_type='mat'
# dataset='indian'

# NUM_CLASS=16
# mdata_name='salinas.mat'
# data_type='mat'
# dataset='salinas'

PATH = os.path.join('../data/' + dataset+'/')  # os.path.join()函数：连接两个或更多的路径名组件
# save_path = os.path.join('../file/' + dataset+'/'+CON_type+'/'+NUM_train+'/'+args.train_label_name.split('_')[-1]+'/')
save_path = os.path.join('../file/' + dataset+'/'+CON_type+'/'+NUM_train+'/')
if os.path.exists(save_path):
    print "Saving Path is no problem."
else:
    print "Saving Path has problem, ready for constructing a new one"
    # os.makedirs(os.path.join('../file/' + dataset+'/'+CON_type+'/'+NUM_train+'/'+args.train_label_name.split('_')[-1]+'/'))
    os.makedirs(os.path.join('../file/' + dataset +
                             '/'+CON_type+'/'+NUM_train+'/'))


def image_pad(data, r):
    if len(data.shape) == 3:
        data_new = np.lib.pad(data, ((r, r), (r, r), (0, 0)), 'symmetric')#数组填充（‘symmetric’——表示对称填充）
        return data_new
    if len(data.shape) == 2:
        data_new = np.lib.pad(data, r, 'constant', constant_values=0) #常数填充模式
        return data_new


def samele_wise_normalization(data):  # 归一化
    """
    normalize each sample to 0-1
    Input:
        sample
    Output:
        Normalized sample
    y=1.0*(x-np.min(x))/(np.max(x)-np.min(x))
    """
    if np.max(data) == np.min(data):  # np.max返回给定参数的最大值
        # np.ones_like返回一个用1填充的跟输入数组形状和类型一样的数组。
        return np.ones_like(data, dtype=np.float32) * 1e-6
    else:
        return 1.0 * (data - np.min(data)) / (np.max(data) - np.min(data))


def sample_wise_standardization(data):  # 标准化
    import math  # 数学模块
    _mean = np.mean(data)
    _std = np.std(data)
    npixel = np.size(data) * 1.0  # data的元素个数
    # 函数 sqrt(npixel) 返回npixel的平方根，此函数不可直接访问，需要导入math模块，然后需要使用math静态对象调用此函数。
    min_stddev = 1.0 / math.sqrt(npixel)
    return (data - _mean) / max(_std, min_stddev)


def normalization(data, style):  # 归一化
    if style == 0:
        mi = np.min(data)
        ma = np.max(data)
        data_new = (data - mi) / (ma - mi)
    else:
        data_new = ldata/np.max(data)
    return data_new


def construct_spatial_patch(mdata, mlabel, r, patch_type):
    # 根据mlabel逐标签点构建HSI空间块(半径为r)和其标签
    # 使用该函数需要预先做好map_train,map_test,分别调用一次本函数
    patch = []
    patch_right = []
    patch_left = []
    patch_bottom = []
    patch_up = []
    patch_center = []
    patch5 = []
    label = []
    result_patchs = []
    result_labels = []
    XR = []
    XL = []
    XU = []
    XB = []
    XC = []
    if patch_type == 'train':
        num_class = np.max(mlabel)
        for c in xrange(1, num_class+1):
            idx, idy = np.where(mlabel == c)
            for i in xrange(len(idx)):
                patch.append(mdata[idx[i]-r:idx[i]+r+1, idy[i]-r:idy[i]+r+1, ...]) #.append在列表末尾添加新的对象。 
                patch_right.append( mdata[idx[i]-1:idx[i]+r+1, idy[i]-r:idy[i]+r+1, ...])
                patch_left.append( mdata[idx[i]-r:idx[i]+2, idy[i]-r:idy[i]+r+1, ...])
                patch_up.append(mdata[idx[i]-r:idx[i]+r+1, idy[i]-1:idy[i]+r+1, ...])
                patch_bottom.append(mdata[idx[i]-r:idx[i]+r+1, idy[i]-r:idy[i]+2, ...])
                patch_center.append(mdata[idx[i]-1:idx[i]+2, idy[i]-1:idy[i]+2, ...])
                label.append(mlabel[idx[i], idy[i]]-1)
        result_patchs = np.asarray(patch, dtype=np.float32)
        result_labels = np.asarray(label, dtype=np.int8)
        XR = np.asarray(patch_right, dtype=np.float32)
        XL = np.asarray(patch_left, dtype=np.float32)
        XU = np.asarray(patch_up, dtype=np.float32)
        XB = np.asarray(patch_bottom, dtype=np.float32)
        XC = np.asarray(patch_center, dtype=np.float32)
        return result_patchs, XR, XL, XU, XB, XC, result_labels
    if patch_type == 'test':
        idx, idy = np.nonzero(mlabel)  # np.nonzero 取出矩阵中的非零元素的坐标
        for i in xrange(len(idx)):
            # append() 方法用于在列表末尾添加新的对象。
            patch.append(mdata[idx[i]-r:idx[i]+r+1, idy[i]-r:idy[i]+r+1, :])
            patch_right.append(mdata[idx[i]-1:idx[i]+r+1, idy[i]-r:idy[i]+r+1, :])
            patch_left.append(mdata[idx[i]-r:idx[i]+2, idy[i]-r:idy[i]+r+1, :])
            patch_up.append(mdata[idx[i]-r:idx[i]+r+1, idy[i]-1:idy[i]+r+1, :])
            patch_bottom.append(mdata[idx[i]-r:idx[i]+r+1, idy[i]-r:idy[i]+2, :])
            patch_center.append(mdata[idx[i]-1:idx[i]+2, idy[i]-1:idy[i]+2, :])
            label.append(mlabel[idx[i], idy[i]]-1)
        # array和asarray都可将结构数据转换为ndarray类型。
        result_patchs = np.asarray(patch, dtype=np.float32)
        # 但是主要区别就是当数据源是ndarray时，
        result_labels = np.asarray(label, dtype=np.int8)
        # array仍会copy出一个副本，占用新的内存，但asarray不会。
        XR = np.asarray(patch_right, dtype=np.float32)
        XL = np.asarray(patch_left, dtype=np.float32)
        XU = np.asarray(patch_up, dtype=np.float32)
        XB = np.asarray(patch_bottom, dtype=np.float32)
        XC = np.asarray(patch_center, dtype=np.float32)
        idx = idx-2*r-1
        idy = idy-2*r-1
        return result_patchs, XR, XL, XU, XB, XC, result_labels, idx, idy


def random_flip(data, xr, xl, xu, xb, xc, label, seed=0): #数据增强
    num = data.shape[0]
    datas = []
    xrs = []
    xls = []
    xus = []
    xbs = []
    xcs = []
    labels = []
    for i in xrange(num):
        datas.append(data[i])
        xrs.append(xr[i])
        xls.append(xl[i])
        xus.append(xu[i])
        xbs.append(xb[i])
        xcs.append(xc[i])
        if len(data[i].shape) == 3:
            noise = np.random.normal(0.0, 0.05, size=(data[i].shape))
            datas.append(np.fliplr(data[i])+noise)
            noise = np.random.normal(0.0, 0.05, size=(xr[i].shape))
            xrs.append(np.fliplr(xr[i])+noise)
            noise = np.random.normal(0.0, 0.05, size=(xl[i].shape))
            xls.append(np.fliplr(xl[i])+noise)
            noise = np.random.normal(0.0, 0.05, size=(xu[i].shape))
            xus.append(np.fliplr(xu[i])+noise)
            noise = np.random.normal(0.0, 0.05, size=(xb[i].shape))
            xbs.append(np.fliplr(xb[i])+noise)
            noise = np.random.normal(0.0, 0.05, size=(xc[i].shape))
            xcs.append(np.fliplr(xc[i])+noise)
        labels.append(label[i])
        labels.append(label[i])
    datas = np.asarray(datas, dtype=np.float32)
    xrs = np.asarray(xrs, dtype=np.float32)
    xls = np.asarray(xls, dtype=np.float32)
    xus = np.asarray(xus, dtype=np.float32)
    xbs = np.asarray(xbs, dtype=np.float32)
    xcs = np.asarray(xcs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    np.random.seed(seed) #如果使用相同的seed，则每次生成的随机数都相同。
    index = np.random.permutation(datas.shape[0]) #随机排列序列，或返回随机范围
    return datas[index], xrs[index], xls[index], xus[index], xbs[index], xcs[index], labels[index]


def read_data(path, file_name, data_name, data_type):
    mdata = []
    if data_type == 'tif':
        mdata = tiff.imread(os.path.join(path, file_name))
        return mdata
    if data_type == 'mat':
        mdata = sio.loadmat(os.path.join(path, file_name))
        mdata = np.array(mdata[data_name])
        return mdata
# In[6]:


def main():
    mdata = read_data(PATH, mdata_name, 'data', data_type)
    mdata = np.asarray(mdata, dtype=np.float32)
    mdata = sample_wise_standardization(mdata)
    mdata = image_pad(mdata, r+r-1)

    mlabel_train = read_data(
        PATH, args.train_label_name, 'mask_train', data_type)
    mlabel_train = image_pad(mlabel_train, r+r-1)
    train_data_H, XR, XL, XU, XB, XC, train_label_H = construct_spatial_patch(
        mdata, mlabel_train, r, 'train')
    train_data_H, XR, XL, XU, XB, XC, train_label_H = random_flip(
        train_data_H, XR, XL, XU, XB, XC, train_label_H)

    print('train data shape:{}'.format(train_data_H.shape))
    print('right data shape:{}'.format(XR.shape))
    print('left  data shape:{}'.format(XL.shape))
    print('up    data shape:{}'.format(XU.shape))
    print('botom data shape:{}'.format(XB.shape))
    print('cnter data shape:{}'.format(XC.shape))
    # shape=n*ksize*ksize*d,n取决于mlabel_train中的非零样本的个数

    # SAVE TRAIN_DATA TO MAT FILE
    print('Saving train data...')
    data = {
        'data': train_data_H,
        'XR': XR,
        'XL': XL,
        'XU': XU,
        'XB': XB,
        'XC': XC,
        'label': train_label_H
    }
    path_train = os.path.join(save_path+'train_data_H.mat')

    sio.savemat(path_train, data)

    # SAVE TEST_DATA TO MAT FILE
    for iclass in xrange(1, NUM_CLASS+1):
        test_label_name = os.path.join('mask_test_patch'+str(iclass)+'.mat')
        mlabel_test = read_data(PATH, test_label_name, 'mask_test', data_type)
        mlabel_test = image_pad(mlabel_test, r+r-1)
        test_data_H, XR, XL, XU, XB, XC, test_label_H, idx, idy = construct_spatial_patch(
            mdata, mlabel_test, r, 'test')
        print('test data shape:{}'.format(test_data_H.shape))
        print('right data shape:{}'.format(XR.shape))
        print('left  data shape:{}'.format(XL.shape))
        print('up    data shape:{}'.format(XU.shape))
        print('botom data shape:{}'.format(XB.shape))
        print('cnter data shape:{}'.format(XC.shape))
        print('Saving test data...')
        data = {
            'hsi': test_data_H,
            'XR': XR,
            'XL': XL,
            'XU': XU,
            'XB': XB,
            'XC': XC,
            'label': test_label_H,
            'idx': idx,
            'idy': idy
        }
        path_test = os.path.join(save_path, 'test_data_H'+str(iclass)+'.mat')
        sio.savemat(path_test, data, format='5')
        test_label_name = []
        mlabel_test = []
        test_data_H = []
        test_label_H = []
        idx = []
        idy = []
        XR = []
        XL = []
        XU = []
        XB = []
        XC = []
    print('Done')


if __name__ == '__main__':
    main()

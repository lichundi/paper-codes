# -*- coding: utf-8 -*-

# In[1]:

import keras as K
import keras.layers as L
import tensorflow as tf
import scipy.io as sio
import argparse
import os
import numpy as np
import h5py
import time
import sys
from data_util import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import plot_model
# In[3]:

parser = argparse.ArgumentParser()#创建对象并添加参数
parser.add_argument('--NUM_EPOCH',
                    type=int,
                    default=500,
                    help='number of epoch')
parser.add_argument('--mode',
                    type=int,
                    default=0,
                    help='train or test mode')
parser.add_argument('--full_net_',
                    type=bool,
                    default=False,
                    help='train or not')
parser.add_argument('--right_net_',
                    type=bool,
                    default=False,
                    help='train or not')

parser.add_argument('--left_net_',
                    type=bool,
                    default=False,
                    help='train or not')
parser.add_argument('--up_net_',
                    type=bool,
                    default=False,
                    help='train or not')
parser.add_argument('--bottom_net_',
                    type=bool,
                    default=False,
                    help='train or not')
parser.add_argument('--center_net_',
                    type=bool,
                    default=False,
                    help='train or not')
args = parser.parse_args() # 使参数创建并生效


# os.path.join()函数：连接两个或更多的路径名组件
model_name_data = os.path.join(weights_path+'data.h5')
model_name_XR = os.path.join(weights_path+'XR.h5')
model_name_XL = os.path.join(weights_path+'XL.h5')
model_name_XU = os.path.join(weights_path+'XU.h5')
model_name_XB = os.path.join(weights_path+'XB.h5')
model_name_XC = os.path.join(weights_path+'XC.h5')

# os.path.join()函数：连接两个或更多的路径名组件
new_model_name = os.path.join(weights_path+'all_mul_cnn_10.h5')


if not os.path.exists('log/'):
    os.makedirs('log/')

# In[4]:


def GW_net(input_spat):
    filters = [32, 64, 128, 256]
    conv0_spat = L.Conv2D(filters[2], (3, 3), padding='valid')(input_spat)
    conv0_spat = L.BatchNormalization(axis=-1)(conv0_spat)
    conv0_spat = L.Activation('relu')(conv0_spat)
    conv1_spat = L.Conv2D(filters[2], (3, 3), padding='valid')(conv0_spat)
    conv1_spat = L.Activation('relu')(conv1_spat)
    conv3_spat = L.Conv2D(
        filters[2], (1, 1), padding='valid', activation='relu')(conv1_spat)
    # conv4_spat=L.Deconv2D(filters[1],(5,5),padding='valid',activation='relu')(conv3_spat)#!!!!!!!!!!!!!
    conv5_spat = L.Conv2D(filters[1], (5, 5), padding='valid', activation='relu')(
        input_spat)  # !!!!!!!!!!!!!
    conv6_spat = L.Conv2D(filters[1], (3, 3), padding='valid', activation='relu')(
        conv0_spat)  # !!!!!!!!!!!!!
    conv3_spat = L.concatenate(
        [conv3_spat, conv6_spat, conv5_spat], axis=-1)  # !!!!!!!!!!!!!
    conv3_spat = L.BatchNormalization(axis=-1)(conv3_spat)  # !!!!!!!!!!!!!

    conv7_spat = L.Flatten()(conv3_spat)
    logits = L.Dense(NUM_CLASS, activation='softmax')(
        conv7_spat)  # L.Dense Tensorflow中dense（全连接层 ）
    model = K.models.Model([input_spat], logits)
    opti = K.optimizers.SGD(lr=0.001, momentum=0.99, decay=1e-3)
    kwargs = K.backend.moving_averages
    model.compile(optimizer=opti, loss='categorical_crossentropy',
                  metrics=['acc'], kwargs=kwargs)
    return model


def SMALL_net(input_spat):
    filters = [32, 64, 128, 256]
    conv0_spat = L.Conv2D(filters[2], (3, 3), padding='valid')(input_spat)
    conv0_spat = L.BatchNormalization(axis=-1)(conv0_spat)
    conv0_spat = L.Activation('relu')(conv0_spat)
    conv1_spat = L.Activation('relu')(conv0_spat)
    conv3_spat = L.Conv2D(
        filters[2], (1, 1), padding='valid', activation='relu')(conv1_spat)
    conv7_spat = L.Flatten()(conv3_spat)
    logits = L.Dense(NUM_CLASS, activation='softmax')(conv7_spat)
    model = K.models.Model([input_spat], logits)
    opti = K.optimizers.SGD(lr=0.001, momentum=0.99, decay=1e-3)
    kwargs = K.backend.moving_averages
    model.compile(optimizer=opti, loss='categorical_crossentropy',
                  metrics=['acc'], kwargs=kwargs)
    return model


def UNION_net():
    input_full = L.Input((11, 11, NUM_CHN))
    full_net = GW_net(input_full)
    full_net.load_weights(model_name_data)
    full_net.layers.pop()
    full_net.trainable = args.full_net_
    full_output = full_net.layers[-1].output

    input_right = L.Input((7, 11, NUM_CHN))
    right_net = GW_net(input_right)
    right_net.load_weights(model_name_XR)
    right_net.layers.pop()
    right_net.trainable = args.right_net_
    right_output = right_net.layers[-1].output

    input_left = L.Input((7, 11, NUM_CHN))
    left_net = GW_net(input_left)
    left_net.load_weights(model_name_XL)
    left_net.layers.pop()
    left_net.trainable = args.left_net_
    left_output = left_net.layers[-1].output

    input_up = L.Input((11, 7, NUM_CHN))
    up_net = GW_net(input_up)
    up_net.load_weights(model_name_XU)
    up_net.layers.pop()
    up_net.trainable = args.up_net_
    up_output = up_net.layers[-1].output

    input_bottom = L.Input((11, 7, NUM_CHN))
    bottom_net = GW_net(input_bottom)
    bottom_net.load_weights(model_name_XB)
    bottom_net.layers.pop()
    bottom_net.trainable = args.bottom_net_
    bottom_output = bottom_net.layers[-1].output

    input_center = L.Input((3, 3, NUM_CHN))
    center_net = SMALL_net(input_center)
    center_net.load_weights(model_name_XC)
    center_net.layers.pop()
    center_net.trainable = args.center_net_
    center_output = center_net.layers[-1].output

    filters = [32, 64, 128, 256]

    # combine all patch
    merge0 = L.concatenate([full_output, right_output, left_output, up_output,
                            bottom_output, center_output], axis=-1)  # axis=-1，意思是从倒数第1个维度进行拼接
    merge1 = L.Dense(filters[2])(merge0)
    merge1 = L.BatchNormalization(axis=-1)(merge1)
    merge2 = L.Activation('relu')(merge1)

    merge3 = L.Dense(filters[3])(merge2)
    # merge4=L.BatchNormalization(axis=-1)(merge3)
    merge4 = L.Activation('relu')(merge3)
    logits = L.Dense(NUM_CLASS, activation='softmax')(merge2)
    new_model = K.models.Model(
        [input_full, input_right, input_left, input_up, input_bottom, input_center], logits)
    sgd = K.optimizers.SGD(lr=0.001, momentum=0.99, decay=1e-4)
    new_model.compile(
        optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])
    return new_model


def train(model, model_name):
    model_ckt = ModelCheckpoint(
        filepath=model_name, verbose=1, save_best_only=True) # ModelCheckpoint该回调函数将在每个epoch后保存模型到filepath
    tensorbd = TensorBoard(log_dir='./log', histogram_freq=0,
                           write_graph=True, write_images=True)
    train_data_full = read_data(PATH, TRAIN_H, 'train', 'data')
    train_data_XR = read_data(PATH, TRAIN_H, 'train', 'XR')
    train_data_XL = read_data(PATH, TRAIN_H, 'train', 'XL')
    train_data_XU = read_data(PATH, TRAIN_H, 'train', 'XU')
    train_data_XB = read_data(PATH, TRAIN_H, 'train', 'XB')
    train_data_XC = read_data(PATH, TRAIN_H, 'train', 'XC')

    train_labels = K.utils.np_utils.to_categorical(
        train_data_XC.labels, NUM_CLASS)
    print('train hsi data shape:{}'.format(train_data_full.hsi.shape))
    print('train XR data shape:{}'.format(train_data_XR.hsi.shape))
    print('train XL data shape:{}'.format(train_data_XL.hsi.shape))
    print('train XU data shape:{}'.format(train_data_XU.hsi.shape))
    print('train XB data shape:{}'.format(train_data_XB.hsi.shape))
    print('train XC data shape:{}'.format(train_data_XC.hsi.shape))
    print('{} train sample'.format(train_data_XC.hsi.shape[0]))
    class_weights = {}
    N = np.sum(train_data_XC.labels != 0)  # np.sum对数组所有元素求和
    for c in xrange(NUM_CLASS):
        n = 1.0*np.sum(train_data_XC.labels == c)
        item = {c: n}
        class_weights.update(item)
    print class_weights
    model.fit([train_data_full.hsi, train_data_XR.hsi, train_data_XL.hsi,
               train_data_XU.hsi, train_data_XB.hsi, train_data_XC.hsi], train_labels,
              batch_size=BATCH_SIZE,
              #  class_weight=class_weights,
              epochs=args.NUM_EPOCH,
              verbose=1,
              validation_split=0.1,
              shuffle=True,
              callbacks=[model_ckt, tensorbd])  # fit 以给定数量的轮次（数据集上的迭代）训练模型。
    model.save(os.path.join(model_name+'_'))


def test(model_name, hsi_data, XR_data, XL_data, XU_data, XB_data, XC_data):
    model = UNION_net()
    model.load_weights(model_name) # model.load_weights 读取权重
    pred = model.predict(
        [hsi_data, XR_data, XL_data, XU_data, XB_data, XC_data], batch_size=BATCH_SIZE) # model.predict 模型预测，输入测试数据,输出预测结果
    return pred


def main(mode=1, show=False):
    if args.mode == 0:
        start_time = time.time()
        model = UNION_net()
        plot_model(model, to_file='model.png', show_shapes=True)
        train(model, new_model_name)
        duration = time.time()-start_time
        print duration
        # train_generator(model,model_name)
    else:
        start_time = time.time()
        prediction = np.zeros(shape=(1, NUM_CLASS), dtype=np.float32)
        idxx = np.zeros(shape=(1,), dtype=np.int64)
        idyy = np.zeros(shape=(1,), dtype=np.int64)
        labels = np.zeros(shape=(1,), dtype=np.int64)
        for iclass in xrange(1, NUM_CLASS+1):
            CTEST = os.path.join('test_data_H'+str(iclass)+'.mat')
            test_data = sio.loadmat(os.path.join(TEST_PATH, CTEST))
            test_data_full = test_data['hsi']
            test_data_XR = test_data['XR']
            test_data_XL = test_data['XL']
            test_data_XU = test_data['XU']
            test_data_XB = test_data['XB']
            test_data_XC = test_data['XC']
            test_labl = test_data['label']
            test_labl = np.reshape(test_labl.T, (test_labl.shape[1]))
            idx = test_data['idx']
            idy = test_data['idy']
            idx = np.reshape(idx.T, (idx.shape[1]))
            idy = np.reshape(idy.T, (idy.shape[1]))

            tmp1 = np.array(test(new_model_name, test_data_full, test_data_XR, test_data_XL,
                                 test_data_XU, test_data_XB, test_data_XC), dtype=np.float32)
            prediction = np.concatenate((prediction, tmp1), axis=0) 
            idxx = np.concatenate((idxx, idx), axis=0)# np.concatenate数组idxx,idx拼接
            idyy = np.concatenate((idyy, idy), axis=0)
            labels = np.concatenate((labels, test_labl), axis=0)
        prediction = np.delete(prediction, 0, axis=0) # np.delete axis=0表示横轴 删除第一行的向量
        duration = time.time()-start_time
        print duration
        idxx = np.delete(idxx, 0, axis=0)
        idyy = np.delete(idyy, 0, axis=0)
        labels = np.delete(labels, 0, axis=0)

        # test_data=sio.loadmat(os.path.join(PATH,CTEST))
        # test_data_full=test_data['hsi']
        # test_data_XR=test_data['XR']
        # test_data_XL=test_data['XL']
        # test_data_XU=test_data['XU']
        # test_data_XB=test_data['XB']
        # test_data_XC=test_data['XC']
        # test_labl=test_data['label']
        # test_labl=np.reshape(test_labl.T,(test_labl.shape[1]))
        # idx=test_data['idx']
        # idy=test_data['idy']
        # idx=np.reshape(idx.T,(idx.shape[1]))
        # idy=np.reshape(idy.T,(idy.shape[1]))
        # del test_data
        # prediction=test(new_model_name,test_data_full,test_data_XR,test_data_XL,
        #             test_data_XU,test_data_XB,test_data_XC)
        # del test_data_full,test_data_XR,test_data_XL
        # del test_data_XU,test_data_XB,test_data_XC
        # f = open(os.path.join('prediction','.txt'), 'w')
        # n = prediction.shape[0]
        # for i in xrange(n):
        #     pre_label = prediction[i]
        #     f.write(str(pre_label)+'\n')

        print(prediction.shape, labels.shape)
        print('OA: {}%'.format(eval(prediction, labels)))

        # generate classification map
        pred_map = generate_map(prediction, idxx, idyy)

        # generate confusion_matrix
        prediction = np.asarray(prediction)
        pred = np.argmax(prediction, axis=1)
        pred = np.asarray(pred, dtype=np.int8)
        print confusion_matrix(labels, pred)

        # generate accuracy
        f = open(os.path.join(str(NUM_CLASS)+'prediction.txt'), 'w')
        n = prediction.shape[0]

        for i in xrange(n):
            pre_label = np.argmax(prediction[i], 0)
            f.write(str(pre_label)+'\n')
        f.close()

        print classification_report(labels, pred)


if __name__ == '__main__':
    main()

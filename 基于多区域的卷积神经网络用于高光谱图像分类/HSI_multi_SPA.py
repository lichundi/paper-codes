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

parser = argparse.ArgumentParser()  # 创建对象并添加参数
parser.add_argument('--NUM_EPOCH',
                    type=int,
                    default=500,
                    help='number of epoch')
parser.add_argument('--mode',
                    type=int,
                    default=0,
                    help='train or test mode')
parser.add_argument('--train_KEY',
                    type=str,
                    default='data',
                    help='train data')
parser.add_argument('--test_KEY',
                    type=str,
                    default='hsi',
                    help='test data')
parser.add_argument('--ksize1',
                    type=int,
                    default='11',
                    help='patch row')
parser.add_argument('--ksize2',
                    type=int,
                    default='11',
                    help='patch column')

args = parser.parse_args() # 使参数创建并生效

# NUM_CLASS=15
# HEIGHT=349
# WIDTH= 1905
# NUM_CHN= 144  #10 for PCA a 10 channels data


if not os.path.exists('log/'):  # 判断是否存在  os.path.exists()就是判断括号里的文件是否存在的意思，括号内的可以是文件路径。
    os.makedirs('log/')  # 不存在则创建

# os.path.join()函数：连接两个或更多的路径名组件
model_name = os.path.join(weights_path+args.train_KEY+'.h5')
# In[4]:


def GW_net(input_spat):  # 多尺度求和
    filters = [32, 64, 128, 256]
    input_spat = L.Input((args.ksize1, args.ksize2, NUM_CHN))
    # 9*9
    # define hsi_spatial convolution　
    conv0_spat = L.Conv2D(filters[2], (3, 3), padding='valid')(input_spat)
    conv0_spat = L.BatchNormalization(axis=-1)(conv0_spat)
    conv0_spat = L.Activation('relu')(conv0_spat)
    # 7*7
    # conv0_spat=L.Conv2D(filters[1],(1,1),padding='valid',activation='relu')(conv0_spat)
    # 7*7
    conv1_spat = L.Conv2D(filters[2], (3, 3), padding='valid')(conv0_spat)
    # 5*5
    # conv1_spat = L.BatchNormalization(
    #     axis=-1, momentum=0.98, gamma_regularizer=K.regularizers.L1L2(l2=1e-3))(conv1_spat)
    conv1_spat = L.Activation('relu')(conv1_spat)
    # conv2_spat=L.Conv2D(filters[3],(3,3),padding='valid',activation='relu')(conv1_spat)
    # 3*3
    conv3_spat = L.Conv2D(
        filters[2], (1, 1), padding='valid', activation='relu')(conv1_spat)

    # conv4_spat=L.Deconv2D(filters[1],(5,5),padding='valid',activation='relu')(conv3_spat)
    conv5_spat = L.Conv2D(filters[1], (5, 5), padding='valid', activation='relu')(
        input_spat)  
    conv6_spat = L.Conv2D(filters[1], (3, 3), padding='valid', activation='relu')(
        conv0_spat)  
    conv3_spat = L.concatenate(
        [conv3_spat, conv6_spat, conv5_spat], axis=-1)  
    conv3_spat = L.BatchNormalization(axis=-1)(conv3_spat)  
    conv7_spat = L.Flatten()(conv3_spat) # Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。
    logits = L.Dense(NUM_CLASS, activation='softmax')(conv7_spat) # Dense全连接层
    model = K.models.Model([input_spat], logits) # 创建模型
    opti = K.optimizers.SGD(lr=0.001, momentum=0.99, decay=1e-3)
    kwargs = K.backend.moving_averages
    model.compile(optimizer=opti, loss='categorical_crossentropy',
                  metrics=['acc'], kwargs=kwargs)  # model.compile对网络的学习过程进行配置
    return model


def SMALL_net(input_spat):  # 提取中心区域
    filters = [32, 64, 128, 256]
    conv0_spat = L.Conv2D(filters[2], (3, 3), padding='valid')(input_spat)
    conv0_spat = L.BatchNormalization(axis=-1)(conv0_spat)
    conv1_spat = L.Activation('relu')(conv0_spat)
    conv2_spat = L.Conv2D(
        filters[2], (1, 1), padding='valid', activation='relu')(conv1_spat)
    conv3_spat = L.Activation('relu')(conv2_spat)
    # Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。
    conv7_spat = L.Flatten()(conv3_spat)
    logits = L.Dense(NUM_CLASS, activation='softmax')(conv7_spat)  # Dense全连接层
    model = K.models.Model([input_spat], logits)  # 创建模型
    opti = K.optimizers.SGD(lr=0.001, momentum=0.99, decay=1e-3)
    kwargs = K.backend.moving_averages
    model.compile(optimizer=opti, loss='categorical_crossentropy', metrics=[
                  'acc'], kwargs=kwargs)  # model.compile对网络的学习过程进行配置
    return model


def SPA_net():
    input_spat = L.Input((args.ksize1, args.ksize2, NUM_CHN))
    if args.ksize1 == 3:
        net = SMALL_net(input_spat)
    else:
        net = GW_net(input_spat)
    return net


def train(model, model_name):
    # ModelCheckpoint该回调函数将在每个epoch后保存模型到filepath
    model_ckt = ModelCheckpoint(
        filepath=model_name, verbose=1, save_best_only=True)
    # TensorBoard展示数据 需要在执行Tensorflow就算图的过程中，将各种类型的数据汇总并记录到日志文件中。然后使用TensorBoard读取这些日志文件，解析数据并生产数据可视化的Web页面，让我们可以在浏览器中观察各种汇总数据。
    tensorbd = TensorBoard(log_dir='./log', histogram_freq=0,
                           write_graph=True, write_images=True)
    train_data = read_data(PATH, TRAIN_H, 'train', args.train_KEY)
    # train_labels=K.utils.np_utils.to_categorical(train_data.labels,NUM_CLASS)
    train_labels = K.utils.np_utils.to_categorical(train_data.labels) #用于将标签转化二值序列
    print('train hsi data shape:{}'.format(train_data.hsi.shape))
    print('{} train sample'.format(train_data.hsi.shape[0]))
    class_weights = {}
    N = np.sum(train_data.labels != 0) # np.sum对数组所有元素求和
    for c in xrange(NUM_CLASS):
        n = 1.0*np.sum(train_data.labels == c)
        item = {c: n}
        class_weights.update(item)
    print class_weights
    model.fit([train_data.hsi], train_labels,  # fit 以给定数量的轮次（数据集上的迭代）训练模型。
              batch_size=BATCH_SIZE,
              #  class_weight=class_weights,
              epochs=args.NUM_EPOCH,
              verbose=1,
              validation_split=0.1,
              shuffle=True,
              callbacks=[model_ckt, tensorbd])
    model.save(os.path.join(model_name+'_'))


def test(model_name, hsi_data):
    model = SPA_net()
    model.load_weights(model_name)  # model.load_weights 读取权重
    # model.predict 模型预测，输入测试数据,输出预测结果
    pred = model.predict([hsi_data], batch_size=BATCH_SIZE)
    return pred


def main(mode=0, show=False):
    if args.mode == 0:
        start_time = time.time()
        model = SPA_net()
        plot_model(model, to_file='model.png',
                   show_shapes=True)  # plot_model 绘制模型图
        train(model, model_name)
        duration = time.time()-start_time
        print duration
        # train_generator(model,model_name)
    else:
        # test_data,idx,idy=read_data(PATH,CTEST,'test',args.test_KEY)
        # prediction=test(model_name,test_data.hsi,test_data.labels)
        start_time = time.time()
        prediction = np.zeros(shape=(1, NUM_CLASS), dtype=np.float32)
        idxx = np.zeros(shape=(1,), dtype=np.int64)  # shape=(1,) 一行
        idyy = np.zeros(shape=(1,), dtype=np.int64)
        labels = np.zeros(shape=(1,), dtype=np.int64)
        for iclass in xrange(1, NUM_CLASS+1):
            CTEST = os.path.join('test_data_H'+str(iclass)+'.mat')
            test_data, idx, idy = read_data(
                TEST_PATH, CTEST, 'test', args.test_KEY)
            tmp1 = np.array(test(model_name, test_data.hsi), dtype=np.float32)
            # print(np.argmax(tmp1,1))
            prediction = np.concatenate((prediction, tmp1), axis=0)
            # np.concatenate数组idxx,idx拼接
            idxx = np.concatenate((idxx, idx), axis=0)
            idyy = np.concatenate((idyy, idy), axis=0)
            tmp_label = test_data.labels
            labels = np.concatenate((labels, tmp_label), axis=0)
        prediction = np.delete(prediction, 0, axis=0)
        duration = time.time()-start_time
        print duration
        idxx = np.delete(idxx, 0, axis=0)  # np.delete axis=0表示横轴 删除第一行的向量
        idyy = np.delete(idyy, 0, axis=0)
        labels = np.delete(labels, 0, axis=0)
        f = open(os.path.join('prediction_'+args.train_KEY+'.txt'), 'w')
        n = prediction.shape[0]  # prediction.shape[0]读取矩阵prediction第一维的长度

        for i in xrange(n):
            pre_label = np.argmax(prediction[i], 0)  # np.argmax返回的是最大数的索引
            f.write(str(pre_label)+'\n')
        f.close()

        print(prediction.shape, labels.shape)
        print('OA: {}%'.format(eval(prediction, labels)))

        # generate classification map
        pred_map = generate_map(prediction, idxx, idyy)

        # generate confusion_matrix
        prediction = np.asarray(prediction)
        pred = np.argmax(prediction, axis=1)  # np.argmax返回的是最大数的索引
        pred = np.asarray(pred, dtype=np.int8)  # np.asarray将结构数据转换为ndarray类型
        print confusion_matrix(labels, pred)

        # generate accuracy
        print classification_report(labels, pred)


if __name__ == '__main__':
    main()

#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
1. train the model
使用 17flowers 作为训练集

'''

from __future__ import division, print_function, absolute_import
import pickle
import numpy as np 
from PIL import Image
import os.path

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


SOURCE='/Users/leidelong/data/'

# 预处理图片函数：
# 首先，读取图片，形成一个Image文件
def load_image(img_path):
    img = Image.open(img_path)
    return img

# 将Image文件给修改成224 * 224的图片大小（当然，RGB三个频道我们保持不变）
def resize_image(in_image, new_width, new_height, out_image=None,
                 resize_mode=Image.ANTIALIAS):
    img = in_image.resize((new_width, new_height), resize_mode)
    if out_image:
        img.save(out_image)
    return img

# 将Image加载后转换成float32格式的tensor
def pil_to_nparray(pil_image):
    pil_image.load()
    return np.asarray(pil_image, dtype="float32")

#加载数据，特征和标签分别以数组形式输出
def load_data(datafile, num_clss, save=False, save_path=SOURCE+'model/rcnn/'+'dataset.pkl'):
    train_list = open(datafile,'r')
    labels = []
    images = []
    #将训练集转为images和labels两个数组（labels为二位数组）
    for line in train_list:
        tmp = line.strip().split(' ')
        fpath = tmp[0]
        print(fpath)

        #图片转数组
        img = load_image(fpath)
        #img输出成 224*224 矩阵
        img = resize_image(img,224,224)
        #矩阵转数组
        np_img = pil_to_nparray(img)
        #单样本加入数组集合
        images.append(np_img)

        # 生成一个num_clss维数的数组，17维
        index = int(tmp[1])
        label = np.zeros(num_clss)
        label[index] = 1
        labels.append(label)
    #是否保存训练数据到 save_path
    if save:
        pickle.dump((images, labels), open(save_path, 'wb'))
    return images, labels


def load_from_pkl(dataset_file):
    X, Y = pickle.load(open(dataset_file, 'rb'))
    return X,Y


# 构建网络，网络结构为 AlexNet
def create_alexnet(num_classes):

    network = input_data(shape=[None, 224, 224, 3])
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, num_classes, activation='softmax')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return network

#根据网络结构、样本集训练
def train(network, X, Y):
    # Training
    model = tflearn.DNN(network, checkpoint_path=SOURCE+'model/rcnn/'+'model_alexnet',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='output')
    if os.path.isfile(SOURCE+'model/rcnn/'+'model_save.model'):
    	model.load(SOURCE+'model/rcnn/'+'model_save.model')
    model.fit(X, Y, n_epoch=10, validation_set=0.1, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='alexnet_oxflowers17') # epoch = 1000
    # Save the model
    model.save(SOURCE+'model/rcnn/'+'model_save.model')

# 我们就是用这个函数来推断输入图片的类别的
def predict(network, modelfile,images):
    model = tflearn.DNN(network)
    model.load(modelfile)
    return model.predict(images)

#只训练模型，不预测
if __name__ == '__main__':
    X, Y = load_data('train_list.txt', 17)
    #X, Y = load_from_pkl('dataset.pkl')
    net = create_alexnet(17)
    train(net,X,Y)


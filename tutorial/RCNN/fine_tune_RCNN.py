#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
2. fine tune the model
使用 2flowers 作为优化数据集

'''


# from __future__ import division, print_function, absolute_import
import pickle
import numpy as np 
import selectivesearch
from PIL import Image
import os.path
import skimage
import preprocessing_RCNN as prep

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# Use a already trained alexnet with the last layer redesigned
# 这里定义了我们的Alexnet的fine tune框架。按照原文，我们需要丢弃alexnet的最后一层，即softmax
# 然后换上一层新的softmax专门针对新的预测的class数+1(因为多出了个背景class)。具体方法为设
# restore为False，这样在最后一层softmax处，我不restore任何数值。
def create_alexnet(num_classes, restore=False):
    # Building 'AlexNet'
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
    network = fully_connected(network, num_classes, activation='softmax', restore=restore)
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return network

# 这里，我们的训练从已经训练好的alexnet开始，即model_save.model开始读取。在训练后，我们
# 将训练资料收录到fine_tune_model_save.model里
def fine_tune_Alexnet(network, X, Y):
    # Training
    model = tflearn.DNN(network, checkpoint_path='rcnn_model_alexnet',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='output_RCNN')
    if os.path.isfile('fine_tune_model_save.model'):
	print("Loading the fine tuned model")
    	model.load('fine_tune_model_save.model')
    elif os.path.isfile('model_save.model'):
	print("Loading the alexnet")
	model.load('model_save.model')
    else:
	print("No file to load, error")
        return False
    model.fit(X, Y, n_epoch=10, validation_set=0.1, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='alexnet_rcnnflowers2') # epoch = 1000
    # Save the model
    model.save('fine_tune_model_save.model')

if __name__ == '__main__':
    if os.path.isfile('dataset.pkl'):
	print("Loading Data")
	X, Y = prep.load_from_pkl('dataset.pkl')
    else:
	print("Reading Data")
    	X, Y = prep.load_train_proposals('refine_list.txt', 2, save=True)
    print("DONE")
    restore = False
    if os.path.isfile('fine_tune_model_save.model'):
	restore = True
        print("Continue training")
    net = create_alexnet(3, restore)
    fine_tune_Alexnet(net,X,Y)



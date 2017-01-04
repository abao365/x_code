#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: leidelong
@license: Apache Licence
@contact: leidl8907@gmail.com
@site: https://github.com/JoneNash
@software: PyCharm Community Edition
@time: 2017/1/3 下午7:12
@source:http://stackoverflow.com/questions/38399609/tensorflow-deep-neural-network-for-regression-always-predict-same-results-in-one

使用深度学习解决回归问题
两个问题：
1）如何做优化，降低cost
2）如何解决报错问题

探索DNN在回归问题中的应用
以boston房屋价格数据为例

"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import learn
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn import datasets, linear_model
from sklearn import cross_validation
import numpy as np

# 处理数据集，分割为训练集和测试集
boston = learn.datasets.load_dataset('boston')
x_all, y_all = boston.data, boston.target
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(
x_all, y_all, test_size=0.2, random_state=42)

total_len = X_train.shape[0]

# 全局参数
learning_rate = 0.001
training_epochs = 500
batch_size = 100
display_step = 1
dropout_rate = 0.9
# 网络参数
n_hidden_1 = 32 # 第一层特征数
n_hidden_2 = 200
n_hidden_3 = 200
n_hidden_4 = 256
n_input = X_train.shape[1]
n_classes = 1

# 定义TF网络的占位符，和数据集结构对应
x = tf.placeholder("float", [None, 13])
y = tf.placeholder("float", [None])


# 定义模型
def multilayer_perceptron(x, weights, biases):
    # 定义4个隐藏层，统一使用relu激励函数
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    # 输出层使用线性回归，前四层用于构建特征
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']

    return out_layer

#初始化权重和偏置，使用random_normal初始化（按照正态分布初始化权重，自定义的两个参数分别是：正态分布的平均值和正态分布的标准差）
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], 0, 0.1)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], 0, 0.1)),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], 0, 0.1)),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], 0, 0.1)),
    'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes], 0, 0.1))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], 0, 0.1)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2], 0, 0.1)),
    'b3': tf.Variable(tf.random_normal([n_hidden_3], 0, 0.1)),
    'b4': tf.Variable(tf.random_normal([n_hidden_4], 0, 0.1)),
    'out': tf.Variable(tf.random_normal([n_classes], 0, 0.1))
}

# 创建模型
pred = multilayer_perceptron(x, weights, biases)
tf.transpose(pred)

# 定义损失函数及优化
cost = tf.reduce_mean(tf.square(pred-y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 模型快跑...
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(total_len/batch_size)
        # Loop over all batches
        for i in range(total_batch-1):
            batch_x = X_train[i*batch_size:(i+1)*batch_size]
            batch_y = Y_train[i*batch_size:(i+1)*batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, p = sess.run([optimizer, cost, pred], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch

        # sample prediction
        label_value = batch_y
        estimate = p
        err = label_value-estimate
        print ("num batch:", total_batch)

        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            print ("[*]----------------------------")
            for i in xrange(3):
                print ("label value:", label_value[i], "estimated value:", estimate[i])
            print ("[*]============================")

    print ("Optimization Finished!")



    # 测试集测试
    biases_value = pred - y
    l1_value =tf.abs(pred - y)
    biases_value_avg = tf.reduce_mean(tf.cast(biases_value, "float"))
    l1_value_avg = tf.reduce_mean(tf.cast(l1_value, "float"))
    print("biases_value_avg:", biases_value_avg.eval({x: X_test, y: Y_test}))
    print("l1_value_avg:", l1_value_avg.eval({x: X_test, y: Y_test}))

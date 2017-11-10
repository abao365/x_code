#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: leidelong
@license: Apache Licence 
@contact: leidl8907@gmail.com
@site: https://github.com/JoneNash
@software: PyCharm Community Edition
@time: 2017/1/1 上午10:48
"""



"""
CNN做数字识别
准确率：99.6%

"""

import tensorflow  as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

#N*784
x = tf.placeholder(tf.float32, [None, 784])
#N*10
y_ = tf.placeholder(tf.float32, [None,10])

#初始化时加入少量的噪声来打破对称性以及避免0梯度
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
#初始化截距0.1
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#卷积核；输入矩阵 W×W；filter矩阵 F×F，卷积核；stride值 S，步长
#valid：new_height = new_width = (W – F + 1) / S （结果向上取整）
#same： new_height = new_width = W / S （结果向上取整），在高度上需要pad的像素数为 pad_needed_height = (new_height – 1)  × S + F - W
#有关padding，参考：http://www.jianshu.com/p/05c4f1621c7e
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 2*2范围内的max pooling
# strides:步长；ksize：池化时各个维度上的大小
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

##第一层卷积
#在5*5的patch中算出32个特征
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

#将数组形式转为多维矩阵形式
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 我们把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数
#新生成的数据为28*28*32（可以理解为一张图片变为32张图片）
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# max pooling
h_pool1 = max_pool_2x2(h_conv1)

##第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

##密集连接层

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

##输出层

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.initialize_all_variables()
sess.run(init)

for i in range(20000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    print "batch_ys:",batch_ys
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

    #评测代码
    if(i%100 == 0):
        print "第", i, "次训练后，在测试集上预测：", sess.run(accuracy,
                                                feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1})
        print ""


#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: leidelong
@license: Apache Licence 
@contact: leidl8907@gmail.com
@site: https://github.com/JoneNash
@software: PyCharm Community Edition
@time: 2017/1/14 下午4:56

An example showing how to save/restore models and retrieve weights.

例子：存储模型，加载模型

 """

from __future__ import absolute_import, division, print_function

import tflearn

import tflearn.datasets.mnist as mnist

SOURCE='/Users/leidelong/data/'


#重构数据加载方法
def load_data(one_hot=False):
    mnist_data = mnist.read_data_sets(SOURCE+'mnist/data/', one_hot=one_hot)
    return mnist_data.train.images, mnist_data.train.labels, mnist_data.test.images, mnist_data.test.labels

# MNIST Data
# X, Y, testX, testY = mnist.load_data(one_hot=True)
X, Y, testX, testY = load_data(one_hot=True)

# Model
input_layer = tflearn.input_data(shape=[None, 784], name='input')
dense1 = tflearn.fully_connected(input_layer, 128, name='dense1')
dense2 = tflearn.fully_connected(dense1, 256, name='dense2')
softmax = tflearn.fully_connected(dense2, 10, activation='softmax')
regression = tflearn.regression(softmax, optimizer='adam',
                                learning_rate=0.001,
                                loss='categorical_crossentropy')

# Define classifier, with model checkpoint (autosave)
model = tflearn.DNN(regression, checkpoint_path='model.tfl.ckpt')

# Train model, with model checkpoint every epoch and every 200 training steps.
model.fit(X, Y, n_epoch=1,
          validation_set=(testX, testY),
          show_metric=True,
          snapshot_epoch=True, # Snapshot (save & evaluate) model every epoch.
          snapshot_step=500, # Snapshot (save & evalaute) model every 500 steps.
          run_id='model_and_weights')


# ---------------------
# Save and load a model
# ---------------------

# Manually save model
model.save(SOURCE+"model/weights_persistence/"+"model.tfl")

# Load a model
model.load(SOURCE+"model/weights_persistence/"+"model.tfl")

# Or Load a model from auto-generated checkpoint
# >> model.load("model.tfl.ckpt-500")

# Resume training
model.fit(X, Y, n_epoch=1,
          validation_set=(testX, testY),
          show_metric=True,
          snapshot_epoch=True,
          run_id='model_and_weights')


# ------------------
# Retrieving weights
# ------------------

# Retrieve a layer weights, by layer name:
dense1_vars = tflearn.variables.get_layer_variables_by_name('dense1')
# Get a variable's value, using model `get_weights` method:
print("Dense1 layer weights:")
print(model.get_weights(dense1_vars[0]))
# Or using generic tflearn function:
print("Dense1 layer biases:")
with model.session.as_default():
    print(tflearn.variables.get_value(dense1_vars[1]))

# It is also possible to retrieve a layer weights through its attributes `W`
# and `b` (if available).
# Get variable's value, using model `get_weights` method:
print("Dense2 layer weights:")
print(model.get_weights(dense2.W))
# Or using generic tflearn function:
print("Dense2 layer biases:")
with model.session.as_default():
    print(tflearn.variables.get_value(dense2.b))

if __name__ == '__main__':
    pass
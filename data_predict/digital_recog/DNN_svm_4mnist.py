#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: leidelong
@license: Apache Licence 
@contact: leidl8907@gmail.com
@site: https://github.com/JoneNash
@software: PyCharm Community Edition
@time: 2016/12/30 下午6:59
"""
"""
未使用TensorFlow
自己写代码实现CNN及应用
"""
import numpy as np
import random
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import gzip

####使用神经网络训练

class NeuralNet(object):
    # 初始化神经网络，sizes是神经网络的层数和每层神经元个数
    def __init__(self, sizes):
        self.sizes_ = sizes
        self.num_layers_ = len(sizes)  # 层数
        self.w_ = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]  # w_、b_初始化为正态分布随机数
        self.b_ = [np.random.randn(y, 1) for y in sizes[1:]]

    # Sigmoid函数，S型曲线
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    # Sigmoid函数的导函数
    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    # 前馈，计算下一个节点的值
    def feedforward(self, x):
        for b, w in zip(self.b_, self.w_):
            x = self.sigmoid(np.dot(w, x) + b)
        return x

    #反向传播
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.b_]
        nabla_w = [np.zeros(w.shape) for w in self.w_]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.b_, self.w_):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * \
                self.sigmoid_prime(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers_):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.w_[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.b_]
        nabla_w = [np.zeros(w.shape) for w in self.w_]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.w_ = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.w_, nabla_w)]
        self.b_ = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.b_, nabla_b)]

    # training_data是训练数据(x, y);epochs是训练次数;mini_batch_size是每次训练样本数;eta是learning rate
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):

        #是否设定test数据
        if test_data:
            n_test = len(test_data)

        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            #设定test数据后，使用当前模型对test数据做评估，输出：训练样本的正确预测数据量，测试数据正确率
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            #未设定test数据，输出：训练样本的正确预测数据量
            else:
                print("Epoch {0} complete".format(j))

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

    # 预测
    def predict(self, data):
        value = self.feedforward(data)
        return value.tolist().index(max(value))

    # 保存训练模型
    def save(self):
        pass  # 把_w和_b保存到文件(pickle)

    def load(self):
        pass


def load_mnist(dataset="training_data", digits=np.arange(10), path="/Users/leidelong/data/mnist"):
    if dataset == "training_data":
        fname_image = os.path.join(path, 'data/train-images-idx3-ubyte.gz')
        fname_label = os.path.join(path, 'data/train-labels-idx1-ubyte.gz')
    elif dataset == "testing_data":
        fname_image = os.path.join(path, 'data/t10k-images-idx3-ubyte.gz')
        fname_label = os.path.join(path, 'data/t10k-labels-idx1-ubyte.gz')
    else:
        raise ValueError("dataset must be 'training_data' or 'testing_data'")

###以下解压mnist数据集
    #flbl = open(fname_label, 'rb')
    #解压label，参数由数据集本身决定
    #'b' 表示 signed char
    flbl= gzip.GzipFile(fname_label,"r")
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    #fimg = open(fname_image, 'rb')
    #解压img，参数由数据集本身决定
    # 'B' 表示 unsigned char
    fimg = gzip.GzipFile(fname_image, 'r')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()



    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels


def load_samples(dataset="training_data"):
    image, label = load_mnist(dataset)

    X = [np.reshape(x, (28 * 28, 1)) for x in image]
    X = [x / 255.0 for x in X]  # 灰度值范围(0-255)，转换为(0-1)

    # 5 -> [0,0,0,0,0,1.0,0,0,0];  1 -> [0,1.0,0,0,0,0,0,0,0]
    def vectorized_Y(y):
        e = np.zeros((10, 1))
        e[y] = 1.0
        return e

    if dataset == "training_data":
        Y = [vectorized_Y(y) for y in label]
        pair = list(zip(X, Y))
        return pair
    elif dataset == 'testing_data':
        pair = list(zip(X, label))
        return pair
    else:
        print('Something wrong')


if __name__ == '__main__':
    INPUT = 28 * 28
    OUTPUT = 10
    net = NeuralNet([INPUT, 40, OUTPUT])

    train_set = load_samples(dataset='training_data')
    test_set = load_samples(dataset='testing_data')

    net.SGD(train_set, 13, 100, 3.0, test_data=test_set)

    # 准确率
    correct = 0.0 ;
    for test_feature in test_set:
        if net.predict(test_feature[0]) == test_feature[1][0]:
            correct += 1.0
    print("准确率: ", correct/len(test_set))



#####使用SVM训练


from sklearn import svm
import numpy as np
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import pickle


def load_samples(dataset="training_data"):
    image, label = load_mnist(dataset)

    # 把28*28二维数据转为一维数据
    X = [np.reshape(x, (28 * 28)) for x in image]
    X = [x / 255.0 for x in X]  # 灰度值范围(0-255)，转换为(0-1)
    # print(X.shape)

    pair = list(zip(X, label))
    return pair


if __name__ == '__main__':

    train_set = load_samples(dataset='training_data')
    test_set = load_samples(dataset='testing_data')

    train_X = []
    train_Y = []
    for feature in train_set:
        train_X.append(feature[0])
        train_Y.append(feature[1][0])

    clf = svm.SVR()
    clf.fit(train_X, train_Y)  # 在这个场景里面，SVM比DNN耗时

    # with open('minst.module', 'wb') as f:
    # pickle.dump(clf, f)

    # with open('minst.module', 'rb') as f:
    #	clf = pickle.load(f)

    test_X = []
    test_Y = []
    for feature in test_set:
        test_X.append(feature[0])
        test_Y.append(feature[1][0])

    # 准确率
    correct = 0.0
    i = 0
    for feature in test_X:
        predict = clf.predict(np.array(feature).reshape(1, -1))
        if round(float(predict)) == test_Y[i]:
            correct += 1
        i = i + 1
    print("准确率: ", correct , len(test_X))



#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: leidelong
@license: Apache Licence 
@contact: leidl8907@gmail.com
@site: https://github.com/JoneNash
@software: PyCharm Community Edition
@time: 2017/2/10 下午2:21
"""
import xgboost as xgb
import numpy as np
import scipy

dtrain = xgb.DMatrix('train.svm.txt')
dtest = xgb.DMatrix('test.svm.buffer')

data = np.random.rand(5,10) # 5 entities, each contains 10 features
label = np.random.randint(2, size=5) # binary target
dtrain = xgb.DMatrix( data, label=label)

csr = scipy.sparse.csr_matrix((data, (row, col)))
dtrain = xgb.DMatrix(csr)



def func():
    pass


class Main():
    def __init__(self):
        pass


if __name__ == '__main__':
    pass
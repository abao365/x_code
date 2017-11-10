#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: leidelong
@license: Apache Licence 
@contact: leidl8907@gmail.com
@site: https://github.com/JoneNash
@software: PyCharm Community Edition
@time: 2017/10/6 下午4:46
"""

import numpy as np
from  sklearn import  datasets
from pandas import read_csv

import  pandas

#读取数据
# train_url="/Users/leidelong/competition/Porto_Seguro_Safe_Driver_Prediction/train.csv"
train_url="sample_submission.csv"
train_data=pandas.read_csv(train_url)



import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

train = pd.read_csv('/Users/leidelong/competition/Porto_Seguro_Safe_Driver_Prediction/train.csv')
test = pd.read_csv('/Users/leidelong/competition/Porto_Seguro_Safe_Driver_Prediction/test.csv')


print train.shape, test.shape

target='target'
IDcol = 'id'

print train['target'].value_counts()
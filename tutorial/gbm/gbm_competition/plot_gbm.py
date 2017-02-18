#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: leidelong
@license: Apache Licence 
@contact: leidl8907@gmail.com
@site: https://github.com/JoneNash
@software: PyCharm Community Edition
@time: 2017/2/16 下午5:52
"""

import time
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import ShuffleSplit
from sklearn.utils import shuffle

import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error,mean_absolute_error


#读取源
original_data = pd.read_csv('/Users/leidelong/data/deliverytime_predict/time_predict_training_data',sep ='\t')
#混洗
original_data=shuffle(original_data, random_state=13)

#修改字段名称
original_data.columns =[ x[31:] for x in original_data.columns ]
target='finished_duration'
IDcol='bm_waybill_id'
#选取所有可用特征
predictors = [x for x in original_data.columns if x not in [target, IDcol]]
#分割训练集和测试集
train_set,test_set=train_test_split(original_data, test_size=0.2, random_state=42)
print train_set.shape,test_set.shape

train_features,train_label=train_set[predictors], train_set['finished_duration']
test_features,test_label=test_set[predictors], test_set['finished_duration']


#构建模型
#固定参数
params = {'n_estimators': 20, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(train_features, train_label)

#速度评估:平均耗时
start_time=time.time()
x=0
while x < test_features.shape[0]:
    clf.predict(test_features[x:x+1])
    x=x+1
end_time=time.time()
avg_time=(end_time - start_time) / test_features.shape[0]
print("avg_time: %.10f" % avg_time)

#训练集
mse = mean_squared_error(train_label, clf.predict(train_features))
print("TRAIN MSE: %.4f" % mse)
mae = mean_absolute_error(train_label, clf.predict(train_features))
print("TRAIN MAE: %.4f" % mae)

#测试集
mse = mean_squared_error(test_label, clf.predict(test_features))
print("TEST MSE: %.4f" % mse)
mae = mean_absolute_error(test_label, clf.predict(test_features))
print("TEST MAE: %.4f" % mae)

#Plot training deviance
# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, test_pred in enumerate(clf.staged_predict(test_features)):
    test_score[i] = clf.loss_(test_label, test_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
plt.show()


#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: leidelong
@license: Apache Licence 
@contact: leidl8907@gmail.com
@site: https://github.com/JoneNash
@software: PyCharm Community Edition
@time: 2017/2/9 下午7:09
"""

from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier

X,y=make_hastie_10_2(random_state=0)
X_train,X_test=X[:2000],X[2000:]
y_train,y_test=y[:2000],y[2000:]
clf=GradientBoostingClassifier(n_estimators=100,learning_rate=1.0,
                               max_depth=1,random_state=0,max_features=10).fit(X_train,y_train)
print clf.score(X_test, y_test)


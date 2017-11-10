#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: leidelong
@license: Apache Licence 
@contact: leidl8907@gmail.com
@site: https://github.com/JoneNash
@software: PyCharm Community Edition
@time: 2017/9/29 下午4:34
"""


import pandas

iris_df = pandas.read_csv("data/iris.csv")

from sklearn2pmml import PMMLPipeline
from sklearn.tree import DecisionTreeClassifier

iris_pipeline = PMMLPipeline([
	("classifier", DecisionTreeClassifier())
])
iris_pipeline.fit(iris_df[iris_df.columns.difference(["species"])], iris_df["species"])

from sklearn2pmml import sklearn2pmml

sklearn2pmml(iris_pipeline, "model/DecisionTreeIris.pmml", with_repr = True)
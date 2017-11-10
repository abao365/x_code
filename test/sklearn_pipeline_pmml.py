#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: leidelong
@license: Apache Licence 
@contact: leidl8907@gmail.com
@site: https://github.com/JoneNash
@software: PyCharm Community Edition
@time: 2017/9/29 下午3:03
"""

import pandas

iris_df = pandas.read_csv("data/iris.csv")

from sklearn2pmml import PMMLPipeline
from sklearn2pmml.decoration import ContinuousDomain
from sklearn_pandas import DataFrameMapper
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression

iris_pipeline = PMMLPipeline([
	("mapper", DataFrameMapper([
		(["sepal_length", "sepal_width", "petal_length", "petal_width"], [ContinuousDomain(), Imputer()])
	])),
	("pca", PCA(n_components = 3)),
	("selector", SelectKBest(k = 2)),
	("classifier", LogisticRegression())
])
iris_pipeline.fit(iris_df, iris_df["species"])

from sklearn2pmml import sklearn2pmml

sklearn2pmml(iris_pipeline, "model/LogisticRegressionIris.pmml", with_repr = True)
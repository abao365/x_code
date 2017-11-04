#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: leidelong
@license: Apache Licence 
@contact: leidl8907@gmail.com
@site: https://github.com/JoneNash
@software: PyCharm Community Edition
@time: 2017/11/1 下午8:07
"""

import pandas as pd
import csv

my_list=list()
my_list.append({'key':"s","value":1})
my_list.append({'key':"w","value":2})
df=pd.DataFrame(my_list)
df.to_csv('./test.txt',index=False,quoting=csv.QUOTE_NONE)


#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: leidelong
@license: Apache Licence 
@contact: leidl8907@gmail.com
@site: https://github.com/JoneNash
@software: PyCharm Community Edition
@time: 2016/12/27 下午4:56
"""

import os
import numpy as np

with open('neg.txt', 'w') as f:
    for img in os.listdir('neg'):
        line = 'neg/' + img + '\n'
        f.write(line)





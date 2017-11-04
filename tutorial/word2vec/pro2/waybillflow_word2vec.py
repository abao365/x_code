#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: leidelong
@license: Apache Licence 
@contact: leidl8907@gmail.com
@site: https://github.com/JoneNash
@software: PyCharm Community Edition
@time: 2017/10/12 下午7:06
"""

import os
import gensim.models
import folium

#write a yielder
class CorpusYielder(object):
    def __init__(self,path):
        self.path=path
    def __iter__(self):
        for line in open(self.path,'r'):
            yield line.split(',')

#train a model
sentenceIterator=CorpusYielder('/Users/leidelong/work/projects/knowledge_graph/2002445_full_waybill_ids.txt')
model=gensim.models.Word2Vec()
model.build_vocab(sentenceIterator)
model.train(sentenceIterator,total_examples=37445,epochs=10)
model.wv.save_word2vec_format('/Users/leidelong/work/projects/knowledge_graph/train_result.txt')




#use the model
print model.similarity('2002445_0.0_33','2002445_0.0_6')
print model.similarity('2002445_0.0_6','2002445_0.0_33')
print model.similarity('2002445_0.0_6','2002445_2.0_33')

print model.most_similar('2002445_0.0_6')
# print model['2002445_2.0_1']



#使用leaflet做可视化




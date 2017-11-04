#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: leidelong
@license: Apache Licence 
@contact: leidl8907@gmail.com
@site: https://github.com/JoneNash
@software: PyCharm Community Edition
@time: 2017/10/27 下午3:37
"""


#########################订单流共现频率与Word2vec的相关性####################

####################订单流共现频率
rawDataDir='/Users/leidelong/work/projects/knowledge_graph/2002445_full_waybill_ids.txt'
import pandas as pd
df=pd.read_csv(rawDataDir,sep=' ')

print df.shape
print df.columns

dict_waybillflow = dict()
dict_waybillflow_combination= dict()

for element_index in df.index:
    str=df.at[element_index, "waybillflowlist"]
    str_arr=str.split(",")

    #统计各个订单流出现的次数
    for ele in str_arr:
        if(""!=ele):
            num=dict_waybillflow.get(ele)
            if(None == num ):
                dict_waybillflow[ele]=1
            else:
                dict_waybillflow[ele] = num+1

    #统计两个订单流的出现次数

    for i in range(len(str_arr)):
        # print "str_arr:",str_arr
        waybillflow_id_left=str_arr[i]

        if("" != waybillflow_id_left):
            for j in range(i+1, len(str_arr)):
                waybillflow_id_right = str_arr[j]
                tmp_str_combination=waybillflow_id_left+","+waybillflow_id_right

                comb_num = dict_waybillflow_combination.get(tmp_str_combination)

                if (None == comb_num):
                    dict_waybillflow_combination[tmp_str_combination] = 1
                else:
                    dict_waybillflow_combination[tmp_str_combination] = comb_num + 1

print "dict_waybillflow:::::::"
for key, value in dict_waybillflow.items():
    print key, ':', value


print "dict_waybillflow_combination:::::::"


dd=sorted(dict_waybillflow_combination.items(), lambda x, y: cmp(x[1], y[1]))

# dict_waybillflow_combination_values = dict_waybillflow_combination.values()
# dict_waybillflow_combination_values.sort()

for key, value in dict_waybillflow_combination.items():
    print key, ':', value

print "########"

for elem in dd:
    if(elem[0].find("2002445_0.0_6")>=0):
        print elem


###########################word2vec


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
model=gensim.models.Word2Vec(size=100)
model.build_vocab(sentenceIterator)
model.train(sentenceIterator,total_examples=37445,epochs=50)
model.wv.save_word2vec_format('train_result.txt')


##########################计算两种方法的相关性
tmp_list=list()
for key, value in dict_waybillflow_combination.items():
    comb=key.split(",")
    tmp_simlarity=0

    start_flow = comb[0]
    end_flow = comb[1]

    start_flow_elem= start_flow.split("_")
    end_flow_elem = end_flow.split("_")

    area_id = int(start_flow_elem[0])
    start_flow_poi_class= int(float(start_flow_elem[1]))
    start_flow_user_class = int(start_flow_elem[2])

    end_flow_poi_class = int(float(end_flow_elem[1]))
    end_flow_user_class = int(end_flow_elem[2])

    try:
        tmp_simlarity=model.similarity(comb[0], comb[1])
    except KeyError:
        print KeyError

    avg_cnt=0

    try:
        value_reverse=dict_waybillflow_combination.get(comb[1]+","+comb[0])
        if(value_reverse <>None):
            avg_cnt= float(value_reverse+value) / (dict_waybillflow.get(comb[0])+dict_waybillflow.get(comb[1]))
        else:
            avg_cnt= 2*value / (dict_waybillflow.get(comb[0])+dict_waybillflow.get(comb[1]))
    except KeyError :
        print KeyError




    tmp_dict={"area_id":area_id,"start_flow_poi_class":start_flow_poi_class,"start_flow_user_class":start_flow_user_class,
              "end_flow_poi_class":end_flow_poi_class,"end_flow_user_class":end_flow_user_class,
              "flow_name":key,"ct":value,"avg_ct":avg_cnt,"simlarity":tmp_simlarity}


    tmp_list.append(tmp_dict)

print tmp_list

df_sim=pd.DataFrame.from_dict(tmp_list)
print df_sim.head(10)

df_sim=df_sim[(df_sim['ct']<>0) &(df_sim['avg_ct']<>0) & (df_sim['simlarity']<>0)]

import scipy.stats as stats

avg_ct_arr=df_sim['avg_ct']
simlarity_arr=df_sim['simlarity']
ct_arr=df_sim['ct']

print stats.pearsonr(avg_ct_arr,simlarity_arr)
print stats.pearsonr(ct_arr,simlarity_arr)

df_sim.to_csv("/Users/leidelong/work/projects/knowledge_graph/df_sim.csv")
print df_sim.shape
print df_sim.columns







#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: leidelong
@license: Apache Licence 
@contact: leidl8907@gmail.com
@site: https://github.com/JoneNash
@software: PyCharm Community Edition
@time: 2017/11/1 上午10:47
"""
import pandas as pd
rawDataDir_sim='/Users/leidelong/work/projects/knowledge_graph/df_sim.csv'
rawDataDir_all='/Users/leidelong/work/projects/knowledge_graph/2002445_flow2flow_info.txt'

df_sim=pd.read_csv(rawDataDir_sim,sep=',')
df_all=pd.read_csv(rawDataDir_all,sep=',')

print df_sim.columns
print df_sim.shape

print df_all.columns
print df_all.shape

df_merge=pd.merge(df_sim,df_all,how='left',
         left_on=['start_flow_poi_class','start_flow_user_class',
                  'end_flow_poi_class','end_flow_user_class'],
         right_on=['start_flow_poi_class_id','start_flow_recipient_class_id',
                   'end_flow_poi_class_id','end_flow_recipient_class_id'])

print df_merge.shape
print df_merge.columns

df_merge.to_csv("/Users/leidelong/work/projects/knowledge_graph/df_sim_packageRate.csv")
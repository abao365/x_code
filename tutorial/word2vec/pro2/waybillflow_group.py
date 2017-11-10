#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: leidelong
@license: Apache Licence
@contact: leidl8907@gmail.com
@site: https://github.com/JoneNash
@software: PyCharm Community Edition
@time: 2017/11/10 下午3:51


1.订单流按照分钟维度单量统计

2.计算每一个订单流每分钟的历史平均单量、顺路单历史平均单量

3.定义对顺路订单流
相同商圈，中心点相距3km以内的餐圈


"""

import pandas as pd
from geopy.distance import vincenty
import time

#订单流之间的关系
rawDataDir_flow2flow_info='/Users/leidelong/work/projects/knowledge_graph/2002445_flow2flow_info.txt'
df_flow2flow=pd.read_csv(rawDataDir_flow2flow_info,sep=',')
#1.定义顺路订单####
df_flow2flow=df_flow2flow[ (df_flow2flow['userclass2userclass_distance'] <2000 ) & (df_flow2flow['start_flow_poi_class_id']==df_flow2flow['end_flow_poi_class_id']) ]




#订单信息
rawDataDir_waybill_info='/Users/leidelong/work/projects/knowledge_graph/2002445_1month.txt'
df_waybill=pd.read_csv(rawDataDir_waybill_info,sep='	')
df_waybill=df_waybill[ (df_waybill['poi_class_id'] >=0 ) & (  df_waybill['recipient_class_id'] >=0 )]
df_waybill=df_waybill[(df_waybill['ctime_hour']>=11) & (df_waybill['ctime_hour']<=13)]
df_waybill.sort_values(["ctime"],ascending=True)
#添加分钟
df_waybill['minutes']=df_waybill['ctime'].apply(lambda x :time.localtime(x).tm_hour*60 +  time.localtime(x).tm_min)

#分钟级别的订单流单量####
df_waybill_baseinfo=df_waybill.loc[:,['bm_delivery_area_id','poi_class_id','recipient_class_id','minutes']]
df_waybillflow_waybillcount=df_waybill_baseinfo.groupby(['bm_delivery_area_id','poi_class_id','recipient_class_id','minutes']).size().reset_index(name='counts')


df_minute=pd.DataFrame(range(11*60,13*60,1),columns=['minutes'])

df_minute['key']=1
df_flow2flow['key']=1
print df_minute.shape
print df_flow2flow.shape
df_flow2flow_merge=pd.merge(df_flow2flow,df_minute,how='outer',left_on='key', right_on='key')
print df_flow2flow_merge.shape

df_flow2flow_count_merge=pd.merge(df_flow2flow_merge,df_waybillflow_waybillcount,how='left',left_on=['end_flow_poi_class_id','end_flow_recipient_class_id','minutes'],right_on=['poi_class_id','recipient_class_id','minutes'])

print df_flow2flow_count_merge.shape
tmp=df_flow2flow_count_merge[df_flow2flow_count_merge['counts']>0]
print 'tmp:',tmp.shape

#join之后的结果落地
df_flow2flow_count_merge.to_csv('/Users/leidelong/work/projects/knowledge_graph/2002445_flow2flow_count_merge.txt',index=False,doublequote=False)

#顺路订单流组的单量总计


# waybillflow_group_ct=df_flow2flow_count_merge.groupby(['start_flow_poi_class_id', 'start_flow_recipient_class_id','minutes'])['counts'].sum()

waybillflow_group_ct=df_flow2flow_count_merge.groupby(by=['start_flow_poi_class_id', 'start_flow_recipient_class_id','minutes']).agg({'counts':sum}).reset_index()

print waybillflow_group_ct.shape

#统计结果落地
waybillflow_group_ct.to_csv('/Users/leidelong/work/projects/knowledge_graph/2002445_waybillflow_group_ct.txt',index=False,doublequote=False)









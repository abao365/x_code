#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: leidelong
@license: Apache Licence 
@contact: leidl8907@gmail.com
@site: https://github.com/JoneNash
@software: PyCharm Community Edition
@time: 2017/10/27 下午3:04
"""


#预处理数据
rawDataDir='/Users/leidelong/work/projects/knowledge_graph/2002445_1month.txt'
import pandas as pd
import csv
df=pd.read_csv(rawDataDir,sep='	')

print df.shape
print df.dtypes
df=df[ (df['poi_class_id'] >=0 ) & (  df['recipient_class_id'] >=0 )]
print df.shape
print df.columns

df=df[(df['ctime_hour']>=11) & (df['ctime_hour']<=13)]
df.sort_values(["ctime"],ascending=True)
print df.shape
wbf_list_df = pd.DataFrame(columns=['waybillflowlist'])

for waybill_index in df.index:
    # print "已扫描运单索引：",waybill_index
    tmp_ctime=df.at[waybill_index,"ctime"]
    tmp_rider=df.at[waybill_index,"rider_id"]
    tmp_wb_dt=df.at[waybill_index,"wb_dt"]


    wbf_list_tmp_str =""
    new_df=df[ (df["ctime"]<tmp_ctime) & (df["finished_time"]>tmp_ctime) & (df["rider_id"]==tmp_rider) & (df["wb_dt"]==tmp_wb_dt) ]
    for tt_index in new_df.index:
        # wbf=new_df.loc[new_df["ctime"]==tt_w, "bm_delivery_area_id"]
        wbf_area_id = new_df.at[tt_index, "bm_delivery_area_id"]
        wbf_poi_class_id = new_df.at[tt_index, "poi_class_id"]
        wbf_recipient_class_id = new_df.at[tt_index, "recipient_class_id"]
        wbf_list_tmp_str=wbf_list_tmp_str+","+str(wbf_area_id)+"_"+str(wbf_poi_class_id)+"_"+str(wbf_recipient_class_id)
    if("" <>wbf_list_tmp_str):
        wbf_list_df.loc[wbf_list_df.shape[0] + 1] = wbf_list_tmp_str


print "wbf_list_df.shape:",wbf_list_df.shape

wbf_list_df.to_csv('/Users/leidelong/work/projects/knowledge_graph/2002445_full_waybill_ids.txt', sep=' ',index=False,quoting=csv.QUOTE_NONE)

#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: leidelong
@license: Apache Licence 
@contact: leidl8907@gmail.com
@site: https://github.com/JoneNash
@software: PyCharm Community Edition
@time: 2017/10/31 下午3:51
"""

import pandas as pd

rawDataDir_poi='/Users/leidelong/work/projects/knowledge_graph/2002445_poi.txt'
rawDataDir_user='/Users/leidelong/work/projects/knowledge_graph/2002445_user.txt'
rawDataDir_waybillflow='/Users/leidelong/work/projects/knowledge_graph/2002445_full_waybill_ids.txt'

df_poi=pd.read_csv(rawDataDir_poi,sep='	')
df_user=pd.read_csv(rawDataDir_user,sep='	')
df_flow=pd.read_csv(rawDataDir_waybillflow,sep='	')

from geopy.distance import vincenty

all_flow_info_list = list()
for poi_index in df_poi.index:
    tmp_poi_class_id = df_poi.at[poi_index, "poi_class_id"]
    tmp_poi_class_center_lat = float(df_poi.at[poi_index, "poi_class_center_lat"]) / 1000000
    tmp_poi_class_center_lng = float(df_poi.at[poi_index, "poi_class_center_lng"]) / 1000000

    for user_index in df_user.index:
        tmp_recipient_class_id = df_user.at[user_index, "recipient_class_id"]
        tmp_recipient_class_center_lat = float(df_user.at[user_index, "recipient_class_center_lat"]) / 1000000
        tmp_recipient_class_center_lng = float(df_user.at[user_index, "recipient_class_center_lng"]) / 1000000

        poi_latlng = (tmp_poi_class_center_lat, tmp_poi_class_center_lng)
        user_latlng = (tmp_recipient_class_center_lat, tmp_recipient_class_center_lng)
        distance = vincenty(poi_latlng, user_latlng).meters

        tmp_flow_dict = {"poi_class_id": tmp_poi_class_id,
                         "poi_class_center_lat": tmp_poi_class_center_lat,
                         "poi_class_center_lng": tmp_poi_class_center_lng,
                         "recipient_class_id": tmp_recipient_class_id,
                         "recipient_class_center_lat": tmp_recipient_class_center_lat,
                         "recipient_class_center_lng": tmp_recipient_class_center_lng,
                         "distance": distance}

        all_flow_info_list.append(tmp_flow_dict)

all_flow_info = pd.DataFrame(all_flow_info_list)
print all_flow_info.shape
all_flow_info.head(10)

flow2flow_packageRate_list = list()
for flow_index in all_flow_info.index:
    start_flow_poi_class_id = all_flow_info.at[flow_index, "poi_class_id"]
    start_flow_poi_class_center_lat = all_flow_info.at[flow_index, "poi_class_center_lat"]
    start_flow_poi_class_center_lng = all_flow_info.at[flow_index, "poi_class_center_lng"]
    start_flow_recipient_class_id = all_flow_info.at[flow_index, "recipient_class_id"]
    start_flow_recipient_class_center_lat = all_flow_info.at[flow_index, "recipient_class_center_lat"]
    start_flow_recipient_class_center_lng = all_flow_info.at[flow_index, "recipient_class_center_lng"]
    start_flow_distance = all_flow_info.at[flow_index, "distance"]

    for flow_index_next in all_flow_info.index:
        # if (flow_index <> flow_index_next):
        end_flow_poi_class_id = all_flow_info.at[flow_index_next, "poi_class_id"]
        end_flow_poi_class_center_lat = all_flow_info.at[flow_index_next, "poi_class_center_lat"]
        end_flow_poi_class_center_lng = all_flow_info.at[flow_index_next, "poi_class_center_lng"]
        end_flow_recipient_class_id = all_flow_info.at[flow_index_next, "recipient_class_id"]
        end_flow_recipient_class_center_lat = all_flow_info.at[flow_index_next, "recipient_class_center_lat"]
        end_flow_recipient_class_center_lng = all_flow_info.at[flow_index_next, "recipient_class_center_lng"]
        end_flow_distance = all_flow_info.at[flow_index_next, "distance"]

        s1e1 = start_flow_distance
        e1s2 = vincenty((start_flow_recipient_class_center_lat, start_flow_recipient_class_center_lng),
                        (end_flow_poi_class_center_lat, end_flow_poi_class_center_lng)).meters
        s2e2 = end_flow_distance
        s1s2 = vincenty((start_flow_poi_class_center_lat, start_flow_poi_class_center_lng),
                        (end_flow_poi_class_center_lat, end_flow_poi_class_center_lng)).meters

        s2e1 = vincenty((end_flow_poi_class_center_lat, end_flow_poi_class_center_lng),
                        (start_flow_recipient_class_center_lat, start_flow_recipient_class_center_lng)).meters

        e1e2 = vincenty((start_flow_recipient_class_center_lat, start_flow_recipient_class_center_lng),
                        (end_flow_recipient_class_center_lat, end_flow_recipient_class_center_lng)).meters

        e2e1 = vincenty((end_flow_recipient_class_center_lat, end_flow_recipient_class_center_lng),
                        (start_flow_recipient_class_center_lat, start_flow_recipient_class_center_lng)).meters

        distance_mode1 = (start_flow_distance + end_flow_distance) / (s1e1 + e1s2 + s2e2)
        distance_mode2 = (start_flow_distance + end_flow_distance) / (s1s2 + s2e1 + e1e2)
        distance_mode3 = (start_flow_distance + end_flow_distance) / (s1s2 + s2e2 + e2e1)

        start_poi_latlng = (start_flow_poi_class_center_lat, start_flow_poi_class_center_lng)
        start_user_latlng = (start_flow_recipient_class_center_lat, start_flow_recipient_class_center_lng)

        end_poi_latlng = (end_flow_poi_class_center_lat, end_flow_poi_class_center_lng)
        end_user_latlng = (end_flow_recipient_class_center_lat,end_flow_recipient_class_center_lng)

        poiclass2poiclass_distance = vincenty(start_poi_latlng, end_poi_latlng).meters
        userclass2userclass_distance = vincenty(start_user_latlng, end_user_latlng).meters

        flow2flow_dict = {"start_flow_poi_class_id": start_flow_poi_class_id,
                          "start_flow_poi_class_center_lat": start_flow_poi_class_center_lat,
                          "start_flow_poi_class_center_lng": start_flow_poi_class_center_lng,
                          "start_flow_recipient_class_id": start_flow_recipient_class_id,
                          "start_flow_recipient_class_center_lat": start_flow_recipient_class_center_lat,
                          "start_flow_recipient_class_center_lng": start_flow_recipient_class_center_lng,
                          "start_flow_distance": start_flow_distance,
                          "end_flow_poi_class_id": end_flow_poi_class_id,
                          "end_flow_poi_class_center_lat": end_flow_poi_class_center_lat,
                          "end_flow_poi_class_center_lng": end_flow_poi_class_center_lng,
                          "end_flow_recipient_class_id": end_flow_recipient_class_id,
                          "end_flow_recipient_class_center_lat": end_flow_recipient_class_center_lat,
                          "end_flow_recipient_class_center_lng": end_flow_recipient_class_center_lng,
                          "end_flow_distance": end_flow_distance,
                          "poiclass2poiclass_distance":poiclass2poiclass_distance,
                          "userclass2userclass_distance": userclass2userclass_distance,

                          "distance_mode1": distance_mode1,
                          "distance_mode2": distance_mode2,
                          "distance_mode3": distance_mode3
                          }
        flow2flow_packageRate_list.append(flow2flow_dict)

flow2flow_info = pd.DataFrame(flow2flow_packageRate_list)
print flow2flow_info.shape
flow2flow_info.head(10)
flow2flow_info.to_csv('/Users/leidelong/work/projects/knowledge_graph/2002445_flow2flow_info.txt',index=False,doublequote=False)

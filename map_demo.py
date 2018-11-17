# -*- coding: utf-8 -*-

"""
Author: kingming

File: map_demo.py

Time: 2018/11/17 下午5:05

License: (C) Copyright 2018, xxx Corporation Limited.

"""


import folium
import pandas as pd


def mark_map(data):
    """
    带有标注的地图
    :param data:
    :return:
    """
    # 地图制作
    myMap = folium.Map(location=[20, 0], tiles="Mapbox Bright", zoom_start=2)

    for i in range(len(data)):
        # 自定义 popup 内容
        test = folium.Html(
            '<b>id:{}</b></br> <b>name:{}</b></br> <b>lon:{}</b></br> <b>lat:{}</b></br> '.format(data.iloc[i]['id'],
                                                                                                  data.iloc[i]['name'],
                                                                                                  data.iloc[i]['lon'],
                                                                                                  data.iloc[i]['lat']),
            script=True)
        popup = folium.Popup(test, max_width=2650)
        folium.Marker([data.iloc[i]['lon'], data.iloc[i]['lat']], popup=popup).add_to(myMap)

    # 保存地图
    myMap.save('testMap.html')


'''创建底层Map对象'''
m = folium.Map(location=[0.5,100.5],
              zoom_start=8,
              control_scale=True)

'''定义geojson图层'''
gj = folium.GeoJson(data={ "type": "LineString",
  "coordinates": [ [100.0, 0.0], [101.0, 1.0] ]
  })

'''为m添加geojson层'''
gj.add_to(m)


if __name__ == '__main__':
    # 地图上的点
    data = pd.DataFrame({
        'lat': [-58, 2, 145, 30.32, -4.03, -73.57, 36.82, -38.5],
        'lon': [-34, 49, -38, 59.93, 5.33, 45.52, -1.29, -12.97],
        'name': ['Buenos Aires', 'Paris', 'melbourne', 'St Petersbourg', 'Abidjan', 'Montreal', 'Nairobi', 'Salvador'],
        'id': [x for x in range(8)]
    })
    mark_map(data)





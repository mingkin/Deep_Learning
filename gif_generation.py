# -*- coding: utf-8 -*-
"""
# Author  : Ming
# File    : d1.py
# Time    : 2019/7/16 0016 上午 11:09
© 2019 Ming. All rights reserved. Powered by King
"""



import requests
from pyquery import PyQuery as pq
import os, sys
import imageio
from PIL import Image

'''
天气预报.gif 生成class
'''


class weatherForecast():
    def __init__(self, weatherSite, path, endpng, savemodel):
        self.savemodel = savemodel
        if not os.path.exists(path):
            os.makedirs(path)

    def getPic(self):
        '''
        获取资源
        '''
        print('获取pic')
        d = pq(weatherSite)
        DomTree = d('#slideform #slide option')  # 获取DOM节点option 标签
        num = 100
        for bigpic in DomTree.items():
            pic = bigpic.attr('bigpic')  # 获取bigpic 属性指
            num += 1
            self.download(pic, 'a' + str(num) + '.png')  # 下载pic
        print('pic下载成功，共下载' + str(num - 100) + '个png')
        self.download(endpng, 'a1200.png')  # 下载end.png
        self.download(endpng, 'a1201.png')
        self.download(endpng, 'a1202.png')
        self.download(endpng, 'a1203.png')

    def download(self, url, fname):
        '''
        下载pic
        :return images size
        '''
        size = 0
        try:
            r = requests.get(url, timeout=3)
            file = open(path + fname, 'wb')
            size = file.write(r.content)
            file.close()
            # 修改图片大小，原：x=640*y=480  = 320*240
            ima = Image.open(path + fname)
            (x, y) = ima.size  # read image size
            x_s = 320
            y_s = int((y * x_s) / x)  # #calc height based on standard width
            out = ima.resize((x_s, y_s), Image.ANTIALIAS)  # resize image with high-quality
            out.save(path + fname)
    
    
        except:
            pass
        return size

    def getGIF(self):
        '''
        生成gif
        '''
        images = []
        print('执行开始')
        self.getPic()  # 获取图片资源
        filenames = sorted(fn for fn in os.listdir(path) if fn.endswith('.png'))
        if self.savemodel == 1:  # imageio方法
            for filename in filenames:
                images.append(imageio.imread(path + filename))
            print('执行conversion操作')
            imageio.mimsave('weather.gif', images, duration=0.5, loop=1)  # duration 每帧间隔时间，loop 循环次数
            print('完成……')
        elif self.savemodel == 2:  # PIL 方法
            imN = 1
            for filename in filenames:
                if imN == 1:  # 执行一次 im的open操作，PIL在保存gif之前，必须先打开一个生成的帧，默认第一个frame的大小、调色
                    im = Image.open(path + filename)
                    imN = 2

                images.append(Image.open(path + filename))
            print('执行conversion操作')
            im.save('weather.gif', save_all=True, append_images=images, loop=1, duration=500,
                    comment=b"this is my weather.gif")
            print('完成……')

    



'''
注：loop循环次数在浏览器有效果，用看图软件不起作用
'''
if __name__ == "__main__":
    weatherSite = "http://products.weather.com.cn/product/radar/index/procode/JC_RADAR_AZ9210_JB"  # 上海南汇
    path = 'images/'  # png 图片存储位置
    endpng = 'http://images.cnblogs.com/cnblogs_com/dcb3688/982266/o_end.png'  # 因gif是循环播放，end png 区分新loop
    savemodel = 1  # 1：imageio保存图片， 2：PIL保存图片
    weatherForecast = weatherForecast(weatherSite, path, endpng, savemodel)
    weatherForecast.getGIF()
    sys.exit()



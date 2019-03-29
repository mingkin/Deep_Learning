# -*- coding: utf-8 -*-
"""
# Author  : Ming
# File    : {NAME}.py
# Time    : 2019/3/28 0028 上午 8:29
"""

import numpy as np
import matplotlib as mpl
import  matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
mpl.rcParams['font.family'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

'一维   y = x^2 - 2x +1'
def f(x):
    return x ** 2-2*x + 1


def graient(x):
    return 2*x -2

x0 =  np.arange(-9,11,0.1)
print(x0.shape)
y0 = f(x0)
print(y0.shape)

def disply():
    x_list = []
    y_list = []
    lr = 0.1
    x = 10
    for i in range(100):
        x_list.append(x)
        y_list.append(f(x))
        x = x - lr*graient(x)
    return x_list, y_list


# x_list, y_list = disply()
# plt.plot(x0, y0)
# plt.plot(x_list, y_list, 'ro--')
# plt.show()


'2维   y =0.2*(x1 +  x2)^2 - 0.3x1x2 +0.4'



x1 = np.arange(-5, 5, 0.1)
x2 = np.arange(-5, 5, 0.1)


def f1(x1, x2):
    return 0.2*(x1 + x2)**2 - 0.3*x1*x2 +0.4

def g_x1(x1, x2):
    return 0.4*x1*(x1+x2) - 0.3*x2

def g_x2(x1, x2):
    return 0.4*x2*(x1+x2) - 0.3*x1



def disply1():
    x1_list = []
    x2_list = []
    y_list = []
    lr = 0.1
    x1, x2 = 4.5, 4.5
    for i in range(100):
        x1_list.append(x1)
        x2_list.append(x2)
        y_list.append(f1(x1, x2))
        x1 = x1 - lr*g_x1(x1, x2)
        x2 = x2 - lr * g_x2(x1, x2)
    return x1_list, x2_list, y_list

x1_list, x2_list, y_list = disply1()
x11, x12 = np.meshgrid(x1, x2)
y = f1(x11, x12)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(x11, x12, y, rstride=5, cstride=5, cmap='rainbow')
ax.plot(x1_list, x2_list, y_list, 'bo--')
plt.show()


# -*- coding: utf-8 -*-
'''
Created on 2018年3月30日

@author: zwp
'''

import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-3, 3, 1000);
noise = np.random.normal(scale=1,size=np.alen(x));
y1 = 3*x+1+noise;
y2 = np.exp(x)+noise;
y3 = np.sin(x)+noise;

dataset_param=[0,3,100];
x1 =  np.linspace(*dataset_param);
x2 =  np.linspace(*dataset_param);
print (x1,x2);
x1,x2 = np.meshgrid(x1,x2)
print(x1,x2);

if __name__ == '__main__':
    
    plt.figure();
#    plt.scatter(x,y2);
    re = x1+x2*2+3;
    C= plt.contour(x1,x2,re,10, colors='black', linewidth=.5);
    plt.clabel(C, inline=True, fontsize=10)
    plt.show();
    pass
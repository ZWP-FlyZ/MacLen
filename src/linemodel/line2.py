# -*- coding: utf-8 -*-
'''
Created on 2018年3月30日

@author: zwp
'''
'''
    多（2）属性线性回归
'''
import numpy as np
import matplotlib.pyplot as plt 

data_size = 100;
dataset_param=[0,3,data_size];
dataset_param2=[0,2,data_size];
noise_rou = 0.8;
x1 =  np.linspace(*dataset_param);
x2 =  np.linspace(*dataset_param2);


def get_origin_model(X):
    return X[:,0]+2*X[:,1]+3;
def get_lable(X):
    x_s = data_size;
    st=1;
    gt=1;
    y = get_origin_model(X);
    gas_noise = np.random.normal(scale=noise_rou,size=x_s);
    sin_noise =np.sin(y);
    return y+st*sin_noise+gt*gas_noise;

def set_dataset(x,y):
    plt.figure(1);
    plt.scatter(x,y);
    oy = get_origin_model(x);
    # plt.plot(x,oy,'b',);
    
def set_line_model(x,w):
    w = np.array(w);
    plt.figure(1);
    y = x * w[0]+w[-1];
    plt.plot(x,y,'r');
    pass;
def show_fig():
    plt.show();

def get_line_w(X,Y):
    X = np.mat(X);
    Y = np.mat(Y);
    # Y = np.log(np.mat(Y));
    xtx = X.T * X;
    w = xtx.I * (X.T * Y);
    return w;


def run():
#     y = get_lable(x);
#     X = np.reshape(x,[-1,1]);
#     X = np.append(X,np.ones([np.alen(x),1]),axis=1);
#     Y = np.reshape(y,[-1,1]);
#     w = get_line_w(X, Y);
#     print(X);
#     print(y);
#     print(w);
#     set_dataset(x, y);
#     set_line_model(x, w);
#     show_fig();
    pass;

if __name__ == '__main__':
    run();
    pass
# -*- coding: utf-8 -*-
'''
Created on 2018年4月2日

@author: zwp
'''

'''
单属性局部线性回归
'''
import numpy as np
import matplotlib.pyplot as plt 

dataset_param=[0,10,100];
noise_rou = 0.9;
x =  np.linspace(*dataset_param);
tx = np.linspace(*[1,8,40]);

def get_origin_model(x):
    return  -2* x +2; 
    # return np.exp(x);
def get_lable(x):
    x_s = np.alen(x);
    gas_noise = np.random.normal(scale=noise_rou,size=x_s);
    sin_noise =np.sin(x);
    st=10;
    gt=1;
    y = get_origin_model(x);
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
    # plt.plot(x,np.exp(y),'y'); 
    pass;
def set_line_model_Y(x,y):
    plt.figure(1);
    plt.scatter(x,y,c='r');
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

def get_line_loc_y(tag_x,X,Y,k):
    X = np.mat(X);
    Y = np.mat(Y);
    tag_x = np.mat(tag_x)
    batch= X.shape[0];
    loc_w= np.mat(np.eye(batch));
    for i in range(batch):
        diff = tag_x - X[i];
        loc_w[i,i]=np.exp(diff*diff.T/(-2.0*(k**2)));
    # Y = np.log(np.mat(Y));d
    xtx = X.T * (loc_w*X);
    w = xtx.I * (X.T *(loc_w *Y));
    return tag_x*w;

def get_line_all_y(testX,X,Y,k=1.0):
    batch = testX.shape[0];
    ally = np.zeros(batch);
    for i in range(batch):
        ally[i]=get_line_loc_y(testX[i],X,Y,k);
        
    return ally;


def run():
    y = get_lable(x);
    X = np.reshape(x,[-1,1]);
    X = np.append(X,np.ones([np.alen(x),1]),axis=1);
    tX = np.reshape(tx,[-1,1]);
    tX = np.append(tX,np.ones([np.alen(tx),1]),axis=1);
    Y = np.reshape(y,[-1,1]);
    PY = get_line_all_y(tX,X,Y,0.3);
    print(X);
    print(y);
    print(PY);
    set_dataset(x, y);
    set_line_model_Y(tx,PY);
    show_fig();

if __name__ == '__main__':
    run();
    pass
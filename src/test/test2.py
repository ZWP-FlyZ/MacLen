# -*- coding: utf-8 -*-
'''
Created on 2018年4月2日

@author: zwp
'''
import numpy as np;


a = np.mat(np.array([[1,2,3],[2,3,4]]));
b = np.array([1,2,3]);
a = np.array(a);


c = np.array([4,5,6]);
print(np.vstack((b,c)));


if __name__ == '__main__':
    pass
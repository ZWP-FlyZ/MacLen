# -*- coding: utf-8 -*-
'''
Created on 2018年4月2日

@author: zwp
'''
import numpy as np;


a = np.array([[1,2,3],[2,3,4]]);
b = np.array([1,2,3]);
a = np.array(a);
print(a.T/b.reshape(3,1))



def getGausP(class_ss_list,target):
    '''
    class_ss_list:每种分类的对应的均值和方差列表
    target：需要求概率的目标值
    '''
    class_ss_list = np.array(class_ss_list);
    means = class_ss_list[:,0];
    variances = class_ss_list[:,1];
    V2 = 2 * variances;
    p = np.exp(-1.0*(means-target)**2/V2)/np.sqrt(V2*np.pi);
    return p;



print(getGausP([[0,1],
                [0,2]], 0))




if __name__ == '__main__':
    pass
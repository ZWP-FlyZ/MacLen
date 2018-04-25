# -*- coding: utf-8 -*-
'''
Created on 2018年4月24日

@author: zwp
'''

'''
朴素贝叶斯模型，前几种特征为任意种离散取值0-n(Xi)，
最后一个特征为连续正态取值，分类结果有多种取值0-C

由公式得p(c/X) = p(X/c)*p(c)/p(X)，各个类型的p(c/X),
以概率最大的作为预测结果

该贝叶斯模型训练与预测采用常规方法
'''

import numpy as np;
import copy ;


class SimpleNaiveBayes():
    '''
    简单贝叶斯模型
    '''
    
    class_cot = 0;# 分类种类计数
    feat_val_list = [];# 各个特征包含值的情况
    
    all_data_size = 0;# 训练集数量总数
    class_size_list = None;
    # 长度为class_cot的整数数组，记录各个类在训练集中的数量
    
    feat_class_val_maps = None;
    # 一个统计数组，数值中每一个项为一张特征与特征值的map，
    # 记录各个分类下的各个特征的数值情况
    
    class_pxc_maps = None;
    # 一个统计数组，形状同feat_class_val_maps，
    # 内部存储着P(X|C)的值，用于预测，与feat_class_val_maps对应
    
    # 各个分类在数据集中所占的比例
    class_p_list = None;
    
    

    def __init__(self,feat_val_list,class_cot):
        '''
        feat_val_list:特征中包含的取值数，
        若大于0表示一个离散值特征
        若小于等于0表示该特征是一个正态连续值特征
        class_cot:分类种类计数,class_cot大于0的整数
        '''
        self.class_cot = class_cot;
        self.feat_val_list = feat_val_list;
        self.init_param();
    pass;

    def init_param(self):
        class_cot = self.class_cot;
        feat_val_list = self.feat_val_list;
        
        self.all_data_size=0;
        self.class_size_list = np.zeros((class_cot,));  
        self.feat_class_val_maps = [];
        for cot in feat_val_list:
            if cot < 1:
                # 存储连续特征的 和 与 平方和
                tmp = np.zeros((class_cot,2),float);
            else:
                # 存储离散特征各个值出现的次数
                tmp = np.zeros((class_cot,cot),float);
            self.feat_class_val_maps.append(tmp);
                            
        self.class_pxc_maps=None;
        self.class_p_list=None;    
        pass;

    def trainNaiveBayes(self,featMat,labelList,update=True):
        '''
        训练贝叶斯模型
        featMat:训练用特征矩阵
        labelList:分类标签列表
        update:增量训练，在原数据集统计情况下更新参数
        '''
        if not update:
            self.init_param();
        
        class_size_list=self.class_size_list;
        feat_class_val_maps = self.feat_class_val_maps; 
        data_size = len(featMat);
        for index in range(data_size):
            self.all_data_size+=1;
            class_ind = int(labelList[index]);
            class_size_list[class_ind]+=1;
            train_item = featMat[index];
            for feat_ind in range(len(self.feat_val_list)):
                value = train_item[feat_ind];
                vect = feat_class_val_maps[feat_ind][class_ind];
                if self.feat_val_list[feat_ind]>=1:
                    vect[int(value)]+=1;
                else:
                    vect[0]+=value;
                    vect[1]+=value**2;
        
        class_cot_list = \
                np.array(class_size_list).reshape([self.class_cot,1]);
        class_pxc_maps = copy.deepcopy(feat_class_val_maps);
        for feat_ind in range(len(self.feat_val_list)):
            tmp = class_pxc_maps[feat_ind];
            feat_num = self.feat_val_list[feat_ind];
            if feat_num>=1:
                class_pxc_maps[feat_ind] = \
                    (tmp+1) /(class_cot_list+feat_num);# 拉普拉斯修正
            else:
                tmp = tmp /class_cot_list;
                tmp[:,1] = tmp[:,1]-tmp[:,0]**2;
                class_pxc_maps[feat_ind]=tmp;
        self.class_pxc_maps = class_pxc_maps;
        self.class_p_list =  \
           (class_size_list+1)/(self.all_data_size+self.class_cot);
        pass;
    
    
    def getGausP(self,class_ss_list,target):
        '''
        class_ss_list:每种分类的对应的均值和方差列表
        target：需要求概率的目标值
        '''
        class_ss_list = np.array(class_ss_list);
        means = class_ss_list[:,0];
        variances = class_ss_list[:,1];
        V2 = 2 * variances;
        p = np.exp(-(means-target)**2/V2)/np.sqrt(V2*np.pi);
        return p;

    

    def classify(self,testVect):
        '''
        将输入项向量进行分类
        '''
        feat_val_list=self.feat_val_list;
        class_pxc_maps = self.class_pxc_maps;
        class_p_list = self.class_p_list;
        feat_size = len(feat_val_list);
        
        test_result = np.zeros([self.class_cot,feat_size]);
        for feat_ind in range(feat_size):
            fvalue = testVect[feat_ind];
            c_pxc = class_pxc_maps[feat_ind];
            if feat_val_list[feat_ind]>0:
                test_result[:,feat_ind] = c_pxc[:,fvalue];
            else:
                test_result[:,feat_ind] = \
                    self.getGausP(c_pxc, fvalue);
        
        test_result = np.log(test_result);
        tmp = np.sum(test_result,axis=1);
        tmp+=np.log(class_p_list);
        print(tmp);
        c = np.argmax(tmp).reshape(());
        return c;

def test():
    f = [[0,0,0],#1
         [0,0,1],#0
         [0,1,0],#1
         [0,1,1],#0
         [1,0,0],#1
         [1,0,1],#0
         [1,1,0],##0
         [1,1,1]]#1
    l = [1,0,1,0,
         1,0,0,1]
    
    snb = SimpleNaiveBayes([3,3,3],2);
    snb.trainNaiveBayes(f, l);
    # 与坐标面平行的超平面分割问题
    print(snb.classify([1,1,0]));


if __name__ == '__main__':
    test();
    pass
# -*- coding: utf-8 -*-
'''
Created on 2018年4月23日

@author: zwp
'''

'''
朴素贝叶斯模型，所有特征为0-1两种状态，分类结果只有0-1两种

由公式得p(c/X) = p(X/c)*p(c)/p(X)，各个类型的p(c/X),
以概率最大的作为预测结果

该贝叶斯模型训练与预测采用常规方法
'''

import numpy as np;





def trainNaiveBayes(featMat,labelList,correction=True):
    '''
    训练贝叶斯模型
    featMat:训练用特征矩阵
    labelList:分类标签列表
    correction:使用拉普拉斯修正
    return:（p0Vect,p1Vect,p1）
        p0Vect分类类型为0时，各个特征上值是1的概率，
        p1Vect分类类型为1时，各个特征上值是1的概率，
        p1表示分类类型为1的概率
    '''
    featMat = np.array(featMat,float);
    labelList = np.array(labelList,float);
    p1index = np.where(labelList==1)[0];
    all_size = np.alen(labelList);
    p1_size = np.alen(p1index);
    p1sum = np.sum(featMat[p1index],axis=0);
    if correction:
        p1 = (p1_size+1)/(all_size+2);
        p1Vect = (p1sum+1) / (p1_size+2);
        p0Vect = (np.sum(featMat,axis=0)-p1sum+1)/(all_size-p1_size+2);
    else:
        p1 = p1_size/all_size;
        p1Vect = p1sum / p1_size;
        p0Vect = (np.sum(featMat,axis=0)-p1sum)/(all_size-p1_size);
    return p0Vect,p1Vect,p1;


def classify(p0Vect,p1Vect,p1,testVect):
    '''
    p0Vect分类类型为0时，各个特征上值是1的概率，
    p1Vect分类类型为1时，各个特征上值是1的概率，
    p1表示分类类型为1的概率
    testVect:测试用特征向量
    '''
    pVect = np.vstack((p0Vect,p1Vect));
    p0p1=np.log(np.array([1-p1,p1]));
    testVect = np.array(testVect);
    tmpP1V = pVect * testVect;
    tmpP0V = (1-pVect) * (1-testVect);
    tmpP = tmpP1V+ tmpP0V;
    tmpP = np.sum(np.log(tmpP),axis=1);
    tmpP+=p0p1;
    print(tmpP);
    if tmpP[0]>=tmpP[1]:
        return 0;
    else:
        return 1;
    pass;



def test():
    f = [[0,0,0],#1
         [0,0,1],#1
         [0,1,0],#1
         [0,1,1],#1
         [1,0,0],#1
         [1,0,1],#0
         [1,1,0],
         [1,1,1]]#0
    l = [1,1,1,1,
         1,0,0,1]
    nb_model = trainNaiveBayes(f,l);
    print(classify(*nb_model,[1,0,1]));
    pass;








if __name__ == '__main__':
    test();
    pass
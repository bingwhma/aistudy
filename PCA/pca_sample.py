
import numpy as np
import matplotlib.pyplot as plt

dataMat=np.array([
    [84,65,61,72,79,81],
    [64,77,77,76,55,70],
    [65,67,63,49,57,67],
    [74,80,69,75,63,74],
    [84,74,70,80,74,82],
    ])
    
#按列求均值，即每一列求一个均值，不同的列代表不同的特征
meanVals = np.mean(dataMat, axis=0)
print(meanVals)
#[74.2 72.6 68.  70.4 65.6 74.8]

#去均值，将样本数据的中心点移到坐标原点
meanAdj = dataMat - meanVals
print(meanAdj.T)

"""
[[  9.8 -10.2  -9.2  -0.2   9.8]
 [ -7.6   4.4  -5.6   7.4   1.4]
 [ -7.    9.   -5.    1.    2. ]
 [  1.6   5.6 -21.4   4.6   9.6]
 [ 13.4 -10.6  -8.6  -2.6   8.4]
 [  6.2  -4.8  -7.8  -0.8   7.2]]
"""

#计算协方差矩阵
covMat = np.cov(meanAdj, rowvar=0)
print(covMat)
"""
[[ 95.2  -13.9  -23.75  62.15 100.35  63.05]
 [-13.9   41.3   32.75  44.95 -26.95  -5.1 ]
 [-23.75  32.75  40.    42.5  -33.    -8.5 ]
 [ 62.15  44.95  42.5  151.3   53.7   53.85]
 [100.35 -26.95 -33.    53.7  110.8   65.9 ]
 [ 63.05  -5.1   -8.5   53.85  65.9   43.7 ]]

*4
[[ 380.8  -55.6  -95.   248.6  401.4  252.2]
 [ -55.6  165.2  131.   179.8 -107.8  -20.4]
 [ -95.   131.   160.   170.  -132.   -34. ]
 [ 248.6  179.8  170.   605.2  214.8  215.4]
 [ 401.4 -107.8 -132.   214.8  443.2  263.6]
 [ 252.2  -20.4  -34.   215.4  263.6  174.8]]
 
 
np.mat([1,2,3])  #创建矩阵
>>> m= np.mat([1,2,3])  #创建矩阵
>>> m
matrix([[1, 2, 3]])
>>> m[0]                #取一行
matrix([[1, 2, 3]])
>>> m[0,1]              #第一行，第2个数据
"""

#计算协方差矩阵的特征值和特征向量
eigVals,eigVects = np.linalg.eig(np.mat(covMat))
print(eigVals)
print(eigVects)

#sort, sort goes smallest to largest  #排序，将特征值按从小到大排列
eigValInd = np.argsort(eigVals)            
print(eigValInd)

#cut off unwanted dimensions      #选择维度为topNfeat的特征值
eigValInd = eigValInd[:-(3+1):-1]  
print(eigValInd)

#reorganize eig vects largest to smallest   #选择与特征值对应的特征向量
redEigVects = eigVects[:,eigValInd]       
print(redEigVects)

#transform data into new dimensions    #将数据映射到新的维度上，lowDDataMat为降维后的数据
lowDDataMat = meanAdj * redEigVects   
print(lowDDataMat)

#对原始数据重构，用于测试
reconMat = (lowDDataMat * redEigVects.T) + meanVals         
print(reconMat)
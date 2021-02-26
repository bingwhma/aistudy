# coding=utf-8

"""
任意一个M*N的矩阵A（M行*N列，M>N），可以被写成三个矩阵的乘机：
1.U：（M行M列的列正交矩阵）
2.S：（M*N的对角线矩阵，矩阵元素非负）
3.V：（N*N的正交矩阵的倒置）
即A=U*S*V‘（注意矩阵V需要倒置）

https://qiita.com/kyoro1/items/4df11e933e737703d549
"""
import numpy as np

#A=UΣVT

dataMat=np.array([
    [1,1],
    [1,1],
    [0,0]
    ])

U, s, V = np.linalg.svd(dataMat, full_matrices=True)

print(u)

print(s)

print(V)


print(np.dot(np.dot(U, np.diag(s)), V)
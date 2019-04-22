import matplotlib.pyplot as plt
import numpy as np

X = np.array([13,15,16,21,22,23,25,29,30,31,36,40,42,55,60,62,64,70,72,100,130])
Y = np.array([11,10,11,12,12,13,13,12,14,16,17,13,14,22,14,21,21,24,17,23,34])


xLen = len(X)

#y=wx+b

w = (xLen*np.sum(X*Y) - np.sum(X) * np.sum(Y)) / (xLen * np.sum(X**2) - np.sum(X)**2)
b = (np.sum(Y) * np.sum(X**2) - np.sum(X) * np.sum(X*Y)) / (xLen * np.sum(X**2) - np.sum(X)**2)


print("p = {:.4f}x + {:.4f} ".format(w,b))

plt.scatter(X, Y, color='b')

Y2 = w * X + b

plt.plot(X, Y2, color='r')

plt.show()

'''
https://blog.csdn.net/jayloncheng/article/details/80300724

用最小二乘法进行线性拟合

https://blog.csdn.net/Jack_Zhao_/article/details/81975405

'''
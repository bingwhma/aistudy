
#Matplotlib and Numpy
import numpy as np
import matplotlib.pyplot as plt

#Chainer Specific
from chainer import FunctionSet, Variable, optimizers, serializers
import chainer.functions as F
import chainer.links as L


# Create 2 chainer variables then sum their squares
# and assign it to a third variable.
a = Variable(np.array([3], dtype=np.float32))
b = Variable(np.array([4], dtype=np.float32))
c = a**2 + b**2

#Inspect the value of your variables.
print("a.data: {0}, b.data: {1}, c.data: {2}".format(a.data, b.data, c.data))

'''
通过调用data属性检查之前定义的变量
使用backward()方法，对变量c进行反向传播
通过在变量中存储的grad属性，检查其导数
'''

#Now call backward() on the sum of squares.
c.backward()
#And inspect the gradients.
print("dc/da = {0}, dc/db = {1}, dc/dc = {2}".format(a.grad, b.grad, c.grad))

#http://imonce.github.io/2016/12/14/Chainer%E5%85%A5%E9%97%A8%E6%95%99%E7%A8%8B(%E4%B8%8A)%EF%BC%9A%E5%9C%A8Chainer%E4%B8%AD%E5%81%9A%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92/
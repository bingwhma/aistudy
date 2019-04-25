import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np


# L.Linear(in_size, out_size, wscale=1, bias=0, nobias=False, 
#          initialW=None, initial_bias=None)
# 第1層
W1 = np.array([[1,-1],[-1,1]]) # weight parameters
b1 = 0.0
a1 = L.Linear(2, 2, initialW = W1, initial_bias = b1) #力2, 出力2
f1 = F.relu # activation function

W2 = np.array([[1, 1]])
b2 = 0.0
a2 = L.Linear(2, 1, initialW = W2, initial_bias = b2)

def XOR(x):
	h1 = f1(a1(x))
	return a2(h1)

X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]], dtype = np.float32)

Z = XOR(X)

print(Z.data)



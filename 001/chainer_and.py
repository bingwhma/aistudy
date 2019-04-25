import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np


# L.Linear(in_size, out_size, wscale=1, bias=0, nobias=False, 
#          initialW=None, initial_bias=None)
W = np.array([1,1])
b = -1.0
a = L.Linear(2, 1, initialW = W, initial_bias = b) # full-connected layer
f = F.relu #activation function

def AND(x):
	return f(a(x))

#use network as a function
x = np.array([[1,1],[1,0],[0,1],[0,0]], dtype = np.float32) #input data
print(AND(x))

'''

•AND関数は入力2ノード、出力1ノードの関数であり、全結合のLinearを使って実装できる。
•initialW, initial_biasを設定することで、ネットワークのパラメータを決め打ちしている。
•Linearはミニバッチを入力とすることに注意する。
•activation関数にはReLUを使った。


'''
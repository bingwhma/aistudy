import chainer
import numpy as np
from chainer import links as L
from chainer import Variable
from chainer import functions as F


'''
L.Linear の第一引数にはNoneを指定することで実際の入力からユニット数を自動で設定してくれます。

'''

class MLP(chainer.Chain):
	def __init__(self,n_units,n_out):
		super(MLP, self).__init__(
		l1 = L.Linear(None, n_units),
		l2 = L.Linear(None, n_units),
		l3 = L.Linear(None, n_out)
		)
	def __call__(self,x):
		h1 = F.relu(self.l1(x))
		h2 = F.relu(self.l2(h1))
		return self.l3(h2)

model = MLP(5,2)

x = Variable(np.ones((3, 5),dtype=np.float32))
print(x)


y = model(x)

print(y)

'''
https://ritsuan.com/blog/5947/

'''
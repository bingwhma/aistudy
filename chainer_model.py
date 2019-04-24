import numpy as np
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import Variable, optimizers


model = L.Linear(1, 1)

opt = optimizers.SGD()
opt.setup(model)


times = 1000


# 入力ベクトル
x = Variable(np.array([[1],[3],[5],[7]], dtype = np.float32))

# 正解ベクトル
t = Variable(np.array([[2],[6],[10], [14]], dtype=np.float32))


# 学習ループ

for i in range(0, times):
	# 勾配を初期化
	#opt.zero_grads() old api
	model.cleargrads()
	# モデルに予測させる
	y = model(x)
	loss = F.mean_squared_error(y, t)
	print("Data: {}, Loss: {}".format(y.data, loss.data) )
	# 逆伝播する
	loss.backward()	
	# optimizer を更新する
	opt.update()

print("Weight : {}".format(model.W.data) )
print("Bias   : {}".format(model.b.data) )

print("---TEST---")
x   = Variable(np.array( [[3],[4],[5]], dtype=np.float32) )
y   = model(x)

print("Test data   : {}".format(x.data) )
print("Test result : {}".format(y.data) )

'''
http://pongsuke.hatenadiary.jp/entry/2017/05/12/101630

'''
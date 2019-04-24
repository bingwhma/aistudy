import numpy as np
import chainer
from chainer import Function, gradient_check,Variable,optimizers, serializers, utils
from chainer import Link, Chain, ChainList
from chainer import functions as F
from chainer import links as L

import matplotlib.pyplot as plt


'''
https://qiita.com/ashitani/items/1dc0a54da218ec224ad8

教師データの作成

まず教師データを出力する関数を作ります。今回は0から1.0までの浮動小数x 
x
に対してe x  
ex
が期待値です。

バッチ学習という手法を使いますが、その際にn 
n
 個の問題・解答のセットを返す関数があると便利です

'''

def get_batch(n):
	x = np.random.random(n)
	y = np.exp(x)
	return x,y

class MyChain(Chain):
	def __init__(self):
		super(MyChain,self).__init__(
			l1=L.Linear(1,16),# 入力1チャネル、出力16チャネル
			l2=L.Linear(16,32),
			l3=L.Linear(32,1)
		)
	
	def __call__(self,x,t):
		# xを入力した際のネットワーク出力と、回答t との差分を返します。
		# 今回は平均二乗誤差を使います。
		return F.mean_squared_error(self.predict(x),t)
	
	def predict(self,x):
		# xを入力した際のネットワーク出力を返します。
		h1 = F.leaky_relu(self.l1(x))
		h2 = F.leaky_relu(self.l2(h1))
		h3 = F.leaky_relu(self.l3(h2))
		return h3
	
	def get(self,x):
		# xを実数で入力したら出力を実数で返すという便利関数です。
		# numpy.ndarrayとVariableを経由するので少々わかりにくいです。
		return self.predict(Variable(np.array([x]).astype(np.float32).reshape(1,1))).data[0][0]

#このモデルのインスタンスを作り、特定の戦略にしたがってパラメータを最適化してくれるオプティマイザの設定をします。
#今回はAdam()というものを使います。
model = MyChain()
optimizer = optimizers.Adam()
optimizer.setup(model)

losses = []
for i in range(10000):
	x,y = get_batch(100)
	x_ = Variable(x.astype(np.float32).reshape(100,1))
	t_= Variable(y.astype(np.float32).reshape(100,1))
	
	model.zerograds()
	loss = model(x_,t_)
	loss.backward()
	optimizer.update()
	
	losses.append(loss.data)

plt.plot(losses)
plt.yscale('log')

#では、出来上がったモデルの出力を確認してみましょう。0.2を入れたらexp(0.2)に近い値が出るでしょうか。
print(model.get(0.2))
print(np.exp(0.2))

x3=np.linspace(0,1,100)
plt.plot(x3, np.exp(x3))
#plt.hold(True)

p = model.predict(Variable(x3.astype(np.float32).reshape(100,1))).data
plt.plot(x3,p, color='r')

plt.show()

'''

次にニューラルネットを設計します。

y=e x  
y=ex
 は非線形関数なので、線形関数だけでの近似では十分な精度が得られません。入力をx 
x
としたとき、y=Wx+b 
y=Wx+b
のようなものを線形関数と呼びます。W 
W
を重み、b 
b
をバイアスと呼びますが、どちらもただの行列です。つまり直線（みたいなもの）ですね。

さて、この線形演算に対して、非線形関数による活性化層が入るだけでもうニューラルネットと呼んでいいらしいです。それを多層にしたものがディープニューラルネット、いわゆるディープラーニングで使われる非線形関数です。どれぐらい深ければディープと呼んでいいのか不明ですが、今回は3段ぐらいにしてみます。

一般的な分類問題だとreluという非線形関数が非常によく使われますが、reluだと微分消失してしまうので今回はleaky_reluを使います（この問題の場合はreluだと収束しませんでした）。leaky_reluは入力が負なら0.2をかける、というだけのシンプルな関数です。

それぞれの線形層のもつW,b 
W,b
というパラメータを最適化することで、y=e x  
y=ex
に相当するような関数を表現してみようということになります。

というわけで、ニューラルネットの構成は下記のようにしてみます。L1,L2,L3はそれぞれ線形関数で、いったん中間層h1,h2 
h1,h2
の次元を16、32と増やしたあとに、最後に1次元に落としています。


'''
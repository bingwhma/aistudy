import chainer
import numpy as np
import chainer.links as L
from chainer import Link, Chain, ChainList
from chainer import Variable


#
Lin = L.Linear(5,2)
#np.onesで入力データを作成し、Variable関数にかける
x = Variable(np.ones((3,5), dtype=np.float32))
print(x)

y = Lin(x)

print(y)
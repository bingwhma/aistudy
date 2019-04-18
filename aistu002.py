import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def numerical_gradient(f,x):
	
	print('.....numerical_gradient......')
	h = 1e-1
	grad = np.zeros_like(x)
	print(x)
	print('....................')
	print('...x.size is ' + str(x.size))
	for idx in range(x.size):
		tmpVal = x[idx]
		x[idx] = float(tmpVal) + h
		fxh1=f(x)
		
		x[idx] = tmpVal - h
		fxh2 = f(x)
		grad[idx] = (fxh1 - fxh2) / (2*h)
		
		x[idx] = tmpVal
	
	return grad



def numerical_gradient_2d(f, X):

	if X.ndim == 1:
		print('.....numerical_gradient_2d...... X.ndim == 1')
		return numerical_gradient(f, X)
	else:
		print('.....numerical_gradient_2d...... X.ndim > 1')
		grad = np.zeros_like(X)
		
		#返回 enumerate(枚举) 对象
		for  idx, x in enumerate(X):
			grad[idx] =numerical_gradient(f, x)

		return grad

def function_2(x):
	print("...........function_2")
	if x.ndim == 1:
		return np.sum(x**2)
	else:
		return np.sum(x**2, axis=1)


print('.......result is.....')
print(numerical_gradient_2d(function_2, np.array([1.0, 1.0])))


batch_x = np.array([
    [1.0, 1.0],
    [2.0, 2.0],
    [3.0, 3.0],
])

print('.......result is.....')
print(numerical_gradient_2d(function_2, batch_x))



x0 = np.arange(-2, 2.5, 0.25)
x1 = np.arange(-2, 2.5, 0.25)

#numpy.meshgrid()——生成网格点坐标矩阵。
#输出的X，Y，就是坐标矩阵。
X, Y = np.meshgrid(x0, x1)



Z = function_2(np.array([X.flatten(), Y.flatten()]).T) 

Z = Z.reshape(X.shape)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(X,Y,Z)

plt.show()





'''
# 坐标向量
a = np.array([1,2,3])
# 坐标向量
b = np.array([7,8])
# 从坐标向量中返回坐标矩阵
# 返回list,有两个元素,第一个元素是X轴的取值,第二个元素是Y轴的取值
res = np.meshgrid(a,b)
#返回结果: [array([ [1,2,3] [1,2,3] ]), array([ [7,7,7] [8,8,8] ])]

'''
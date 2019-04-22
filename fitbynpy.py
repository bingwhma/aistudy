
import numpy as np
import matplotlib.pyplot as plt

'''

https://www.cnblogs.com/gslyyq/p/5043847.html

'''
def linefit(x,y):
	dataLen = float(len(x))
	sx,sy,sxx,syy,sxy = 0,0,0,0,0
	
	for i in range(0, int(dataLen)):
		sx += x[i]
		sy += y[i]
		
		sxx += x[i]**2
		syy += y[i]**2
		sxy += x[i]*y[i]
	
	#w = (dataLen * sxy - sx*sy) / (dataLen * sxx - sx * sx)
	#w = (sx*sy/dataLen - sxy) / (sx * sx / dataLen - sxx)
	w = (sxy - sx*sy / dataLen) / (sxx - sx * sx / dataLen)
	b = (sy - w * sx) / dataLen
	
	return w,b



def linear_regression(x,y):
	dataLen=len(x)
	w=(np.sum(x*y) - dataLen*np.mean(x)*np.mean(y)) / (np.sum(x*x) - dataLen * np.mean(x)**2)
	b=np.mean(y) - w * np.mean(x)
	
	return w,b


#X = np.linspace(-5, 5, 20)
#Y = 3 * X + 5


X=np.array([ 1 ,2  ,3 ,4 ,5 ,6])
Y=np.array([ 2.5 ,3.51 ,4.45 ,5.52 ,6.47 ,7.51])

#plt.plot(X, Y)
plt.scatter(X, Y, color='b')

w1,b1 = linefit(X,Y)
print("result: y = %10.5f x + %10.5f , r=%10.5f", (w1,b1))
Y1 = w1 * np.asarray(X) + b1
plt.plot(X, Y1, color='g')


w2,b2 = linear_regression(X,Y)
print("result: y = %10.5f x + %10.5f , r=%10.5f", (w2,b2))
Y2 = w2 * np.asarray(X) + b2
plt.plot(X, Y2, color='r')

z1=np.polyfit(X,Y,1) # #一次多项式拟合，相当于线性拟合
p1 = np.poly1d(z1)

print(p1)
Y3=p1(X)#x1代入多项式，得到pp1,代入matplotlib中画多项式曲线
print(Y3)

plt.show()

'''

https://blog.csdn.net/your_answer/article/details/79195428

https://blog.csdn.net/starter_____/article/details/82011772



'''

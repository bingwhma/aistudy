import math
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
	
	w = (dataLen * sxy - sx*sy) / (dataLen * sxx - sx * sx)
	b = (sy - w * sx) / dataLen
	
	return w,b


X = np.linspace(-5, 5, 100)
Y = 3 * X + 5


#X=[ 1 ,2  ,3 ,4 ,5 ,6]
#Y=[ 2.5 ,3.51 ,4.45 ,5.52 ,6.47 ,7.51]

#plt.plot(X, Y)
plt.scatter(X, Y, color='b')

w,b = linefit(X,Y)
print("result: y = %10.5f x + %10.5f , r=%10.5f", (w,b,))
Y2 = w * np.asarray(X) + b
plt.plot(X, Y2)

plt.show()

'''

https://blog.csdn.net/your_answer/article/details/79195428

https://blog.csdn.net/starter_____/article/details/82011772



'''

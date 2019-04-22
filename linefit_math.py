import math

'''

https://www.cnblogs.com/gslyyq/p/5043847.html

'''
def linefit(x,y)
	dataLen = float(len(x))
	sx,sy,sxx,syy,sxy = 0,0,0,0,0
	
	for i in range(0, int(dateLen)):
		sx += x[i]
		sy += y[i]
		
		sxx += x[i]**2
		syy += y[i]**2
		sxy += x[i]*y[i]
	
	w = (dataLen * sx * sy - sy*sx)


'''

https://blog.csdn.net/your_answer/article/details/79195428

https://blog.csdn.net/starter_____/article/details/82011772



'''

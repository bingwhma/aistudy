
import numpy as np
import matplotlib.pyplot as plt


x = np.arange(1,17,1)
y = np.array([4.00, 6.40, 8.00, 8.80, 9.22, 9.50, 9.70, 9.86, 10.00, 10.20, 10.32, 10.42, 10.50, 10.55, 10.58, 10.60])

z=np.polyfit(x,y,4) #第三个参数为多项式最高次幂，结果为多项式的各个系数

#最高次幂3，得到4个系数,从高次到低次排列
#最高次幂取几要视情况而定
p=np.poly1d(z)  #将系数代入方程，得到函式p1
print(z) #多项式系数
print(p) #多项式方程

print(p(18))

x1=np.linspace(x.min(), x.max(), 100)
pp1=p(x1)

plt.scatter(x, y, color='g')
plt.plot(x1, pp1, color='b')

plt.show()
'''
# Author: yangbinchao
# Date:   2021-11-29
# Email:  heroybc@qq.com
# Describe: 对激活函数进行可视化
'''


import matplotlib.pyplot as plt
import numpy as np
import math

title = "Leaky ReLU"

def elu(x,alpha=1):
    a = x[x>0]
    b = alpha*(math.e**(x[x<0])-1)
    result=np.concatenate((b,a),axis=0)
    return result

def relu(x):
    if x >= 0:
        return x
    else :
        return np.exp(x)-1
    #return np.maximum(0,x)

def leakyRelu(x,alpha=1):
    aa = 0.01
    a = x[x>0]
    b = alpha*(0.01*(x[x<0]))
    result=np.concatenate((b,a),axis=0)
    return result

x = np.arange(-10, 10, 0.1)
y=leakyRelu(x)


plt.title(title)
plt.plot(x, y)
plt.savefig("test.png")
# plt.show()

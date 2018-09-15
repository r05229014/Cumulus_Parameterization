import numpy as np
import sys
import random
import matplotlib.pyplot as plt
import os
a=[0,100,200,300,400,500,600,700,800,900,1000]
b=[7,10,23,30,7,5,8,12,16,25]
c=[5,3,26,30,7,5,8,16,25,12]

y = np.load('../../data_8km_mean_y/wqv_32.npy')
z = np.load('../data/z.npy')


#for i in range(10):    
plt.figure(i)
plt.plot(y[a[6],:,b[6],c[6]]*2.5*10**6, z, label='True')
plt.legend()
plt.savefig('./img_%s' %i)
    

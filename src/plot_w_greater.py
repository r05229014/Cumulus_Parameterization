import numpy as np
import matplotlib.pyplot as plt

y = np.load('../../data_8km_mean_y/wqv_32.npy')
index = np.load('../data/target_w_index.npy')
z = np.load('../data/z.npy')
print(z[32])
print(index[487])
i = index[487]
print(np.argmax(y[i[0], :, i[1], i[2]] * 2.5*10**6))
#print(index[0][0])
#plt.plot(y[index[0][0],:, index[0][1], index[0][2]] * 2.5*10**6, z/1000)
#plt.show()
#dir_ = '../img/w_greater/'
#count = 0
#for i in index:
#    plt.figure(figsize=(10,10))
#    plt.xlabel('Wqv')
#    plt.ylabel('Height(m)')
#    plt.plot(y[i[0], :, i[1], i[2]]*2.5*10**6, z/1000)
#    plt.savefig(dir_ + '%s.png' %count)
#    plt.close()
#    count += 1 
    
    

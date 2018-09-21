import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab

y = np.load('../../data_8km_mean_y/wqv_32.npy')
z = np.load('../data/z.npy')

hor_3000 = y[:, 24, :, :]
x = np.arange(32)
print(x.shape)
xx, yy = np.meshgrid(x,x)
print(hor_3000[1].shape)

for i in range(1423):
    plt.figure(i)
    plt.title('t = %s' %i)
    cs = plt.contourf(xx,yy,hor_3000[i]*2.5*10**6, cmap=cm.coolwarm, vmax=10000, vmin=-10000, levels = np.arange(-10000, 10001, 10))
    m = plt.cm.ScalarMappable(cmap=cm.coolwarm)
    m.set_array(hor_3000)
    m.set_clim(-10000, 10000)
    cbar = plt.colorbar(cs)
    cbar.set_ticks([-10000, -8000,-6000,-4000,-2000,0,2000,4000,6000,8000,10000])
    cbar.set_ticklabels([-10000, -8000,-6000,-4000,-2000,0,2000,4000,6000,8000,10000])
    plt.savefig('../img/hor_true/'+'{:0>5d}'.format(i) + '.png')
    plt.close()

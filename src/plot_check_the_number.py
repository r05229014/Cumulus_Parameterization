import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab


y = np.load('../../data_8km_mean_y/wqv_32.npy')
index = np.load('../data/target_w_index.npy')
z = np.load('../data/z.npy')
w_8 = np.load('/home/ericakcc/Desktop/data_8km_mean_X/w_8km_mean.npy')[398]
w_all = np.load('../data/w.npy')[398]
qv_all = np.load('../data/qv.npy')[398]

w_avg = np.mean(np.mean(w_all[:,:,:], -1),-1)
qv_avg = np.mean(np.mean(qv_all[:,:,:], -1),-1)

#plt.figure()
#plt.plot(qv_avg, z)
#plt.title('qv_mean (t = 398)')
#plt.ylabel('z(m)')
#plt.savefig('tmp/qv_avg.png')


x = np.linspace(0,256,256)
X,Y = np.meshgrid(x,x)

for zz in range(70):
    print(max(qv_all[zz,:,:].flatten()))
    plt.figure()
    plt.title('z = %s(m)' %z[zz])
    cs = plt.contourf(X,Y,qv_all[zz, :, :], cmap=cm.coolwarm, vmax=0.2, vmin=0)
    m = cm.ScalarMappable(cmap=cm.coolwarm)
    m.set_array(qv_all[zz])
    m.set_clim(0,0.02)
    plt.colorbar(cs)
    plt.hlines(12*8, 24*8, 25*8, color='r')
    plt.hlines(13*8, 24*8, 25*8, color='r')
    plt.vlines(24*8, 12*8, 13*8, color='r')
    plt.vlines(25*8, 12*8, 13*8, color='r')
    plt.text(120,200,'qv_avg = %s' %qv_avg[zz], fontsize=12)
    plt.savefig('tmp/qv%s' %zz)
#w_ = w_all[:,:,:] - w_avg
#plt.plot(w_avg, z)
#plt.show()


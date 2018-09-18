import numpy as np
import sys
import random
import matplotlib.pyplot as plt
from keras.models import load_model
import os
from sklearn.preprocessing import StandardScaler

def load_data():
    th = np.load('../../data_8km_mean_X/th_8km_mean.npy')
    w = np.load('../../data_8km_mean_X/w_8km_mean.npy')
    qv = np.load('../../data_8km_mean_X/qv_8km_mean.npy')
    u = np.load('../../data_8km_mean_X/u_8km_mean.npy')
    v = np.load('../../data_8km_mean_X/v_8km_mean.npy')

    y = np.load('../../data_8km_mean_y/wqv_32.npy')
    
    th = np.swapaxes(th, 0,1)
    w = np.swapaxes(w, 0,1)
    qv = np.swapaxes(qv, 0,1)
    u = np.swapaxes(u, 0,1)
    v = np.swapaxes(v, 0,1)

    th = th.reshape(1423*70*32*32,1)
    w = w.reshape(1423*70*32*32,1)
    qv = qv.reshape(1423*70*32*32,1)
    u = u.reshape(1423*70*32*32,1)
    v = v.reshape(1423*70*32*32,1)

    sc = StandardScaler()
    th = sc.fit_transform(th)
    w = sc.fit_transform(w)
    qv = sc.fit_transform(qv)
    u = sc.fit_transform(u)
    v = sc.fit_transform(v)

    # swapaxex to fit in RNN input
    y = np.swapaxes(y, 0,1)
    y = y.reshape(70, -1)
    y = np.swapaxes(y, 0,1)

    th = th.reshape(70, -1, 1)
    w = w.reshape(70, -1, 1)
    qv = qv.reshape(70 ,-1 ,1)
    u = u.reshape(70, -1, 1)
    v = v.reshape(70, -1 ,1)

    th = np.swapaxes(th, 0,1)
    w = np.swapaxes(w, 0,1)
    qv = np.swapaxes(qv, 0,1)
    u = np.swapaxes(u, 0,1)
    v = np.swapaxes(v, 0,1)

    X = np.concatenate((th, w, qv, u, v), axis=-1)
    return X, y

X, y = load_data()
model = load_model('/home/ericakcc/Desktop/Cumulus_Parameterization/model/RNN/weights-improvement-020-4.668e-09.hdf5')

z = np.load('../data/z.npy')
print(z.shape)

img_dir = '/home/ericakcc/Desktop/Cumulus_Parameterization/img/RNN2/'
pre = model.predict(X, batch_size=2048)
print(pre.shape)
if not os.path.exists(img_dir):
    os.mkdir(img_dir)

for i in range(1000): 
    plt.figure(i)
    plt.plot(pre[random.randint(0, 1457152)]*2.5*10**6, z, label='Pre')
    plt.plot(y[random.randint(0, 1457152)]*2.5*10**6, z, label='True')
    #plt.xlim(-500, 500)
    plt.legend()
    plt.savefig(img_dir + 'img_%s' %i)
    plt.close()

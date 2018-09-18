import numpy as np
import sys
import random

def load_data():
    th = np.load('../../data_8km_mean_X/th_8km_mean.npy')
    w = np.load('../../data_8km_mean_X/w_8km_mean.npy')
    qv = np.load('../../data_8km_mean_X/qv_8km_mean.npy')
    u = np.load('../../data_8km_mean_X/u_8km_mean.npy')
    v = np.load('../../data_8km_mean_X/v_8km_mean.npy')

    # y preprocessing
    y = np.load('../../data_8km_mean_y/wqv_32.npy')
    y = np.swapaxes(y, 1, -1)
    y = y.reshape(-1, 70, 1)
    print('y shape is : ', y.shape)
    
    # X reshape and standardrization
    th_m = np.mean(th)
    th_s = np.std(th)
    th = (th-th_m)/th_s
    th = np.swapaxes(th, 1, -1)
    th = th.reshape(-1, 70, 1)
    
    w_m = np.mean(w)
    w_s = np.std(w)
    w = (w-w_m)/w_s
    w = np.swapaxes(w, 1, -1)
    w = w.reshape(-1, 70, 1)
    
    u_m = np.mean(u)
    u_s = np.std(u)
    u = (u-u_m)/u_s
    u = np.swapaxes(u, 1, -1)
    u = u.reshape(-1, 70, 1)
    
    qv_m = np.mean(qv)
    qv_s = np.std(qv)
    qv = (qv-qv_m)/qv_s
    qv = np.swapaxes(qv, 1, -1)
    qv = qv.reshape(-1, 70, 1)
    
    v_m = np.mean(v)
    v_s = np.std(v)
    v = (v-v_m)/v_s
    v = np.swapaxes(v, 1, -1)
    v = v.reshape(-1, 70, 1)
    
    X = np.concatenate((th, w, qv, u, v), axis=-1)
    print('X shape is : ', X.shape)
    
    return X, y 

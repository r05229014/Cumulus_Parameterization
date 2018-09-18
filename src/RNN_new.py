import numpy as np
import sys
import random
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras import Model
from keras.models import Sequential, load_model
from keras.layers import LeakyReLU, TimeDistributed
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers import LSTM
from keras.layers.convolutional import Convolution3D, MaxPooling3D, Convolution2D, MaxPooling2D
from keras.callbacks import *
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.utils import multi_gpu_model
import os 
import time


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
    y = y.reshape(1457152, 70, 1)
    print('y shape:', y.shape)

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
    print('X shape:', X)
    return X, y

class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)
            
        return super(ModelMGPU, self).__getattribute__(attrname)


def RNN():
    print("Build model!!")
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(70,5)))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    #model.add(LSTM(128, return_sequences=True))
    #model.add(LSTM(70, return_sequences=True))
    #model.add(Dense(128, activation='linear'))
    #model.add(Dense(70, activation='linear'))
    return model

tStart = time.time()

X, y = load_data()
#model = RNN()
#parallel_model = ModelMGPU(model, 3)
#parallel_model.compile(optimizer = 'adam', loss='mean_squared_error')
#print(model.summary())
#dirpath = "../model/RNN/"
#if not os.path.exists(dirpath):
#    os.mkdir(dirpath)

#filepath= dirpath + "/weights-improvement-{epoch:03d}-{loss:.3e}.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_loss', 
#                            save_best_only=False, period=2)
#earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
#parallel_model.fit(X,y, validation_split=0.1 , batch_size=256, epochs=150, shuffle=True, callbacks = [checkpoint, earlystopper])

tEnd = time.time()

print("It cost %f sec" %(tEnd - tStart))

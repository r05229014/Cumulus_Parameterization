from preprocessing import load_data
import random
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
import pickle
import time


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
    model.add(LSTM(256, return_sequences=True, input_shape=(70,5)))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    #model.add(LSTM(128, return_sequences=True))
    #model.add(LSTM(70, return_sequences=True))
    #model.add(Dense(128, activation='linear'))
    #model.add(Dense(70, activation='linear'))
    return model

tStart = time.time()

X, y = load_data()
model = RNN()
parallel_model = ModelMGPU(model, 3)
parallel_model.compile(optimizer = 'adam', loss='mean_squared_error')
print(model.summary())
dirpath = "../model/RNN_3_256/"
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

filepath= dirpath + "/weights-improvement-{epoch:03d}-{loss:.3e}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', 
                            save_best_only=False, period=2)
history = earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
parallel_model.fit(X,y, validation_split=0.1 , batch_size=256, epochs=150, shuffle=True, callbacks = [checkpoint])

with open('../loss/RNN_3_256', 'wb') as f:
    pickle.dump(history.history, f)

tEnd = time.time()

print("It cost %f sec" %(tEnd - tStart))

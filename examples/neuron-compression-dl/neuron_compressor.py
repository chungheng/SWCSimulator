from __future__ import print_function
import keras
from keras.layers import SimpleRNN, LSTM, GRU
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop, Adam
from keras.models import Sequential
from keras import initializers
import numpy as np

def simplernn_model(N, X, Y):
    X = np.transpose(X, (0, 2, 1))
    Y = np.transpose(Y, (0, 2, 1))
    X = X.astype('float32')
    Y = Y.astype('float32')
    model = Sequential()
    model.add(SimpleRNN(N,
                        activation='relu',
                        return_sequences = True,
                        input_shape=X.shape[1:]))
    model.add(Dense(Y.shape[-1]))
    model.summary()
    metatype = {}

    metatype['model'] = model
    metatype['X'] = X
    metatype['Y'] = Y
    return metatype

def simplegru_model(N, X, Y):
    X = np.transpose(X, (0, 2, 1))
    Y = np.transpose(Y, (0, 2, 1))
    X = X.astype('float32')
    Y = Y.astype('float32')
    model = Sequential()
    model.add(GRU(N,
                  activation='relu',
                  return_sequences = True,
                  input_shape=X.shape[1:]))
    model.add(Dense(Y.shape[-1]))
    model.summary()
    metatype = {}

    metatype['model'] = model
    metatype['X'] = X
    metatype['Y'] = Y
    return metatype

def neuron_compressor(metatype, batch_size = 32, epochs = 1, optimizer=None, compile_model = True, validation_split = 0.25):
    if optimizer == None:
        optimizer = Adam()
    metatype['model'].compile(loss='mse',
              optimizer=optimizer)
    metatype['model'].fit(metatype['X'], metatype['Y'],
          batch_size=batch_size,
          epochs=epochs,
          validation_split=validation_split)
    return metatype['model']

from __future__ import print_function
from compact_dependencies import *

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
    model.add(LSTM(N,
                  return_sequences = True,
                  input_shape=X.shape[1:],
                  implementation = 2))
    #model.add(Dense(Y.shape[-1]))
    model.summary()
    metatype = {}

    metatype['model'] = model
    metatype['X'] = X
    metatype['Y'] = Y
    return metatype

def simplelstmdense_model(N, X, Y):
    X = np.transpose(X, (0, 2, 1))
    Y = np.transpose(Y, (0, 2, 1))
    X = X.astype('float32')
    Y = Y.astype('float32')
    model = Sequential()
    model.add(Bidirectional(LSTM(N,
                  return_sequences = True,
                  implementation = 2,
                  dropout=0.00, recurrent_dropout=0.00), input_shape=X.shape[1:]))
    model.add(Dense(Y.shape[-1]))
    model.summary()
    metatype = {}

    metatype['model'] = model
    metatype['X'] = X
    metatype['Y'] = Y
    return metatype

def hhn_recovery_model(N, X, Y):
    X = np.transpose(X, (0, 2, 1))
    Y = np.transpose(Y, (0, 2, 1))
    X = X.astype('float32')
    Y = Y.astype('float32')
    model = Sequential()
    model.add(BatchNormalization(input_shape=X.shape[1:]))
    model.add(Bidirectional(LSTM(N,
                  return_sequences = True,
                  implementation = 2,
                  dropout=0.00, recurrent_dropout=0.00)))
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
    if compile_model == True:
        metatype['model'].compile(loss='mse',
                  optimizer=optimizer)
    metatype['model'].fit(metatype['X'], metatype['Y'],
          batch_size=batch_size,
          epochs=epochs,
          validation_split=validation_split)
    return metatype

#imports
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Lambda
from keras.layers import Dropout, add, Input, Layer
from keras.layers.recurrent import LSTM, GRU
from keras import callbacks, optimizers
from keras import backend as K
from keras.regularizers import l2
from keras.layers import Bidirectional
import random
import time
import csv
import json

from keras.layers.merge import Concatenate
from keras.initializers import glorot_uniform,he_uniform

from scipy import stats
from scipy.stats import iqr, skew, kurtosis

try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

print("imports successful")


#reshape data to (,,,1) format
def reshapeData(data):
  initShape=list(data.shape)
  initShape.append(1)
  return data.reshape(tuple(initShape))


#load data
def load_data():
    print('Loading data...')
    X_train = reshapeData(np.load('x_train.npy'))
    X_test = reshapeData(np.load('xauth_test.npy'))
    Y_train = reshapeData(np.load('y_train.npy'))
    Y_test = reshapeData(np.load('yauth_test.npy'))

    print('X_train shape:', X_train.shape)
    print('y_train shape:', Y_train.shape)
    print('X_test shape:', X_test.shape)
    print('y_test shape:', Y_test.shape)
    
    return (X_train, Y_train), (X_test, Y_test)

def createDict(x,y,seq_network_trained, N=10):

  dicts = {}
  bins = []

  Y = y
  y_temp=list(Y.reshape((len(Y),)))
  noOfDiffLabels=len(set(y_temp))

  #bins contain the training example indexes of each label
  for k in range(0,noOfDiffLabels):
    row = [i for i,x in enumerate(Y) if x == k]
    bins.append(row)

  for k in range(0,noOfDiffLabels):
    pos = []
    for l in range(0,N):
      pos.append(random.choice([i for i in bins[k] if i not in pos]))
    embed = seq_network_trained.predict([x[0,pos,:,:],x[1,pos,:,:],x[2,pos,:,:]])
    dicts[k] = embed

  return dicts

def getEval(x_test, y_test,seq_network_trained,dicts):
  predictions=seq_network_trained.predict([x_test[0,:,:,:],x_test[1,:,:,:],x_test[2,:,:,:]])
  result = []
  Y = y_test #creating y with class index
  for i in range(0,len(predictions)):
    means = []
    for j in range(0,len(dicts.keys())):
      means.append(np.mean([np.linalg.norm(predictions[i]-sample) for sample in dicts[j]]))
    #idx = [i for i, j in enumerate(means) if j == min(means)] 
    means=np.asarray(means)
    idx=np.where(means==np.amin(means))
    # print("predicted: ",idx[0][0])
    # print("actual:", Y[i])
    result.append(idx[0][0]==Y[i])
  return result

#Define identity loss
def identity_loss(y_true, y_pred):
    return K.mean(y_pred)



(X_train, Y_train), (X_test, Y_test) = load_data()

# Load saved siamese network
siamese_path='model-01_corrected_small_1Dconv_98Acc_SEQ.h5'
seq_network_trained=models.load_model(siamese_path,custom_objects={'identity_loss':identity_loss})
dictionary = createDict(X_train,Y_train,seq_network_trained,100)
predictions=getEval(X_test,Y_test,seq_network_trained,dictionary)
print("Accuracy: ",sum(predictions)/len(Y_test))
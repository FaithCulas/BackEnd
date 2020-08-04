import numpy as np
import pandas as pd
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
from collections import Counter
import csv

from keras.layers.merge import Concatenate
from keras.initializers import glorot_uniform,he_uniform

from scipy import stats
from scipy.stats import iqr, skew, kurtosis

try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle
import sys
np.set_printoptions(threshold=sys.maxsize)

def Authentication(CSI, dicti):
    #presence detection
    presence = getSVM(CSI)

    #authentication
    if presence == 1:
      feature = getFeatures(CSI,30,150)
      embeds = getEmbeddings([feature[0],feature[1],feature[2]])
      #print(res)
      #print(res.count(0),res.count(1),res.count(2),res.count(3))
      res = getEval(embeds,dictionary)
      user = most_frequent(res)

    # WORDS=pd.read_csv("users.csv")
    # values=WORDS.iloc[-1,:].values
    # CSI=values[-1]   
    #CSI = predictModel()[1]
    return user

def getSVM(arr):
    #using CSI values get features
    #using the features send into SVM
    #SVM predicts 1/0 for presence detection
    prediction = 1
    return prediction

#reshape data to (,,,1) format
def reshapeData(data):
  initShape=list(data.shape)
  initShape.append(1)
  return data.reshape(tuple(initShape))

def getFeatures(data,wind,stride):
  noReciever = data.shape[0]
  noPackets = data.shape[1]
  noSub = data.shape[2]

  #No of windows
  NoOfWindows=((noPackets-wind)//stride)+1

  features = 61
  res = np.empty((NoOfWindows,features))
  fin = np.empty((noReciever,NoOfWindows,features))
  ind = 0
  for a in data:
    
    start = 0
    end = wind
    for b in range(0,NoOfWindows):
      avg =  np.array([])
      n = np.array([])
      ent = 0
      win = a[start:end,:]
      start = start + stride
      end = end + stride

      avgSub = np.mean(win, axis=0)   #52 features
      ###################FEATURES#####################
      inter = np.append(avg,(avgSub))
      mean = np.mean(inter)
      std = np.std(inter)
      med_ad = stats.median_absolute_deviation(inter,axis=None)
      mad = np.mean(np.abs(inter - np.mean(inter, None)), None)
      intqua = iqr(inter)
      rms = np.sqrt(np.mean(inter**2))
      sk = skew(inter,None)
      kur = kurtosis(inter,None)
      ######################ENTROPY###################
      arr = np.reshape(inter, (1,len(inter)))
      m = np.amin(arr)
      M = np.amax(arr)
      sorted_arr = np.sort(arr)
      n = np.asarray(np.histogram(sorted_arr[0], bins=10, density=False))
      p = n/(noSub)
      for prob in p[0]:
        if (prob == 0):
          ent = ent + 0 
        else:
          ent = ent -1*(prob*np.log10(prob))
      avg = np.append(inter,[mean,std,med_ad,mad,intqua,rms,sk,kur,ent]) 
      res[b,:]=avg
    
    #Normalizing result
    res=np.asarray(res)
    # maxes=np.max(res,axis=0)
    # mins=np.min(res,axis=0)
    # diff=maxes-mins
    if 0 in diff:
        print("div by ZERO")
        print(diff)
    interMat=(2*res-(maxx+minn))/diff

    fin[ind,:]=interMat
    #fin[ind,:]=res
    ind = ind + 1
  
  return fin

#Define identity loss
def identity_loss(y_true, y_pred):
    return K.mean(y_pred)

def getEmbeddings(arr):
    # Load saved siamese network
    siamese_path='authstuff/model-01_corrected_small_1Dconv_98Acc_SEQ.h5'
    seq_network_trained=models.load_model(siamese_path,custom_objects={'identity_loss':identity_loss})
    #siamese_network_trained.layers
    #seq_network_trained.summary()
    embeddings=seq_network_trained.predict(arr)
    return embeddings

def most_frequent(List): 
    occurence_count = Counter(List) 
    return occurence_count.most_common(1)[0][0] 

def getEval(predictions,dicts):
  result = []
  for i in range(0,len(predictions)):
    means = []
    for j in range(0,len(dicts.keys())):
      means.append(np.mean([np.linalg.norm(predictions[i]-sample) for sample in dicts[j]]))
    #idx = [i for i, j in enumerate(means) if j == min(means)] 
    means=np.asarray(means)
    idx=np.where(means==np.amin(means))
    result.append(idx[0][0])
  return result

# def addUser(name):
#     WORDS=pd.read_csv("users.csv")
#     values=WORDS.iloc[-1,:].values
#     CSI=values[-1] 
#     presence = getSVM(CSI)
#     if presence == 1:
# 	    embed = getEmbeddings(CSI)
#     #add this embed value to the list of embeddings
#     return CSI

#getting the saved dictionary of embeddings
with open('authstuff/createDict/dict.p', 'rb') as fp:
      dictionary = pickle.load(fp)

#getting maxx,minn,diff data
maxx = np.load('authstuff/maxes.npy')
minn = np.load('authstuff/mins.npy')
diff = np.load('authstuff/diff.npy')

acc=[]
for i in range(1,5):
  for j in range(12):
    dat = np.load('authstuff/preprocessed_user'+str(i)+'/preprocessed_loc_'+str(j)+".npy")
    id = Authentication(dat,dictionary)
    acc.append(id==i-1)
    with open("predicted_user.csv","a") as fo:
      fo.write(str(id))
      fo.write("\n")
    print(id,i-1)

print("Accuracy: ",sum(acc)/len(acc))
import numpy as np
import random
from flask import json


def Authentication(CSI):
    #getting preprocessed data
    #CSI = np.ndarray(10, dtype=np.complex128) 
    amp = np.real(CSI)
    ph = np.imag(CSI)

    #presence detection
    presence = getSVM(CSI)

    #authentication
    if presence == 1:
        embed = getEmbeddings(CSI)
        user = predictUser(embed)
        #send user to front end        
    return user

def getSVM(arr):
    #using CSI values get features
    #using the features send into SVM
    #SVM predicts 1/0 for presence detection
    prediction = 1
    return prediction

def getEmbeddings(arr):
    #using CSI to get features
    #using features get embeddings
    embed = []
    return embed

def predictUser(arr):
    #checking distance make prediction
    #output user name or 'unknown'
    prediction = "roz"
    return prediction

def addUser(name, CSI):
    presence = getSVM(CSI)
    if presence == 1:
	    embed = getEmbeddings(CSI)
    #add this embed value to the list of embeddings
    return embed
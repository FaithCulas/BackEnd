import numpy as np
import random
from flask import json
import pandas as pd

def Authentication():
    WORDS=pd.read_csv("predicted_user.csv")
    values=WORDS.iloc[-1,:].values
    auth=values[-1]
    return auth

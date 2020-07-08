import pandas as pd
import numpy as np
def Activity():
    """ activities=["running1","jumping"]
    activity=activities[1]
    return activity """
    #WORDS = []
    """ with open("A:\\GIT\\Backend\\env1\\test2.csv", "r") as file:
        for line in file.readlines():
            WORDS.append(line.rstrip()) """
    WORDS=pd.read_csv('/home/lahiru/Documents/Git/BackEnd/test2.csv')
    values=WORDS.iloc[-1,:].values
    activity=values[-1]
    return activity
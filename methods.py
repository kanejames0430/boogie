# Imports
import numpy as np


def sigmoid(x):
    '''
    Used to normalize data. X is a float value. Called in normData()
    '''
    return 1/(1 + np.exp(-x))


def normData(data):
    '''
    normalizes the entries in the data matrix, such that all entries are approximately in [0,1]
    '''
    for att in data:

        # we do not want to normalize PassengerId or Survived Status
        if att != 'PassengerId' and att != 'Survived':
            mean = data[att].mean()
            std = data[att].std()
            norm_hash = [(h - mean) / std for h in data[att]]
            mapped_val = [sigmoid(hv) for hv in norm_hash]
            data[att] = mapped_val
    return data

def condit(i):
    '''
    Quick conditional for lambda function used to recategorize data.
    '''
    if i == "S":
        return 0
    elif i == "Q":
        return 1
    else:
        return 2
    
def strToFloat(data):
    '''
    Converts all entries in a dataframe to floats
    '''
    for att in data:
        data[att] = data[att].apply(lambda x: float(x))
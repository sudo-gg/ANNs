import gzip
import pickle
import numpy as np

def loadData():
    f = gzip.open('/data/mnist.pkl.gz','rb')
    trainingData, validationData, testData = pickle.load(f)
    f.close()
    return trainingData, validationData, testData

traindata, valdata, testdata = loadData()
print(traindata)
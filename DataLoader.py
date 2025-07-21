import gzip
import pickle
import numpy as np

def loadData():
    f = gzip.open('data/mnist.pkl.gz','rb')
    trainingData, validationData, testData = pickle.load(f, encoding='bytes')
    f.close()
    return trainingData, validationData, testData

def vectorizedResult(j):
    e = np.zeros((10, 1))
    e[j] = 1
    return e

def loadDataWrapper():
    """
    Returns a tuple containing the training data, validation data, and test data.
    Each of these is a list of tuples (input, result), where input is a
    784-dimensional vector (representing a flattened 28x28 image) and result
    is a 10-dimensional vector representing the one-hot encoded label.
    """
    # 50000 training samples, 10000 validation samples, 10000 test samples
    # trainingData[0] is a list of 50000 784-dimensional vectors
    # trainingData[1] is a list of 50000 labels (0-9)
    trainingData, validationData, testData = loadData()
    # Reshape is necessary to convert each 784-dimensional vector into a column vector
    trainingInputs = [np.reshape(x, (784, 1)) for x in trainingData[0]]
    trainingResults = [vectorizedResult(y) for y in trainingData[1]]
    trainingData = zip(trainingInputs, trainingResults)
    # for validiation and test data we keep the classification labels as is (an integer as opposed to a one-hot vector)
    validationInputs = [np.reshape(x, (784, 1)) for x in validationData[0]]
    validationData = zip(validationInputs, validationData[1])
    testInputs = [np.reshape(x, (784, 1)) for x in testData[0]]
    testData = zip(testInputs, testData[1])
    return list(trainingData), list(validationData), list(testData)


if __name__ == "__main__":
    traindata, valdata, testdata = loadDataWrapper()
    print(traindata[2][1]) 
    # print(vectorizedResult(5))
    # print(np.reshape(traindata[0][0], (784, 1)))
    # print(valdata)
    # print(testdata)
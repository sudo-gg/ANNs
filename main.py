import DataLoader
import ANN
import gzip
import pickle
trainingData, validationData, testData = DataLoader.loadDataWrapper()

# net = ANN.Network([784, 30, 10])

with open('network.pkl', 'rb') as f:
    net = pickle.load(f)

net.SGD(trainingData,epochs=30, miniBatchSize=100, eta=2, testData=validationData)

with open('network.pkl', 'wb') as f:
    pickle.dump(net,f, protocol=pickle.HIGHEST_PROTOCOL)

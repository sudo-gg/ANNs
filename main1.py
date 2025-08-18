import DataLoader
import ANN
import gzip
import pickle
trainingData, validationData, testData = DataLoader.loadDataWrapper()
new = True
if new:
    net = ANN.Network([784, 30, 10])
else:
    with open('network.pkl', 'rb') as f:
        net = pickle.load(f)


print(net.evaluate(testData,test=True))
net.SGD(trainingData,epochs=5, miniBatchSize=30, eta=0.5, lmbda=0.1, testData=validationData, monitorTrainingCost=True, monitorTestCost=True, monitorTrainingAccuracy=True, monitorTestAccuracy=True)
print(net.evaluate(testData,test=True))
with open('network.pkl', 'wb') as f:
    pickle.dump(net, f, protocol=pickle.HIGHEST_PROTOCOL)

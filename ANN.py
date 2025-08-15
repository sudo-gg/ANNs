import re
import numpy as np
import random

# I will look at making this more modular later
def activationFunc(z):
    return sigmoid(z)

def activationFunc_(z):
    return sigmoid_(z)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_(z):
    return sigmoid(z)*(1-sigmoid(z))

def ReLU(z):
    return np.maximum(z, 0)

def ReLU_(z):
    return (z > 0).astype(int)

def LeakyRelu(z):
    return max(0.01*z,z)

def LeakyRelu_(z):
    return 1 if z > 0 else 0.01

def tanh(z):
    return np.tanh(z)

def tanh_(z):
    return 1 - np.tanh(z)**2

def softmax(z):
    # z is a vector
    expz = np.exp(z - np.max(z)) # subtract max for numerical stability (reduces overflow risk)
    return expz / expz.sum(axis=0) # sum along the first axis (columns) (all elements in the vector)

class CrossEntropyCost:

    @staticmethod
    def fn(a,y):
        # accepts y as a one-hot vector
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))
    
    # returns the derivative of the cost function w.r.t. z (our delta vector)
    @staticmethod
    def delta(z,a,y):
        # accepts y as a one-hot vector
        # z included for generalisation
        return a-y
    
class QuadraticCost:

    @staticmethod
    def fn(a,y):
        return 0.5*np.linalg.norm(a-y)**2 
    
    @staticmethod
    def delta(z, a, y):
        return (a-y) * sigmoid_(z)

class Network:
    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.LeCunInit()
        self.cost = cost

    def largeRandomInit(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
    
    def LeCunInit(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x,y in zip(self.sizes[:-1],self.sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = activationFunc(np.dot(w, a) + b)
        return a
    
    def total_cost(self, data, test=False):
        """Calculate total cost across all samples in data"""
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if test:  # For test/validation data with integer labels
                y_vec = np.zeros((self.sizes[-1], 1))
                y_vec[y] = 1
                cost += self.cost.fn(a, y_vec)
            else:  # For training data with one-hot vectors
                cost += self.cost.fn(a, y)
        return cost / len(data)

    def SGD(self, trainingData, epochs, miniBatchSize, eta,
            lmbda=0,
            testData=None,
            monitorTestCost=False,
            monitorTestAccuracy=False,
            monitorTrainingCost=False,
            monitorTrainingAccuracy=False):

        
        if testData: 
            nTest = len(testData)
        n = len(trainingData)
        testCost, testAccuracy, trainingCost, trainingAccuracy = [], [], [], []
        
        for i in range(epochs):
            random.shuffle(trainingData)
            miniBatches = [trainingData[k:k+miniBatchSize]
                           for k in range(0, n, miniBatchSize)]
            
            for miniBatch in miniBatches:
                self.updateMiniBatch(miniBatch, eta, lmbda)
            
            print(f"Epoch {i} complete", end="")
            
            if testData:
                if monitorTestCost:
                    cost = self.total_cost(testData, test=True)
                    testCost.append(cost)
                    print(f" - Test cost: {cost:.4f}", end="")
                
                if monitorTestAccuracy:
                    acc = self.evaluate(testData)
                    testAccuracy.append(acc)
                    print(f" - Test accuracy: {acc}", end="")
                
                if monitorTrainingCost:
                    cost = self.total_cost(trainingData[:1000])  # Sample for speed
                    trainingCost.append(cost)
                    print(f" - Training cost: {cost:.4f}", end="")
                
                if monitorTrainingAccuracy:
                    acc = self.evaluate(trainingData, test=False)  # Sample for speed
                    trainingAccuracy.append(acc)
                    print(f" - Training accuracy: {acc}", end="")
            
            print() 
            
        return testCost, testAccuracy, trainingCost, trainingAccuracy
    
    def updateMiniBatch(self, miniBatch, eta, lmbda):
        nablaB = [np.zeros(b.shape) for b in self.biases]
        nablaW = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in miniBatch:
            dNablaB, dNablaW = self.backprop(x, y) 
            nablaB = [nb + dnb for nb, dnb in zip(nablaB, dNablaB)]
            nablaW = [nw + dnw for nw, dnw in zip(nablaW, dNablaW)]
        
        self.weights = [(1-eta*(lmbda/len(miniBatch)))*w - (eta/len(miniBatch))*nw
                        for w, nw in zip(self.weights, nablaW)]
        self.biases = [b - (eta/len(miniBatch))*nb
                       for b, nb in zip(self.biases, nablaB)]

    def backprop(self, x, y):
        nablaB = [np.zeros(b.shape) for b in self.biases]
        nablaW = [np.zeros(w.shape) for w in self.weights]
        
        activation = x
        activations = [x]
        zs = []
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = activationFunc(z)
            activations.append(activation)
        
        deltaL = self.cost.delta(zs[-1], activations[-1], y)
        nablaB[-1] = deltaL
        nablaW[-1] = np.dot(deltaL, activations[-2].transpose())
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            deltaL = np.dot(self.weights[-l+1].transpose(), deltaL) * activationFunc_(z)
            nablaB[-l] = deltaL
            nablaW[-l] = np.dot(deltaL, activations[-l-1].transpose())
        
        return (nablaB, nablaW)
    
    def evaluate(self, data, test=True):
        if test:
            # Test/validation data: y is integer label
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        else:
            # Training data: y is one-hot vector
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        
        return sum(int(x == y) for (x, y) in results) / len(results)
     
import numpy as np
import random

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_(z):
    return sigmoid(z)*(1-sigmoid(z))

def Relu(z):
    return max(0,z)

def LeakyRelu(z):
    return max(0.01*z,z)

class Network:
    def __init__(self,sizes):
        # so sizes = [sizelayer1,sizelayer2,ect...]
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # self.weights are matrices of weights
        self.weights = [np.random.randn(y, x) for x,y in zip(sizes[:-1],sizes[1:])]
        # y is the number of neurons in the next layer
        # x is the number of neurons in the current layer


    def feedforward(self,a):
        # applies the activation function to each layer and returns the output
        # a is a (n,1) numpy nD array input
        # (basically run the ANN on the input function)
        for b,w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def SGD(self,trainingData,epochs,miniBatchSize,eta,testData=None):
        # trainingData is a list of tuples (x,y)
        # epochs is the number of times the trainingData is passed through the network
        # eta (η) is the learning rate (the step size)
        if testData: 
            nTest = len(testData)
        n = len(trainingData)
        for i in range(epochs):
            random.shuffle(trainingData)
            miniBatches = [trainingData[k:k+miniBatchSize]
                           for k in range(0,n,miniBatchSize)]
            for miniBatch in miniBatches:
                # updates the weights and biases according to a single iteration
                # of mini-batch gradient descent
                self.updateMiniBatch(miniBatch,eta)
            if testData:
                print(f"Epoch {i}: {self.evaluate(testData)} / {nTest}")
            else:
                print(f"Epoch {i} complete")
    
    def updateMiniBatch(self, miniBatch, eta):
        # ∇b and ∇w are lists of numpy arrays the same length as self.biases and self.weights
        # np.zeros accepts a tuple as an argument and returns an array of zeros with the same shape
        nablaB = [np.zeros(b.shape) for b in self.biases]
        nablaW = [np.zeros(w.shape) for w in self.weights]
        for x, y in miniBatch:
            # so x is the training example and y is the training label (x is input and y is output)
            dNablaB, dNablaW = self.backprop(x, y) 
            #adds all the contributions to the nablaB and nablaW which were initally empty as
            # this is the change in our biases and weights
            nablaB = [nb + dnb for nb, dnb in zip(nablaB, dNablaB)]
            nablaW = [nw + dnw for nw, dnw in zip(nablaW, dNablaW)]
        # update the weights and biases based on our calculated step
        self.weights = [w-(eta/len(miniBatch))*nw
                        for w, nw in zip(self.weights, nablaW)]
        self.biases = [b - (eta/len(miniBatch))*nb
                       for b, nb in zip(self.biases, nablaB)]

    def backprop(self,x, y):
        # returns a tuple (NablaB, NablaW) representing the gradient for the cost function C_x
        # where NablaB and NablaW are lists of numpy arrays with the same dimensions as self.weights and self.biases to be 
        # added as you can see above in updateMiniBatch
        nablaB = [np.zeros(b.shape) for b in self.biases]
        nablaW = [np.zeros(w.shape) for w in self.weights]
        # 1)activation contains the outputs of each layer (a(z))
        activation = x
        activations = [x] # list to store all the activations
        zs = [] # list to store all the z vectors
        # 2) feedforward (essentially we just let the neural network do its thing and record the values
        # of each layer)
        for b, w in zip(self.biases,self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # 3) output error (error for last layer hence last
        # activation being compared with desired output)
        # so this error vector is the proportional change to the biases
        # (and weights when accounting for the activation of the previous layer)
        # refer to below for what partial C_x / partial a is
        deltaL = self.cost_(activations[-1],y) * sigmoid_(zs[-1])
        # 4) backward pass
        nablaB[-1] = deltaL
        nablaW[-1] = np.dot(deltaL,activations[-2].transpose())
        # -l = -2, -3, -4, etc. so we are going backwards through the layers
        # we start at the second to last layer and go backwards (we have last layers error)
        for l in range(2, self.num_layers):
            z = zs[-l]
            delta_l = np.dot(self.weights[-l+1].transpose(), deltaL) * sigmoid_(z)
            nablaB[-l] = delta_l # grad for the biases
            nablaW[-l] = np.dot(delta_l, activations[-l-1].transpose())
        return (nablaB, nablaW)

    def cost_(self, outputActivations, y):
        # partial C_x / partial a
        return (outputActivations-y)
    
  # Will add cross entropy cost function later

    def evaluate(self,testData):
        testResults = [(np.argmax(self.feedforward(x)), y) for (x, y) in testData]
        return sum(int(x == y) for (x, y) in testResults)
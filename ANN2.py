import re
import numpy as np
import random
import matplotlib.pyplot as plt
import json
from datetime import datetime

from DataLoader import loadData

def activationFunc(z):
    return sigmoid(z)

def activationFunc_(z):
    return sigmoid_(z)

def sigmoid(z):
    # Clip z to prevent overflow/underflow (sigmoid saturation)
    z = np.clip(z, -5, 5)  
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_(z):
    s = sigmoid(z)
    return s * (1 - s)

def ReLU(z):
    return np.maximum(z, 0)

def ReLU_(z):
    # Derivative of ReLU: 1 if z > 0, else 0
    return (z > 0).astype(float)

def LeakyRelu(z):
    return np.maximum(0.01*z, z)

def LeakyRelu_(z):
    # Derivative: 1 if z > 0, else 0.01
    return np.where(z > 0, 1.0, 0.01)

def tanh(z):
    return np.tanh(z)

def tanh_(z):
    return 1 - np.tanh(z)**2

def softmax(z):
    # z is a vector
    expz = np.exp(z - np.max(z)) # subtract max for numerical stability (reduces overflow risk)
    return expz / expz.sum(axis=0) # sum along the first axis (columns) (all elements in the vector)

# Activation function mapping for easy switching between different activation types
ACTIVATIONS = {
    'sigmoid': (sigmoid, sigmoid_),
    'relu': (ReLU, ReLU_),
    'tanh': (tanh, tanh_),
    'leaky_relu': (LeakyRelu, LeakyRelu_)
}

REGULARIZERS = {
    'l1': lambda w, lmbda: lmbda * np.sign(w),
    'l2': lambda w, lmbda: lmbda * w
}

class CrossEntropyCost:
    @staticmethod
    def fn(a, y):
        # accepts y as a one-hot vector
        # Clip a to prevent log(0) which would cause NaN values
        # Claude reccommended this as opposed to np.nan_to_num
        a = np.clip(a, 1e-15, 1 - 1e-15)  
        return np.sum(-y * np.log(a) - (1-y) * np.log(1-a))
    
    # returns the derivative of the cost function w.r.t. z (our delta vector)
    @staticmethod
    def delta(z, a, y):
        # accepts y as a one-hot vector
        # z included for generalisation
        return a - y
    
class QuadraticCost:
    @staticmethod
    def fn(a, y):
        return 0.5 * np.linalg.norm(a - y)**2 
    
    @staticmethod
    def delta(z, a, y):
        return (a - y) * sigmoid_(z)

class SGDOptimizer:
    """Standard SGD optimizer - basic gradient descent with optional regularization"""
    def update(self, weights, biases, nablaW, nablaB, eta, lmbda, miniBatchSize, regularizer=None):
        # nablaW/B (nw/nb) are the sums of the gradients over the mini-batch so still need to divide by miniBatchSize
        if regularizer == 'l1':
            newWeights = [w - eta(nw/miniBatchSize + lmbda * np.sign(w)/n) for w, nw in zip(weights, nablaW)]
        elif regularizer == 'l2':
            # Standard weight update with L2 regularization term
            newWeights = [(1 - eta * (lmbda / n)) * w - (eta / miniBatchSize) * nw
                        for w, nw in zip(weights, nablaW)]
        else:
            newWeights = [w - (eta / miniBatchSize) * nw
                          for w, nw in zip(weights, nablaW)]
        newBiases = [b - (eta / miniBatchSize) * nb
                     for b, nb in zip(biases, nablaB)]
        return newWeights, newBiases

class MomentumOptimizer:
    """SGD with momentum - helps escape local minima and accelerates convergence
    by accumulating gradients from previous steps"""
    def __init__(self, momentum=0.85):
        self.momentum = momentum
        self.velocityW = None  # velocity for weights
        self.velocityB = None  # velocity for biases
    
    def update(self, weights, biases, nablaW, nablaB, eta, lmbda, miniBatchSize, regularizer=None):
        # Initialize velocities on first call
        if self.velocityW is None:
            self.velocityW = [np.zeros_like(w) for w in weights]
            self.velocityB = [np.zeros_like(b) for b in biases]
        
        # Update bias velocities: v = momentum * v - learning_rate * gradient
        self.velocityB = [self.momentum * vb - (eta / miniBatchSize) * nb
                    for vb, nb in zip(self.velocityB, nablaB)]
        
        # Update weights: w = (1 - regularization) * w + velocity
        if regularizer == 'l2':
            self.velocityW = [self.momentum * vw - eta*(nw / miniBatchSize + (lmbda / n) * w) for vw, nw, w in zip(self.velocityW, nablaW, weights)]
        elif regularizer == 'l1':
            self.velocityW = [self.momentum * vw - eta*(nw / miniBatchSize + (lmbda / n) * np.sign(w)) for vw, nw, w in zip(self.velocityW, nablaW, weights)]
        else:
            self.velocityW = [self.momentum * vw - (eta / miniBatchSize) * nw
                    for vw, nw in zip(self.velocityW, nablaW)]
        newWeights = [w + vw for w, vw in zip(weights, self.velocityW)]

        newBiases = [b + vb for b, vb in zip(biases, self.velocityB)]
        
        return newWeights, newBiases

class AdamOptimizer:
    """Adam optimizer - combines momentum with adaptive learning rates
    Maintains running averages of both gradients (first moment) and squared gradients (second moment)
    No L1/L2 regularization implemented"""
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1      # decay rate for first moment
        self.beta2 = beta2      # decay rate for second moment  
        self.epsilon = epsilon  # small constant to prevent division by zero
        self.mW = None         # first moment estimate for weights
        self.vW = None         # second moment estimate for weights
        self.mB = None         # first moment estimate for biases
        self.vB = None         # second moment estimate for biases
        self.t = 0             # time step counter
    
    def update(self, weights, biases, nablaW, nablaB, eta, lmbda, miniBatchSize, regularizer=None):
        # REGULARIZER FOR GENERALITY NOT ACTUALLY USED
        # Initialize moment estimates on first call
        if self.mW is None:
            self.mW = [np.zeros_like(w) for w in weights]
            self.vW = [np.zeros_like(w) for w in weights]
            self.mB = [np.zeros_like(b) for b in biases]
            self.vB = [np.zeros_like(b) for b in biases]
        
        self.t += 1
        
        # Update biased first and second moment estimates
        # m_t = beta1 * m_{t-1} + (1 - beta1) * gradient
        self.mW = [self.beta1 * mw + (1 - self.beta1) * nw
                   for mw, nw in zip(self.mW, nablaW)]
        # v_t = beta2 * v_{t-1} + (1 - beta2) * gradient^2
        self.vW = [self.beta2 * vw + (1 - self.beta2) * (nw ** 2)
                   for vw, nw in zip(self.vW, nablaW)]
        self.mB = [self.beta1 * mb + (1 - self.beta1) * nb
                   for mb, nb in zip(self.mB, nablaB)]
        self.vB = [self.beta2 * vb + (1 - self.beta2) * (nb ** 2)
                   for vb, nb in zip(self.vB, nablaB)]
        
        # Bias correction - compensates for initialization bias towards zero
        mWHat = [mw / (1 - self.beta1 ** self.t) for mw in self.mW]
        vWHat = [vw / (1 - self.beta2 ** self.t) for vw in self.vW]
        mBHat = [mb / (1 - self.beta1 ** self.t) for mb in self.mB]
        vBHat = [vb / (1 - self.beta2 ** self.t) for vb in self.vB]
        
        # Update weights: w = w - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
        newWeights = [(1 - eta * (lmbda / miniBatchSize)) * w - 
                      eta * mwHat / (np.sqrt(vwHat) + self.epsilon)
                      for w, mwHat, vwHat in zip(weights, mWHat, vWHat)]
        newBiases = [b - eta * mbHat / (np.sqrt(vbHat) + self.epsilon)
                     for b, mbHat, vbHat in zip(biases, mBHat, vBHat)]
        
        return newWeights, newBiases

class LearningRateScheduler:
    """Learning rate scheduling strategies - helps fine-tune training by adjusting learning rate over time"""
    
    @staticmethod
    def exponentialDecay(initialLr, epoch, decayRate=0.95):
        # Exponentially decrease learning rate: lr = initial_lr * decay_rate^epoch
        return initialLr * (decayRate ** epoch)
    
    @staticmethod
    def stepDecay(initialLr, epoch, dropRate=0.5, epochsDrop=10):
        # Step-wise decay: drop learning rate by dropRate every epochsDrop epochs
        return initialLr * (dropRate ** (epoch // epochsDrop))
    
    @staticmethod
    def cosineAnnealing(initialLr, epoch, maxEpochs):
        # Smooth cosine decay from initial_lr to 0 over maxEpochs
        return initialLr * 0.5 * (1 + np.cos(np.pi * epoch / maxEpochs))

class Network:
    
    def __init__(self, sizes, cost=CrossEntropyCost, activation='sigmoid'):
        # so sizes = [sizelayer1,sizelayer2,ect...]
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost = cost
        
        # Set activation functions allows switching between sigmoid, ReLU, tanh, etc.
        if activation in ACTIVATIONS:
            self.activationFunc, self.activationFunc_ = ACTIVATIONS[activation]
        else:
            self.activationFunc, self.activationFunc_ = sigmoid, sigmoid_
        
        self.LeCunInit()
        
        # Training history stores metrics for visualization and analysis
        self.history = {
            'trainCost': [],
            'trainAccuracy': [],
            'testCost': [],
            'testAccuracy': [],
            'learningRates': []
        }

    # Weight Initializers
    def largeRandomInit(self):
        # self.weights are matrices of weights
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
        # y is the number of neurons in the next layer
        # x is the number of neurons in the current layer

    def LeCunInit(self):
        # This division by np.sqrt(x) makes the weights more evenly distributed and reduces the chance of overfitting/saturation
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) 
                       for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def HeInit(self):
        """Initialize weights using He initialization - better for ReLU networks apparently
        Uses sqrt(2/fanIn)"""
        self.biases = [np.zeros((y, 1)) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) * np.sqrt(2.0 / x) 
                       for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def XavierInit(self):
        """Initialize weights using Xavier initialization - good for tanh/sigmoid
        Uses sqrt(1/fanIn)"""
        self.biases = [np.zeros((y, 1)) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) * np.sqrt(1.0 / x) 
                       for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        # applies the activation function to each layer and returns the output
        # a is a (n,1) numpy nD array input
        # (basically run the ANN on the input function)
        for b, w in zip(self.biases, self.weights):
            a = self.activationFunc(np.dot(w, a) + b)
        return a
    
    def totalCost(self, regularizer, optimizer, data, test=False, lmbda=0):
        """Calculate total cost across all samples including L2 regularization
        test=True means data has integer labels, test=False means one-hot vectors"""
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if test:  # For test/validation data with integer labels
                yVec = np.zeros((self.sizes[-1], 1))
                yVec[y] = 1
                cost += self.cost.fn(a, yVec)
            else:  # For training data with one-hot vectors
                cost += self.cost.fn(a, y)
        
        # Add L2 regularization term: lambda/2 * sum(w^2) for all weights
        if regularizer and optimizer == 'SGD':
            if regularizer == 'l1':
                regularizationCost = 0.5 * lmbda * sum(np.sum(w**2) for w in self.weights)
            elif regularizer == 'l2':
                regularizationCost = 0.5 * lmbda * sum(np.sum(w**2) for w in self.weights)
        else:
            regularizationCost = 0
        return (cost + regularizationCost) / len(data)

    def SGD(self, trainingData, epochs, miniBatchSize, eta,
            lmbda=0,
            testData=None,
            lrScheduler=None,
            optimizer = None,
            optimizerParams = {},
            regularizer = None,
            earlyStopping=None,
            monitorTestCost=False,
            monitorTestAccuracy=False,
            monitorTrainingCost=False,
            monitorTrainingAccuracy=False):
        # trainingData is a list of tuples (x,y)
        # epochs is the number of times the trainingData is passed through the network
        # eta (η) is the learning rate (the step size)
        # lmbda (λ) is the regularization parameter
        # lrScheduler: function that takes (initialLr, epoch) and returns new learning rate
        # earlyStopping: number of epochs to wait for improvement before stopping
        
        # Initialize optimizer based on choice
        # optimizer: 'sgd', 'momentum', or 'adam'
        # optimizerParams: dictionary of optimizer-specific parameters
        if optimizer == 'momentum':
            opt = MomentumOptimizer(**(optimizerParams or {})) # ** unpacks the dictionary into arguments
        elif optimizer == 'adam':
            opt = AdamOptimizer(**(optimizerParams or {}))
        else:
            opt = SGDOptimizer()
        
        if testData: 
            nTest = len(testData)
        global n
        n = len(trainingData)
        
        # Early stopping variables - tracks best performance to prevent overfitting
        bestAccuracy = 0
        patienceCounter = 0
        
        for i in range(epochs):
            # Learning rate scheduling - allows dynamic adjustment of learning rate
            currentLr = eta
            if lrScheduler:
                currentLr = lrScheduler(eta, i)
            
            random.shuffle(trainingData)
            miniBatches = [trainingData[k:k+miniBatchSize]
                           for k in range(0, n, miniBatchSize)]
            
            for miniBatch in miniBatches:
                # updates the weights and biases according to a single iteration
                # of mini-batch gradient descent using the chosen optimizer
                self.updateMiniBatch(miniBatch, currentLr, lmbda, opt, regularizer)
            
            # Monitoring and logging training progress
            print(f"Epoch {i+1:3d}", end="")
            
            if monitorTrainingAccuracy:
                trainAcc = self.evaluate(trainingData, test=False)
                self.history['trainAccuracy'].append(trainAcc)
                print(f" | Train Acc: {trainAcc:.3f}", end="")
            
            if monitorTrainingCost:
                trainCost = self.totalCost(data=trainingData[:5000], regularizer=regularizer, optimizer=optimizer, lmbda=lmbda)  # Sample for speed
                self.history['trainCost'].append(trainCost)
                print(f" | Train Cost: {trainCost:.4f}", end="")
            
            if testData:
                if monitorTestAccuracy:
                    testAcc = self.evaluate(testData)
                    self.history['testAccuracy'].append(testAcc)
                    print(f" | Test Acc: {testAcc:.3f}", end="")
                    
                    # Early stopping implementation - stops training if no improvement
                    if earlyStopping and testAcc > bestAccuracy:
                        bestAccuracy = testAcc
                        patienceCounter = 0
                    elif earlyStopping:
                        patienceCounter += 1
                        if patienceCounter >= earlyStopping:
                            print(f"\nEarly stopping at epoch {i+1}")
                            break
                
                if monitorTestCost:
                    testCost = self.totalCost(data=testData,regularizer=regularizer, optimizer=optimizer, test=True, lmbda=lmbda)
                    self.history['testCost'].append(testCost)
                    print(f" | Test Cost: {testCost:.4f}", end="")
            
            # Store learning rate for plotting
            self.history['learningRates'].append(currentLr)
            print(f" | LR: {currentLr:.4f}")
        
        return self.history
    
    def updateMiniBatch(self, miniBatch, eta, lmbda, optimizer, regularizer=None):
        # ∇b and ∇w are lists of numpy arrays the same length as self.biases and self.weights
        # np.zeros accepts a tuple as an argument and returns an array of zeros with the same shape
        # lambda is the regularization parameter
        nablaB = [np.zeros(b.shape) for b in self.biases]
        nablaW = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in miniBatch:
            # so x is the training example and y is the training label (x is input and y is output)
            dNablaB, dNablaW = self.backprop(x, y) 
            #adds all the contributions to the nablaB and nablaW which were initally empty as
            # this is the change in our biases and weights
            nablaB = [nb + dnb for nb, dnb in zip(nablaB, dNablaB)]
            nablaW = [nw + dnw for nw, dnw in zip(nablaW, dNablaW)]
        
        # Use selected optimizer to update weights and biases
        self.weights, self.biases = optimizer.update(
            self.weights, self.biases, nablaW, nablaB, eta, lmbda, len(miniBatch), regularizer
        )

    def backprop(self, x, y):
        # returns a tuple (NablaB, NablaW) representing the gradient for the cost function C_x wrt self.biases and self.weights
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
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activationFunc(z)
            activations.append(activation)
        
        # 3) output error (error for last layer hence last
        # activation being compared with desired output)
        # so this error vector is the proportional change to the biases
        # (and weights when accounting for the activation of the previous layer)
        # refer to below for what partial C_x / partial a is
        deltaL = self.cost.delta(zs[-1], activations[-1], y)
        # 4) backward pass
        # derivatives of biases and weights using the formula, [-2] as l-1 layer
        nablaB[-1] = deltaL
        nablaW[-1] = np.dot(deltaL, activations[-2].transpose())
        # -l = -2, -3, -4, etc. so we are going backwards through the layers
        # we start at the second to last layer and go backwards (we have last layers error)
        for l in range(2, self.num_layers):
            z = zs[-l]
            deltaL = np.dot(self.weights[-l+1].transpose(), deltaL) * self.activationFunc_(z)
            nablaB[-l] = deltaL # grad for the biases
            nablaW[-l] = np.dot(deltaL, activations[-l-1].transpose())
        
        return (nablaB, nablaW)
    
    def evaluate(self, Data, test=True):
        # Gives the number of correct predictions (the ANNs output is the neuron with the highest activation)
        # In the test/validation data our y is one hot encoded so we compare the argmax of the output with the argmax of the label (so comparing the class as a number)
        # in the training data the y is the class (so comparing the class as a number)
        if test:
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in Data]
        else:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in Data]
        
        return sum(int(x == y) for (x, y) in results) / len(Data)
    
    def plotTrainingHistory(self, savePath=None):
        """Plot comprehensive training history with 4 subplots showing different metrics
        savePath: optional path to save the plot as an image file"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot accuracy over epochs - shows learning progress
        if self.history['trainAccuracy']:
            axes[0, 0].plot(self.history['trainAccuracy'], label='Training Accuracy', color='blue')
        if self.history['testAccuracy']:
            axes[0, 0].plot(self.history['testAccuracy'], label='Test Accuracy', color='red')
        axes[0, 0].set_title('Accuracy vs Epochs')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot cost/loss over epochs - shows convergence
        if self.history['trainCost']:
            axes[0, 1].plot(self.history['trainCost'], label='Training Cost', color='blue')
        if self.history['testCost']:
            axes[0, 1].plot(self.history['testCost'], label='Test Cost', color='red')
        axes[0, 1].set_title('Cost vs Epochs')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Cost')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot learning rate schedule - shows how lr changes over time
        if self.history['learningRates']:
            axes[1, 0].plot(self.history['learningRates'], color='green')
            axes[1, 0].set_title('Learning Rate vs Epochs')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True)
        
        # Plot overfitting indicator - train accuracy - test accuracy
        # Positive values indicate overfitting (model memorizing training data)
        print(self.history['trainAccuracy'], self.history['testAccuracy'])
        if self.history['trainAccuracy'] and self.history['testAccuracy']:
            diff = [train - test for train, test in 
                   zip(self.history['trainAccuracy'], self.history['testAccuracy'])]
            axes[1, 1].plot(diff, color='purple')
            axes[1, 1].set_title('Training - Test Accuracy (Overfitting Indicator)')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy Difference')
            axes[1, 1].grid(True)
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if savePath:
            plt.savefig(savePath, dpi=300, bbox_inches='tight')
        plt.show()
    
    def saveModel(self, filename):
        """Save model weights, biases, architecture and training history to JSON file
        
        Why JSON instead of pickle?
        1. Human readable - you can open and inspect the file in any text editor
        2. Language independent - can be loaded by other programming languages  
        3. Safer - pickle can execute arbitrary code when loading, JSON cannot
        4. Version stable - pickle files can break between Python versions
        5. Debugging friendly - easy to examine weights/structure manually
        6. Smaller file size for network data compared to pickle's overhead
        
        The only downside is JSON doesn't natively support NumPy arrays, so we convert
        to Python lists (arrays are reconstructed when loading)"""
        modelData = {
            'weights': [w.tolist() for w in self.weights],  # Convert numpy arrays to lists
            'biases': [b.tolist() for b in self.biases],
            'sizes': self.sizes,
            'history': self.history,
            'timestamp': datetime.now().isoformat()  # Track when model was saved
        }
        
        with open(filename, 'w') as f:
            json.dump(modelData, f, indent=2)  # indent=2 makes it more readable according to claude
        print(f"Model saved to {filename}")
    
    def loadModel(self, filename):
        """Load model weights, biases, architecture and training history from JSON file
        Reconstructs NumPy arrays from the stored lists"""
        with open(filename, 'r') as f:
            modelData = json.load(f)
        
        # Reconstruct numpy arrays from lists
        self.weights = [np.array(w) for w in modelData['weights']]
        self.biases = [np.array(b) for b in modelData['biases']]
        self.sizes = modelData['sizes']
        self.history = modelData['history']
        print(f"Model loaded from {filename}")

# Example usage functions showing different training configurations

def trainWithMomentum(trainingData, testData):
    """Example: Training with momentum optimizer - helps escape local minima"""
    net = Network([784, 100, 30, 10], activation='sigmoid')
    
    return net.SGD(
        trainingData, epochs=20, miniBatchSize=32, eta=0.5,
        lmbda=0.1, testData=testData,
        optimizerParams={'momentum': 0.9},  # momentum coefficient
        monitorTestAccuracy=True,
        monitorTrainingAccuracy=True,
        monitorTestCost=True,
        monitorTrainingCost=True
    )

def trainWithAdamAndScheduling(trainingData, testData):
    """Example: Training with Adam optimizer and exponential learning rate decay"""
    net = Network([784, 128, 64, 10], activation='relu')
    net.HeInit()  # He initialization works better with ReLU
    
    return net.SGD(
        trainingData, epochs=30, miniBatchSize=64, eta=0.001,
        lmbda=0.01, testData=testData,
        lrScheduler=lambda lr, epoch: LearningRateScheduler.exponentialDecay(lr, epoch, 0.95), # lambda so doesnt run here
        earlyStopping=5,  # Stop if no improvement for 5 epochs
        monitorTestAccuracy=True,
        optimizer='adam',
        optimizerParams={'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8},
        monitorTrainingAccuracy=True,
        monitorTestCost=True,
        monitorTrainingCost=True
    )

def compareOptimizers(trainingData, testData):
    """Compare performance of different optimizers on the same network architecture
    Returns dictionary with accuracy curves for each optimizer"""
    results = {}
    
    # Standard SGD baseline
    print("Training with SGD...")
    netSgd = Network([784, 30, 10])
    histSgd = netSgd.SGD(trainingData, 15, 32, 0.5, testData=testData,
                         monitorTestAccuracy=True)
    results['SGD'] = histSgd['testAccuracy']
    
    # SGD with Momentum
    print("Training with Momentum...")
    netMomentum = Network([784, 30, 10])
    histMomentum = netMomentum.SGD(trainingData, 15, 32, 0.5, testData=testData,
                                   optimizer='momentum', monitorTestAccuracy=True)
    results['Momentum'] = histMomentum['testAccuracy']
    
    # Adam optimizer
    print("Training with Adam...")
    netAdam = Network([784, 30, 10])
    histAdam = netAdam.SGD(trainingData, 15, 32, 0.001, testData=testData,
                           optimizer='adam', monitorTestAccuracy=True)
    results['Adam'] = histAdam['testAccuracy']
    
    # Plot comparison of all optimizers
    plt.figure(figsize=(10, 6))
    for name, accuracy in results.items():
        plt.plot(accuracy, label=name, linewidth=2)
    plt.title('Optimizer Comparison - Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return results


if __name__ == "__main__":
    # Example usage
    from DataLoader import loadDataWrapper
    trainingData, valData, testData = loadDataWrapper()
    net = Network([784, 30, 10])
    
    # Train with momentum
    print("Training with momentum...")
    histMomentum = net.SGD(trainingData, 15, 32, 0.1, testData=testData,
                           optimizer='momentum',regularizer='L2', lmbda=0.01, monitorTestAccuracy=True, monitorTestCost=True, monitorTrainingAccuracy=True, monitorTrainingCost=True)
    net.plotTrainingHistory()
    # Compare optimizers
    # compareOptimizers(trainingData, testData)
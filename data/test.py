import numpy as np
def Relu(z):
    return np.maximum(z, 0)

def Relu_(z):
    return (z > 0).astype(int)

x = np.array([-1, 0, 1, 2])
print("Relu:", Relu(x))
print("Relu_:", Relu_(x))

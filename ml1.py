import numpy as np
np.random.seed(0)

xs = np.arange(-8, 9) * 0.25                             
ys = xs ** 2   

H = 8 # number of hidden neurons

w1 = np.random.uniform(-1, 1, H)    
b1 = np.random.uniform(-1, 1, H)
w2 = np.random.uniform(-1, 1, H)
b2 = 0.0
def relu(z):
    return np.maximum(0.0, z)
 
def relu_grad(z):
    return (z > 0).astype(float)
lr = 0.02
for epoch in range(1,3001):
    total_loss = 0.0
    for x,y in zip(xs, ys):
        z = w1 * x + b1                                
        h = relu(z)
        out = w2 @ h + b2   
        loss = 0.5 * (out - y) ** 2
        total_loss =+ loss
        d2 = out - y
        d1 = w2 * d2 * relu_grad(z)
        w2 = w2 - lr * d2 * h
        w1 = w1 - lr * d1 * x
        b1 = b1 - lr * d1
        b2 = b2 - lr * d2
    
  


def predict(x):
    h = relu(w1 * x + b1)
    return w2 @ h + b2


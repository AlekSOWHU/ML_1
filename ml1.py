import numpy as np                      # arrays
np.random.seed(0)                       # repeatable randomness

xs = np.arange(-8, 9) * 0.25            # inputs −2…2
ys = xs ** 2                            # targets x²

H = 8                                   # hidden neurons

w1 = np.random.uniform(-1, 1, H)        # input weights
b1 = np.random.uniform(-1, 1, H)        # hidden biases
w2 = np.random.uniform(-1, 1, H)        # output weights
b2 = 0.0                                # output bias

def relu(z):                            # activation
    return np.maximum(0.0, z)           # cut negatives

def relu_grad(z):                       # ReLU slope
    return (z > 0).astype(float)        # 1 or 0

lr = 0.02                               # step size
for epoch in range(1, 3001):            # epochs
    total_loss = 0.0                    # reset sum
    for x, y in zip(xs, ys):            # each example
        z = w1 * x + b1                 # hidden input
        h = relu(z)                     # hidden output
        out = w2 @ h + b2               # network output
        loss = 0.5 * (out - y) ** 2     # squared error
        total_loss += loss              # add up  
        d2 = out - y                    # output error
        d1 = w2 * d2 * relu_grad(z)     # hidden errors
        w2 = w2 - lr * d2 * h           # update w2
        w1 = w1 - lr * d1 * x           # update w1
        b1 = b1 - lr * d1               # update b1
        b2 = b2 - lr * d2               # update b2

def predict(x):                         # test function
    h = relu(w1 * x + b1)               # hidden output
    return w2 @ h + b2                  # network output

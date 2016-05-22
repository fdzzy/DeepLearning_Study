#!/usr/bin/python

'''
http://cs231n.github.io/neural-networks-case-study/
'''

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline  

N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K, D)) # data matrix (eacho row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels

# some hyperparameters
step_size = 1e-0 # learning rate
reg = 1e-3  # regularization strength
training_steps = 200 # number of training steps

# initialize parameters randomly
W = 0.01 * np.random.randn(D, K)
b = np.zeros((1, K))

def generateData():
    for j in xrange(K):
        idx = range(N*j, N*(j+1))
        r = np.linspace(0.0, 1, N) # radius
        t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2 # theta
        X[idx] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[idx] = j

    # lets visualize the data
    #plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    #print X, y

def softmaxLinearClassifierTraining():
    global X, y, W, b

    num_examples = X.shape[0]
    for i in xrange(training_steps):
        # evaluate class scores, [N x K]
        scores = np.dot(X, W) + b
        
        # compute the class probabilities
        # get unnomalized probabilities
        exp_scores = np.exp(scores)
        # normalize them for each example
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims = True)
        
        #compute the loss: average cross-entropy loss and regularization
        correct_logprobs = -np.log(probs[range(num_examples), y])    
        data_loss = np.sum(correct_logprobs) / num_examples
        reg_loss = 0.5 * reg * np.sum(W*W)
        loss = data_loss + reg_loss
        if i % 10 == 0:
            print "iteration %d: loss %f" % (i, loss)
        
        # compute the gradient on scores
        dscores = probs
        dscores[range(num_examples), y] -= 1
        dscores /= num_examples
        
        # backpropgate the gradient to the parameters(W, b)
        dW = np.dot(X.T, dscores)
        db = np.sum(dscores, axis=0, keepdims=True)
        dW += reg * W # don't forget the regularization gradient
        
        # perform a parameter update
        W += - step_size * dW
        b += - step_size * db
    
    # evaluate training set accuracy
    scores = np.dot(X, W) + b
    predicted_class = np.argmax(scores, axis=1)
    print 'training accuracy: %.2f' % (np.mean(predicted_class == y))
        
    
def main():
    generateData()
    softmaxLinearClassifierTraining()
    
main()
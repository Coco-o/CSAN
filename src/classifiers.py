from __future__ import division
from __future__ import print_function
import numpy as np
from util import *

activation_dict = {
    'sigmoid': (sigmoid, sigmoid_grad),
    'tanh': (tanh, tanh_grad),
    'relu': (relu, relu_grad)
}

class WeightedBinaryNN(object):
    '''
    Fully connected neural network with importance weights over samples
    required parameters: layers, weight_decay, activation
    '''
    def __init__(self, parameters):
        layers = parameters['layers']
        self.layers = layers
        self.layer_num = len(layers) # num of hidden layers + 2 (including the input x and output y)
        self.weights = []
        self.biases = []
        self.weight_decay = parameters['weight_decay'] # weight_decay is per-point decay
        self.activate = activation_dict[parameters['activation']][0]
        self.activation_grad = activation_dict[parameters['activation']][1]
        # initialize weights:
        for i in range(self.layer_num - 1):
            self.weights.append(np.zeros((layers[i+1], layers[i])))
            self.biases.append(np.zeros(layers[i+1]))

    def random_init(self):
        for i in range(self.layer_num - 1):
            shape = self.weights[i].shape
            b = np.sqrt(6 / (shape[0]+shape[1]))
            self.weights[i] = np.random.uniform(-b, b, shape)
            self.biases[i] = np.random.uniform(-b, b, shape[0])

    def train_batch(self, X, y, importance, learning_rate):
        '''
        mini-batch with immportance weights
        X: batch_size * dimension
        y: batch_size * 1
        importance: batch_size * 1 = data_size * normalized_weightes
        '''
        batch_size = X.shape[0]
        layer_inputs = []
        layer_inputs.append(X)
        # forward computation:
        for i in range(self.layer_num - 2):
            W = self.weights[i]
            b = self.biases[i]
            output = self.activate(np.dot(layer_inputs[-1], W.T) + np.tile(b, (batch_size, 1)))
            layer_inputs.append(output)
        # forward computation: prediction layer
        W = self.weights[-1]
        b = self.biases[-1]
        y_pred = sigmoid(np.dot(layer_inputs[-1], W.T) + np.tile(b, (batch_size, 1)))
        # back propagation:
        pre_activation_grad = (y_pred - y) * importance / X.shape[0]
        W_grad = np.dot(pre_activation_grad.T, layer_inputs[-1]) + self.weight_decay * self.weights[-1]
        b_grad = np.sum(pre_activation_grad, 0)
        post_activation_grad = np.dot(pre_activation_grad, self.weights[-1])
        self.weights[-1] -= learning_rate * W_grad
        self.biases[-1] -= learning_rate * b_grad
        for i in range(self.layer_num-2, 0, -1):
            pre_activation_grad = self.activation_grad(layer_inputs[i]) * post_activation_grad
            W_grad = np.dot(pre_activation_grad.T, layer_inputs[i-1]) + self.weight_decay * self.weights[i-1]
            b_grad = np.sum(pre_activation_grad, 0)
            post_activation_grad = np.dot(pre_activation_grad, self.weights[i-1])
            self.weights[i-1] -= learning_rate * W_grad
            self.biases[i-1] -= learning_rate * b_grad

    def train_epoch(self, X, y, importance, learning_rate, batch_size):
        rounds = X.shape[0] // batch_size
        importance = importance / np.sum(importance) * X.shape[0]
        for i in range(rounds):
            st, ed = i * batch_size, i * batch_size + batch_size
            self.train_batch(X[st:ed, :], y[st:ed, :], importance[st:ed, :], learning_rate)


    def predict(self, X):
        '''
        input: X, batch_size * dim
        output: y, batch_size * 1
        '''
        batch_size = X.shape[0]
        layer_input = X
        for i in range(self.layer_num - 2):
            W = self.weights[i]
            b = self.biases[i]
            layer_input = self.activate(np.dot(layer_input, W.T) + np.tile(b, (batch_size, 1)))
        # prediction layer
        W = self.weights[-1]
        b = self.biases[-1]
        y_pred = sigmoid(np.dot(layer_input, W.T) + np.tile(b, (batch_size, 1)))
        return y_pred

    def update_with_grad_on_pred(self, x, pred_grad, learning_rate):
        '''
        Given x, gradient on prediction(x), update weights using backprop
        '''
        batch_size = x.shape[0]
        layer_inputs = []
        layer_inputs.append(x)
        # forward computation:
        for i in range(self.layer_num - 2):
            W = self.weights[i]
            b = self.biases[i]
            output = self.activate(np.dot(layer_inputs[-1], W.T) + np.tile(b, (batch_size, 1)))
            layer_inputs.append(output)
        # forward computation: prediction layer
        W = self.weights[-1]
        b = self.biases[-1]
        y_pred = sigmoid(np.dot(layer_inputs[-1], W.T) + np.tile(b, (batch_size, 1)))
        # back propagation:
        # only the starting gradient is different from the stantard cross_entropy minimization
        pre_activation_grad = sigmoid_grad(y_pred) * pred_grad / x.shape[0] # normalize
        W_grad = np.dot(pre_activation_grad.T, layer_inputs[-1]) + self.weight_decay * self.weights[-1]
        b_grad = np.sum(pre_activation_grad, 0)
        post_activation_grad = np.dot(pre_activation_grad, self.weights[-1])
        self.weights[-1] -= learning_rate * W_grad
        self.biases[-1] -= learning_rate * b_grad
        for i in range(self.layer_num-2, 0, -1):
            pre_activation_grad = self.activation_grad(layer_inputs[i]) * post_activation_grad
            W_grad = np.dot(pre_activation_grad.T, layer_inputs[i-1]) + self.weight_decay * self.weights[i-1]
            b_grad = np.sum(pre_activation_grad, 0)
            post_activation_grad = np.dot(pre_activation_grad, self.weights[i-1])
            self.weights[i-1] -= learning_rate * W_grad
            self.biases[i-1] -= learning_rate * b_grad

#class MulticlassNN(object):





'''
class DAE(object):
    #Autoencoder
    def __init__(self):
        self.W = np.zeros((100, 784))
        self.hidden = 100
        self.dim = 784
        self.b = np.ones(100)
        self.c = np.ones(784)
        self.log = []
        self.drop_prob = 0


    def config(self, p):
        self.hidden = p['hidden']
        self.dim = p['dim']
        self.drop_prob = p['drop_prob']
        self.W = np.zeros((self.hidden, self.dim))
        self.b = np.ones(self.hidden)
        self.c = np.ones(self.dim)

    def random_init(self):
        shape = self.W.shape
        b = np.sqrt(6 / (shape[0]+shape[1]))
        self.W = np.random.uniform(-b, b, shape)
        self.b = np.random.uniform(-b, b, self.hidden)
        self.c = np.random.uniform(-b, b, self.dim)

    def h_output(self, X):
        N = data_size(X)
        if N == 1:
            b = self.b
        else:
            b = np.tile(self.b, (N,1))
        return sigmoid(np.dot(X, self.W.T) + b)

    def x_output(self, H):
        N = data_size(H)
        if N == 1:
            c = self.c
        else:
            c = np.tile(self.c, (N,1))
        return sigmoid(np.dot(H, self.W) + c)


    def train_epoch(self, Xtrain, rate):
        for i in range(Xtrain.shape[0]):
            x = Xtrain[i,:]
            #add dropout
            mask = np.random.binomial(1, 1-self.drop_prob, self.dim)
            x_drop = x * mask
            #forward:
            h = self.h_output(x_drop)
            y = self.x_output(h)
            #back prop
            pre_y_grad = y - x
            W_grad_1 = np.outer(h, pre_y_grad)
            h_grad = np.dot(self.W, pre_y_grad)
            pre_h_grad = h_grad * h * (1-h)
            W_grad_2 = np.outer(pre_h_grad, x_drop)
            #update
            self.W -= rate * (W_grad_1 + W_grad_2)
            self.c -= rate * pre_y_grad
            self.b -= rate * pre_h_grad


    def train(self, Xtrain, Xvalid, rate, epoch, modelfile):
        train_size = Xtrain.shape[0]
        idx = np.random.permutation(train_size)
        Xtrain = Xtrain[idx,:]
        for i in range(epoch):
            self.train_epoch(Xtrain, rate)
            #cross entropies:
            ytrain = self.x_output(self.h_output(Xtrain))
            yvalid = self.x_output(self.h_output(Xvalid))
            loss_train = cross_entropy(ytrain, Xtrain)
            loss_valid = cross_entropy(yvalid, Xvalid)

            print('epoch ', i, ': training_loss ', loss_train,
                ',validation_loss ', loss_valid)
            self.log.append([loss_train, loss_valid])
            if modelfile != '':
                self.write_model(modelfile)

    def write_model(self, filename):
        w_file = filename + '.W'
        b_file = filename + '.b'
        c_file = filename + '.c'
        np.savetxt(w_file, self.W, delimiter=',')
        np.savetxt(b_file, self.b, delimiter=',')
        np.savetxt(c_file, self.c, delimiter=',')


    def write_log(self, filename):
        f = open(filename, 'w')
        for i in range(len(self.log)):
            f.write(str(i+1) + ',' + ','.join(str(x) for x in self.log[i]) + '\n')

'''

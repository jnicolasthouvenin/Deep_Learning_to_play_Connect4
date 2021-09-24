
import numpy as np
import math

from encoder import *

class NeuralNetwork:
    
    def __init__(self, *args):
        if len(args) == 1:
            shape, learning_rate = args[0], 0.05

            self.size = len(shape)
            self.shape = shape
            self.l_r = learning_rate
            self.biases = []
            self.weights = []
            for prev_layer, layer in zip(self.shape[:-1], self.shape[1:]):
                b = np.squeeze(np.random.randn(layer, 1))
                self.biases.append(b)
                w = np.random.randn(layer, prev_layer)
                self.weights.append(w)
        elif len(args) == 2:
            shape, learning_rate = args[0], args[1]
            
            self.size = len(shape)
            self.shape = shape
            self.l_r = learning_rate
            self.biases = []
            self.weights = []
            for prev_layer, layer in zip(self.shape[:-1], self.shape[1:]):
                b = np.squeeze(np.random.randn(layer, 1))
                self.biases.append(b)
                w = np.random.randn(layer, prev_layer)
                self.weights.append(w)
        else:
            self.size = args[0]
            self.shape = args[1]
            self.l_r = args[2]
            self.biases = args[3]
            self.weights = args[4]

    def train(self, x, y):
        y_pred = self.forward(x)
        nabla_b, nabla_w = self.backprop(x, y)
        self.update(nabla_b, nabla_w)
        return y_pred

    def forward(self, a):
        self.zs = []
        self.activations = [np.array(a)]
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            self.zs.append(z)
            a = sigmoid(z)
            self.activations.append(np.array(a))
        return a

    def backprop(self, x, y):
        self.forward(x)

        gradient_bias = [np.zeros(b.shape) for b in self.biases]
        gradient_weights = [np.zeros(w.shape) for w in self.weights]

        # last layer
        delta = cost_derivative(self.activations[-1], y) * sigmoid_derivative(self.zs[-1])
        gradient_bias[-1] = delta
        gradient_weights[-1] = computeGradientW(self.activations[-2],delta,len(self.zs[-1]))

        # from before last layer to first layer
        # last layer is self.size-2
        # before last layer is self.size-3
        for l in range(self.size - 3, -1, -1):
            delta = np.dot(self.weights[l + 1].T, delta) * sigmoid_derivative(self.zs[l])
            gradient_bias[l] = delta
            # len(activation) == len(weights)+1
            # activation[i] is the previous activations to the layer weights[i]
            #delta_w = np.dot(delta, self.activations[l].T)
            gradient_weights[l] = computeGradientW(self.activations[l],delta,len(self.zs[l]))

        return gradient_bias, gradient_weights

    def update(self, nabla_b, nabla_w):
        self.biases = [b - self.l_r * nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - self.l_r * nw for w, nw in zip(self.weights, nabla_w)]

    def train_sgd(self, x_train, y_train, batch_size=20):
        x_batches = [ x_train[i : i + batch_size] for i in range(0, len(x_train), batch_size) ]
        y_batches = [ y_train[i : i + batch_size] for i in range(0, len(y_train), batch_size) ]

        for x_batch, y_batch in zip(x_batches,y_batches):
            gradient_bias = [np.zeros(b.shape) for b in self.biases]
            gradient_weights = [np.zeros(w.shape) for w in self.weights]

            for x, y in zip(x_batch, y_batch):
                delta_grad_b, delta_grad_w = self.backprop(x, y)
                gradient_bias = [ nb + dnb for nb, dnb in zip(gradient_bias, delta_grad_b) ]
                gradient_weights = [ nw + dnw for nw, dnw in zip(gradient_weights, delta_grad_w) ]
            
            gradient_weights = [nw / batch_size for nw in gradient_weights]
            gradient_bias = [nb / batch_size for nb in gradient_bias]

            self.weights = [ w - self.l_r * nw for w, nw in zip(self.weights, gradient_weights) ]
            self.biases = [ b - self.l_r * nb for b, nb in zip(self.biases, gradient_bias) ]

    def supervised_learning(self,x_train,y_train,x_test,y_test,lenTest,it,EPOCH=100,batch_size=1000,dataset="classic",file="networks/",write=False):
    
        print("[INIT] - classification rate =",self.evaluate(x_test,y_test))

        for j in range(EPOCH+1):
            # train
            shuffler = np.random.permutation(x_train.shape[0])
            x_train = x_train[shuffler]
            y_train = y_train[shuffler]
            self.train_sgd(x_train,y_train,batch_size=batch_size)

            # test
            goodPred = 0
            preds = 0
            for i in range(lenTest):
                label_pred = ENCODER.encode_prediction(self.forward(x_test[i]))
                if (label_pred == y_test[i]).all():
                    goodPred += 1
                preds += 1

            print(j," - classification rate =",self.evaluate(x_test,y_test))
            if j%10 == 0:
                if write:
                    self.save((file+dataset+"_"+str(it)+"_"+str(j)))

    def evaluate(self, x_test, y_test):
        test_results = [ (ENCODER.encode_prediction(self.forward(_x)), (_y)) for _x, _y in zip(x_test, y_test) ]
        result = sum(int(_y_pred == _y) for (_y_pred, _y) in test_results)
        result /= len(y_test)
        return round(result, 3)

    def save(self, fileName):
        file = open(fileName, "w")
        file.write(str(self.size)+"\n")
        for i in range(self.size):
            file.write(str(self.shape[i])+"\n")
        file.write(str(self.l_r)+"\n")
        for i in range(1, self.size):
            for j in range(self.shape[i]):
                for k in range(self.shape[i-1]):
                    file.write(str(self.weights[i-1][j][k])+"\n")
        for i in range(self.size-1):
            for j in range(self.shape[i+1]):
                file.write(str(self.biases[i][j])+"\n")
        
        file.close()

def readNeuralNetwork(fileName):
    file = open(fileName, "r")
    size = int(file.readline())
    shape = []
    for i in range(size):
        shape.append(int(file.readline()))
    l_r = float(file.readline())
    weights = []
    for i in range(1, size):
        weights.append([])
        for j in range(shape[i]):
            weights[i-1].append([])
            for k in range(shape[i-1]):
                weights[i-1][j].append(float(file.readline()))
                
    biases = []
    for i in range(size-1):
        biases.append([])
        for j in range(shape[i+1]):
            biases[i].append(float(file.readline()))
    
    file.close()
    return NeuralNetwork(size, shape, l_r, [np.array(obj) for obj in biases], [np.array(obj) for obj in weights])

def cost(a, y):
    return (a - y) ** 2


def cost_derivative(a, y):
    return 2*(a - y)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def computeGradientW(a,delta,repeat):
    aBuffer = a.transpose()
    aBuffer = np.tile(a,repeat).reshape(repeat,len(a)).transpose()
    g_w = np.multiply(aBuffer,delta).transpose()
    return g_w
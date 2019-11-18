import math
import sys
import operator
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def sigmoid(linear_combination):
    return 1 / (1 + np.exp(np.negative(linear_combination)))

def sigmoid_deriv(output):
    return sigmoid(output) * (1 - sigmoid(output))

def evaluate(y_pred, y_true):
    return ((y_pred - y_true)**2).mean(1)
    #TODO: Generic scoring the network's output versus the y for classification or regression (MSE or log loss)

class NN():
    def __init__(self, shapes, lr, epoch):
        """
        shape is passed in as an array of n
        """
        self.epsilon = lr #Learning rate
        self.epochs = epoch

        self.depth = len(shapes)
        self.shapes = shapes

        self.init_weights()
        self.init_bias()

    def init_weights(self):
        #Reset the weights of the neural network
        self.weights = []
        for i in range(1, self.depth):
            weight = np.random.randn(self.shapes[i], self.shapes[i-1])*np.sqrt(2/self.shapes[i-1])
            self.weights.append(weight) 
        self.weights = np.array(self.weights)
            
    def init_bias(self, bias=1):
        self.bias = []
        for i in range(1, self.depth):
            self.bias.append(np.random.randn(self.shapes[i], 1)*np.sqrt(2))
        self.bias = np.array(self.bias)

    def forward_pass(self, inputs):
        #Passes input through the neural net, saves the activations, and returns output values
        if(len(inputs) != self.shapes[0]):
            raise ValueError('Wrong dimension of input passed')
            
        activation = inputs
        activations = [inputs]
        combinations = []
        for w, b in zip(self.weights, self.bias):
            z = np.add(np.dot(w, activation), b)
            
            combinations.append(z)
            activation = sigmoid(z)
            
            activations.append(activation)
        return combinations, activations
    
    def backward_pass(self, x, y): 
        """
        returns: tuple (grad_w, grad_b) where elements are the gradient
        for the cost wrt weight and bias respectively. grad_w and grad_b are lists of numpy arrays.
        """
        delta_b = [np.zeros(b.shape) for b in self.bias]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        
        combinations, activations = self.forward_pass(x)
        delta = (activations[-1] - y) * sigmoid_deriv(combinations[-1])
        delta_w[-1] = np.matmul(delta, activations[-2].transpose())
        delta_b[-1] = delta

        ### TODO: Error testing
        #Iterating through deep layers       
        for i in range(2, self.depth): 
            z = combinations[-i]
            
            temp = np.dot(self.weights[-i+1].transpose(), delta)
            delta = temp * sigmoid_deriv(combinations[-i])

            delta_w[-i] = np.matmul(delta, activations[-i-1].transpose())
            delta_b[-i] = delta
            
        return delta_w, delta_b

    def predict(self, inputs):
        #A quick and light version of forward pass: no instance attributes changed
        if(len(inputs) != self.shapes[0]):
            raise ValueError('Wrong dimension of input passed')
            
        activation = inputs

        for w, b in zip(self.weights, self.bias):
            z = np.dot(w, activation)+ b
            activation = sigmoid(z)               
        return activation
    
    def train_nn(self, x, y):
        for k in range(self.epochs+1):
            delta_weights, delta_bias = self.backward_pass(x,y)
            
            #Initialize Adam params 
            beta_1 = 0.8
            beta_2 = 0.99
            mw = [np.zeros(w.shape) for w in self.weights]
            vw = [np.zeros(w.shape) for w in self.weights]
            mb = [np.zeros(b.shape) for b in self.bias]
            vb = [np.zeros(w.shape) for w in self.bias]
            eps = 1e-4

            for i in range(self.depth-1):
                mw[i] = beta_1*mw[i] + (1-beta_1)*delta_weights[i]
                vw[i] = beta_2*vw[i] + (1-beta_2)*(np.square(delta_weights[i]))
                self.weights[i] = self.weights[i] - self.epsilon* mw[i] / (np.sqrt(vw[i]) + eps)

            for i in range(self.depth-1):
                mb[i] = beta_1*mb[i] + (1-beta_1)*delta_bias[i]
                vb[i] = beta_2*vb[i] + (1-beta_2)*(np.square(delta_bias[i]))
                self.bias[i] = self.bias[i] - self.epsilon* mb[i] / (np.sqrt(vb[i]) + eps)
            
            for i in range(self.depth-1):
                self.weights[i] = self.weights[i] - self.epsilon * delta_weights[i]
                self.bias[i] = self.bias[i] - self.epsilon * delta_bias[i]
                
            if k % 50 == 0:
                combinations, activations = self.forward_pass(x) 
                y_pred = activations[-1]
                print("MSE", evaluate(y_pred, y))
                
        combinations, activations = self.forward_pass(x) 
        return activations[-1]    

    def train(self, train, epochs, batch_size, test_data=None):
        if test_data: 
            pass

        for i in range(epochs):
            random.shuffle(train)
            mini_batch_list = []
            for j in range(0, len(train), batch_size): #Iterate through all data by interval of batch size
                mini_batch_list.append(training_data[j:j+batch_size])

                for batch in mini_batch_list():
                    self.adam_optimize(batch, lr=0.2)

def adam_optimize(self, batch, lr):
        """
        Backpropogates the valeus for each batch, averages the gradients, and runs and adam optimizer to update
        the instance's weight matrix
        lr: learn rate
        """
        avg_delta_w = np.zeros(self.weights.shape)
        avg_delta_b = np.zeros(self.bias.shape)

        for x in batch:###Find out how data is formatted (for x and y)
            new_d_w, new_d_b = backward_pass(self, x, y)
            avg_delta_w = avg_delta_w + new_d_w
            avg_delta_b = avg_delta_b + new_d_b

        avg_delta_w /= len(batch)
        avg_delta_b /= len(batch) #By size of batch we just iterated through

        #Initialize Adam params 
        beta_1 = 0.9
        beta_2 = 0.999
        m = np.zeros(self.weights.shape)
        v = np.zeros(self.weights.shape)

        for i in range(self.depth):
            m[i] = beta_1*m[i] + (1-beta_1)*avg_delta_w[i]
            v[i] = beta_2*v[i] + (1-beta_2)*(np.square(avg_delta_w[i]))
            self.weights[i] = self.weights[i] - lr*(m[i] / (v[i] + 0.01))

        for i in range(self.depth):
            m[i] = beta_1*m[i] + (1-beta_1)*avg_delta_b[i]
            v[i] = beta_2*v[i] + (1-beta_2)*(np.square(avg_delta_b[i]))
            self.bias[i] = self.bias[i] - lr*(m[i] / (v[i] + 0.01))

def test_0():
    test_net = NN([2, 128, 128, 3], 0.1, 1000)
    
    test_x = np.array([[0.5, 1/2], [0.92, 0.73]])
    test_y = np.array([[1/3, 1/1, 1/2],[0.65, 0.25, 0.15]])
    
    test_x = np.asarray(test_x).T  
    test_y = np.asarray(test_y).T 
    print("data shape", test_x.shape, test_y.shape)

    y_pred = test_net.train_nn(test_x, test_y)
    print("Test 0\n----y_pred----\n", y_pred, "\n----y_true----\n", test_y)

def test_1():
    data = loadmat('hw2_data.mat')

    test_x = data['X1']
    test_y = data['Y1']/255.0
    test_x[:, 0] = (test_x[:, 0] - np.mean(test_x[:, 0])) / np.std(test_x[:, 0])
    test_x[:, 1] = (test_x[:, 1] - np.mean(test_x[:, 1])) / np.std(test_x[:, 1])
    
    test_x = np.asarray(test_x).T  
    test_y = np.asarray(test_y).T  

    print("data shape", test_x.shape, test_y.shape)
 
    test_net = NN([2, 256, 128, 3], 0.001, 1000)
    y_pred = test_net.train_nn(test_x, test_y)
    
    row = max(data['X1'][:,0])
    col = max(data['X1'][:,1])
    print("row", row)
    print("column", col)

    global matrix
    matrix = np.zeros((int(row)+1, int(col)+1, 3))

    for i in range(int(row)+1):
        for j in range(int(col)+1):
            for k in range(3):
                 matrix[i][j][k] = y_pred[k, i*(int(col)+1) + j]
                
    print("RGB", matrix)
    plt.imshow(matrix)

def test_2():
    data = loadmat('data.mat')

    test_x = data['X2']
    test_y = data['Y2']/255.0
    test_x[:, 0] = (test_x[:, 0] - np.mean(test_x[:, 0])) / np.std(test_x[:, 0])
    test_x[:, 1] = (test_x[:, 1] - np.mean(test_x[:, 1])) / np.std(test_x[:, 1])
    
    test_x = np.asarray(test_x).T  
    test_y = np.asarray(test_y).T  

    print("data shape", test_x.shape, test_y.shape)
 
    test_net = NN([2, 256, 128, 3], 0.001, 1000)
    
    #Iterate through all data by interval of batch size
    batch_size = 1024
    for i in range(len(data) // batch_size):
        x_batch = test_x[i*batch_size, (i+1)*batch_size]
        y_batch = test_y[i*batch_size, (i+1)*batch_size]        
 
        mini_batch_list.append(training_data[j:j+batch_size])
        batch_pred = test_net.train_nn(x_batch, y_batch)
        y_pred = np.vstack((y_pred, batch_pred.T))
        
    row = max(data['X2'][:,0])
    col = max(data['X2'][:,1])
    print("row", row)
    print("column", col)

    global matrix
    matrix = np.zeros((int(row)+1, int(col)+1, 3))

    for i in range(int(row)+1):
        for j in range(int(col)+1):
            for k in range(3):
                 matrix[i][j][k] = y_pred[k, i*(int(col)+1) + j]
                
    print("RGB", matrix)
    
    plt.imshow(matrix)

if __name__ == "__main__":
    test_0()
  #  test_1()
  #  test_2()
    

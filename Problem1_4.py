"""
Problem 1.4 and Problem 1.5
N= {Set of 5}
p = 5
"""
import numpy as np
import mnist_reader
from mlxtend.preprocessing import one_hot
import math
import matplotlib.pyplot as plt

class NN(object):
    def __init__(self,hidden_dims=(512, 512),n_hidden=2,mode='train',datapath=None,model_path=None, mini_batch_size=64):
        self.total_layers = 2 + n_hidden # input + ouput+ n_hidden
        self.mini_batch_size = mini_batch_size
        self.neuron_counts = np.array([784, hidden_dims[0], hidden_dims[1], 10])
        self.initialize_weights_uniform_distribution()
        return
    
    def initialize_weights_uniform_distribution(self):
        self.biases = [np.zeros((y, 1)) for y in self.neuron_counts[1:]]
        self.weights = [np.random.uniform(-(math.sqrt(6/(x+y))), (math.sqrt(6/(x+y))), (y, x)) for x, y in zip(self.neuron_counts[:-1], self.neuron_counts[1:])]
        return
        
    def forward(self,input):
        activation = input
        count = 1
        activations = [input]        
        pre_activations = []     
        for bias, wweight in zip(self.biases, self.weights):
            pre_activation = np.dot(wweight, activation)+bias
            pre_activations.append(pre_activation)
            if(count == self.total_layers-1): #Ouput layer is softmax 
                activation = self.softmax_activation(pre_activation)
            else:
                activation = self.relu_activation(pre_activation)
            count = count + 1
            activations.append(activation)
        return(activations, pre_activations)
    def feed_forward(self, input):
        activation = input
        count = 1       
        for bias, wweight in zip(self.biases, self.weights):
            pre_activation = np.dot(wweight, activation)+bias
            if(count == self.total_layers-1): #Ouput layer is softmax 
                activation = self.softmax_activation(pre_activation)
            else:
                activation = self.relu_activation(pre_activation)
            count = count + 1
        return(activation)
    def relu_derivative(self, input):
        return np.greater(input, 0).astype(int) # Relu
    def relu_activation(self,input):
        return np.maximum(input, 0) #Relu
    def softmax_activation(self,input):
        ex = np.exp(input - np.max(input)) #Numerically stable version of softmax
        return ex / ex.sum()
    def backward(self,input,label):
        local_biases = [np.zeros(b.shape) for b in self.biases]
        local_weights = [np.zeros(w.shape) for w in self.weights]
        input = input.reshape(len(input), 1)
        #Forward pass
        activations, pre_activations = self.forward(input)    
        #Error at last layer
        delta = activations[-1] - one_hot([label], 10).reshape(10, 1)
        local_biases[-1] = delta
        local_weights[-1] = np.dot(delta, activations[-2].transpose())    
        for l in range(2, self.total_layers):
            z = pre_activations[-l]
            sp = self.relu_derivative(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            local_biases[-l] = delta
            local_weights[-l] = np.dot(delta, activations[-l-1].transpose())
        return (local_biases, local_weights)
    def update_mini_batch(self, image_batches, labels_batches, learning_rate):
        local_biases = [np.zeros(b.shape) for b in self.biases]
        local_weights = [np.zeros(w.shape) for w in self.weights]
        for x, y in zip(image_batches, labels_batches):
            delta_local_biases, delta_local_weights = self.backward(x, y)
            local_biases = [nb+dnb for nb, dnb in zip(local_biases, delta_local_biases)]
            local_weights = [nw+dnw for nw, dnw in zip(local_weights, delta_local_weights)]
        local_weights = np.divide(local_weights, len(labels_batches))
        return local_weights    
    def cross_entropy(self, activation, label):
        target = one_hot([label], 10).reshape(10,1)
        return -np.sum(np.log(activation) * target)
def PrintAnalytical_Gradients(images, labels, object):
    learning_rate = 0.001
    weights_gradients = object.update_mini_batch(images, labels, learning_rate);
    grad= np.hstack([weights_gradients[1].ravel()])
    return grad
def GetCost_Batch(images, labels, object):
    cost = 0.0
    for i in range(len(labels)):
        input = images[i].reshape(len(images[i]), 1)
        activation = object.feed_forward(input)
        cost = cost + object.cross_entropy(activation, labels[i])
    cost = cost/len(labels)
    return -cost
def PrintNumerical_Gradients(images, labels, object):
    actual_weights = object.weights.copy()
    actual_baises = object.biases.copy()
    weights_change = np.zeros(actual_weights[1].shape)
    gradient_matrix = np.zeros(actual_weights[1].shape)
    x, y = weights_change.shape 
    for i in range(x):
        for j in range(y):
            change_small = 1/len(labels)
            object.weights = actual_weights.copy()
            object.biases = actual_baises.copy()
            weights_change[i][j] = change_small
            object.weights[1] = object.weights[1] + weights_change
            loss_1 = GetCost_Batch(images, labels, object)
            object.weights = actual_weights.copy()
            object.biases = actual_baises.copy()
            object.weights[1] = object.weights[1] - weights_change
            loss_2 = GetCost_Batch(images, labels, object)
            gradient_matrix[i][j] = (loss_2 - loss_1) / (2*change_small)
            weights_change[i][j] = 0    
    grad= np.hstack([gradient_matrix.ravel()])
    return grad

X_train, y_train = mnist_reader.load_mnist (r'', kind = 'train')
net = NN()
p = 5
input_array = np.array([10, 20, 30, 40, 50])
max_difference = []
for j in range(len(input_array)):
    layer_2_analytical = PrintAnalytical_Gradients(X_train[:input_array[j]], y_train[:input_array[j]], net)
    layer_2_numerical = PrintNumerical_Gradients(X_train[:input_array[j]], y_train[:input_array[j]], net)
    print("\nN = ", input_array[j])
    print('Analytical Gradient', 'Numerical Gradient')
    for i in range(len(layer_2_analytical[:p])):
        print(layer_2_analytical[i], layer_2_numerical[i])
    max_difference.append(abs(np.max(layer_2_analytical- layer_2_numerical)))
plt.plot(input_array,max_difference, 'r--')
plt.ylabel('max_difference')
plt.xlabel('N')
plt.show()
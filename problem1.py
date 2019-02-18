"""
Problem 1.1
"""
import numpy as np
import mnist_reader
from mlxtend.preprocessing import one_hot

class NN(object):
    def __init__(self,hidden_dims=(512, 512),n_hidden=2,mode='train',datapath=None,model_path=None):
        self.total_layers = 2 + n_hidden # input + ouput+ n_hidden
        self.neuron_counts = np.array([784, hidden_dims[0], hidden_dims[1], 10])
        self.initialize_weights()
        return
    def initialize_weights(self):
        self.biases = [np.random.uniform(low=0.0005, high=0.001, size=(y,1)) for y in self.neuron_counts[1:]]
        self.weights = [np.random.uniform(low=0.0005, high=0.001, size=(y,x)) for x, y in zip(self.neuron_counts[:-1], self.neuron_counts[1:])]
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
                activation = self.sigmoid_activation(pre_activation)
            count = count + 1
            activations.append(activation)
        return(activations, pre_activations)
    def sigmoid_derivative(self, input):
        return self.sigmoid_activation(input)*(1-self.sigmoid_activation(input))
    def softmax_derivative(self, input): #Need to implement
        soft_deriv = np.zeros([len(input), 1])
        for i in range(len(input)):
            soft_deriv[i] = input[i][0]*(1-input[i][0])
        return soft_deriv
    def sigmoid_activation(self,input):
        return 1.0/(1.0+np.exp(-input))
    def crossentropy_derivative(self, output_activations, y):
        act_output = one_hot([y], 10)
        return((-np.divide(act_output,output_activations.reshape(1,10))).reshape(10, 1)) 
    def softmax_activation(self,input):
        ex = np.exp(input - np.max(input))
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
            sp = self.sigmoid_derivative(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            local_biases[-l] = delta
            local_weights[-l] = np.dot(delta, activations[-l-1].transpose())
        return (local_biases, local_weights)
    def update(self,bias_grads, weight_grads, learning_rate, input_len):
        self.biases = [b-(learning_rate/input_len)*nb for b, nb in zip(self.biases, bias_grads)]
        self.weights = [w-(learning_rate/input_len)*nw for w, nw in zip(self.weights, weight_grads)]
        return
    def train(self, images, labels, learning_rate=0.001):
        local_biases = [np.zeros(b.shape) for b in self.biases]
        local_weights = [np.zeros(w.shape) for w in self.weights]
        for x, y in zip(images, labels):
            delta_local_biases, delta_local_weights = self.backward(x, y)
            local_biases = [nb+dnb for nb, dnb in zip(local_biases, delta_local_biases)]
            local_weights = [nw+dnw for nw, dnw in zip(local_weights, delta_local_weights)]
        self.update(local_biases, local_weights, learning_rate, len(images))
        return
    def train_SGD(self, images, labels, learning_rate=0.001):
        local_biases = [np.zeros(b.shape) for b in self.biases]
        local_weights = [np.zeros(w.shape) for w in self.weights]
        for x, y in zip(images, labels):
            delta_local_biases, delta_local_weights = self.backward(x, y)
            #Update parameter after finding gradients for every sample
            self.update(local_biases, local_weights, learning_rate, len(images))
        return
    def test(self):
        return
X_train, y_train = mnist_reader.load_mnist (r'', kind = 'train')

net = NN()
net.train(X_train, y_train)
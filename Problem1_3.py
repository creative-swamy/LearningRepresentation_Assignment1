"""
Problem 1.3
Batch Size = 64
Number hidden layers : 2 "512 *512"
Learning Rate : 0.001
Validation Set: 10000

"""
import numpy as np
import mnist_reader
from mlxtend.preprocessing import one_hot
from sklearn.utils import shuffle
import math
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
    def backward(self,input,label, lambda_hyper = 0.001):
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
    def update(self,bias_grads, weight_grads, learning_rate, input_len):
        self.biases = [b-(learning_rate/input_len)*nb for b, nb in zip(self.biases, bias_grads)]
        self.weights = [w-(learning_rate/input_len)*nw for w, nw in zip(self.weights, weight_grads)]
        return    
    def update_mini_batch(self, image_batches, labels_batches, learning_rate):
        local_biases = [np.zeros(b.shape) for b in self.biases]
        local_weights = [np.zeros(w.shape) for w in self.weights]
        for x, y in zip(image_batches, labels_batches):
            delta_local_biases, delta_local_weights = self.backward(x, y)
            local_biases = [nb+dnb for nb, dnb in zip(local_biases, delta_local_biases)]
            local_weights = [nw+dnw for nw, dnw in zip(local_weights, delta_local_weights)]
        self.update(local_biases, local_weights, learning_rate, len(labels_batches))        
        return    
    def train_SGD_min_batch(self, images, labels, validation_img, validation_labels, learning_rate=0.001, epochs=100):
        total_images = len(labels)
        for i in range(epochs):
            print("Epoch:", i)
            training_data, training_label = shuffle(images, labels, random_state=0)
            image_batches = [training_data[k:k+self.mini_batch_size] for k in range(0, total_images, self.mini_batch_size)]
            labels_batches = [training_label[l:l+self.mini_batch_size] for l in range(0, total_images , self.mini_batch_size)]
            #Update parameters for every batch
            for x, y in zip(image_batches, labels_batches):
                self.update_mini_batch(x, y, learning_rate)
            print("Correct predictions training", self.test(images, labels))
            print("Correct predictions validation", self.test(validation_img, validation_labels))
        return    
    def cross_entropy(self, activation, label):
        target = one_hot([label], 10).reshape(10,1)
        return -np.sum(np.log(activation) * target)    
    def test(self, data, labels):
        test_results = [(np.argmax(self.feed_forward(x.reshape(len(x), 1))), y) for (x, y) in zip(data, labels)]
        return sum(int(x == y) for (x, y) in test_results)
X_train, y_train = mnist_reader.load_mnist (r'', kind = 'train')
net = NN()
net.train_SGD_min_batch(X_train[:50000], y_train[:50000],X_train[50000:60000], y_train[50000:60000])



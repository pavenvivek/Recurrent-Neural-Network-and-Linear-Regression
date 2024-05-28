#!/usr/local/bin/python3
 
import numpy as np


def construct_data(sequence_len):
    
    time_x = []
    value_y = []
    xmin = 0
    xmax = 12.8 #16.28
    num_points = 200

    data = np.sin(np.linspace(xmin,xmax,num_points))
    
    for i in range(100):
        time_x.append(data[i:i+sequence_len])
        value_y.append(data[i+sequence_len])
        
    time_x = np.array(time_x)
    value_y = np.array(value_y)
    
    return (time_x, value_y)

class neural_network(object):      
    
    def __init__(self, seq_len):

        self.input_neurons = seq_len 
        self.hidden_layer_neurons = 1200   
        self.output_neurons = 1
        
        self.weights_U = np.ones([self.hidden_layer_neurons, seq_len])
        self.weights_W = np.random.uniform(0, 1, (self.hidden_layer_neurons, self.hidden_layer_neurons))
        self.weights_V = np.random.uniform(0, 1, (self.output_neurons, self.hidden_layer_neurons))

        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoidPrime(self, x):
        return x * (1 - x)
    
    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def tanhPrime(self, x):
        return 1 - (tanh(x) ** 2)
    

def feed_forward_and_back_propogate(rnn, time_x, value_y, b, c, sequence_len):

    EPOCHS = 25
    learning_rate = 0.0001
    gradient_lower_lmt = -3
    gradient_upper_lmt = 3
    e = 0

    while e < EPOCHS:        
        error = 0
        for i in range(value_y.shape[0]):
            x, y = time_x[i], value_y[i]
        
            prev_h = np.zeros((rnn.hidden_layer_neurons))
            hidden_layer_ouput = []
            
            # feed-forward step
            for j in range(0, sequence_len):
                input_x = np.zeros(x.shape)
                input_x[j] = x[j]
                U_x = np.dot(rnn.weights_U, input_x)
                W_h_prev = np.dot(rnn.weights_W, prev_h)
                a = W_h_prev + U_x + b
                h_curr = rnn.sigmoid(a)
                o = np.dot(rnn.weights_V, h_curr)
                yhat = o + c  # activation is identity function at output layer
                hidden_layer_ouput.append(h_curr)
                prev_h = h_curr
                
            # back-propogate step
            d1 = (yhat - y)
            d2 = np.zeros(rnn.weights_V.shape)
            gradient = np.zeros(rnn.weights_V.shape)
            
            for j in range(0, sequence_len):
                d2 = hidden_layer_ouput[j] * d1[0]
                gradient = gradient + d2
                
                #adjusting gradients to prevent from exploding
                if gradient.max() > gradient_upper_lmt:
                    gradient[gradient > gradient_upper_lmt] = gradient_upper_lmt
                if gradient.min() < gradient_lower_lmt:
                    gradient[gradient < gradient_lower_lmt] = gradient_lower_lmt
            
            # updating weights V
            rnn.weights_V = rnn.weights_V - (learning_rate * gradient)
            
            #error calculation
            sample_error = 0.5 * ((y - yhat) ** 2)
            error = error + sample_error                

        print("itr -> {}, error -> {}".format(e + 1, error))
        e = e + 1
    
    
# Main Function
if __name__ == "__main__":

    print("Recurrent Neural Networks:")

    b = 0 #bias 1
    c = 0 #bias 2
    seq_len = 10 #input sequence length
    
    (time_x, value_y) = construct_data(seq_len)
    rnn = neural_network(seq_len)
    feed_forward_and_back_propogate(rnn, time_x, value_y, b, c, seq_len)

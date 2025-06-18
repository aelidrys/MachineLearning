import random
import math
import numpy as np



class NeuralNetwork:
    def __init__(self, hidden_layer_weights, output_layer_weights, activation_type = 'poly', lr = 0.00001):
        self.lr = lr  # learning rate
        self.hidden_layer = NeuronLayer(hidden_layer_weights, activation_type)
        self.output_layer = NeuronLayer(output_layer_weights, activation_type)

    def feed_forward(self, input):
        outputs = self.hidden_layer.feed_forward(input)
        # print(f'outputs: {outputs}')
        outputs = self.output_layer.feed_forward(outputs)
        # print(f'outputs: {outputs}')
        return outputs

    def compute_delta(self, target):
        # Base case rule for output layer   
        self.dE_dO_out = np.zeros(len(self.output_layer.neurons), dtype=float)
        self.dE_dO_net = np.zeros(len(self.output_layer.neurons), dtype=float)
        for i, neuron in enumerate(self.output_layer.neurons):
            self.dE_dO_out[i] = neuron.output - target[i]
            self.dE_dO_net[i] = self.dE_dO_out * neuron.activation_derv()
        # print(f'dE_dO_net = {self.dE_dO_net}')

        # Sum on neighbours formula for hidden neuron
        self.dE_dH_out = np.zeros(len(self.hidden_layer.neurons), dtype=float)
        self.dE_dH_net = np.zeros(len(self.hidden_layer.neurons), dtype=float)
        for i, neuron in enumerate(self.hidden_layer.neurons):
            self.dE_dH_out = np.dot(self.dE_dO_net, self.output_layer.weights[:,i])
            self.dE_dH_net[i] = self.dE_dH_out  * neuron.activation_derv()
        # print(f'dE_dH_net = {self.dE_dH_net}')

    def update_weights(self):
        # Update output layer
        for i, neuron in enumerate(self.output_layer.neurons):
            dE_dW = self.dE_dO_net[i] * neuron.input
            # print(f'\nold_weights: {self.output_layer.weights[i]}')
            self.output_layer.weights[i] = self.output_layer.weights[i] - dE_dW * self.lr
            # print(f'new_weights: {self.output_layer.weights[i]}')
            
            
        # Update hidden layer
        for i, neuron in enumerate(self.hidden_layer.neurons):
            dE_dW = self.dE_dH_net[i] * neuron.input
            # print(f'\nold_weights: {self.hidden_layer.weights[i]}')
            self.hidden_layer.weights[i] = self.hidden_layer.weights[i] - dE_dW * self.lr
            # print(f'new_weights: {self.hidden_layer.weights[i]}')
            
        

    def train_step(self, input, target):
        output = self.feed_forward(input)
        print('network output:', output)
        self.compute_delta(target)
        self.update_weights()
        return self.hidden_layer.weights, self.output_layer.weights
             


class NeuronLayer:
    def __init__(self, weights, activation_type):
        self.weights = weights
        # Consist of list of Neuron
        self.neurons = [None] * weights.shape[0]
        for i, weight in enumerate(weights):
            self.neurons[i] = Neuron(weight, activation_type)
            

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            neuron.calc_net_out(inputs)
            outputs.append(neuron.output)
            
        return outputs


class Neuron:
    def __init__(self, weights, activation_type):
        self.weights = weights
        self.activation_type = activation_type

    def calc_net_out(self, input):
        self.input = np.array(input)
        self.net = np.dot(self.weights,input)
        self.output = self.activation(self.net)

    def activation(self, net):
        if self.activation_type == 'poly':
            return net**2
        if self.activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-net))

    def activation_derv(self):
        if self.activation_type == 'poly':
            return self.net*2
        if self.activation_type == 'sigmoid':
            return self.activation(self.net) * (1 - self.activation(self.net))





def poly():     # 2 x 2 x 2
    hidden_layer_weights = np.array([[1, 1], # 1, 1
                                     [2, 1]]) # 1, 2
    output_layer_weights = np.array([[2, 1],
                                     [1, 0]])

    nn = NeuralNetwork(hidden_layer_weights, output_layer_weights, 'poly')

    nn.train_step([1, 1], [290, 14])

    '''
    network output: [289, 16]
    Delta o[0]: -34.0
    Delta o[1]: 16.0
    Delta h[0]: -208.0
    Delta h[1]: -204.0
    node o: 0 - w_ho: 0: Delata -136.0 => new w = 70.0
    node o: 0 - w_ho: 1: Delata -306.0 => new w = 154.0
    node o: 1 - w_ho: 0: Delata 64.0 => new w = -31.0
    node o: 1 - w_ho: 1: Delata 144.0 => new w = -72.0
    node h: 0 - w_ih: 0: Delata -208.0 => new w = 105.0
    node h: 0 - w_ih: 1: Delata -208.0 => new w = 105.0
    node h: 1 - w_ih: 0: Delata -204.0 => new w = 104.0
    node h: 1 - w_ih: 1: Delata -204.0 => new w = 103.0
    '''


def sigm():     # 2 4 3
    hidden_layer_weights = np.array([[0.1, 0.1],      # 4x2 NOT 2x4
                                     [0.2, 0.1],
                                     [0.1, 0.3],
                                     [0.5, 0.01]])

    output_layer_weights = np.array([[0.1, 0.2, 0.1, 0.2],
                                     [0.1, 0.1, 0.1, 0.5],
                                     [0.1, 0.4, 0.3, 0.2]])

    nn = NeuralNetwork(hidden_layer_weights, output_layer_weights, 'sigmoid')

    nn.train_step([1, 2], [0.4, 0.7, 0.6])

    '''
    network output: [0.5913212667539777, 0.6219200057374265, 0.6508562785102494]
    Delta o[0]: 0.04623477887224621
    Delta o[1]: -0.01835937944358026
    Delta o[2]: 0.011556701931083076
    Delta h[0]: 0.000963950492482261
    Delta h[1]: 0.0028912254002713203
    Delta h[2]: 0.001386714367431997
    Delta h[3]: 0.000556197739142091
    node o: 0 - w_ho: 0: Delata 0.026559222739603632 => new w = 0.0867203886301982
    node o: 0 - w_ho: 1: Delata 0.027680191578841717 => new w = 0.18615990421057915
    node o: 0 - w_ho: 2: Delata 0.030893513891333994 => new w = 0.08455324305433301
    node o: 0 - w_ho: 3: Delata 0.028996038295713737 => new w = 0.18550198085214314
    node o: 1 - w_ho: 0: Delata -0.010546408134670482 => new w = 0.10527320406733524
    node o: 1 - w_ho: 1: Delata -0.010991533920193718 => new w = 0.10549576696009687
    node o: 1 - w_ho: 2: Delata -0.01226751284879592 => new w = 0.10613375642439797
    node o: 1 - w_ho: 3: Delata -0.01151404380893776 => new w = 0.5057570219044689
    node o: 2 - w_ho: 0: Delata 0.006638660943333523 => new w = 0.09668066952833325
    node o: 2 - w_ho: 1: Delata 0.006918854837737182 => new w = 0.39654057258113146
    node o: 2 - w_ho: 2: Delata 0.007722046916941944 => new w = 0.29613897654152904
    node o: 2 - w_ho: 3: Delata 0.007247759802026145 => new w = 0.19637612009898694
    node h: 0 - w_ih: 0: Delata 0.000963950492482261 => new w = 0.09951802475375887
    node h: 0 - w_ih: 1: Delata 0.001927900984964522 => new w = 0.09903604950751775
    node h: 1 - w_ih: 0: Delata 0.0028912254002713203 => new w = 0.19855438729986435
    node h: 1 - w_ih: 1: Delata 0.005782450800542641 => new w = 0.09710877459972869
    node h: 2 - w_ih: 0: Delata 0.001386714367431997 => new w = 0.09930664281628401
    node h: 2 - w_ih: 1: Delata 0.002773428734863994 => new w = 0.298613285632568
    node h: 3 - w_ih: 0: Delata 0.000556197739142091 => new w = 0.49972190113042897
    node h: 3 - w_ih: 1: Delata 0.001112395478284182 => new w = 0.00944380226085791
    '''



if __name__ == '__main__':
    poly()
    #sigm()

import numpy as np
import pandas as pd
from neuralnet import NeuralNetwork
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    
    h_l_weights = np.array([[1, 1],
                            [2, 1]], dtype=float)
    o_l_weights = np.array([[2, 1],
                            [1, 0]], dtype=float)
    input = np.array([1, 1], dtype=float)
    target = np.array([290, 14], dtype=float)
    precession = 0.000000001
    nn = NeuralNetwork(h_l_weights, o_l_weights, 'sigmoid')
    old_w_m1 = h_l_weights + 100
    old_w_m2 = o_l_weights + 100
    iter = 0
    while (norm(old_w_m1 - nn.hidden_layer.weights) > precession or norm(old_w_m2 - nn.output_layer.weights) > precession) and iter < 100:
        old_w_m1 = nn.hidden_layer.weights.copy()
        old_w_m1 = nn.output_layer.weights.copy()
        nn.train_step(input, target)
        print(f'iter: {iter}')
        print(f'new h_layer_weights : \n{nn.hidden_layer.weights}')
        print(f'new o_layer_weights : \n{nn.output_layer.weights}')
        iter += 1


 
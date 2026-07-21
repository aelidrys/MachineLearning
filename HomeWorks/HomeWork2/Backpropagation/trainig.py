import numpy as np
import pandas as pd
from NeuralNetwork import NeuralNetwork
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import argparse
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Backpropagation")

    parser.add_argument("--activation", type=str, default="poly",
            help='poly fo poly activation'
                 'sigmoid for sigmoid activation')
    
    parser.add_argument("--max_iter", type=int, default=3)
    
    args = parser.parse_args()
    activation = args.activation
    max_iter = args.max_iter

    if activation == "poly":
        print("--------------- Activation Function: Poly ---------------")
        h_l_weights = np.array([[1, 1],
                                [2, 1]], dtype=float)
        o_l_weights = np.array([[2, 1],
                                [1, 0]], dtype=float)
        input = np.array([1, 1], dtype=float)
        target = np.array([290, 14], dtype=float)
    if activation == "sigmoid":
        # 2 4 3
        print("--------------- Activation Function: Sigmoid -------------")
        h_l_weights = np.array([[0.1, 0.1],      # 4x2 NOT 2x4
                                [0.2, 0.1],
                                [0.1, 0.3],
                                [0.5, 0.01]])
        o_l_weights = np.array([[0.1, 0.2, 0.1, 0.2], # 3x4
                                [0.1, 0.1, 0.1, 0.5],
                                [0.1, 0.4, 0.3, 0.2]])
        input = np.array([1, 2], dtype=float)
        target = np.array([0.4, 0.7, 0.6], dtype=float)
    print("------------------------------------------------------------------------")
    p = 0.000000001
    nn = NeuralNetwork(h_l_weights, o_l_weights, 'sigmoid', lr=0.5)
    old_hl_w = h_l_weights + 100
    old_ol_w = o_l_weights + 100
    iter = 0
    while (norm(old_hl_w - nn.hidden_layer.weights) > p or norm(old_ol_w - nn.output_layer.weights) > p) and iter < max_iter:
        print(f'iter: {iter}')
        old_hl_w = nn.hidden_layer.weights.copy()
        old_ol_w = nn.output_layer.weights.copy()
        nn.train_step(input, target)
        print("------------------------------------------------------------------------")
        iter += 1


 
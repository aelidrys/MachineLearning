import numpy as np
import math
from numpy.linalg import norm

class LogisticRegression():
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def cost(self, X, Y, W):
        n = X.shape[0] # number of rows
        
        predict = X.dot(W)
        error = predict - Y
        cost = error.T @ (error) / (2 * n)
        return cost[0,0]
    
    def predict(self,X):
        predict = np.dot(X,self.W)
        predict = self.sigmoid(predict)
        return predict
    
    def cost_drive(self, X, Y, W):
        n = X.shape[0] # number of rows
        
        predict = np.dot(X,W)
        prop = self.sigmoid(predict)
        error = prop - Y
        grad = X.T @ error / n
        return grad
    
    def fit(self, X, t, _step_size=0.01, _precision=0.0001, _max_iter=1000):
        # curr_p = np.array([1,1,1,1]).reshape(-1,1)
        curr_p = np.random.rand(X.shape[1]).reshape(-1,1)
        last_p = curr_p + 100
        lrn_list = [curr_p]
        
        
        iter = 0
        while norm(curr_p - last_p) > _precision and iter < _max_iter:
            last_p = curr_p.copy()
            gr =  self.cost_drive(X,t,curr_p)
            curr_p = curr_p - gr * _step_size
            lrn_list.append(curr_p)
            iter += 11
        self.W = curr_p.copy()
        return curr_p ,iter, lrn_list
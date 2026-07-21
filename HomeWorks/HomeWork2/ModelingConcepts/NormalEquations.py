import numpy as np


class NormalEquations:
    _wights = None
    _alpha = 0

    def __init__(self, alpha=None):
        if alpha:
            self._alpha = alpha 

    # (X.T @ X)^-1 @ X.T @ Y
    def fit(self, X, Y):
        X1 = np.dot(X.T,X) + self._alpha
        self._wights = np.linalg.inv(X1) @ X.T @ Y
        return self._wights.copy()
    
    def predict(self, X):
        if self._wights is None:
            raise("No wights founded use fit(X,Y) to generate wights")
        p = np.dot(X,self._wights)
        return p






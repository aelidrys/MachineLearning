import numpy as np


def normal_equations_solution(X, Y):
    X1 = np.dot(X.T,X)
    wights = np.linalg.inv(X1) @ X.T @ Y
    return wights
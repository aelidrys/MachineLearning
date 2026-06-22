import numpy as np

def f(X, t, weights):
    n = X.shape[0]
    y_prdct = X @ weights
    error = y_prdct - t
    cost = error.T @ error /(2*n)
    return cost



def f_dervative(X, t, weights):
    pass
    n = X.shape[0]
    y_prdct = X @ weights
    error = y_prdct - t
    cost_dr = X.T @ error / n
    return cost_dr




if __name__ == '__main__':
    # Input is 1D feature, e.g. the price
    X = np.array([0, 0.2, 0.4, 0.8, 1.0])
    t = 5 + X   # Output linear, no noise

    X = X.reshape((-1, 1))  # let's reshape in 2D
    X = np.hstack([np.ones((X.shape[0], 1)), X])    # add 1 for c

    weights = np.array([1.0, 1.0])

    print(f"weights = {weights}")
    print(X)


    print(f(X, t, weights)) # cost: 8

    print(f_dervative(X, t, weights))  # dervative: [-4.   -1.92]
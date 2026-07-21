import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def split_ans_scale_data(X, Y, scaler_type=1):
    # Manual split 50% - 50%
    X_train = X[:100]
    X_test = X[100:]
    Y_train = Y[:100]
    Y_test = Y[100:]

    # Scaling
    if scaler_type == 1:
        scaler = MinMaxScaler().fit(X_train)
    else:
        scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, Y_train, X_test, Y_test
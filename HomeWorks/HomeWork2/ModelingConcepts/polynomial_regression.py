import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from data_helper import split_ans_scale_data

def monomials_poly_features(X, degree):
    i = 2
    X1 = X
    while i <= degree:
        X = np.hstack([X,X1**i])
        i +=1
    return X




def do_polynomailReg(X,Y,degree, poly=False):
    TrainError = []
    TestError = []
    for i in range(degree):
        if poly:
            X_new = PolynomialFeatures(i+1).fit_transform(X)
        else:
            X_new = monomials_poly_features(X,i+1)
        X_train, Y_train, X_test, Y_test = split_ans_scale_data(X_new, Y)
        
        # Train
        LReg = LinearRegression(fit_intercept=True).fit(X_train,Y_train)
        if i+1 == 3:
            print("-----------------------------------------------")
        print(f"- Degree: {i+1}")
        print(f"\tX_new features number: {X_new.shape[1]}")
        print(f"\tintercept: {LReg.intercept_}")
        Predict = LReg.predict(X_train)
        trainRMSE = np.sqrt(mean_squared_error(Predict, Y_train))
        print(f"\tTrain RMSE: {trainRMSE}")
        TrainError.append(trainRMSE)
        
        Predict = LReg.predict(X_test)
        testRMSE = np.sqrt(mean_squared_error(Predict, Y_test))
        print(f"\tTest RMSE: {testRMSE}")
        TestError.append(testRMSE)
        if i+1 == 3:
            print("-----------------------------------------------\n")
        else:
            print("")
        
    return TrainError, TestError



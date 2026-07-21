import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn import random
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score
from visualization import error_VS_degree, FeaurVsError
from regularized_polynomial import do_ridgeWithPoly, do_lassoWithPoly
from polynomial_regression import do_polynomailReg
from data_helper import split_ans_scale_data
from NormalEquations import NormalEquations
import warnings

warnings.filterwarnings('ignore')
np.random.seed(17)
random.seed(17)

# Create the parser
parser = argparse.ArgumentParser(description="A simple argument parser example")

parser.add_argument('--dataset', type=str, default='data2_200x30.csv')

parser.add_argument('--preprocessing', type=int, default=1,
    help='1 for min/max,'
         '2 for standrizing')

parser.add_argument('--choice', type=int, default=2,
    help="1 for simple linerar regression"
         "2 for polynomial on all featurs with degrre [1,2,3,4] with cross featurs"
         "3 for polynomial on all featurs with degrre [1,2,3,4] with monomial featurs"
         "4 individual feature test"
         "5 for find best lambda with grid search"
         "6 Lasso selection")

parser.add_argument('--extra_args', type=str, default='0,3,6', help="extra values passed")

args = parser.parse_args()

extra_args = args.extra_args
dataset = args.dataset
preprocessing = args.preprocessing
choice = args.choice

df = pd.read_csv(dataset)

X = np.array(df.drop(columns="Target"))
Y = np.array(df[["Target"]])


# Simple linear regression
if choice == 1:

    # Manual split 50% - 50%
    X_train, Y_train, X_test, Y_test = split_ans_scale_data(X,Y)
    LReg = LinearRegression(fit_intercept=True).fit(X_train,Y_train)

    print("intercept          : ", LReg.intercept_[0])
    Predict1 = LReg.predict(X_train)
    wights_avg = sum(abs(LReg.coef_[0]))/len(LReg.coef_[0])
    print(f"wights abs average :  {wights_avg}")
    print("Train RMSE         : ", np.sqrt(mean_squared_error(Predict1, Y_train)))
    Predict2 = LReg.predict(X_test)
    print("Test RMSE          : ", np.sqrt(mean_squared_error(Predict2, Y_test)))


# Polynomail regresion
if choice == 2:
    TrainError, TestError = do_polynomailReg(X,Y,4,True)   
 
    # Visualisation
    error_VS_degree(TrainError, TestError, 4)


# Mononmail regression
if choice == 3:
    TrainError, TestError = do_polynomailReg(X,Y,4,False)

    # Visualisation
    error_VS_degree(TrainError, TestError, 4)


# Individual fetures
if choice == 4: 
    Dgr1Error = []
    Dgr2Error = []
    Dgr3Error = []
    for i in range(9):
        F1TrErr, F1TsErr = do_polynomailReg(X[:,i].reshape(-1,1),Y,3,True)
        Dgr1Error.append([F1TrErr[0],F1TsErr[0]])
        Dgr2Error.append([F1TrErr[1],F1TsErr[1]])
        Dgr3Error.append([F1TrErr[2],F1TsErr[2]])
        
    FsErr = np.array(Dgr1Error)
    FeaurVsError(FsErr," 1", "orange", "gray")
    FsErr = np.array(Dgr2Error)
    FeaurVsError(FsErr," 2", "brown", "yellow")
    FsErr = np.array(Dgr3Error)
    FeaurVsError(FsErr," 3")


# Ridge with all  Features and cross val
if choice == 5:
    do_ridgeWithPoly(X,Y)


# Lasso Selection
if choice == 6:
    best_model = do_lassoWithPoly(X,Y)

    data=df.drop(columns="Target")
    select_featurs = SelectFromModel(best_model)
    selected_features = data.columns[(select_featurs.get_support())]
    print("selected_features: \n\t", selected_features)

    df_new = data[selected_features]
    X_new = np.array(df_new)
    do_ridgeWithPoly(X_new,Y, degree_=3)


if choice == 7:

    X_train, t_train, X_val, t_val = split_ans_scale_data(X,Y)

    alphas = [0.1, 0, 1, 10, 100, 1000]
    for alpha in alphas:
        model = NormalEquations(alpha=alpha)

        model.fit(X_train, t_train)

        t_prdct = model.predict(X_val)
        error = mean_squared_error(t_val, t_prdct)
        print(f"error = {error}")

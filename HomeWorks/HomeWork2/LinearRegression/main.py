import argparse
import numpy as np
import pandas as pd
from linear_regression import LinearReg
from visualization import display_points, costs_VS_iters, featurs_Vs_target
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score
from Normal_Equation import normal_equations_solution
import warnings
warnings.filterwarnings('ignore')


# Early verifications 
def early_verification():
    x = np.array([0, 0.2, 0.4, 0.8, 1.0])
    t=5+x
    ones = np.ones((5,1))
    x = x.reshape(-1,1)
    t = t.reshape(-1,1)
    x = np.hstack([ones, x])
    wights, costs, wights_list = LR.gradient_descent(x, t, _step_size=step_size,
            _precision=precision, _max_iter=max_iter)
    print(f"cost after learning: {LR.cost(x,t,wights)}\n")
    # print(f"wights: {wights}\n")
    y_p = LR.predict(x,wights)
    display_points(x[:,1],y_p)


# Trian with All Featurs
def all_features():
    wights , costs, wights_list = LR.gradient_descent(x, t,
            _step_size=step_size, _precision=precision, _max_iter=max_iter)
    iters = len(costs)
    print(f"iterations = {iters}")
    # print(f"wights: {wights[:,0]}")
    p = LR.predict(x, wights)
    print(f"r2_score: {r2_score(t,p):.2f}")
    costs_VS_iters(costs, iters, 'g')
    


# P2: Optimizing the hyperparameters
def optimizing_hyperparameters():
    step_sizes = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.0000001]
    precisions = [0.01, 0.001, 0.0001, 0.00001]
    iterations = 0
    best_prms = {"cost": 10000, "step_size": 0, "precision": 0}
    for step_size in step_sizes:
        for precision in precisions:
            for itr in range(3):
                wights , costs, wights_list = LR.gradient_descent(x, t, _step_size=step_size,
                    _precision=precision, _max_iter=max_iter)
                cost = LR.cost(x, t, wights)
                iterations += 1
                if best_prms["cost"] > cost:
                    best_prms["cost"] = cost
                    best_prms['step_size'] = step_size
                    best_prms['precision'] = precision
    print(f"iterations = {iterations}")
    print(f"\tcost: {best_prms["cost"]:.2f}")
    print(f"\tstep_size: {best_prms['step_size']}")
    print(f"\tprecision: {best_prms['precision']}")



# Trian with Best Featurs
def best_feature():
    df1 = df[['Feat1', 'Target']]
    df2 = df[['Feat2', 'Target']]
    df3 = df[['Feat3', 'Target']]
    featurs_Vs_target(df1,df2,df3)
    x1 = x[:,:2]
    wights , costs, wights_list = LR.gradient_descent(x1, t, _step_size=step_size,
            _precision=precision, _max_iter=max_iter)
    iters = len(costs)
    print(f"iterations = {iters}")
    # print(f"wights: {wights[:,0]}")
    p = LR.predict(x1, wights)
    print(f"r2_score: {r2_score(t,p):.2f}")
    costs_VS_iters(costs, iters, 'g')


# Normal Equation
def normal_equation():
    wights = normal_equations_solution(x,t)
    # print(f"wights: {wights[:,0]}")
    p = LR.predict(x, wights)
    print(f"r2_score: {r2_score(t,p):.2f}")


# sikit linear regression
def sikit_linear():
    LReg = LinearRegression(fit_intercept=False).fit(x,t)
    wights = LReg.coef_
    # print(f"wights: {wights}")
    p = LReg.predict(x)
    print(f"r2_score: {r2_score(t,p):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser example")

    parser.add_argument('--dataset', type=str, default='dataset_200x4_regression.csv')

    parser.add_argument('--preprocessing', type=int, default=1, #P4
                    help='0 for no processing,'
                        '1 for min/max,'
                        '2 for standrizing')

    parser.add_argument('--choice', type=int, default=2,
        help="0 for linerar verification"            #P0
            "1 for training wih all feature"         #P1 / P3 / P7
            "2 for training with the best features"  #P5
            "3 for normal equation"                  #p6
            "4 for sikit")

    parser.add_argument('--step_size', type=float, default=0.01,help="Learning Rate default(0.01)")

    parser.add_argument('--precision', type=float, default=0.000001, help="Precision defualt(0.0001)")

    parser.add_argument('--max_iter', type=float, default=10000, help="number of iteration to learn defualt(1000)")

    args = parser.parse_args()

    dataset = args.dataset
    preprocessing = args.preprocessing
    choice = args.choice
    step_size = args.step_size
    precision = args.precision
    max_iter = args.max_iter

    LR = LinearReg()

    if choice == 0:
        early_verification()
    
    df = pd.read_csv("./dataset_200x4_regression.csv")
    x = np.array(df[['Feat1', 'Feat2', 'Feat3']])

    if preprocessing == 1:
        scaler = MinMaxScaler().fit(x)
        x = scaler.transform(x)
        
    if preprocessing == 2:
        scaler = StandardScaler().fit(x)
        x = scaler.transform(x)
        
    ones = np.ones(x.shape[0]).reshape(-1,1)
    x = np.hstack([ones,x])
    t = np.array(df[['Target']])


    #----- Choices -------
    if choice == 1:
        all_features()

    if choice == 2:
        best_feature()

    if choice == 3:
        normal_equation()

    if choice == 4:
        sikit_linear()

    if choice == 5:
        optimizing_hyperparameters()





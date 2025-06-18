import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

def load_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Split the data into features and target
    X = data.iloc[:, :-1].values  # All columns except the last one
    y = data.iloc[:, -1].values   # Last column
    
    return X, y




def preprocess_data(X, y):
    # Normalize the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test



def training(X_train, y_train):
    regr = MLPRegressor(random_state=1, max_iter=10000, tol=0.01,
            learning_rate_init=0.01, hidden_layer_sizes=(150,))
    regr.fit(X_train, y_train)
    return regr
    
    
if __name__ == "__main__":
    # Load and preprocess the data
    X, y = load_data('../../LinearRegression/HWork2/H2Part2/data2_200x30.csv')
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    regr = training(X_train, y_train)
    predict = regr.predict(X_test[:5])
    for i in range(5):
        print(f'predict = {predict[i]}')
        print(f'target = {y_test[i]}')
        print(f'error = {abs(predict[i] - y_test[i])}\n')
    print(f'score = {regr.score(X_test, y_test)}')
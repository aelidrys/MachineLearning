import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from datetime import datetime as dt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, PolynomialFeatures
from geopy.distance import geodesic as GD
from scipy.stats import zscore
from visualize import simple_graph, histogram
from tools import wescr
import argparse
from data_preprocessing import data_preprocess
import pickle


parcer = argparse.ArgumentParser(description='ArgumentParser')
parcer.add_argument('--train', type=int, default=0,
                    help='0: no Train, 1: Train')
parcer.add_argument('--degree', type=int, default=4)
parcer.add_argument('--alpha', type=int, default=1)
args = parcer.parse_args()
train = args.train
alpha = args.alpha
degree_ = args.degree


# Load & Preprocess data
df, val_df = data_preprocess()


# Training
def training(df, val_df, degree_):

    X_train = np.array(df.drop(columns='trip_duration'))
    Y_train = np.array(df['trip_duration'])

    # print(f'X_train: {X_train[:2]}')
    # Scaling
    # X_train = np.log1p(X_train)
    Scaler = MinMaxScaler().fit(X_train)
    X_train = Scaler.transform(X_train)

    X_train = PolynomialFeatures(degree=degree_).fit_transform(X_train)

    # print("X_train: ", X_train)
    # print("Y_train: ", Y_train)
    model = Ridge(fit_intercept=True, alpha=alpha).fit(X_train,Y_train)
    print('----------train---------')
    print('\tScore: ', model.score(X_train, Y_train))

    # Validation
    X_val = np.array(val_df.drop(columns='trip_duration'))
    Y_val = np.array(val_df['trip_duration'])

    # Scaling in Val
    # X_val = np.log1p(X_val)
    X_val = Scaler.transform(X_val)

    X_val = PolynomialFeatures(degree=degree_).fit_transform(X_val)

    print('----------Val---------')
    print('\tScore: ', model.score(X_val, Y_val))
    
    # Save the model as pickle
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    

if train:
    training(df, val_df, degree_)



# # Visualize
# histogram(df['dis_pass'], title='Histogram of passenger_count', xlabel='passenger_count')
# dfx = df[['distance', 'trip_duration']].sort_values(by='distance')
# simple_graph((dfx['distance']), (dfx['trip_duration']),xlabel='distance', ylabel='trip_duration')



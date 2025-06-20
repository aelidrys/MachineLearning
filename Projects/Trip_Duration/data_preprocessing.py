import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from datetime import datetime as dt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, PolynomialFeatures
# from geopy.distance import geodesic as GD
from scipy.stats import zscore
from tools import wescr
# from haversine import haversine, Unit



def data_encoding(df, val_df):
    
    df = df.copy()
    val_df = val_df.copy()
    # Data Encoding
    df['store_and_fwd_flag'] = LabelEncoder().fit_transform(df['store_and_fwd_flag'])
    val_df['store_and_fwd_flag'] = LabelEncoder().fit_transform(val_df['store_and_fwd_flag'])
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    val_df['pickup_datetime'] = pd.to_datetime(val_df['pickup_datetime'])
    return df, val_df



def outliers_treatment(df, val_df):
    
    df = df[df['trip_duration'] > 20]
    val_df = val_df[val_df['trip_duration'] > 20]
    # Data Encoding
    clmns = ['distance', 'trip_duration', 'passenger_count']
    for i in clmns:
        lw, uw = wescr(df[i])
        df[i] = np.where(df[i]<lw, lw, df[i])
        df[i] = np.where(df[i]>uw, uw, df[i])
        
    for i in clmns:
        lw, uw = wescr(val_df[i])
        val_df[i] = np.where(val_df[i]<lw, lw, val_df[i])
        val_df[i] = np.where(val_df[i]>uw, uw, val_df[i])
    return df, val_df


    # df['z_score'] = np.abs(zscore(df['trip_duration']))
    # df = df[df['z_score'] < 3]  # Remove values where z-score is greater than 3
    # df['z_score'] = np.abs(zscore(df['distance']))
    # df = df[df['z_score'] < 3]  # Remove values where z-score is greater than 3
    # df = df.drop(columns=['z_score'])
    # df = df[(df['passenger_count'] > 0) & (df['passenger_count'] < 7)]  # Remove passenger_count outliers

    # val_df['z_score'] = np.abs(zscore(val_df['trip_duration']))
    # val_df = val_df[val_df['z_score'] < 3]  # Remove values where z-score is greater than 3
    # val_df['z_score'] = np.abs(zscore(val_df['distance']))
    # val_df = val_df[val_df['z_score'] < 3]  # Remove values where z-score is greater than 3
    # val_df = val_df.drop(columns=['z_score'])
    # val_df = val_df[(val_df['passenger_count'] > 0) & (val_df['passenger_count'] < 7)]  # Remove passenger_count outliers



def feature_engineering(df, val_df):

    # Add features
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['pickup_month'] = df['pickup_datetime'].dt.month
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_weekday'] = df['pickup_datetime'].dt.weekday
    val_df['pickup_datetime'] = pd.to_datetime(val_df['pickup_datetime'])
    val_df['pickup_month'] = val_df['pickup_datetime'].dt.month
    val_df['pickup_hour'] = val_df['pickup_datetime'].dt.hour
    val_df['pickup_weekday'] = val_df['pickup_datetime'].dt.weekday
    
    # Selecting features
    # corr = df.corr()
    # corr['selected'] = corr['trip_duration'].apply(lambda x: abs(x) > 0.08)
    # df_selcted = corr[corr['selected'] == True]
    # print(f'train corrolation matrix:\n{df_selcted[['trip_duration', 'selected']]}')
    selected_columns = ['distance', 'pickup_longitude', 'pickup_latitude', 'dropoff_latitude', 'passenger_count', 'pickup_month',
                    'pickup_hour', 'pickup_weekday', 'trip_duration']
    df = df[selected_columns]
    val_df = val_df[selected_columns]
    corr = df.corr()
    print(f'Selected Columns and its corroluation with target:\n{corr["trip_duration"]}')
    
    return df, val_df



def data_preprocess():
    df = pd.read_csv('data/Train.csv')
    val_df = pd.read_csv('data/Val.csv')
    
    df, val_df = data_encoding(df, val_df)
    df, val_df = outliers_treatment(df.copy(), val_df.copy())
    # df, val_df = drop_columns(df, val_df)
    df, val_df = feature_engineering(df, val_df)


    return df, val_df

if __name__ == '__main__':
    data_preprocess()



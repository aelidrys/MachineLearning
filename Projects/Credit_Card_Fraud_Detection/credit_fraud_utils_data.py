import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler



def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


# Inter quartile outliers treatment
def inter_quartile_range_treatment(data, columns_to_treat):
    for column_name in columns_to_treat:
        Q1 = data[column_name].quantile(0.25)
        Q3 = data[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[column_name] = np.where(data[column_name] < lower_bound, lower_bound, data[column_name])
        data[column_name] = np.where(data[column_name] > upper_bound, upper_bound, data[column_name])
    return data


# Z-score treatment
def z_score_treatment(data, columns_to_treat):
    for column_name in columns_to_treat:
        mean = data[column_name].mean()
        std_dev = data[column_name].std()
        z_scores = (data[column_name] - mean) / std_dev
        data[column_name] = np.where(z_scores < -3, mean - 3 * std_dev, data[column_name])
        data[column_name] = np.where(z_scores > 3, mean + 3 * std_dev, data[column_name])
    return data


# Outliers treatment
def treat_outliers(data):
    columns_to_treat = [col for col in data.columns if col != 'Class']
    try:
        data = inter_quartile_range_treatment(data, columns_to_treat)
        # data = z_score_treatment(data, columns_to_treat)
        return data
    except Exception as e:
        print(f"Error treating outliers: {e}")
        raise
    
    
def features_selection(data):
    columns_to_drop = ['Time', 'V13', 'V15', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V27', 'V28', 'Amount']
    data = data.drop(columns=columns_to_drop, errors='ignore')
    return data


def preprocess_data(data):
    data = features_selection(data)
    X = data.drop(columns=['Class'], errors='ignore')
    Y = data['Class']
    return X, Y


def split_data(X, Y):
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42) 
    return X_train, X_val, Y_train, Y_val




# SMOTE oversampling
def oversample_data(X, Y, random_state=42):
    counter = Counter(Y)
    factor, majority_size = 15, counter[0]
    new_size = int(majority_size / factor)
    print(f"over new_size: {new_size}")
    
    oversample = SMOTE(sampling_strategy={1: new_size}, random_state=random_state, k_neighbors=3)
    X_os, Y_os = oversample.fit_resample(X, Y)
    counter_os = Counter(Y_os)
    print(f"Before oversampling: {counter}, After oversampling: {counter_os}")
    return X_os, Y_os


# Undesampling data
def undersample_data(X, Y, random_state=42):
    counter = Counter(Y)
    factor, minority_size = 10, counter[1]
    new_size = int(minority_size * factor)
    print(f"under new_size: {new_size}")
    
    rus = RandomUnderSampler(sampling_strategy={0: new_size}, random_state=random_state)
    X_us, Y_us = rus.fit_resample(X, Y)
    counter_us = Counter(Y_us)
    print(f"Before undersampling: {counter}, After undersampling: {counter_us}")
    return X_us, Y_us


# Oversampling and Undersampling
def sampling_data(X, Y, random_state=42):
    X, Y = oversample_data(X, Y, random_state)
    X_s, Y_s = undersample_data(X, Y, random_state)
    return X_s, Y_s


# Save the model to a file using pickle
def save_model(model, file_path, threshold=0.5, model_name='model'):

    try:
        model_dict = {
            'model': model,
            'threshold': threshold,
            'model_name': model_name
        }
        with open(file_path, 'wb') as file:
            pickle.dump(model_dict, file)
        print(f"Model saved to {file_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
        raise
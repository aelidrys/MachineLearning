import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler



def load_data(file_path) -> pd.DataFrame:
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise FileNotFoundError(f"Error loading data: {e}")


# Inter quartile outliers treatment
def inter_quartile_range_treatment(data, columns_to_treat):
    bounds = []
    for column_name in columns_to_treat:
        Q1 = data[column_name].quantile(0.25)
        Q3 = data[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[column_name] = np.where(data[column_name] < lower_bound, lower_bound, data[column_name])
        data[column_name] = np.where(data[column_name] > upper_bound, upper_bound, data[column_name])
        bounds.append((lower_bound, upper_bound))
    return data, bounds


# Outliers treatment
def treat_outliers(data, bounds=[]):
    data = data.copy()
    columns_to_treat = [col for col in data.columns if col != 'Class']
    if bounds:
        for i, column_name in enumerate(columns_to_treat):
            lower_bound, upper_bound = bounds[i]
            data[column_name] = np.where(data[column_name] < lower_bound, lower_bound, data[column_name])
            data[column_name] = np.where(data[column_name] > upper_bound, upper_bound, data[column_name])
        return data, bounds
    try:
        data, bounds = inter_quartile_range_treatment(data, columns_to_treat)
        return data, bounds
    except Exception as e:
        raise ValueError(f"Error treating outliers: {e}")


# Determine relevant features using RandomForestClassifier
def feature_importances(X_train, Y_train):
    print("Determining relevant features using RandomForestClassifier...")
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)

    feature_importances = model.feature_importances_
    threshold = np.mean(feature_importances)
    print(f"threshold: {threshold}")
    selected_features = X_train.columns[feature_importances > threshold]
    print(f"selected_features: {selected_features}")
    return selected_features    


# Select relevant features based on the provided list of selected features
def features_selection(X, selected_features=None):
    if selected_features is None:
        raise ValueError("selected_features must be provided for feature selection.")
    X = X[selected_features]
    return X


# Split the data into training and validation sets
def split_data(X, Y):
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42) 
    return X_train, X_val, Y_train, Y_val


# Preprocess the data by treating outliers, splitting data, and selecting relevant features
def preprocess_data(data):
    selected_features = ['V7', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18']
    selected_features.append('Class')
    data = features_selection(data, selected_features)

    X = data.drop(columns=['Class'], errors='ignore')
    Y = data['Class']
    X_train, X_val, Y_train, Y_val = split_data(X, Y)
    X_train, bounds = treat_outliers(X_train)
    X_val, _ = treat_outliers(X_val, bounds)

    return X_train, X_val, Y_train, Y_val, bounds


# Preprocess the test data by treating outliers and selecting relevant features
def preprocess_eval_data(data, bounds=[]):
    selected_features = ['V7', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18']
    selected_features.append('Class')
    data = features_selection(data, selected_features)
    data, bounds = treat_outliers(data, bounds)
    X = data.drop(columns=['Class'], errors='ignore')
    Y = data['Class']
    return X, Y


# SMOTE oversampling
def oversample_data(X, Y, random_state=42):
    counter = Counter(Y)
    factor, majority_size = 100, counter[0]
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
    factor, minority_size = 80, counter[1]
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
    X, Y = undersample_data(X, Y, random_state)
    return X, Y


# Save the model to a file using pickle
def save_model(model, threshold=0.5, model_name='model', bounds=[], file_path='model.pkl'):

    try:
        model_dict = {
            'model': model,
            'threshold': threshold,
            'model_name': model_name,
            'bounds': bounds
        }
        with open(file_path, 'wb') as file:
            pickle.dump(model_dict, file)
        print(f"Model saved to {file_path}")
    except Exception as e:
        raise ValueError(f"Error saving model: {e}")

import pickle
import numpy as np
import pandas as pd
from preprocessingData import preprocess_data
from sklearn.model_selection import train_test_split

# load the model from the pickle file
def load_model(model_path='model.pkl'):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Load Data from csv files
df = pd.read_csv('data/HPrice_train.csv')
df = df.drop(columns=['Id'], axis=1)  # Drop Id column from training data


# Preprocessing Data
df = preprocess_data(df)


# Training
df_y = np.array(df["SalePrice"]).reshape(-1, 1)
df_x = np.array(df.drop(columns=["SalePrice"], axis=1)).reshape(1460, df.shape[1] - 1)
print(f"Train Dataset shape: {df_x.shape}")



# Train test split
X_train, X_val, y_train, y_val = train_test_split(df_x, df_y, test_size=0.30, random_state=42) 


model = load_model()
score_val = model.score(X_val, y_val)

print("Validation Dataset")
print(f'\tscore: {score_val}') 
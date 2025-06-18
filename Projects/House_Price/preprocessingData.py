import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

def wescr(column):
    q1, q3 = np.percentile(column, [25, 75])
    iqr = q3 - q1
    lw = q1 - 1.5 * iqr
    uw = q3 + 1.5 * iqr
    return lw, uw


def outleir_treatment(df: pd.DataFrame):
    # int features outlier treatment
    # df = df.copy()
    columns = df.select_dtypes(include="int64").columns
    for i in columns:
        df[i] = df[i].astype(float)  # Convert int64 to float64 for consistency
        lw, uw = wescr(df[i])
        if lw == uw:
            continue
        df.loc[df[i] < lw, i] = lw
        df.loc[df[i] > uw, i] = uw

    # float features outlier treatment
    clmns2 = ['LotFrontage', 'GarageYrBlt', 'MasVnrArea']
    for i in clmns2:
        lw, uw = wescr(df[i])
        df.loc[df[i] < lw, i] = lw
        df.loc[df[i] > uw, i] = uw
    # return df


def encoding_data(df):
    
    df = df.copy()
    # training data encoding
    label_encoder = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = label_encoder.fit_transform(df[col])
        
    return df


def missing_values_treatment(df):
    # Drop the columns that have more than 30% of values missing 
    df = df.loc[:, [col for col in df.columns if df[col].isnull().sum() < 0.3 * df.shape[0]]]

    # Use Mean or Median or Mode to fill the remaining missing values or None Available
    df.loc[:, 'LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())
    df.loc[:, 'MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].mean())
    df.fillna(df.mode().iloc[0], inplace=True)
    # df.loc[:, 'Electrical'] = df['Electrical'].fillna("SBrkr")
    # df.loc[:, 'GarageYrBlt'] = df['GarageYrBlt'].fillna(2005.0)
    df.fillna({'GarageType': 'NA', 'GarageFinish': 'NA', 'GarageQual': 'NA', 'GarageCond': 'NA'}, inplace=True)
    df.fillna({'BsmtQual': 'NA', 'BsmtCond': 'NA', 'BsmtExposure': 'NA',
               'BsmtFinType1': 'NA', 'BsmtFinType2': 'NA'}, inplace=True)
    return df


def selected_features(df):

    df = df.copy()
    # Select Features based on correlation with SalePrice
    corr = df.corr()
    selected_columns = corr[abs(corr.iloc[-1])>0.25].index
    df = df[selected_columns]

    return df



def preprocess_data(df):
    # Missing Values Treatment
    df = missing_values_treatment(df)
    
    # Outleirs Treatment
    outleir_treatment(df)

    # Data Encoding
    df = encoding_data(df)

    # Selected Features
    df = selected_features(df)

    return df
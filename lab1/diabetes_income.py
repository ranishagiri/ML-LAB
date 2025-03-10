import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load the datasets
adult_df = pd.read_csv('/adult.csv')
diabetes_df = pd.read_csv('/content/Dataset of Diabetes .csv')

# Function to handle missing values
def handle_missing_values(df):
    df = df.fillna(df.median(numeric_only=True))  # Fill numeric NaNs with median
    df = df.fillna(df.mode().iloc[0])  # Fill categorical NaNs with mode
    return df

# Function to handle categorical data
def encode_categorical(df):
    df = pd.get_dummies(df, drop_first=True)  # One-hot encoding
    return df

# Function to handle outliers using IQR method
def remove_outliers(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where((df[col] < lower_bound) | (df[col] > upper_bound), np.nan, df[col])
    df = handle_missing_values(df)  # Refill outlier values
    return df

# Function to apply normalization and standardization
def apply_scaling(df):
    scaler_minmax = MinMaxScaler()
    scaler_standard = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    df[numeric_cols] = scaler_minmax.fit_transform(df[numeric_cols])  # Min-Max Scaling
    df[numeric_cols] = scaler_standard.fit_transform(df[numeric_cols])  # Standardization
    
    return df

# Apply preprocessing to both datasets
adult_df = handle_missing_values(adult_df)
adult_df = encode_categorical(adult_df)
adult_df = remove_outliers(adult_df)
adult_df = apply_scaling(adult_df)

diabetes_df = handle_missing_values(diabetes_df)
diabetes_df = encode_categorical(diabetes_df)
diabetes_df = remove_outliers(diabetes_df)
diabetes_df = apply_scaling(diabetes_df)

# Create the directory to save the preprocessed datasets
!mkdir -p /mnt/data

# Save the preprocessed datasets
adult_df.to_csv('/mnt/data/adult_preprocessed.csv', index=False)
diabetes_df.to_csv('/mnt/data/diabetes_preprocessed.csv', index=False)

print("Preprocessing completed. Processed files saved as 'adult_preprocessed.csv' and 'diabetes_preprocessed.csv'")

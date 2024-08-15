import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from pennylane import numpy as np

def load_dataset():
    filepath = 'load csv file path here'
    
    # Load and shuffle the dataset
    df = pd.read_csv(filepath)
    df = df.sample(frac=1)

    # Separate fraud and regular transactions
    df_fraud = df[df['Class'] == 1]  # Fraud transactions
    df_regular = df[df['Class'] == 0]  # Regular transactions

    # Create a balanced dataset by undersampling the regular transactions
    df_sample = pd.concat([df_fraud, df_regular[:len(df_fraud)]])

    # Select relevant columns
    keep_cols = ['V14', 'V4', 'V12', 'Amount', 'Class']
    df = df[keep_cols]

    # Scale the 'Amount' feature using RobustScaler
    rob_scaler = RobustScaler()
    df['Amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

    # Split the dataset into features (X) and target (y)
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Split the data into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Apply Random Under-Sampling to the training data
    rus = RandomUnderSampler()
    X_rus, y_rus = rus.fit_resample(X_train, y_train)

    # Normalize the features to be within the range [-1, 1]
    normalize = lambda x: 2 / (1 + np.exp(-x)) - 1
    X_rus = normalize(X_rus.to_numpy())
    X_test = normalize(X_test.to_numpy())

    # Convert target arrays to NumPy arrays
    y_rus = y_rus.to_numpy()
    y_test = y_test.to_numpy()

    return X_rus, X_test, y_rus, y_test

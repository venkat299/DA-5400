import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def load_df(path):
    _dir = os.path.dirname(__file__)
    return  pd.read_csv(os.path.join(_dir, path), header=None)  

def load_train_df():
    return load_df("./FMLA1Q1Data_train.csv")

def load_test_df():
    return load_df("./FMLA1Q1Data_test.csv")


def get_data():

    # load the train data
    train_df = load_train_df()
    train_df.columns = ["x1", "x2", "y"]

    # add bias column
    train_df.insert(0, "bias", 1)

    # convert the DataFrame to a NumPy matrix
    train = train_df.values  # Convert the dataframe to a numpy array

    # separate features (X) and target (y), assuming the last column is the target:
    X_train = train[:, :-1]  # All rows, all columns except the last one
    y_train = train[:, -1] # All rows, last column (target)

    # print("Design matrix (X) shape:", X_train.shape)
    # print("Target vector (y) shape:", y_train.shape)

    # load the test data
    test_df = load_test_df()

    test_df.columns = ["x1", "x2", "y"]

    # add bias column
    test_df.insert(0, "bias", 1)

    # convert the DataFrame to a NumPy matrix
    test = test_df.values  # Convert the dataframe to a numpy array

    # separate features (X) and target (y), assuming the last column is the target:
    X_test = test[:, :-1]  # All rows, all columns except the last one
    y_test = test[:, -1] # All rows, last column (target)

    # print("test data Design matrix (X) shape:", X_test.shape)
    # print("test data Target vector (y) shape:", y_test.shape)

    return X_train, y_train, X_test, y_test

# compute the Mean Squared Error
def calculate_mse(y_true:np.ndarray, y_pred:np.ndarray) -> np.float64:
    return np.mean((y_true - y_pred) ** 2)

def train_test_split(X, y, test_size=0.2, random_state=None):
    n_samples = len(y)
    n_test = int(n_samples * test_size)
    if random_state is not None:
        np.random.seed(random_state)
    shuffled_indices = np.random.permutation(n_samples)
    test_indices = shuffled_indices[:n_test]
    train_indices = shuffled_indices[n_test:]
    X_train = X[train_indices]
    X_eval = X[test_indices]
    y_train = y[train_indices]
    y_eval = y[test_indices]
    return X_train, X_eval, y_train, y_eval
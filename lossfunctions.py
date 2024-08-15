
from pennylane import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler

import keras
from keras.models import Sequential
from keras.layers import Dense
import qiskit
print(qiskit.__version__)

from qiskit_aer.noise import NoiseModel
# from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2

import time
import warnings
import matplotlib.colors
warnings.filterwarnings('ignore')
def mse_loss(y_real: np.array, y_pred: np.array) -> float:

    if len(y_real) != len(y_pred):
        raise ValueError("The length of y_real and y_pred must be the same.")
    loss = np.mean((y_real - y_pred) ** 2)
    return loss


def accuracy_loss(y_real: np.array, y_pred: np.array, threshold: float = 0.5) -> float:
    """
    Calculates the accuracy loss between the real and predicted values.

    Parameters:
    - y_real (np.array): The real values.
    - y_pred (np.array): The predicted values.
    - threshold (float): The threshold value for classification (default: 0.5).

    Returns:
    - accuracy (float): The accuracy loss between the real and predicted values.
    """

    if len(y_real) != len(y_pred):
        raise ValueError("The length of y_real and y_pred must be the same.")
    y_ = [int(y >= threshold) for y in y_pred]
    accuracy = np.mean([y1 == y2 for y1, y2 in zip(y_real, y_)])
    return accuracy


def bin_loss(y_real: np.array, y_pred: np.array, eps: float = 1e-8) -> float:

    if len(y_real) != len(y_pred):
        raise ValueError("The length of y_real and y_pred must be the same.")
    loss = -np.mean([y1 * np.log(y2 + eps) + (1 - y1) * np.log(1 - y2 + eps) for y1, y2 in zip(y_real, y_pred)])
    return loss

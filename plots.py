from pennylane import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

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
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2

import time
import warnings
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score
import matplotlib.colors
warnings.filterwarnings('ignore') 


def plot_roc_curve(fpr, tpr, roc_auc):
        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

def calculate_metrics(y_test, y_hat_test):
        conf_matrix = confusion_matrix(y_test, y_hat_test)
        TN, FP, FN, TP = conf_matrix.ravel()

        accuracy = accuracy_score(y_test, y_hat_test)
        recall = recall_score(y_test, y_hat_test)  # Same as sensitivity
        precision = precision_score(y_test, y_hat_test)
        misclassification_rate = 1 - accuracy
        specificity = TN / (TN + FP)
        g_mean = np.sqrt(recall * specificity)

        # ROC AUC score requires probabilities, but here we are using it directly from the confusion matrix
        # This is less accurate than using probability scores, but demonstrates the functionality
        roc_auc = roc_auc_score(y_test, y_hat_test)

        # Print all calculated metrics
        print("Confusion Matrix:\n", conf_matrix)
        print("Accuracy:", accuracy)
        print("Recall (Sensitivity):", recall)
        print("Precision:", precision)
        print("Misclassification Rate:", misclassification_rate)
        print("Specificity:", specificity)
        print("G-Mean:", g_mean)
        print("ROC/AUC Score:", roc_auc)

import pennylane as qml
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
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2

import time
import warnings
import matplotlib.colors
warnings.filterwarnings('ignore') 


def features_encoding(x, wires):
    for i, qb in enumerate(wires):
        qml.RX(np.arccos(x[i % len(x)]), wires=qb)


def trainable_circuit(params, wires):
    for qb in wires:
        qml.RX(params[qb, 0], wires=qb)
        qml.RY(params[qb, 1], wires=qb)
        qml.RZ(params[qb, 2], wires=qb)

    qml.broadcast(qml.CNOT, wires=wires, pattern="ring")


def encoding(n_qubits=4, backend='default.qubit', noise_model=None, x=None):
    dev = qml.device(backend, wires=n_qubits, noise_model=noise_model)
    features_encoding(x, wires=list(range(n_qubits)))

    @qml.qnode(dev)
    def pqc(weights, x):
        features_encoding(x, wires=list(range(n_qubits)))
        for params in weights:
            trainable_circuit(params, wires=list(range(n_qubits)))
        return qml.expval(qml.PauliZ(0))

    return pqc
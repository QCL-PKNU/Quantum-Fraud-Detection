
import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pennylane as qml
from pennylane import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler

import keras
from keras.models import Sequential
from keras.layers import Dense

print(qiskit.__version__)

import time
import warnings

warnings.filterwarnings('ignore')
from lossfunctions import mse_loss, bin_loss, accuracy_loss
from utils import train_per_epoch

class QAECircuit:

    def __init__(self, qnode: qml.qnode, weights_shape: tuple, loss_function=bin_loss):
        self.qnode = qnode
        self.weights = np.random.randn(*weights_shape, requires_grad=True)
        self.loss_function = loss_function

    def get_circuit_specs(self, weights=None, circuit_input=None):
        if weights is None:
            weights = self.weights
        if circuit_input is None:
            circuit_input = np.random.randn(self.weights.shape[1])
        specs_func = qml.specs(self.qnode)
        return specs_func(weights, circuit_input)

    def display_circuit(self, weights=None, circuit_input=None):
        if weights is None:
            weights = self.weights
        if circuit_input is None:
            circuit_input = np.random.randn(self.weights.shape[1])
        drawer = qml.draw_mpl(self.qnode, show_all_wires=True)
        print(drawer(weights, circuit_input))

    def fit(self, x_train: np.array, y_train: np.array, epochs: int, batch_size: int, optimizer,
            learning_rate: float, threshold: float = 0.5, verbose: bool = True, x_test=None, y_test=None):
        opt = optimizer(learning_rate)
        train_losses, test_losses = [], []

        plt.figure(figsize=(10, 5))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Testing Losses')

        for epoch in range(epochs):
            self.weights = train_per_epoch(qnode=self.qnode, weights=self.weights, 
                                     loss_function=self.loss_function, opt=opt,
                                     batch_size=batch_size, x_train=x_train, y_train=y_train,
                                     epoch_label=str(epoch + 1))

            pred_epoch = self.predict_probas(x_train)
            train_loss = self.loss_function(y_train, pred_epoch)
            train_acc = accuracy_loss(y_train, [int(x >= threshold) for x in pred_epoch])
            train_losses.append(train_loss)

            if x_test is not None:
                y_hat = self.predict_probas(x_test)
                test_loss = self.loss_function(y_test, y_hat)
                test_acc = accuracy_loss(y_test, [int(x >= threshold) for x in y_hat])
                test_losses.append(test_loss)

            if verbose:
                if x_test is None:
                    print(f'Epoch {epoch + 1}/{epochs}, train loss = {train_loss}, train accuracy = {train_acc}')
                else:
                    print(f'Epoch {epoch + 1}/{epochs}, train loss = {train_loss}, train accuracy = {train_acc}, test loss = {test_loss}, test accuracy = {test_acc}')

            with open('losses.txt', 'w') as f:
                f.write('Training Losses:\n')
                f.write('\n'.join(map(str, train_losses)))
                f.write('\nTest Losses:\n')
                f.write('\n'.join(map(str, test_losses)))

            plt.plot(train_losses, label='Train Loss' if epoch == 0 else "", color='blue')
            if x_test is not None:
                plt.plot(test_losses, label='Test Loss' if epoch == 0 else "", color='red')
            plt.legend(loc='upper right')
            plt.pause(0.05)

        plt.show()
        plt.savefig('train_loss.png')
        return train_losses, test_losses

    def predict(self, x_list: np.array):
        return [(self.qnode(self.weights, x) + 1) / 2 for x in x_list]

    def inference(self, x_list: np.array, threshold: float = 0.5):
        probas = self.predict_probas(x_list)
        return [int(x >= threshold) for x in probas]




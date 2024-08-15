from data import load_dataset
import pennylane as qml
from qae import QAECircuit 
from circuits import  *
from utils import calculate_metrics


def main():
    X_train,y_train,X_test, y_test = load_dataset() 
    ### Consturct pqc 
    n_layers = 4
    n_qubits = 4 
    dev = qml.device('default.qubit', wires=n_qubits)


    @qml.qnode(dev)
    def quantum_encoder_circuit(weights, x):
        features_encoding(x, wires=list(range(n_qubits)))

        for params in weights:
                trainable_circuit(params, wires=list(range(n_qubits)))

        return qml.expval(qml.PauliZ(0))

    model = QAECircuit(qnode=quantum_encoder_circuit, weights_shape=[n_layers, n_qubits, 3])
    train_losses, _ = model.fit(x_train=X_train, y_train=y_train, epochs=25, batch_size=16, optimizer=qml.AdamOptimizer,
                            learning_rate=0.001, verbose=True) 
    y_hat_train = model.predict(X_train, threshold=0.7) 
    y_hat_test = model.predict(X_test, threshold=0.7)
    calculate_metrics(y_test, y_hat_test)
    
    
    
    
if __name__ == "__main__":
    main()
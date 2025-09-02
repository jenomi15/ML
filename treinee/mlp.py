import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Carrega o MNIST
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.to_numpy() / 255.0
y = mnist.target.astype(int).to_numpy().reshape(-1, 1)

# One-hot encoding das classes
encoder = OneHotEncoder(sparse_output=False, categories='auto')

y_encoded = encoder.fit_transform(y)

# Divide em treino e teste
x_train, x_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Inicialização dos pesos
def init_weights(shape):
    return np.random.randn(*shape) * 0.01

# Ativações
def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return z > 0

def softmax(z):
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

# Perda
def cross_entropy(y_pred, y_true):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# Inicializa pesos e biases
input_size = 784
hidden1 = 128
hidden2 = 64
output_size = 10

W1 = init_weights((input_size, hidden1))
b1 = np.zeros((1, hidden1))
W2 = init_weights((hidden1, hidden2))
b2 = np.zeros((1, hidden2))
W3 = init_weights((hidden2, output_size))
b3 = np.zeros((1, output_size))

# Treinamento
epochs = 10
learning_rate = 0.1
batch_size = 64
loss_hist = []
acc_hist = []

for epoch in range(epochs):
    perm = np.random.permutation(x_train.shape[0])
    x_train = x_train[perm]
    y_train = y_train[perm]

    for i in range(0, x_train.shape[0], batch_size):
        X_batch = x_train[i:i+batch_size]
        Y_batch = y_train[i:i+batch_size]

        # Forward
        Z1 = X_batch @ W1 + b1
        A1 = relu(Z1)
        Z2 = A1 @ W2 + b2
        A2 = relu(Z2)
        Z3 = A2 @ W3 + b3
        A3 = softmax(Z3)

        # Backward
        dZ3 = A3 - Y_batch
        dW3 = A2.T @ dZ3 / batch_size
        db3 = np.sum(dZ3, axis=0, keepdims=True) / batch_size

        dA2 = dZ3 @ W3.T
        dZ2 = dA2 * relu_deriv(Z2)
        dW2 = A1.T @ dZ2 / batch_size
        db2 = np.sum(dZ2, axis=0, keepdims=True) / batch_size

        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * relu_deriv(Z1)
        dW1 = X_batch.T @ dZ1 / batch_size
        db1 = np.sum(dZ1, axis=0, keepdims=True) / batch_size

        # Atualiza pesos
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    # Avaliação
    Z1 = x_test @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)
    Z3 = A2 @ W3 + b3
    A3 = softmax(Z3)

    loss = cross_entropy(A3, y_test)
    predictions = np.argmax(A3, axis=1)
    labels = np.argmax(y_test, axis=1)
    acc = np.mean(predictions == labels)

    loss_hist.append(loss)
    acc_hist.append(acc)
    print(f"Época {epoch+1}: Perda = {loss:.4f}, Acurácia = {acc:.4f}")

# Gráficos
plt.plot(acc_hist, label='Acurácia')
plt.title('Acurácia por Época')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(loss_hist, label='Perda')
plt.title('Perda por Época')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.grid(True)
plt.legend()
plt.show()

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x = np.concatenate((x_train, x_test)) / 255.0
y = np.concatenate((y_train, y_test))
x = x.reshape((-1, 28 * 28))
y = to_categorical(y, 10)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
def avaliar_mlp(hparams):
    
    neurons = int(hparams[0])  
    neurons2 = int(hparams[3])
    learning_rate = hparams[1]
    dropout_rate = hparams[2]
    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_shape=(784,)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons2, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))


    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=6, batch_size=128, verbose=0)  
    loss, acc = model.evaluate(x_val, y_val, verbose=0)    
    return -acc  



n_particles = 10
n_iterations = 10
dim = 4  
bounds_min = np.array([32, 0.0001, 0.0,32])
bounds_max = np.array([256, 0.01, 0.5,256])
np.random.seed(42)
positions = np.random.uniform(bounds_min, bounds_max, (n_particles, dim)) 
velocities = np.zeros((n_particles, dim)) 
melhores_posico_pesso = positions.copy()
melhor_fit_pess = np.array([avaliar_mlp(p) for p in positions]) 
index_melhor = np.argmin(melhor_fit_pess)
melhor_posi_global= melhores_posico_pesso[index_melhor]
melhor_score_global = melhor_fit_pess[index_melhor]


w = 0.9  
c1 = 1.5  
c2 = 2.6  
for t in range(n_iterations):  
    for i in range(n_particles):
        r1 = np.random.rand(dim)  
        r2 = np.random.rand(dim)

        cognitive = c1 * r1 * (melhores_posico_pesso[i] - positions[i])
        social = c2 * r2 * (melhor_posi_global- positions[i])
        velocities[i] = w * velocities[i] + cognitive + social
        positions[i] += velocities[i]
        positions[i] = np.clip(positions[i], bounds_min, bounds_max)
        score = avaliar_mlp(positions[i])

        if score < melhor_fit_pess[i]:
            melhor_fit_pess[i] = score
            melhores_posico_pesso[i] = positions[i]
        if score < melhor_score_global:
            melhor_score_global = score
            melhor_posi_global= positions[i]

    print(f"iteraçao {t+1}/{n_iterations} - Melhor score: {-melhor_score_global:.4f}")


print("\nMelhores combinacoes :")
print(f"- N na camada : {int(melhor_posi_global[0])}")
print(f"- N na 2ª camada : {int(melhor_posi_global[3])}")
print(f"- Lr: {melhor_posi_global[1]:.5f}")
print(f"- Dropout: {melhor_posi_global[2]:.2f}")

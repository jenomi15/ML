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

# 1. Carregar e preparar os dados
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x = np.concatenate((x_train, x_test)) / 255.0
y = np.concatenate((y_train, y_test))

x = x.reshape((-1, 28 * 28))
y = to_categorical(y, 10)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# 2. Função para treinar e avaliar a MLP
def train_evaluate_mlp(hparams):
    neurons = int(hparams[0])  # os parametros mdo PSO
    learning_rate = hparams[1]
    dropout_rate = hparams[2]

    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_shape=(784,)))   #inicializacao do mlp com os parametros
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=3, batch_size=128, verbose=0)  
    loss, acc = model.evaluate(x_val, y_val, verbose=0)    #treinamento com esses parametros
    return -acc  # PSO minimiza  #volta o fitness score 


# 3. Parâmetros do PSO
n_particles = 10
n_iterations = 10
dim = 3  # [neurônios, lr, dropout]

# Limites dos hiperparâmetros
bounds_min = np.array([32, 0.0001, 0.0])
bounds_max = np.array([256, 0.01, 0.5])

# Inicialização de partículas
np.random.seed(42)
positions = np.random.uniform(bounds_min, bounds_max, (n_particles, dim)) #inicializa as particulas aleatoriamente
velocities = np.zeros((n_particles, dim)) # inicializa as velocidades com 0's

# Inicialização dos valores pessoais e global
personal_best_positions = positions.copy()
personal_best_scores = np.array([train_evaluate_mlp(p) for p in positions]) # ja verifica o group best na inicializacao
global_best_index = np.argmin(personal_best_scores)
global_best_position = personal_best_positions[global_best_index]
global_best_score = personal_best_scores[global_best_index]

# Hiperparâmetros do PSO
w = 0.7   # inércia  # quanto da velocidade anterior a particula mantem (se alto as part exploram mais, se nao , exploram menos)
#como se fosse o momemtum da part, se ela estiver mt rapida, ela continua indo longe

c1 = 1.5  # componente cognitiva ( quanto maior, mais a particula vai dar importancia para o seu melhor ponto e valorizam mais a sua propria experiencia)

c2 = 2.0  # componente social ( o quanto as particulas observam/copiam/seguem o grupo e o otimo global, quanto maior, mais elas vao ir em direcao do melhor do grupo)

# 4. Loop do PSO
for t in range(n_iterations):  # quantidade de iteracoes desejadas
    for i in range(n_particles):
        # Atualizar velocidade
        r1 = np.random.rand(dim)  #sem eles as particulas se moveriam de forma deterministica e igual
        r2 = np.random.rand(dim)

        cognitive = c1 * r1 * (personal_best_positions[i] - positions[i])
        social = c2 * r2 * (global_best_position - positions[i])
        velocities[i] = w * velocities[i] + cognitive + social

        # Atualizar posição
        positions[i] += velocities[i]
        positions[i] = np.clip(positions[i], bounds_min, bounds_max)

        # Avaliar nova posição
        score = train_evaluate_mlp(positions[i])

        # Atualizar melhor pessoal
        if score < personal_best_scores[i]:
            personal_best_scores[i] = score
            personal_best_positions[i] = positions[i]

        # Atualizar melhor global
        if score < global_best_score:
            global_best_score = score
            global_best_position = positions[i]

    print(f"Iteração {t+1}/{n_iterations} - Melhor score: {-global_best_score:.4f}")

# 5. Exibir resultado final
print("\nMelhor configuração encontrada:")
print(f"- Neurônios na camada oculta: {int(global_best_position[0])}")
print(f"- Learning rate: {global_best_position[1]:.5f}")
print(f"- Dropout: {global_best_position[2]:.2f}")

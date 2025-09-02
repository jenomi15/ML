
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)
model = Sequential([
    Flatten(input_shape=(28, 28)),  
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train_cat, epochs=15, validation_data=(x_test, y_test_cat))
test_loss, test_acc = model.evaluate(x_test, y_test_cat)
print(f"acuracia no teste: {test_acc:.4f}")
plt.plot(history.history['accuracy'], label='acuracia treino')
plt.plot(history.history['val_accuracy'], label='acuracia aalidacão')
plt.title('acuracia por epoca')
plt.xlabel('epoca')
plt.ylabel('acuracia')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(history.history['loss'], label='perda treino')
plt.plot(history.history['val_loss'], label='perda validacão')
plt.title('perda por epoca')
plt.xlabel('epoca')
plt.ylabel('perda')
plt.legend()
plt.grid(True)
plt.show()
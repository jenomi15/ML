import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Carregar os dados
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", 
                 names=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"], 
                 na_values="?")

# Substituir valores ausentes com a mediana
imputer = SimpleImputer(strategy="median")
df.iloc[:, :-1] = imputer.fit_transform(df.iloc[:, :-1])

# Normalizar os dados
scaler = StandardScaler()
X = scaler.fit_transform(df.iloc[:, :-1])
y = (df["target"] > 0).astype(int)  # Converter a saída para binária (0: sem doença, 1: com doença)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo com hiperparâmetros fixos
rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=2, criterion="gini", random_state=42)
rf.fit(X_train, y_train)

# Fazer previsões e calcular a acurácia
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Acurácia no conjunto de teste: {acc:.4f}")
import matplotlib.pyplot as plt
import numpy as np

# Importância das features
importances = rf.feature_importances_
feature_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# Plotar a importância das features
plt.figure(figsize=(10, 5))
plt.barh(feature_names, importances, color="royalblue")
plt.xlabel("Importância")
plt.ylabel("Feature")
plt.title("Importância das Features no Random Forest")
plt.show()
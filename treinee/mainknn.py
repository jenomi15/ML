import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Carregar dados
titanic = pd.read_csv("titanic_train.csv")
titanic_clean = titanic[["Survived", "Age", "Fare"]]

titanic_clean.info()

# Tratar valores nulos
titanic_clean["Age"] = titanic_clean["Age"].fillna(titanic_clean["Age"].mean())

# Definir variáveis preditoras e alvo
X = titanic_clean[["Age", "Fare"]] #variavel preditora
Y = titanic_clean["Survived"]  #variavel alvo


# Divisão dos dados em treino e teste
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, random_state=123)

# Converter `y_train` e `y_test` para 1D
y_train = y_train.to_numpy().ravel()
y_test = y_test.to_numpy().ravel()

# Criar modelo KNN e treinar
knn = KNN(n_neighbors=2)
knn.fit(x_train, y_train)

# Fazer previsões
results = knn.predict(x_test)
print("Previsões:", results[:10])  # Mostrar algumas previsões

# Validação com diferentes valores de K
x_train_modeltuning, x_validation, y_train_modeltuning, y_validation = train_test_split(
    x_train, y_train, train_size=0.90, test_size=0.10
)

scores = []
k_s = []

for k in range(2, 41, 2):
    knn = KNN(n_neighbors=k)
    knn.fit(x_train_modeltuning, y_train_modeltuning)
    results_k = knn.predict(x_validation)
    
    score = accuracy_score(y_validation, results_k)
    scores.append(score)
    k_s.append(k)

# Plotar gráfico
sb.lineplot(x=k_s, y=scores)
plt.xlabel("Número de vizinhos (K)")
plt.ylabel("Acurácia")
plt.title("Acurácia do KNN para diferentes valores de K")
plt.show()

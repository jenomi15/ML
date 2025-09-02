from ucimlrepo import fetch_ucirepo 
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = fetch_ucirepo(id=53) 
print(iris.metadata) 
print(iris.variables) 
import pandas as pd
url = "https://archive.ics.uci.edu/static/public/53/data.csv"
iris = pd.read_csv(url)
iris_clean = iris[["sepal length", "sepal width", "class","petal length", "petal width"]]  
iris_clean.info()
X = iris_clean [["sepal length", "sepal width","petal length", "petal width"]] 
Y = iris_clean [["class"]]#variavel alvo
x_train , x_test, y_train , y_test = train_test_split(X,Y,train_size=0.75 , random_state=123)
y_train = y_train.to_numpy().ravel()
y_test = y_test.to_numpy().ravel()

parametros = [
    {"n_neighbors": 3, "metric": "euclidean", "weights": "uniform"},
    {"n_neighbors": 5, "metric": "manhattan", "weights": "distance"},
    {"n_neighbors": 25, "metric": "minkowski", "weights": "distance"}
]
k_list = []
score_list = []

for i in parametros:
    print(f"\nTreinando modelo com n_neighbors={i['n_neighbors']}, metric={i['metric']}, weights={i['weights']}")
    x_train_modeltuning, x_validation, y_train_modeltuning, y_validation = train_test_split(x_train, y_train, train_size=0.90, random_state=123)
    knn = KNN(n_neighbors=i["n_neighbors"], weights=i["weights"], metric=i["metric"])
    knn.fit(x_train_modeltuning, y_train_modeltuning)
    results_k = knn.predict(x_validation)
    score = accuracy_score(y_validation, results_k)
    print(f"Acurácia do modelo: {score:.4f}")
    k_list.append(i["n_neighbors"])
    score_list.append(score)
knn = KNN(n_neighbors= 5 ) 
knn.fit(x_train, y_train)  
results = knn.predict(x_test)
print("previsoes:", results[:10]) 
x_train_modeltuning , x_validation , y_train_modeltuning , y_validation = train_test_split(x_train , y_train , train_size= 0.90 , random_state=123)
score_list = []
k_list = []
for k in range(1,100,1):   #testando ks para 1 a 40
         knn = KNN(n_neighbors= k)
         knn.fit(x_train_modeltuning, y_train_modeltuning)  
         results_k = knn.predict(x_validation)
         score = accuracy_score(y_validation , results_k)
         score_list.append(score)
         k_list.append(k)
sb.lineplot(x=k_list, y=score_list)
plt.xlabel("Número de vizinhos (K)")
plt.ylabel("Acurácia")
plt.title("Acurácia do KNN para diferentes valores de K")
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.impute import SimpleImputer

listaparam = ["gini", "entropy", "log_loss"]
listaparam1 = np.array([100, 200, 300], dtype=int)
numerosrdm = [i for i in range(1, 101)]
listaacc = []
parametros = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10, 8, 9, 21],
    "min_samples_leaf": [1, 2, 4, 8, 10],
    "criterion": ["gini", "entropy", "log_loss"]
}

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
colunas = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
df = pd.read_csv(url, names=colunas)
df.replace("?", np.nan, inplace=True)

imputer = SimpleImputer(strategy="median")
df.iloc[:, :-1] = imputer.fit_transform(df.iloc[:, :-1])

x = df.drop(columns=['target'])
y = df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)

# Decision Tree
dt = DecisionTreeClassifier(random_state=1, min_samples_leaf=4, min_samples_split=2)
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Acurácia do modelo Decision Tree: {accuracy_dt}")

# KNN
knn = KNN(n_neighbors=10)
knn.fit(x_train, y_train)
results = knn.predict(x_test)
acc = accuracy_score(results, y_test)
print(f"{acc} knn foi de ")

def imputar_valores(X_train, proximity_matrix):
    X_train_numeric = X_train.select_dtypes(include=[np.number])  
    X_imputed = X_train_numeric.copy().values 

    for col in range(X_train_numeric.shape[1]):  
        missing_indices = np.where(np.isnan(X_train_numeric.iloc[:, col]))[0]  
        for i in missing_indices:
            weights = proximity_matrix[i, :]
            valid_indices = ~np.isnan(X_train_numeric.iloc[:, col].values)
            known_values = X_train_numeric.iloc[:, col].values[valid_indices]  
            known_weights = weights[valid_indices] 
            X_imputed[i, col] = np.average(known_values, weights=known_weights)  

    return pd.DataFrame(X_imputed, columns=X_train_numeric.columns)

rf = RandomForestClassifier(criterion="gini", n_estimators=100, random_state=1, n_jobs=-1)
rf.fit(x_train, y_train)
leaf_indices = rf.apply(x_train)
proximity_matrix = np.zeros((len(x_train), len(x_train)))

for tree in leaf_indices.T:
    for i in range(len(x_train)):
        for j in range(i, len(x_train)):
            if tree[i] == tree[j]: 
                proximity_matrix[i, j] += 1
                proximity_matrix[j, i] += 1

proximity_matrix /= rf.n_estimators
x_train_imputed = imputar_valores(x_train, proximity_matrix)

for i in listaparam:
    for j in listaparam1:
        for h in numerosrdm:
            rf_imputed = RandomForestClassifier(n_estimators=100, random_state=1)
            rf_imputed.fit(x_train_imputed, y_train)
            accuracy_imputed = rf_imputed.score(x_test[x_train_imputed.columns], y_test)
            print("Acurácia após imputação:", accuracy_imputed)

rf1 = RandomForestClassifier(random_state=1)
grid_search = GridSearchCV(rf1, parametros, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(x_train, y_train)

print("Executando GridSearchCV...")
best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(grid_search.best_params_)

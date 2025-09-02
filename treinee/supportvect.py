import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
colunas = [f"feature_{i}" for i in range(34)] + ["target"]
df = pd.read_csv(url, names=colunas)
# transformar a coluna alvo ('g' e 'b') para binária (1 = good, 0 = bad)
df["target"] = df["target"].map({"g": 1, "b": 0})
x = df.drop(columns=["target"])
y = df["target"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)
x_train_scaled = scale(x_train)
x_test_scaled = scale(x_test)
svm = SVC(random_state=42)
svm.fit(x_train_scaled, y_train)
y_pred = svm.predict(x_test_scaled)
print("acc:", accuracy_score(y_test, y_pred))
print("\nrelatório de classificação:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("predito/resultados")
plt.ylabel("verdadeiro")
plt.title("matriz de Confusão")
plt.show()


param_grid = {
    'C': [0.5,3, 1, 10, 100],
    'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf', 'linear','poly']
}
parametros_optimais = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', verbose=0)
parametros_optimais.fit(x_train_scaled, y_train)
print("melhores parâmetros:", parametros_optimais.best_params_)
melhor_modelo = parametros_optimais.best_estimator_
y_pred_melhor = melhor_modelo.predict(x_test_scaled)
print("acurácia com melhores parâmetros:", accuracy_score(y_test, y_pred_melhor))



resultados = pd.DataFrame(parametros_optimais.cv_results_)
for kernel in resultados["param_kernel"].unique():
    print(f"\n Kernel: {kernel}")
    df_kernel = resultados[resultados["param_kernel"] == kernel]

    if kernel == "linear":
        # Apenas plota acurácia média em função do C
        plt.figure(figsize=(8, 4))
        sns.lineplot(x="param_C", y="mean_test_score", data=df_kernel, marker="o")
        plt.title("acuracia média - Kernel Linear")
        plt.xlabel("Parametro C")
        plt.ylabel("acurácia madia (cross validation)")
        plt.grid(True)
        plt.show()
    else:
        heatmap_data = df_kernel.pivot_table(
            index="param_gamma", 
            columns="param_C", 
            values="mean_test_score",
            aggfunc="mean"
        )
        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu")
        plt.title(f"acurácia média - Kernel {kernel}")
        plt.xlabel("aarâmetro C")
        plt.ylabel("aarâmetro gamma")
        plt.show()

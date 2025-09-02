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

listaparam = [ "gini" , "entropy" , "log_loss"]
listaparam1 = np.array([100, 200, 300], dtype=int)
numerosrdm = [i for i in range(1,101)]
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
#df = df.apply(pd.to_numeric, errors='ignore')   
imputer = SimpleImputer(strategy="median")
df.iloc[:, :-1] = imputer.fit_transform(df.iloc[:, :-1])
#df.fillna(df.mean(), inplace=True)  
#df.dropna(inplace=True) # dropar esses valores ? / nan
x = df.drop(columns=['target'])
y = (df["target"] > 0).astype(int)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)
somatorio = 0
#for i in listaparam:
 #     for j in listaparam1:
  #        for h in numerosrdm:
   #             rdm = RandomForestClassifier(criterion=i,n_estimators=j, random_state=h)
    #            rdm.fit(x_train, y_train)
     #           y_pred = rdm.predict(x_test)
      #          accuracy_random_forest = accuracy_score(y_test, y_pred)
       #         listaacc.append(accuracy_random_forest)
        #        somatorio = somatorio + accuracy_random_forest
         #       print(f"acuracia do criterion {i} {j} {h} foi de {accuracy_random_forest}")
#accmedia = somatorio/9
#print(accmedia)
dt = DecisionTreeClassifier(random_state=1 , min_samples_leaf=4 , min_samples_split=10 , max_depth= None , criterion="entropy")
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)  
print(f"Acurácia do modelo Decision Tree: {accuracy_dt}")
#knn = KNN(n_neighbors=10)
#knn.fit(x_train, y_train)
#results = knn.predict(x_test)
#acc = accuracy_score(results,y_test)
#print(f"{acc} knn foi de ")
dt1 = RandomForestClassifier(random_state=1, min_samples_leaf=4 , min_samples_split=2 , n_estimators= 300, max_depth= None , criterion="entropy")
dt1.fit(x_train, y_train)
y_pred_dt1 = dt1.predict(x_test)
accuracy_dt1 = accuracy_score(y_test, y_pred_dt1)  
print(f"Acurácia do modelo rdm: {accuracy_dt1}")


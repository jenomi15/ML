import pandas as pd 
import numpy as np 
import matplotlib.pyplot as ptl   
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
import glob
caminho =  r"C:\Users\joao\Desktop\decision tree\pastaatp\*.csv"
arquivos_csv = glob.glob(caminho)
df = pd.concat((pd.read_csv(arquivo) for arquivo in arquivos_csv), ignore_index=True)
#print(df)
print(df.dtypes)
#print(df.isnull().sum())
for col in df.select_dtypes(include=['object']).columns:
    print(col)
print((df == "?").sum())
#usar apenas as colunas de winner name
X = df.drop(columns=['winner_name', 'tourney_date','tourney_id' ,'winner_entry','winner_ioc','loser_entry','loser_ioc','score'])
X = pd.get_dummies(X, columns=['tourney_name', 'surface', 'tourney_level', 'winner_hand', 'loser_name', 'loser_hand', 'round'], drop_first=False)
Y = df['winner_name']
Y.fillna(Y.mean(),inplace = True)
X.fillna(X.mean(),inplace = True)
print(Y)
print(X)
u = input("continuar?")
X_train , X_test , Y_train, Y_test = train_test_split(X , Y , test_size= 0.2, random_state=1)
rdm = RandomForestClassifier(n_estimators= 100, random_state= 1)
rdm.fit(X_train , Y_train)
Y_pred = rdm.predict(X_test)
accuracy = accuracy_score(Y_test , Y_pred)
print(accuracy)

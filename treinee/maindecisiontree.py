import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score


wine = load_wine()


print(wine.target) 
print(wine.target_names)  
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target  
input(" ENTER ")
print(df.head()) 
X = df.drop(columns=['target'])  
y = df['target']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

clf1 = DecisionTreeClassifier(max_depth=1, random_state=42)
clf1.fit(X_train, y_train)
y_pred1 = clf1.predict(X_test)

clf2 = DecisionTreeClassifier()
clf2.fit(X_train, y_train)
y_pred2 = clf2.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
accuracy1 = accuracy_score(y_test, y_pred1)
accuracy2 = accuracy_score(y_test, y_pred2)
print(f"Acurácia do modelo: {accuracy:.2%}")
print(f"Acurácia do modelo: {accuracy1:.2%}")
print(f"Acurácia do modelo: {accuracy2:.2%}")
plt.figure(figsize=(15, 10))
plot_tree(clf, feature_names=X.columns, class_names=wine.target_names, filled=True)
plt.show()

plt.figure(figsize=(15, 10))
plot_tree(clf1, feature_names=X.columns, class_names=wine.target_names, filled=True)
plt.show()

plt.figure(figsize=(15, 10))
plot_tree(clf2, feature_names=X.columns, class_names=wine.target_names, filled=True)
plt.show()

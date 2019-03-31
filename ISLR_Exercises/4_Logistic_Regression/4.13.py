import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis 
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_boston

boston= load_boston()

df= pd.DataFrame(boston.data, columns= boston.feature_names)
df.head()

df['CRIM01']= np.where(df['CRIM'] > df['CRIM'].median(), 1, 0)

df= df.drop('CRIM', axis= 1)
print(df.head())

x_all = df.iloc[:,:-1].values
x_6 = df[['INDUS','NOX','AGE','DIS','RAD','TAX']].values
y = df['CRIM01'].values

x_all_train, x_all_test, y_train, y_test = train_test_split(x_all, y, random_state=1)
x_6_train, x_6_test, y_train, y_test = train_test_split(x_6, y, random_state=1)

lr = LogisticRegression()
lr.fit(x_6_train, y_train)
print("Accuracy with x_6  :",accuracy_score(y_test, lr.predict(x_6_test)))
lr.fit(x_all_train, y_train)
print("Accuracy with x_all:",accuracy_score(y_test, lr.predict(x_all_test)))


lda = LinearDiscriminantAnalysis()
lda.fit(x_6_train, y_train)
print("Accuracy with x_6  :",accuracy_score(y_test, lda.predict(x_6_test)))
lda.fit(x_all_train, y_train)
print("Accuracy with x_all:",accuracy_score(y_test, lda.predict(x_all_test)))


qda = QuadraticDiscriminantAnalysis()
qda.fit(x_6_train, y_train)
print("Accuracy with x_6  :",accuracy_score(y_test, qda.predict(x_6_test)))
qda.fit(x_all_train, y_train)
print("Accuracy with x_all:",accuracy_score(y_test, qda.predict(x_all_test)))


best_acc = 0
best_k = 0
for K in range(1,101):
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(x_all_train, y_train)
    acc = accuracy_score(y_test, knn.predict(x_all_test))
    if acc > best_acc:
        best_acc, best_k = acc, K

print('Best accuracy = {:.4f} with K = {:3}'.format(best_acc, best_k))


best_acc = 0
best_k = 0
for K in range(1,101):
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(x_6_train, y_train)
    acc = accuracy_score(y_test, knn.predict(x_6_test))
    if acc > best_acc:
        best_acc, best_k = acc, K

print('Best accuracy = {:.4f} with K = {:3}'.format(best_acc, best_k))
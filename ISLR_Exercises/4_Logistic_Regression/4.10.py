import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns #visualization library

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression #problem will be solved with scikit
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #linear discriminant analysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis #quadratic discriminant analysis
from sklearn.neighbors import KNeighborsClassifier #K nearest neighbours (KNN)

import statsmodels.api as sm #to compute p-values
from patsy import dmatrices

#Part a
df = pd.read_csv('D:/Project/ISLR/ISLR_Exercises/4_Logistic_Regression/Data/Auto.csv')

print(df.head())

df['mpg01']= np.where(df['mpg'] > df['mpg'].median(), 1, 0)

df= df.drop('mpg', axis= 1)
df.head()

#Part b
g= sns.PairGrid(df, size= 2)
g.map_upper(plt.scatter, s=3)
g.map_diag(plt.hist)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.fig.set_size_inches(12, 12)

df.corr()

#Part c
x= df[['cylinders', 'displacement', 'weight']].values
y= df['mpg01'].values
x_train, x_test, y_train, y_test= train_test_split(x, y, random_state= 1)

#Part d
lda= LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)
print(accuracy_score(y_test, lda.predict(x_test)))

#Part e
qda= QuadraticDiscriminantAnalysis()
qda.fit(x_train, y_train)
print(accuracy_score(y_test, qda.predict(x_test)))

#Part f
lr= LogisticRegression()
lr.fit(x_train, y_train)
print(accuracy_score(y_test, lr.predict(x_test)))

#Part g
for K in range(1, 101):
    knn= KNeighborsClassifier(n_neighbors= K)
    knn.fit(x_train, y_train)
    acc= accuracy_score(y_test, knn.predict(x_test))
    print('K = {:3}, accuracy = {:.4f}'.format(K, acc))
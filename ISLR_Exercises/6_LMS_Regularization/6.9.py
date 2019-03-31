import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV

from sklearn.model_selection import train_test_split

df = pd.read_csv('D:/Project/ISLR/ISLR_Exercises/6_LMS_Regularization/Data/College.csv')

df.head()

df.describe()

X= df.iloc[:, 3:]
X= X.iloc[:, :-1]

y= df['Apps']

print(y.head())

X_train, X_test, y_train, y_test= train_test_split(X, y, random_state= 0)

print(X_train, X_test, y_train, y_test)


#Linear Regression

lr= LinearRegression()

lr.fit(X_train, y_train)

y_pred= lr.predict(X_test)

error_lr= np.mean((y_pred - y_test) ** 2)

print(error_lr)

lr.score(X_test, y_test)


# Ridge Regression and Lasso

# Need to standardize the inputs and the output

scaler = StandardScaler()

scaler = StandardScaler().fit(X_train)
X_train_scaled= scaler.transform(X_train)
X_test_scaled= scaler.transform(X_test)

rcv= RidgeCV(alphas= np.linspace(0.01, 100, 1000), cv= 10)

rcv.fit(X_train_scaled, y_train)

RidgeCV(alphas=np.array([1.00000e-02, 1.10090e-01, ..., 9.98999e+01, 1.00000e+02]),
    cv=10, fit_intercept=True, gcv_mode=None, normalize=False,
    scoring=None, store_cv_values=False)

print(rcv.score(X_test_scaled, y_test))

errorRCV= np.mean((rcv.predict(X_test_scaled) - y_test) ** 2)
print(errorRCV)


# Lasso Problemo Similar to Ridge

las= LassoCV(alphas= np.linspace(0.01, 100, 1000), cv= 10)

las.fit(X_train_scaled, y_train)

LassoCV(alphas=np.array([1.00000e-02, 1.10090e-01, ..., 9.98999e+01, 1.00000e+02]),
    copy_X=True, cv=10, eps=0.001, fit_intercept=True, max_iter=1000,
    n_alphas=100, n_jobs=1, normalize=False, positive=False,
    precompute='auto', random_state=None, selection='cyclic', tol=0.0001,
    verbose=False)

print(las.score(X_test_scaled, y_test))

errorLasso= np.mean((las.predict(X_test_scaled) - y_test) ** 2)
print(errorLasso)
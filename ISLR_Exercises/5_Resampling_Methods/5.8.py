import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut  # To use cross-validation in (c); only available after scikit v0.17.1
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm  # To fit models using least squares

np.random.seed(1)

y= np.random.normal(size = 100)

x= np.random.normal(size= 100)

epsilon= np.random.normal(size= 100)

y= x - 2 * x**2 + epsilon

plt.scatter(x, y)

np.random.seed(5)

loo= LeaveOneOut()

df= pd.DataFrame({'x': x, 'y': y})

min_deg= 1
max_deg= 4 + 1
scores= []

for i in range(min_deg, max_deg):
    
    for train, test in loo.split(df):
        
        X_train= df['x'][train]
        y_train= df['y'][train]
        
        X_test= df['x'][test]
        y_test= df['y'][test]
        
        model= Pipeline([('poly', PolynomialFeatures(degree = i)), ('linear', LinearRegression())])
        model.fit(X_train[:, np.newaxis], y_train)
        
        #MSE
        score= mean_squared_error(y_test, model.predict(X_test[:, np.newaxis]))
        scores.append(score)
        
    print('Model %i (MSE): %f' %(i, np.mean(scores)))
    scores= []
    
    
    
# Models with polynomial features
min_deg = 1  
max_deg = 4+1 

for i in range(min_deg, max_deg):
    pol = PolynomialFeatures(degree = i)
    X_pol = pol.fit_transform(df['x'][:,np.newaxis])
    y = df['y']

    model = sm.OLS(y, X_pol)
    results = model.fit()

    print(results.summary())  

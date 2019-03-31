import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm


df = pd.read_csv('D:/Project/ISLR/ISLR_Exercises/3_Linear_Regression/Data/Carseats.csv')
df.head()

#part a multiple regressor for sales with price, urban, US

mod= smf.ols(formula='Sales ~ Price + Urban + US', data=df)
res= mod.fit()
print(res.summary())


#part e to derive a lower order model
mod= smf.ols(formula= 'Sales ~ Price + US', data=df)
res= mod.fit()
print(res.summary())

res_fitted_y= res.fittedvalues

model_residuals= res.resid

model_norm_residuals = res.get_influence().resid_studentized_internal

# absolute squared normalized residuals
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

# absolute residuals
model_abs_resid = np.abs(model_residuals)

# leverage, from statsmodels internals
model_leverage = res.get_influence().hat_matrix_diag

# cook's distance, from statsmodels internals
model_cooks = res.get_influence().cooks_distance[0]
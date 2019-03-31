import pandas as pd
import numpy as np
import patsy
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# statsmodels issue: https://github.com/statsmodels/statsmodels/issues/3931
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

sns.set(style="white")

np.random.seed(1)

df = pd.ExcelFile('D:/Project/ISLR/ISLR_Exercises/5_Resampling_Methods/Data/Default.xlsx')

df['default_yes'] = (df['default'] == 'Yes').astype('int')
df.head()
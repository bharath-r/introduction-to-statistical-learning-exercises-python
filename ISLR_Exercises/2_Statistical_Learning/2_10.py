import pandas as py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Loading the famous Boston dataset
from sklearn.datasets import load_boston

boston= load_boston()

df= pd.DataFrame(boston.data, columns= boston.feature_names)
df['target']= boston.target

print(df)

g= sns.PairGrid(df)
g.map_upper(plt.scatter)
g.map_diag(plt.hist)
g.map_lower(plt.scatter, s=3)
g.fig.set_size_inches(12, 12)

#print correlation
print(df.corrwith(df['CRIM']).sort_values())

ax= sns.boxplot(x="RAD", y="CRIM", data=df)

print(df.loc[df['CRIM'].nlargest(5).index])
print(df.loc[df['TAX'].nlargest(5).index])
print(df.loc[df['PTRATIO'].nlargest(5).index])


df['CHAS'].value_counts()[1]

df['PTRATIO'].median()

df['target'].idxmin()

a = df.describe()
a.loc['range'] = a.loc['max'] - a.loc['min']
a.loc[398] = df.iloc[398]
print(a)


len(df[df['RM']>7])

len(df[df['RM']>8])

df[df['RM']>8].describe()

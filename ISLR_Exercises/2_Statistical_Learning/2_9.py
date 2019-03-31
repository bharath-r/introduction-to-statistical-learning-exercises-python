# -*- coding: utf-8 -*
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('D:/Project/ISLR/ISLR_Exercises/2_Statistical_Learning/Data/Auto.csv')
print(df)

print(df.info())

print(df.horsepower.unique())

df= df[df.horsepower != '?'].copy()
df['horsepower']= pd.to_numeric(df['horsepower'])

print(df.info())

print(df.head())

quantitative = df.select_dtypes(include=['number']).columns
print(quantitative)

qualitative = df.select_dtypes(exclude=['number']).columns
print(qualitative)

a= df.describe()
a.loc['range']= a.loc['max'] - a.loc['min']

print(a.loc[['mean','std','range']])

#Removing 10th through 85th observations
df_b= df.drop(df.index[10:85])
b= df_b.describe()

b.loc['range']= b.loc['max'] - b.loc['min']
print(b.loc[['mean', 'std', 'range']])


g= sns.PairGrid(df, size=2)
g.map_upper(plt.scatter, s=3)
g.map_diag(plt.hist)
g.map_lower(sns.kdeplot, cmap="Blues_d")

g.fig.set_size_inches(12, 12)
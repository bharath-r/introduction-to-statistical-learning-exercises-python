import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

college = pd.read_csv('D:/Project/ISLR/ISLR_Exercises/2_Statistical_Learning/Data/College.csv')
print(college)

college = college.set_index("Unnamed: 0") # The default option 'drop=True', deletes the column
college.index.name = 'Names'
print(college.head())

print(college.describe(include= 'all'))

print(college.describe(include=['number']))

print(college.describe(include= ['object']))

g= sns.PairGrid(college, vars=college.iloc[:,1:11], hue='Private')
g.map_upper(plt.scatter, s=3)
g.map_diag(plt.hist)
g.map_lower(plt.scatter, s=3)
g.fig.set_size_inches(12, 12)

sns.boxplot(x='Private', y='Outstate', data=college)

college.loc[college['Top10perc']>50, 'Elite'] = 'Yes'
college['Elite']= college['Elite'].fillna('No')

sns.boxplot(x='Elite', y='Outstate', data=college)


# Bins creation
college['PhD'] = pd.cut(college['PhD'], 3, labels=['Low', 'Medium', 'High'])
college['Grad.Rate'] = pd.cut(college['Grad.Rate'], 5, labels=['Very low', 'Low', 'Medium', 'High', 'Very high'])
college['Books'] = pd.cut(college['Books'], 2, labels=['Low', 'High'])
college['Enroll'] = pd.cut(college['Enroll'], 4, labels=['Very low', 'Low', 'High', 'Very high'])

fig = plt.figure()

plt.subplot(221)
college['PhD'].value_counts().plot(kind='bar', title = 'Private');
plt.subplot(222)
college['Grad.Rate'].value_counts().plot(kind='bar', title = 'Grad.Rate');
plt.subplot(223)
college['Books'].value_counts().plot(kind='bar', title = 'Books');
plt.subplot(224)
college['Enroll'].value_counts().plot(kind='bar', title = 'Enroll');

fig.subplots_adjust(hspace=1) # To add space between subplots

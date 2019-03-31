import numpy as np
import pandas as pd

d= {
    'X1': pd.Series([0,2,0,0,-1,1]),
    'X2': pd.Series([3,0,1,1,0,1]),
    'X3': pd.Series([0,0,3,2,1,1]),
    'Y': pd.Series(['Red', 'Red', 'Red', 'Green', 'Green', 'Red'])
    }

df= pd.DataFrame(d)
df.index= np.arange(1, len(df) + 1)
#print(df)

#%% Euclidean Distance

#from math import sqrt
df['distance']= np.sqrt(df['X1']**2 + df['X2']**2 + df['X3']**2)
#print(df)

# Sort distance
df.sort_values(['distance'])
print(df)
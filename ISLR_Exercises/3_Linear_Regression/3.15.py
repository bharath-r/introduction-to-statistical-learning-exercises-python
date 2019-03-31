import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm

from pandas.tools.plotting import scatter_matrix 
from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston

boston= load_boston()

print(boston['DESCR'])

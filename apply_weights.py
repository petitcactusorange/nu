
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets, linear_model

from math import log10, floor
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter


sns.set(style="ticks")

from sklearn.externals import joblib
clf = joblib.load('weights/filename.pkl')
file_location = 'data/solid-data-cosmics-v1.csv'
df = pd.read_csv(file_location)
data = df
features_forlearning = ["ix1","iy1","iz1","E111","E112","E113","E121","E122","E123","E131"
          ,"E132","E133","E211","E212","E213","E221","E222","E223","E231","E232",
          "E233","E311","E312","E313","E321","E322","E323","E331","E332","E333"]
X = data[features_forlearning]

predicted_energy= clf.predict(X)
print len(predicted_energy)

#-----------------------------
file_location_ibd = 'data/solid-data-ibd-v2.csv'
df_ibd = pd.read_csv(file_location_ibd)
data_ibd = df_ibd
true_energy = data_ibd["trueE"]
sumE = data['sumE']
ECube2= data ['ECube2']

opts = dict(alpha=0.6, bins=100, range=(-5,20),  normed=1 , histtype='stepfilled')

#plt.figure(facecolor="white")
#plt.figure.add_subplot(1, 1, 1, axisbg='red')
plt.hist(predicted_energy,color = 'orange',  label="'Learned' Positron energy - Cosmics", **opts)
plt.hist(true_energy, color = 'red', label = 'True IBD', **opts)
plt.hist(sumE,  color = 'yellow', label = 'sumE', **opts)
#plt.hist(ECube2, color = 'blue', label = 'ECube2', **opts)
plt.xlabel("Positron energy [MeV] ")

plt.legend(loc='right')
plt.title("data IBD events")
#plt.show()
plt.savefig("figures/predicted_energy.pdf")
plt.clf()





print "really you managed?"

#from operator import itemgetter

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

print "-----------------------------------------------------------------------------------"
file_location = 'data/solid-data-ibd-v2.csv'
print 'We will use here the file : '+file_location

df = pd.read_csv(file_location)




print 'The number of entries in this files is  : '+str( len (df.index))
#print len(df.index)
print 'Here is the list of variables in this file : '
print ', '.join(df.columns)
print "------------------------------------------------------------------------------------"



debug = False



features_forlearning = ["ix1","iy1","iz1","E111","E112","E113","E121","E122","E123","E131"
          ,"E132","E133","E211","E212","E213","E221","E222","E223","E231","E232",
          "E233","E311","E312","E313","E321","E322","E323","E331","E332","E333"]

print "This is the list of  variables (features) that we will use for leaning : "
print ', '.join(features_forlearning)
print "------------------------------------------------------------------------------------"

data = df[ df.trueE >-500. ] #reduce to keep only the data

# X is the list of features that we can use for learning
X = data[features_forlearning]


# trueE is the variable that we want to learn "target"
Positron_Energy_trueE = data['trueE']  #Now "Positron_Energy_trueE_trueE is a numpy array ohle "
#ECube2, and sumE are two sets of variables that we can compare
Positron_Energy_ECube2 = data['ECube2']
Positron_Energy_sumE = data['sumE']










# First we split our sample in Eval sample and Dev Sample
X_dev,X_eval,  Positron_Energy_trueE_dev, Positron_Energy_trueE_eval = train_test_split(X.as_matrix(),Positron_Energy_trueE.as_matrix(), train_size =0.5,   random_state=9548)
# Second we split the Dev Sample in Train Sample and Test Sample
X_train,X_test,  Positron_Energy_trueE_train, Positron_Energy_trueE_test = train_test_split(X_dev,  Positron_Energy_trueE_dev, train_size =0.5, random_state=2171)


# we will do something similar with the other estimators just to keep the nparrays of the same size
XX_dev, XX_eval, Positron_Energy_ECube2_dev, Positron_Energy_ECube2_eval = train_test_split(X.as_matrix(), Positron_Energy_ECube2.as_matrix(), train_size =0.5, random_state = 9548)
XXX_dev, XXX_eval, Positron_Energy_sumE_dev, Positron_Energy_sumE_eval = train_test_split(X.as_matrix(), Positron_Energy_sumE.as_matrix(), train_size =0.5, random_state = 9548)
XX_train,XX_test,  Positron_Energy_ECube2_train, Positron_Energy_ECube2_test = train_test_split(XX_dev,  Positron_Energy_ECube2_dev, train_size =0.5, random_state=2171)
XXX_train,XXX_test,  Positron_Energy_ECube2_train, Positron_Energy_ECube2_test = train_test_split(XXX_dev,  Positron_Energy_ECube2_dev, train_size =0.5, random_state=2171)

#Always useful to show some information
total_sample_size = float (Positron_Energy_trueE.size)
print "dev  sample fraction  : "+ str('%.1f' %(Positron_Energy_trueE_dev.size*100/total_sample_size))+ "%"
print "eval sample fraction  : "+str('%.1f' % (Positron_Energy_trueE_eval.size*100/total_sample_size))+ "%"
print "train sample fraction : "+str('%.1f' % (Positron_Energy_trueE_train.size*100/total_sample_size))+ "%"
print "test sample fraction  : "+str('%.1f' %(Positron_Energy_trueE_test.size*100/total_sample_size))+ "%"

#  ---- > Now we are cooking with Gas
#GradientBoostingRegressor
clf = GradientBoostingRegressor(n_estimators=2000,
                                learning_rate=0.05,
                                subsample=0.7296635812586656,
                                max_features=0.5779758681446561,
                                max_depth=5,
                                loss='ls')



# have a look at GridSearchCV

#RandomForestRegressor
"""
clf = RandomForestRegressor(n_estimators=500, criterion='mse', max_depth=None, min_samples_split=2,
                            min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                            max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1,
                            random_state=None,verbose=2, warm_start=False)
"""





print "------------------------------------------------------------------------------------"
print "start the training ... "
clf.fit(X_train, Positron_Energy_trueE_train)
print "done with the training ... "

Positron_Energy_trueE_pred_train = clf.predict(X_train)
Positron_Energy_trueE_pred_test = clf.predict(X_test)
Positron_Energy_trueE_pred_eval = clf.predict(X_eval)

print "evaluating..."
from sklearn.externals import joblib
joblib.dump(clf, 'weights/filename.pkl')
print "Save the output of the training to an external file !"

print "MSE on training set:",
print ('%.3f' % mean_squared_error(Positron_Energy_trueE_train, Positron_Energy_trueE_pred_train))
print "MSE on testing set:",
print ('%.3f' % mean_squared_error(Positron_Energy_trueE_test, Positron_Energy_trueE_pred_test))
print "MSE on evaluation set:",
print ('%.3f' %mean_squared_error(Positron_Energy_trueE_eval, Positron_Energy_trueE_pred_eval))

print "------------------------------------------------------------------------------------"



#figure = Figure()
#canvas = FigureCanvas(figure)
#axes = figure.add_subplot(1, 1, 1, axisbg='red')
#axes.plot([1,2,3])


opts = dict(alpha=0.6, bins=100, range=(-5,20),  normed=1 , histtype='stepfilled')

#plt.figure(facecolor="white")
#plt.figure.add_subplot(1, 1, 1, axisbg='red')
plt.hist(Positron_Energy_trueE_eval,color = 'red',  label="'True' Positron energy ", **opts)
plt.hist(Positron_Energy_trueE_pred_eval, color = 'purple', label="'Learned' Positron energy", **opts)
plt.hist(Positron_Energy_sumE_eval , color = 'yellow', label = " Sum all energy", **opts)
plt.xlabel("Positron Energy [MeV]")

plt.legend(loc='right')
plt.title("data IBD events")
plt.savefig("figures/positron_energy.pdf")
plt.clf()

#--------------------------------------------------------------
def mean_and_std(values):
    average = np.average(values)
    variance = np.average((values-average)**2)

    return (average, np.sqrt(variance))
#--------------------------------------------------------------
# Make the residual plot true - predicted
#--------------------------------------------------------------
delta_true_learned = Positron_Energy_trueE_eval - Positron_Energy_trueE_pred_eval
delta_true_sumE = Positron_Energy_trueE_eval - Positron_Energy_sumE_eval
delta_true_ECube2 = Positron_Energy_trueE_eval - Positron_Energy_ECube2_eval



#fig = figure(figsize=(8,6), dpi=80)

_=plt.hist(delta_true_learned,alpha=0.6,bins=40, color = "purple",  range=(-10,10),normed=True , histtype='stepfilled')
plt.legend(loc='best')
plt.xlabel("(True Energy - Pred energy) MeV/c2 ")
plt.grid = True
#ax = plt.axes()
#ax.yaxis.grid() # horizontal lines
#ax.xaxis.grid() # vertical lines

plt.savefig("figures/residual_true_learned.pdf")
plt.clf()


_=plt.hist(delta_true_sumE,alpha=0.6,bins=40, color = "yellow",  range=(-10,10),normed=True, histtype='stepfilled')
plt.legend(loc='best')
plt.xlabel("(True Energy - SumE ) MeV/c2 ")
plt.savefig("figures/residual_true_sumE.pdf")
plt.clf()



_=plt.hist(delta_true_ECube2,alpha=0.6,bins=40, color = "blue",  range=(-10,10),normed=True, histtype='stepfilled')
plt.legend(loc='best')
plt.xlabel("(True Energy -  EnergyCube2) MeV/c2 ")
plt.savefig("figures/residual_true_ECube2.pdf")
plt.clf()




#delta_inbins =  Positron_Energy_trueE_eval - Positron_Energy_trueE_pred_eval




x = Positron_Energy_trueE_eval
y = Positron_Energy_trueE_eval - Positron_Energy_trueE_pred_eval



plt.hist2d(x,y, bins=50, cmap=plt.cm.YlOrRd)
plt.colorbar()
plt.xlabel(" True Energy [MeV/c2]")
plt.ylabel(" True Energy - Predicted Energy [MeV/c2] ")
plt.savefig("figures/two_dim_trueE_vs_residual_trueE_learned.pdf")
plt.clf()




x = Positron_Energy_trueE_eval
y = Positron_Energy_trueE_eval - Positron_Energy_sumE_eval
plt.hist2d(x,y, bins=50, cmap=plt.cm.YlOrRd)
plt.colorbar()
plt.xlabel(" True Energy [MeV/c2]")
plt.ylabel(" True Energy - SumE [MeV/c2] ")
plt.savefig("figures/two_dim_trueE_vs_residual_trueE_sumE.pdf")
plt.clf()



x = Positron_Energy_trueE_eval
y = Positron_Energy_trueE_eval - Positron_Energy_ECube2_eval

plt.hist2d(x,y, bins=50, cmap=plt.cm.YlOrRd)
plt.colorbar()
#plt.scatter(x, y, c='blue', alpha=0.5, marker = 'o')
plt.xlabel(" True Energy [MeV/c2]")
plt.ylabel(" True Energy - ECube2[MeV/c2] ")
plt.savefig("figures/two_dim_trueE_vs_residual_trueE_Ecube2.pdf")
plt.clf()








print "noow we will a simple linear regression ... "


# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets




# First we split our sample in Eval sample and Dev Sample
#X_dev,X_eval,  Positron_Energy_trueE_dev, Positron_Energy_trueE_eval = train_test_split(X.as_matrix(),Positron_Energy_trueE.as_matrix(), train_size =0.5,   random_state=9548)
# Second we split the Dev Sample in Train Sample and Test Sample
#X_train,X_test,  Positron_Energy_trueE_train, Positron_Energy_trueE_test = train_test_split(X_dev,  Positron_Energy_trueE_dev, train_size =0.5, random_state=2171)






reg_data_x = Positron_Energy_trueE_eval
reg_data_y = Positron_Energy_trueE_eval - Positron_Energy_trueE_pred_eval




# Split the data into training/testing sets
reg_data_x_train = reg_data_x[:-20]
reg_data_x_test = reg_data_x[-20:]

# Split the targets into training/testing sets
#diabetes_y_train = diabetes.target[:-20]
#diabetes_y_test = diabetes.target[-20:]


reg_data_y_train = reg_data_y[:-20]
reg_data_y_test = reg_data_y[-20:]



print len (reg_data_x_train)
print len (reg_data_y_train)



#regr.fit(reg_data_x_train, reg_data_y_train)
#regr.fit(X_train, Positron_Energy_trueE_train)

# The coefficients
#print('Coefficients: \n', regr.coef_)
# The mean square error
#print("Residual sum of squares: %.2f"
#      % np.mean((regr.predict(x_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % regr.score(x_test, y_test))

# Plot outputs
#plt.scatter(x_test, y_test,  color='black')
#plt.plot(x_test, regr.predict(y_test), color='blue',linewidth=3)

#plt.xticks(())
#plt.yticks(())


#plt.show()
#plt.hist2d(x, y, bins=40)
#plt.colorbar()
#plt.show()



#print ('%.3f' % mean_and_std(delta))




#plt.rc('grid', linestyle="-", color='#316931', linewidth =0.1 )

#fig = plt.figure()

#fig.savefig("bloodyfigure.pdf")
#plt.hist(trueE.,color = 'red',  label="'True' Positron energy ", **opts)

#df['trueE'].plot()
#fig.show()
#fig.clf()



'''
x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)

y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
y2 = np.cos(2 * np.pi * x2)

plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'yo-')
plt.title('A tale of 2 subplots')
plt.ylabel('Damped oscillation')

plt.subplot(2, 1, 2)
plt.plot(x2, y2, 'r.-')
plt.xlabel('time (s)')
plt.ylabel('Undamped')

plt.clf()

"""


'''

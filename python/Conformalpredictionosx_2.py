# %%

import sys
sys.path.append('/Users/williamlopez/Documents/Maastricht University/Internship paper/William/Slobs/new/Python')

import os
import six
import numpy as np
import pandas as pd
import seaborn as sns
import time
import matplotlib.pyplot as plt
from sklearn import tree
import joblib
from imblearn.over_sampling import SMOTE
import lime
import lime.lime_tabular
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn import ensemble
import tqdm
from nonconformist.cp import TcpClassifier
from nonconformist.nc import NcFactory
from nonconformist.cp import IcpClassifier
from sklearn.ensemble import RandomForestClassifier
from nonconformist.evaluation import cross_val_score
from nonconformist.evaluation import ClassIcpCvHelper, RegIcpCvHelper
from nonconformist.evaluation import class_avg_c, class_mean_errors, class_one_err, class_one_c, class_empty, class_two_c
from nonconformist.evaluation import reg_mean_errors, reg_median_size

from nonconformist.base import ClassifierAdapter, RegressorAdapter
from nonconformist.icp import IcpClassifier, IcpRegressor
from nonconformist.nc import MarginErrFunc
from nonconformist.nc import ClassifierNc, RegressorNc, RegressorNormalizer
from nonconformist.nc import AbsErrorErrFunc, SignErrorErrFunc

from nonconformist.base import OobClassifierAdapter, OobRegressorAdapter
from nonconformist.icp import OobCpClassifier, OobCpRegressor

from Utilities import Preprocessing as preprocessing
# %%
#* Loaing and formatting data 
path = os.getcwd()

path =  '/Users/williamlopez/Documents/Maastricht University/Internship paper/William/Slobs/new'
test = pd.read_csv(path + '/data/slobs_2019_06_test_all.csv')
cvtrain = pd.read_csv(path + '/data/slobs_2019_05_cvtrain_all.csv')
train = pd.read_csv(path + '/data/slobs_2019_04_train.csv')
val = pd.read_csv(path + '/data/slobs_2019_05_val.csv')

prepro = preprocessing()

x_train, x_val, x_test, y_train, y_val, y_test, x_cvtrain, y_cvtrain, x_train_unbalanced, y_train_unbalanced, x_cvtrain_unbalanced, y_cvtrain_unbalanced, og_features, og_features_train, catcolumns, numcolumns = prepro.Get_sets(train, val, test, cvtrain)

# %%
#*loading feature sets from the feature selection results

x_cvtrain, x_cvtrain_unbalanced, x_train, x_train_unbalanced, x_val, x_test = prepro.FeatureSelection(og_features, og_features_train, x_cvtrain, x_cvtrain_unbalanced, x_train, x_train_unbalanced, x_val, x_test)


# %%
model = RandomForestClassifier(n_estimators = 300, n_jobs =-1)

# %%
nc = NcFactory.create_nc(model)	# Create a default nonconformity function
icp = IcpClassifier(nc)			# Create an inductive conformal classifier

x_train_np = x_cvtrain_unbalanced.to_numpy(copy = True)
x_val_np = x_val.to_numpy(copy = True)
x_test_np = x_test.to_numpy(copy = True)

y_train_np = y_cvtrain_unbalanced.to_numpy(copy = True)
y_val_np = y_val.to_numpy(copy = True)
y_test_np = y_test.to_numpy(copy = True)

# %%
print('fitting inductive conformal predictor')
# Fit the ICP using the proper training set
icp.fit(x_train_np, y_train_np)
print('calibrating inductive conformal prediction')
# Calibrate the ICP using the calibration set
icp.calibrate(x_val_np, y_val_np)
print('predicting inductive conformal prediction')
# Produce predictions for the test set, with confidence 95%
prediction = icp.predict(x_test_np, significance=0.05)
prediction_conf_cred = pd.DataFrame(icp.predict_conf(x_test_np),
                                    columns=['Label', 'Confidence', 'Credibility'])
# %%
#Cross validation of the conformal predictor

#icp = IcpClassifier(ClassifierNc(ClassifierAdapter(model),MarginErrFunc()))

icp = OobCpClassifier(ClassifierNc(OobClassifierAdapter(RandomForestClassifier(n_estimators=300, oob_score=True))))

significance = np.arange(0,1,0.025)
significance[0] = 0.01
icp_cv = ClassIcpCvHelper(icp)

scores = cross_val_score(icp_cv,
                         x_train_np,
                         y_train_np,
                         iterations=1,
                         folds=5,
                         scoring_funcs=[class_mean_errors, class_one_err, class_avg_c, class_one_c, class_empty, class_two_c],
                         significance_levels=significance,
                         verbose = True)

print('Classification: SLOBS')
scores = scores.drop(['fold', 'iter'], axis=1)

scores2 = scores.groupby(['significance']).mean()
print(scores2)

#%%

scores2 = scores2.reset_index()


# %%
#cross validated cp

icp = OobCpClassifier(ClassifierNc(OobClassifierAdapter(RandomForestClassifier(n_estimators=100, oob_score=True))))


x_train_np = x_train.to_numpy(copy = True)
x_val_np = x_val.to_numpy(copy = True)
x_test_np = x_test.to_numpy(copy = True)

y_train_np = y_train.to_numpy(copy = True)
y_val_np = y_val.to_numpy(copy = True)
y_test_np = y_test.to_numpy(copy = True)


print('fitting inductive conformal predictor')
# Fit the ICP using the proper training set
icp.fit(x_train_np, y_train_np)

print('predicting inductive conformal prediction')
# Produce predictions for the test set, with confidence 95%
prediction = icp.predict(x_test_np, significance=0.05)

# %%
scores3 = scores2.copy(deep = True)
scores3.reset_index(inplace = True)
# %%

#standard icp
icp = IcpClassifier(ClassifierNc(ClassifierAdapter(model)))
icp.fit(x_train_np, y_train_np)
icp.calibrate(x_val_np, y_val_np)
predictions_knncv_icp = icp.predict(x_test_np, significance = None)

predictions_knncv_icp_df = pd.DataFrame(data = predictions_knncv_icp, columns = ['0','1'])

predictions_knncv_icp_df['target'] = y_test_np
scoring_funcs = [class_mean_errors, class_one_err, class_avg_c, class_one_c, class_empty, class_two_c]

j=0
predictions_knncv_icp =  predictions_knncv_icp_df.to_numpy(copy = True)

columns = ['fold','significance',] + [f.__name__ for f in scoring_funcs]
df2 = pd.DataFrame()

for k, s in enumerate(significance):

    #********Knn****************#
    scores2 = [scoring_func(predictions_knncv_icp[:,:2], predictions_knncv_icp[:,2], s) for scoring_func in scoring_funcs]
    df_score2 = pd.DataFrame([[j, s] + scores2], columns=columns)
    df2 = df2.append(df_score2, ignore_index=True)    
    
        
    

df2 = df2.drop(['fold'], axis=1)
resultscv_icp = df2.copy(deep = True)











#%%



plt.title('Error rate of predictions containing only a single output label')

plt.scatter( x = scores2.significance, y = scores2.class_one_err)
plt.plot(scores2.significance, scores2.class_one_err)
plt.xlabel('significance level')
plt.ylabel('Class one errors')
plt.show()
# %%

plt.title('Average number of classes per prediction')
plt.scatter( x = scores2.Significance, y = scores2.class_avg_c)
plt.plot(scores2.significance, scores2.class_avg_c)
plt.xlabel('significance level')
plt.ylabel('Avg N classes')
plt.show()


# %%

plt.title('Rate of prediction sets containing only a single class label')
plt.scatter( x = scores2.significance, y = scores2.class_one_c)
plt.plot(scores2.significance, scores2.class_one_c)
plt.xlabel('significance level')
plt.ylabel('One class rate')
plt.show()


# %%

plt.title('Rate of empty prediction sets ')
plt.scatter( x = scores2.significance, y = scores2.class_empty)
plt.plot(scores2.significance, scores2.class_empty)
plt.xlabel('significance level')
plt.ylabel('Empty sets')
plt.show()

# %%


f, ax = plt.subplots(figsize=(6, 6))
plt.title('Average error rate')
ax.scatter(x = scores2.significance, y = scores2.class_mean_errors)
ax.plot(scores2.significance, scores2.class_mean_errors)
##ax.plot([0, 1], [0, 1],ls="--", transform=ax.transAxes)
lims = [np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes 
        np.max([ax.get_xlim(),ax.get_ylim()]),  # max of both axes
        ]
ax.plot(lims, lims, ls="--",  alpha=0.75, zorder=0) #ax1.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
plt.xlabel('Significance level')
plt.ylabel('Average error rate')
plt.show()

#%%
#printing all at the same time

scores2 = resultscv_icp.copy(deep = True)


x1 = scores2['significance']
x2 = scores2['significance']
x3 = scores2['significance']

y1 = scores2['class_two_c']
y2 = scores2['class_one_c']
y3 = scores2['class_empty']

n1 = 'Two class rate'
n2 ='One class rate'
n3 = 'Empty sets'


fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.title('Output set rate ICP')

ax1.scatter(x1, y1, s = 10, marker = "s", label = n1)
ax1.plot(x1, y1)


ax1.scatter(x2, y2, s = 10, marker = "o", label = n2)
ax1.plot(x2, y2)

ax1.scatter(x3, y3, s = 10, marker = "*", label = n3)
ax1.plot(x3, y3)

plt.legend(loc='upper right')
plt.xlabel('significance level')
#plt.ylabel(self.y_ax)
plt.show()


#%%
scores2.to_excel('/Users/williamlopez/Documents/Maastricht University/Internship paper/William/Slobs/new/output/ICPtestscores.xlsx')

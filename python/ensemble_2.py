#%%


import sys

sys.path.append("C:\\William\\Slobs\\new\\Python\\")
import numpy as np
import pandas as pd
import seaborn as sns
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,  cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.ensemble import AdaBoostClassifier
from scipy.optimize import minimize
from sklearn import ensemble
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
plt.style.use('default')
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from minepy import MINE
import lightgbm as lgb
from sklearn.externals import joblib
from scipy.spatial import distance
import scipy.stats as stats
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
from skopt.space import Integer, Categorical, Real
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt import callbacks


from sklearn.metrics import classification_report, roc_curve
from sklearn.metrics import roc_auc_score

from sklearn.calibration import calibration_curve, CalibratedClassifierCV

from Utilities import Preprocessing as preprocessing

#%%
#* Loaing and formatting data 
path = os.getcwd()

test = pd.read_csv(path + '\\data\\slobs_2019_06_test_all.csv')
cvtrain = pd.read_csv(path + '\\data\\slobs_2019_05_cvtrain_all.csv')
train = pd.read_csv(path + '\\data\\slobs_2019_04_train.csv')
val = pd.read_csv(path + '\\data\\slobs_2019_05_val.csv')

prepro = preprocessing()

x_train, x_val, x_test, y_train, y_val, y_test, x_cvtrain, y_cvtrain, x_train_unbalanced, y_train_unbalanced, x_cvtrain_unbalanced, y_cvtrain_unbalanced, og_features, og_features_train, catcolumns, numcolumns = prepro.Get_sets(train, val, test, cvtrain)

# %%
#*loading feature sets from the feature selection results

x_cvtrain, x_cvtrain_unbalanced, x_train, x_train_unbalanced, x_val, x_test = prepro.FeatureSelection(og_features, og_features_train, x_cvtrain, x_cvtrain_unbalanced, x_train, x_train_unbalanced, x_val, x_test)
# %%
#*loading models
lgbm_calibrator = joblib.load()




# %%
#*loading results
#! this are the names to keep
model_evalcv = pd.read_excel(path + '//output//final_model_comparison.xlsx', index_col= 0)
model_predict_test = pd.read_excel(path + '//output//final_model_predictions_test.xlsx', index_col= 0)

# %%
#majority voting
#weights=[0.253,0,0.47,0, 0.258]
#weights=[0.35,0,0.35,0, 0.3]
weights=[0.46,0,0.3,0, 0.24]
#eclf_hard = EnsembleVoteClassifier(clfs=[lgbm, knn, rf, ada, extra], weights = weights, refit=False, voting = 'hard')
eclf_soft = EnsembleVoteClassifier(clfs=[ lgbm_calibrator, knn_calibrator, rf_calibrator, ada_calibrator, extra_calibrator], weights = weights, refit=False, voting = 'soft', verbose = 1)

print('fitting')
#eclf_hard.fit(train_xm, train_y)
eclf_soft.fit(train_xm, train_y)
print('predicting')
#eclf_hard_pred = eclf_hard.predict(val_xm)
#eclf_hard_pred_pr = eclf_hard.predict_proba(val_xm)

eclf_soft_pred = eclf_soft.predict(val_xm)
eclf_soft_pred_pr = eclf_soft.predict_proba(val_xm)

#evaluating majority voting
voter_pred = eclf_soft_pred
voter_pred_pr = eclf_soft_pred_pr

acc_voter = accuracy_score(val_y, voter_pred)
roc_voter = roc_auc_score(val_y, voter_pred_pr[:,1])
f1_voter = f1_score(val_y, voter_pred)
precision_voter = precision_score(val_y, voter_pred)
recall_voter = recall_score(val_y, voter_pred)
log_loss_voter = log_loss(val_y, voter_pred_pr)
print('accuracy voter: ',  acc_voter)
print('roc voter: ',  roc_voter)
print('f1 voter: ',  f1_voter)
print('precision voter: ',  precision_voter)
print('recall voter: ',  recall_voter)
print('log loss voter: ',  log_loss_voter)

#print('accuracy:', np.mean(y == eclf.predict(X)))

# %%
#gettin dem preds

clfs = ['LGBMClassifier_pr_cal', 'RandomForestClassifier_pr_cal', 'AdaBoostClassifier_pr_cal', 'ExtraTreesClassifier_pr_cal']

Nfeval = 1


predictions = []
for clf in clfs:
    predictions.append(model_predict_test[clf])

# %%
#*optimizing the weights
Nfeval = 1
def log_loss_func(weights, info):
    ''' scipy minimize will pass the weights as a numpy array '''
    
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction
    if info['Nfeval']%10 == 0:
        print ('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f} {5: 3.6f}  '.format(info['Nfeval'], weights[0], weights[1], weights[2], weights[3], log_loss(y_test, final_prediction)))
    info['Nfeval'] += 1
            


    return log_loss(y_test, final_prediction)
    #return -1*roc_auc_score(val_y, final_prediction)


#the algorithms need a starting value, right not we chose 0.5 for all weights
#its better to choose many random starting points and run minimize a few times
#starting_values = [0.5]*len(predictions)
starting_values =[0.2,0.2,0.2,0.2]


cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
#our weights are bound between 0 and 1
bounds = [(0,1)]*len(predictions)
#print ( '{0}   {1}   {2}   {3}   {4} {5} {6} '.format('Iter', 'X1', 'X2', 'X3', 'X4', 'X5', 'Fx'))
res = minimize(log_loss_func, starting_values, method='SLSQP', args=({'Nfeval':0},), bounds=bounds, constraints=cons, options = {'maxiter':1000})

print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
print('Best Weights: {weights}'.format(weights=res['x']))
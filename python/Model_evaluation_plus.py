#%%

import sys
sys.path.append('/Users/williamlopez/Documents/Maastricht University/Internship paper/William/Slobs/new/Python')

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,  cross_val_score
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn import ensemble
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
plt.style.use('default')
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
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
#path = os.getcwd()
#path = path + '/Documents/Maastricht University/Internship paper/William/Slobs/new/'
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

#*loading best parameters per model

lgbm_results = load(path + '//models//lgbm_hpo_results_gp.pkl')
rf_results = load(path + '//models//rf_hpo_results_gp.pkl')
ada_results = load(path + '//models//ada_hpo_results_gp.pkl')
extra_results = load(path + '//models//extra_hpo_results_gp.pkl')



# %%
mla = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(n_estimators = 500),
    ensemble.ExtraTreesClassifier(n_estimators = 500, n_jobs =-1),
    ensemble.RandomForestClassifier(n_estimators = 500, n_jobs =-1),
    lgb.LGBMClassifier(boosting_type = 'gbdt',
                      objective ='binary',
                      metric = 'auc',
                      num_threads = 4,
                      learning_rate = 0.21183,
                      max_depth = 15,
                      num_leaves = 41,
                      min_data_in_leaf = 298,
                      feature_fraction = 0.623818,
                      subsample = 0.583636)   
    ]

# %%
#train all models

#cv_split = model_selection.StratifiedShuffleSplit(n_splits = 5, test_size = .3, train_size = .7, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%

#create table to compare mla metrics
mla_columns = ['Model', 
               'Model_Parameters',
               'Train_Accuracy_Mean',
               'Test_Accuracy_Mean',
               'Train_Roc_auc_Mean',
               'Test_Roc_auc_Mean' ,
               'Train_f1_Mean',
               'Test_f1_Mean',
               'Train_precision_Mean',
               'Test_precision_Mean',
               'Train_recall_Mean',
               'Test_recall_Mean',
               'Test_Roc_auc_3*STD',
               'Test_Accuracy_3*STD',
               'Test_f1 3*STD',
               'Test_precision_3*STD',
               'Test_recall_3*STD', 
               'Time'
               ]
mla_compare = pd.DataFrame(columns = mla_columns)
temp = y_test.copy(deep = True)
temp2 = y_val.copy(deep = True)
#create table to compare mla predictions
mla_test_predict = temp.to_frame()
mla_val_predict = temp2.to_frame()

# %%
#index through mla and save performance to table
row_index = 0
for alg in mla:
    #mla_compare.dropna(inplace = True)
    
    #set name and parameters
    mla_name = alg.__class__.__name__

    print('training cv: ', mla_name)
    mla_compare.loc[row_index, 'Model'] = mla_name
    mla_compare.loc[row_index, 'Model_Parameters'] = str(alg.get_params())


    cv_results = model_selection.cross_validate(alg, 
                                                x_cvtrain_unbalanced, 
                                                y_cvtrain_unbalanced, 
                                                cv  = 5, 
                                                return_train_score = True,
                                                 scoring = ('accuracy', 'roc_auc', 'f1', 'precision','recall'),
                                                  verbose = 2,
                                                   n_jobs = -1)
    mla_compare.loc[row_index, 'Time'] = cv_results['fit_time'].mean()
    mla_compare.loc[row_index, 'Train_Accuracy_Mean'] = cv_results['train_accuracy'].mean()
    mla_compare.loc[row_index, 'Test_Accuracy_Mean'] = cv_results['test_accuracy'].mean()
    mla_compare.loc[row_index, 'Train_Roc_auc_Mean'] = cv_results['train_roc_auc'].mean()
    mla_compare.loc[row_index, 'Test_Roc_auc_Mean'] = cv_results['test_roc_auc'].mean()
    mla_compare.loc[row_index, 'Train_f1_Mean'] = cv_results['train_f1'].mean()
    mla_compare.loc[row_index, 'Test_f1_Mean'] = cv_results['test_f1'].mean()
    mla_compare.loc[row_index, 'Train_precision_Mean'] = cv_results['train_precision'].mean()
    mla_compare.loc[row_index, 'Test_precision_Mean'] = cv_results['test_precision'].mean() 
    mla_compare.loc[row_index, 'Train_recall_Mean'] = cv_results['train_recall'].mean()
    mla_compare.loc[row_index, 'Test_recall_Mean'] = cv_results['test_recall'].mean()       
    #lets see what happens when shit hits the fan
    mla_compare.loc[row_index, 'Test_Roc_auc_3*STD'] = cv_results['test_roc_auc'].std()*3
    mla_compare.loc[row_index, 'Test_Accuracy_3*STD'] = cv_results['test_accuracy'].std()*3
    mla_compare.loc[row_index, 'Test_f1 3*STD'] = cv_results['test_f1'].std()*3
    mla_compare.loc[row_index, 'Test_precision_3*STD'] = cv_results['test_precision'].std()*3
    mla_compare.loc[row_index, 'Test_recall_3*STD'] = cv_results['test_recall'].std()*3  
    
    #save mla predictions 
    print('training on cvtrain: ', mla_name)
    alg.fit(x_cvtrain, y_cvtrain)
    mla_test_predict[mla_name] = alg.predict(x_test)
    joblib.dump(alg, path + '//models//cv//' + mla_name + '.gz', compress=('gzip', 3))
    #
    #print('training on train: ', mla_name)
    alg.fit(x_train, y_train)
    mla_val_predict[mla_name] = alg.predict(x_val)
    joblib.dump(alg, path + '//models//trainval//' + mla_name + '.gz', compress=('gzip', 3))
    
    
    
    mla_compare.to_excel(path + '//output//final_model_comparison.xlsx')
    mla_test_predict.to_excel(path + '//output//final_model_predictions_test.xlsx')
    mla_val_predict.to_excel(path + '//output//final_model_predictions_val.xlsx')
    row_index+=1
    
        
# %%
#*just to fit
row_index = 0
for alg in mla:
    #mla_compare.dropna(inplace = True)
    
    #set name and parameters
    mla_name = alg.__class__.__name__

    print('training on train: ', mla_name)
    alg.fit(x_train, y_train)
    mla_test_predict[mla_name] = alg.predict_proba(x_test)[:,1]
    
    #joblib.dump(alg, path + '//models//trainval//' + mla_name + '.gz', compress=('gzip', 3))
    
    #save mla predictions 
    #print('training on cvtrain: ', mla_name)
    #alg.fit(x_cvtrain, y_cvtrain)
    #mla_test_predict[mla_name] = alg.predict(x_test)
    #joblib.dump(alg, path + '//models//cv//' + mla_name + '.gz', compress=('gzip', 3))
    

    #mla_compare.to_excel(path + '//output//final_model_comparisonextra.xlsx')
    #mla_test_predict.to_excel(path + '//output//final_model_predictions_test.xlsx')
    mla_test_predict.to_excel(path + '/output/final_model_predictions_val_test.xlsx')
    row_index+=1





# %%
#loading results
#! this are the names to keep
model_evalcv = pd.read_excel(path + '//output//final_model_comparison.xlsx', index_col= 0)
model_predict_test = pd.read_excel(path + '//output//final_model_predictions_test.xlsx', index_col= 0)


# %%
#Loading models
print('loading models')
rf = joblib.load(path + '//models//cv//RandomForestClassifier.gz')
ada = joblib.load(path + '//models//cv//AdaBoostClassifier.gz')
extra = joblib.load(path + '//models//cv//ExtraTreesClassifier.gz')
lgbm = joblib.load(path + '//models//cv//LGBMClassifier.gz')
        



# %% 
#correlations
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    aa = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of models', y=1.05, size=15)
    bottom, top = aa.get_ylim()
    aa.set_ylim(bottom + 0.5, top - 0.5)


correlation_heatmap(mla_val_predict)

# %% 
#* predicting probabilities memory friendly

print('loading extra')
extra = joblib.load(path + '//models//cv//ExtraTreesClassifier.gz')
print('extra predicting')
extra_predict_pr = extra.predict_proba(x_test)
del extra

print('loading rf')
rf = joblib.load(path + '//models//cv//RandomForestClassifier.gz')
rf_predict_pr = rf.predict_proba(x_test)
del rf

print('loading ada')
ada = joblib.load(path + '//models//cv//AdaBoostClassifier.gz')
print('ada predicting')
ada_predict_pr = ada.predict_proba(x_test)
del ada


print('loading lgbm')
lgbm = joblib.load(path + '//models//cv//LGBMClassifier.gz')
print('lgbm predicting')
lgbm_predict_pr = lgbm.predict_proba(x_test)
del lgbm


#*metrics for models on val set

indices = model_predict_test.index.values
model_predict_test.loc[indices, 'RandomForestClassifier_pr'] = rf_predict_pr[:,1]
model_predict_test.loc[indices, 'AdaBoostClassifier_pr'] = ada_predict_pr[:,1]
model_predict_test.loc[indices, 'ExtraTreesClassifier_pr'] = extra_predict_pr[:,1]
model_predict_test.loc[indices, 'LGBMClassifier_pr'] = lgbm_predict_pr[:,1]


model_evalcv.loc[0, 'Roc_auc_testset'] = roc_auc_score(y_test, model_predict_test['AdaBoostClassifier_pr'])
model_evalcv.loc[1, 'Roc_auc_testset'] = roc_auc_score(y_test, model_predict_test['ExtraTreesClassifier_pr'])
model_evalcv.loc[2, 'Roc_auc_testset'] = roc_auc_score(y_test, model_predict_test['RandomForestClassifier_pr'])
model_evalcv.loc[3, 'Roc_auc_testset'] = roc_auc_score(y_test, model_predict_test['LGBMClassifier_pr'])


model_evalcv.loc[0, 'Accuracy_testset'] = accuracy_score(y_test, model_predict_test['AdaBoostClassifier'])
model_evalcv.loc[1, 'Accuracy_testset'] = accuracy_score(y_test, model_predict_test['ExtraTreesClassifier'])
model_evalcv.loc[2, 'Accuracy_testset'] = accuracy_score(y_test, model_predict_test['RandomForestClassifier'])
model_evalcv.loc[3, 'Accuracy_testset'] = accuracy_score(y_test, model_predict_test['LGBMClassifier'])


model_evalcv.loc[0, 'F1_testset'] = f1_score(y_test, model_predict_test['AdaBoostClassifier'])
model_evalcv.loc[1, 'F1_testset'] = f1_score(y_test, model_predict_test['ExtraTreesClassifier'])
model_evalcv.loc[2, 'F1_testset'] = f1_score(y_test, model_predict_test['RandomForestClassifier'])
model_evalcv.loc[3, 'F1_testset'] = f1_score(y_test, model_predict_test['LGBMClassifier'])


model_evalcv.loc[0, 'Precision_testset'] = precision_score(y_test, model_predict_test['AdaBoostClassifier'])
model_evalcv.loc[1, 'Precision_testset'] = precision_score(y_test, model_predict_test['ExtraTreesClassifier'])
model_evalcv.loc[2, 'Precision_testset'] = precision_score(y_test, model_predict_test['RandomForestClassifier'])
model_evalcv.loc[3, 'Precision_testset'] = precision_score(y_test, model_predict_test['LGBMClassifier'])


model_evalcv.loc[0, 'Recall_testset'] = recall_score(y_test, model_predict_test['AdaBoostClassifier'])
model_evalcv.loc[1, 'Recall_testset'] = recall_score(y_test, model_predict_test['ExtraTreesClassifier'])
model_evalcv.loc[2, 'Recall_testset'] = recall_score(y_test, model_predict_test['RandomForestClassifier'])
model_evalcv.loc[3, 'Recall_testset'] = recall_score(y_test, model_predict_test['LGBMClassifier'])




model_evalcv.to_excel(path + '//output//final_model_comparison.xlsx')
model_predict_test.to_excel(path + '//output//final_model_predictions_test.xlsx')


extra = ensemble.ExtraTreesClassifier(n_estimators = 500, n_jobs =-1)


print('training extra on cvtrain: ')
extra.fit(x_cvtrain, y_cvtrain)
extra_predict_pr = extra.predict_proba(x_test)
del extra

model_predict_test.loc[indices, 'ExtraTreesClassifier_pr'] = extra_predict_pr[:,1]
model_evalcv.loc[1, 'Roc_auc_testset'] = roc_auc_score(y_test, model_predict_test['ExtraTreesClassifier_pr'])
model_evalcv.loc[1, 'Accuracy_testset'] = accuracy_score(y_test, model_predict_test['ExtraTreesClassifier'])
model_evalcv.loc[1, 'F1_testset'] = f1_score(y_test, model_predict_test['ExtraTreesClassifier'])
model_evalcv.loc[1, 'Precision_testset'] = precision_score(y_test, model_predict_test['ExtraTreesClassifier'])
model_evalcv.loc[1, 'Recall_testset'] = recall_score(y_test, model_predict_test['ExtraTreesClassifier'])

model_evalcv.to_excel(path + '//output//final_model_comparison.xlsx')
model_predict_test.to_excel(path + '//output//final_model_predictions_test.xlsx')



# %%
#*loading results
#! this are the names to keep
model_evalcv = pd.read_excel(path + '//output//final_model_comparison.xlsx', index_col= 0)
model_predict_test = pd.read_excel(path + '//output//final_model_predictions_test.xlsx', index_col= 0)


# %%
#*lets check some roc curves

classifiers = ['LGBMClassifier_pr',  'AdaBoostClassifier_pr', 'ExtraTreesClassifier_pr', 'RandomForestClassifier_pr']

result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

for cl in classifiers:
    
    yproba = model_predict_test[cl]
    
    fpr, tpr, _ = roc_curve(y_test,  yproba)
    auc = roc_auc_score(y_test, yproba)
    
    result_table = result_table.append({'classifiers':cl.replace('Classifier_pr',''),
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)

# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)

fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()


# %%
#*lets check how bad the classifiers are predicting probs

# reliability diagram
fop, mpv = calibration_curve(y_test, model_predict_test['AdaBoostClassifier_pr'], n_bins=10)
# plot perfectly calibrated
plt.plot([0, 1], [0, 1], linestyle='--')
# plot model reliability
plt.plot(mpv, fop, marker='.')

fop_extra, mpv_extra = calibration_curve(y_test, model_predict_test['ExtraTreesClassifier_pr'], n_bins=10)
# plot model reliability
plt.plot(mpv_extra, fop_extra, marker='.')

fop_rf, mpv_rf = calibration_curve(y_test, model_predict_test['RandomForestClassifier_pr'], n_bins=10)
# plot model reliability
plt.plot(mpv_rf, fop_rf, marker='.')


fop_lgbm, mpv_lgbm = calibration_curve(y_test, model_predict_test['LGBMClassifier_pr'], n_bins=10)
# plot model reliability
plt.plot(mpv_lgbm, fop_lgbm, marker='.')

plt.title('Calibration plots (reliability diagram)', fontweight='bold', fontsize=12)
plt.legend(['Perfect', 'AdaBoost', 'ExtraTrees', 'RandomForest', 'LGBM'], loc='upper left')

plt.show()


#!everything is not good, ada is absolute dog shit


# %%

ada = ensemble.AdaBoostClassifier(n_estimators = 500)
extra = ensemble.ExtraTreesClassifier(n_estimators = 500, n_jobs =-1)
rf = ensemble.RandomForestClassifier(n_estimators = 500, n_jobs =-1)
lgbm = lgb.LGBMClassifier(boosting_type = 'gbdt',
                      objective ='binary',
                      metric = 'auc',
                      num_threads = 4,
                      learning_rate = 0.21183,
                      max_depth = 15,
                      num_leaves = 41,
                      min_data_in_leaf = 298,
                      feature_fraction = 0.623818,
                      subsample = 0.583636)   

# %%

#?calibrate model on cross validation using sigmoid
indices = model_predict_test.index.values
model_evalcv.loc[4, 'Model'] = 'LGBMClassifier_cal'
model_evalcv.loc[5, 'Model'] = 'AdaBoostClassifier_cal'
model_evalcv.loc[6, 'Model'] = 'ExtraTreesClassifier_cal'
model_evalcv.loc[7, 'Model'] = 'RandomForestClassifier_cal'
model_evalcv.loc[8, 'Model'] = 'LGBMClassifier_cal_iso'
model_evalcv.loc[9, 'Model'] = 'AdaBoostClassifier_cal_iso'
model_evalcv.loc[10, 'Model'] = 'ExtraTreesClassifier_cal_iso'
model_evalcv.loc[11, 'Model'] = 'RandomForestClassifier_cal_iso'

print('fitting lgbm')
# fit and calibrate model on training data
lgbm_calibrator = CalibratedClassifierCV(lgbm, cv=3)
lgbm_calibrator.fit(x_cvtrain_unbalanced, y_cvtrain_unbalanced)
# evaluate the model
print('predicting lgbm')
lgbm_predict_cal = lgbm_calibrator.predict(x_test)
lgbm_predict_pr_cal = lgbm_calibrator.predict_proba(x_test)
joblib.dump(lgbm_calibrator, path +'//models//trainval//LGBMClassifier_cal.sav')
del  lgbm_calibrator

model_predict_test.loc[indices, 'LGBMClassifier_cal'] = lgbm_predict_cal
model_predict_test.loc[indices, 'LGBMClassifier_pr_cal'] = lgbm_predict_pr_cal[:,1]
model_evalcv.loc[4, 'Roc_auc_testset'] = roc_auc_score(y_test, model_predict_test['LGBMClassifier_pr_cal'])
model_evalcv.loc[4, 'Accuracy_testset'] = accuracy_score(y_test, model_predict_test['LGBMClassifier_cal'])
model_evalcv.loc[4, 'F1_testset'] = f1_score(y_test, model_predict_test['LGBMClassifier_cal'])
model_evalcv.loc[4, 'Precision_testset'] = precision_score(y_test, model_predict_test['LGBMClassifier_cal'])
model_evalcv.loc[4, 'Recall_testset'] = recall_score(y_test, model_predict_test['LGBMClassifier_cal'])

model_evalcv.to_excel(path + '//output//final_model_comparison.xlsx')
model_predict_test.to_excel(path + '//output//final_model_predictions_val.xlsx')


print('fitting ada')
# fit and calibrate model on training data
ada_calibrator = CalibratedClassifierCV(ada, cv=3)
ada_calibrator.fit(x_cvtrain_unbalanced, y_cvtrain_unbalanced)
# evaluate the model
print('predicting ada')
ada_predict_cal = ada_calibrator.predict(x_test)
ada_predict_pr_cal = ada_calibrator.predict_proba(x_test)

joblib.dump(ada_calibrator, path +'//models//trainval//AdaBoostClassifier_cal.sav')
del  ada_calibrator
model_predict_test.loc[indices, 'AdaBoostClassifier_cal'] = ada_predict_cal
model_predict_test.loc[indices, 'AdaBoostClassifier_pr_cal'] = ada_predict_pr_cal[:,1]
model_evalcv.loc[5, 'Roc_auc_testset'] = roc_auc_score(y_test, model_predict_test['AdaBoostClassifier_pr_cal'])
model_evalcv.loc[5, 'Accuracy_testset'] = accuracy_score(y_test, model_predict_test['AdaBoostClassifier_cal'])
model_evalcv.loc[5, 'F1_testset'] = f1_score(y_test, model_predict_test['AdaBoostClassifier_cal'])
model_evalcv.loc[5, 'Precision_testset'] = precision_score(y_test, model_predict_test['AdaBoostClassifier_cal'])
model_evalcv.loc[5, 'Recall_testset'] = recall_score(y_test, model_predict_test['AdaBoostClassifier_cal'])

model_evalcv.to_excel(path + '//output//final_model_comparison.xlsx')
model_predict_test.to_excel(path + '//output//final_model_predictions_val.xlsx')


print('fitting extra')
# fit and calibrate model on training data
extra_calibrator = CalibratedClassifierCV(extra, cv=3)
extra_calibrator.fit(x_cvtrain_unbalanced, y_cvtrain_unbalanced)
# evaluate the model
print('predicting extra')
extra_predict_cal = extra_calibrator.predict(x_test)
extra_predict_pr_cal = extra_calibrator.predict_proba(x_test)
joblib.dump(extra_calibrator, path +'//models//trainval//ExtraTreesClassifier_cal.sav')
del  extra_calibrator

model_predict_test.loc[indices, 'ExtraTreesClassifier_pr_cal'] = extra_predict_pr_cal[:,1]
model_predict_test.loc[indices, 'ExtraTreesClassifier_cal'] = extra_predict_cal
model_evalcv.loc[6, 'Roc_auc_testset'] = roc_auc_score(y_test, model_predict_test['ExtraTreesClassifier_pr_cal'])
model_evalcv.loc[6, 'Accuracy_testset'] = accuracy_score(y_test, model_predict_test['ExtraTreesClassifier_cal'])
model_evalcv.loc[6, 'F1_testset'] = f1_score(y_test, model_predict_test['ExtraTreesClassifier_cal'])
model_evalcv.loc[6, 'Precision_testset'] = precision_score(y_test, model_predict_test['ExtraTreesClassifier_cal'])
model_evalcv.loc[6, 'Recall_testset'] = recall_score(y_test, model_predict_test['ExtraTreesClassifier_cal'])

model_evalcv.to_excel(path + '//output//final_model_comparison.xlsx')
model_predict_test.to_excel(path + '//output//final_model_predictions_val.xlsx')


print('fitting rf')
# fit and calibrate model on training data
rf_calibrator = CalibratedClassifierCV(rf, cv=3)
rf_calibrator.fit(x_cvtrain_unbalanced, y_cvtrain_unbalanced)
# evaluate the model
print('predicting rf')
rf_predict_cal = rf_calibrator.predict(x_test)
rf_predict_pr_cal = rf_calibrator.predict_proba(x_test)

joblib.dump(rf_calibrator, path +'//models//trainval//RandomForestClassifier_cal.sav')
del  rf_calibrator

model_predict_test.loc[indices, 'RandomForestClassifier_cal'] = rf_predict_cal
model_predict_test.loc[indices, 'RandomForestClassifier_pr_cal'] = rf_predict_pr_cal[:,1]
model_evalcv.loc[7, 'Roc_auc_testset'] = roc_auc_score(y_test, model_predict_test['RandomForestClassifier_pr_cal'])
model_evalcv.loc[7, 'Accuracy_testset'] = accuracy_score(y_test, model_predict_test['RandomForestClassifier_cal'])
model_evalcv.loc[7, 'F1_testset'] = f1_score(y_test, model_predict_test['RandomForestClassifier_cal'])
model_evalcv.loc[7, 'Precision_testset'] = precision_score(y_test, model_predict_test['RandomForestClassifier_cal'])
model_evalcv.loc[7, 'Recall_testset'] = recall_score(y_test, model_predict_test['RandomForestClassifier_cal'])

model_evalcv.to_excel(path + '//output//final_model_comparison.xlsx')
model_predict_test.to_excel(path + '//output//final_model_predictions_val.xlsx')

print('Trying isotonic regression')

#?calibrate model on cross validation using isotonic regresion

print('fitting lgbm')
# fit and calibrate model on training data
lgbm_calibrator_iso = CalibratedClassifierCV(lgbm, method ='isotonic', cv=3)
lgbm_calibrator_iso.fit(x_cvtrain_unbalanced, y_cvtrain_unbalanced)
# evaluate the model
print('predicting lgbm')
lgbm_predict_cal_iso = lgbm_calibrator_iso.predict(x_test)
lgbm_predict_pr_cal_iso = lgbm_calibrator_iso.predict_proba(x_test)

joblib.dump(lgbm_calibrator_iso, path +'//models//trainval//LGBMClassifier_cal_iso.sav')
del lgbm_calibrator_iso

model_predict_test.loc[indices, 'LGBMClassifier_cal_iso'] = lgbm_predict_cal_iso
model_predict_test.loc[indices, 'LGBMClassifier_pr_cal_iso'] = lgbm_predict_pr_cal_iso[:,1]

model_evalcv.loc[8, 'Roc_auc_testset'] = roc_auc_score(y_test, model_predict_test['LGBMClassifier_pr_cal_iso'])
model_evalcv.loc[8, 'Accuracy_testset'] = accuracy_score(y_test, model_predict_test['LGBMClassifier_cal_iso'])
model_evalcv.loc[8, 'F1_testset'] = f1_score(y_test, model_predict_test['LGBMClassifier_cal_iso'])
model_evalcv.loc[8, 'Precision_testset'] = precision_score(y_test, model_predict_test['LGBMClassifier_cal_iso'])
model_evalcv.loc[8, 'Recall_testset'] = recall_score(y_test, model_predict_test['LGBMClassifier_cal_iso'])

model_evalcv.to_excel(path + '//output//final_model_comparison.xlsx')
model_predict_test.to_excel(path + '//output//final_model_predictions_val.xlsx')


print('fitting ada')
# fit and calibrate model on training data
ada_calibrator_iso = CalibratedClassifierCV(ada, method ='isotonic', cv=3)
ada_calibrator_iso.fit(x_cvtrain_unbalanced, y_cvtrain_unbalanced)
# evaluate the model
print('predicting ada')
ada_predict_cal_iso = ada_calibrator_iso.predict(x_test)
ada_predict_pr_cal_iso = ada_calibrator_iso.predict_proba(x_test)

joblib.dump(ada_calibrator_iso, path +'//models//trainval//AdaBoostClassifier_cal_iso.sav')

del ada_calibrator_iso


model_predict_test.loc[indices, 'AdaBoostClassifier_cal_iso'] = ada_predict_cal_iso
model_predict_test.loc[indices, 'AdaBoostClassifier_pr_cal_iso'] = ada_predict_pr_cal_iso[:,1]
model_evalcv.loc[9, 'Roc_auc_testset'] = roc_auc_score(y_test, model_predict_test['AdaBoostClassifier_pr_cal_iso'])
model_evalcv.loc[9, 'Accuracy_testset'] = accuracy_score(y_test, model_predict_test['AdaBoostClassifier_cal_iso'])
model_evalcv.loc[9, 'F1_testset'] = f1_score(y_test, model_predict_test['AdaBoostClassifier_cal_iso'])
model_evalcv.loc[9, 'Precision_testset'] = precision_score(y_test, model_predict_test['AdaBoostClassifier_cal_iso'])
model_evalcv.loc[9, 'Recall_testset'] = recall_score(y_test, model_predict_test['AdaBoostClassifier_cal_iso'])

model_evalcv.to_excel(path + '//output//final_model_comparison.xlsx')
model_predict_test.to_excel(path + '//output//final_model_predictions_val.xlsx')


print('fitting extra')
# fit and calibrate model on training data
extra_calibrator_iso = CalibratedClassifierCV(extra, method ='isotonic', cv=3)
extra_calibrator_iso.fit(x_cvtrain_unbalanced, y_cvtrain_unbalanced)
# evaluate the model
print('predicting extra')
extra_predict_cal_iso = extra_calibrator_iso.predict(x_test)
extra_predict_pr_cal_iso = extra_calibrator_iso.predict_proba(x_test)
joblib.dump(extra_calibrator_iso, path +'//models//trainval//ExtraTreesClassifier_cal_iso.sav')
del extra_calibrator_iso

model_predict_test.loc[indices, 'ExtraTreesClassifier_pr_cal_iso'] = extra_predict_pr_cal_iso[:,1]
model_predict_test.loc[indices, 'ExtraTreesClassifier_cal_iso'] = extra_predict_cal_iso
model_evalcv.loc[10, 'Roc_auc_testset'] = roc_auc_score(y_test, model_predict_test['ExtraTreesClassifier_pr_cal_iso'])
model_evalcv.loc[10, 'Accuracy_testset'] = accuracy_score(y_test, model_predict_test['ExtraTreesClassifier_cal_iso'])
model_evalcv.loc[10, 'F1_testset'] = f1_score(y_test, model_predict_test['ExtraTreesClassifier_cal_iso'])
model_evalcv.loc[10, 'Precision_testset'] = precision_score(y_test, model_predict_test['ExtraTreesClassifier_cal_iso'])
model_evalcv.loc[10, 'Recall_testset'] = recall_score(y_test, model_predict_test['ExtraTreesClassifier_cal_iso'])

model_evalcv.to_excel(path + '//output//final_model_comparison.xlsx')
model_predict_test.to_excel(path + '//output//final_model_predictions_val.xlsx')


print('fitting rf')
# fit and calibrate model on training data
rf_calibrator_iso = CalibratedClassifierCV(rf, method ='isotonic', cv=3)
rf_calibrator_iso.fit(x_cvtrain_unbalanced, y_cvtrain_unbalanced)
# evaluate the model
print('predicting rf')
rf_predict_cal_iso = rf_calibrator_iso.predict(x_test)
rf_predict_pr_cal_iso = rf_calibrator_iso.predict_proba(x_test)
joblib.dump(rf_calibrator_iso, path +'//models//trainval//RandomForestClassifier_cal_iso.sav')
del rf_calibrator_iso


model_predict_test.loc[indices, 'RandomForestClassifier_cal_iso'] = rf_predict_cal_iso
model_predict_test.loc[indices, 'RandomForestClassifier_pr_cal_iso'] = rf_predict_pr_cal_iso[:,1]
model_evalcv.loc[11, 'Roc_auc_testset'] = roc_auc_score(y_test, model_predict_test['RandomForestClassifier_pr_cal_iso'])
model_evalcv.loc[11, 'Accuracy_testset'] = accuracy_score(y_test, model_predict_test['RandomForestClassifier_cal_iso'])
model_evalcv.loc[11, 'F1_testset'] = f1_score(y_test, model_predict_test['RandomForestClassifier_cal_iso'])
model_evalcv.loc[11, 'Precision_testset'] = precision_score(y_test, model_predict_test['RandomForestClassifier_cal_iso'])
model_evalcv.loc[11, 'Recall_testset'] = recall_score(y_test, model_predict_test['RandomForestClassifier_cal_iso'])

model_evalcv.to_excel(path + '//output//final_model_comparison.xlsx')
model_predict_test.to_excel(path + '//output//final_model_predictions_val.xlsx')



# %%
#*saving stats
#! this are the names to keep
model_evalcv.to_excel(path + '//output//final_model_comparison.xlsx')
model_predict_test.to_excel(path + '//output//final_model_predictions_val.xlsx')

# %%
#*loading results
#! this are the names to keep
model_evalcv = pd.read_excel(path + '//output//final_model_comparison.xlsx', index_col= 0)
model_predict_val = pd.read_excel(path + '//output//final_model_predictions_val.xlsx', index_col= 0)

# %%
#*lets check the calibrated prs for sigmoid models

# reliability diagram
fop, mpv = calibration_curve(y_test, model_predict_test['AdaBoostClassifier_pr_cal'], n_bins=10)
# plot perfectly calibrated
plt.plot([0, 1], [0, 1], linestyle='--')
# plot model reliability
plt.plot(mpv, fop, marker='.')

fop_extra, mpv_extra = calibration_curve(y_test, model_predict_test['ExtraTreesClassifier_pr_cal'], n_bins=10)
# plot model reliability
plt.plot(mpv_extra, fop_extra, marker='.')

fop_rf, mpv_rf = calibration_curve(y_test, model_predict_test['RandomForestClassifier_pr_cal'], n_bins=10)
# plot model reliability
plt.plot(mpv_rf, fop_rf, marker='.')


fop_lgbm, mpv_lgbm = calibration_curve(y_test, model_predict_test['LGBMClassifier_pr_cal'], n_bins=10)
# plot model reliability
plt.plot(mpv_lgbm, fop_lgbm, marker='.')
plt.title('Calibration plots Sigmoid calibrated models', fontweight='bold', fontsize=12)
plt.legend(['perfect', 'Ada', 'Extra', 'Rf',  'lgbm'], loc='upper left')

plt.show()

# %%
#*lets check the calibrated prs for iso models

# reliability diagram
fop, mpv = calibration_curve(y_test, model_predict_test['AdaBoostClassifier_pr_cal_iso'], n_bins=10)
# plot perfectly calibrated
plt.plot([0, 1], [0, 1], linestyle='--')
# plot model reliability
plt.plot(mpv, fop, marker='.')

fop_extra, mpv_extra = calibration_curve(y_test, model_predict_test['ExtraTreesClassifier_pr_cal_iso'], n_bins=10)
# plot model reliability
plt.plot(mpv_extra, fop_extra, marker='.')

fop_rf, mpv_rf = calibration_curve(y_test, model_predict_test['RandomForestClassifier_pr_cal_iso'], n_bins=10)
# plot model reliability
plt.plot(mpv_rf, fop_rf, marker='.')


fop_lgbm, mpv_lgbm = calibration_curve(y_test, model_predict_test['LGBMClassifier_pr_cal_iso'], n_bins=10)
# plot model reliability
plt.plot(mpv_lgbm, fop_lgbm, marker='.')
plt.title('Calibration plots Isotonic regression calibrated models', fontweight='bold', fontsize=12)
plt.legend(['perfect', 'Ada', 'Extra', 'Rf',  'lgbm'], loc='upper left')

plt.show()

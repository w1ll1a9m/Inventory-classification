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
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from dataframe_column_identifier import DataFrameColumnIdentifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
plt.style.use('default')
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN
from minepy import MINE
import lightgbm as lgb
from sklearn.externals import joblib
from scipy.spatial import distance
import scipy.stats as stats
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
import os
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
#all models ever known to mankind
mla = [
    
    #lgbm
    
    lgb.LGBMClassifier(boosting_type = 'gbdt',
                       objective ='binary',
                       metric = 'auc',
                       num_threads = 4),
    
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.ExtraTreesClassifier(n_jobs = -1),
    ensemble.RandomForestClassifier(n_jobs = -1),
 
    #GLM
    linear_model.LogisticRegressionCV(max_iter = 200, n_jobs = -1),
    linear_model.RidgeClassifierCV(),
 
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
 
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(n_jobs = -1)
    
       
    ]

# %%
#train all models

cv_split = model_selection.StratifiedShuffleSplit(n_splits = 5, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%

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
temp = y_val.copy(deep = True)
#create table to compare mla predictions
mla_predict = temp.to_frame()


# %%
#index through mla and save performance to table
row_index = 0
for alg in mla:
    #mla_compare.dropna(inplace = True)
    
    #set name and parameters
    mla_name = alg.__class__.__name__
    
    if mla_name  not in mla_compare['Model'].unique():
        
    
        print('training cv: ', mla_name)
        mla_compare.loc[row_index, 'Model'] = mla_name
        mla_compare.loc[row_index, 'Model_Parameters'] = str(alg.get_params())
    
    
        cv_results = model_selection.cross_validate(alg, 
                                                    x_cvtrain, 
                                                    y_cvtrain, 
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
        print('training on val: ', mla_name)
        alg.fit(x_train, y_train)
        mla_predict[mla_name] = alg.predict(x_val)
        joblib.dump(alg, 'C:/ART_House/IHART/Python/models/' + mla_name + '.sav')
        mla_compare.to_excel(path + '//output//model_evaluation.xlsx')
        mla_predict.to_excel(path + '//output//model_predictions.xlsx')
        row_index+=1
    else:
        row_index+=1
        


# %%  
#*load mla_compare and mla_predict

mla_compare = pd.read_excel(path + '//output//model_evaluation.xlsx', index_col= 0)
mla_predict = pd.read_excel(path + '//output//model_predictions.xlsx', index_col= 0)
# %%  
mla_compare.sort_values(by = ['Test_f1_Mean'], ascending = False, inplace = True)
#mla_compare
#mla_predict

sns.barplot(x='Test_Roc_auc_Mean', y = 'Model', data = mla_compare, color = 'm')

#prettify using pyplot: https://matplotlib.org/api/pyplot_api.html
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')
# %%
#correlation of predictions on val set
#plt.style.use('default')

corr = mla_predict.corr()
plt.figure(figsize=(10, 10))

ax = sns.heatmap(corr, 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 5}, square=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)



# %%

report = classification_report(y_val, mla_predict['AdaBoostClassifier'], output_dict = True)

pd_report = pd.DataFrame.from_dict(report)


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
from minepy import MINE
import lightgbm as lgb
from sklearn.externals import joblib
from scipy.spatial import distance
import scipy.stats as stats
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
#*Feature selection for numerical features based on correlation
plt.style.use('default')
#correlated numerical features
correlated_features = set()
correlation_matrix = cvtrain[numcolumns].corr()
for i in range(len(correlation_matrix .columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)

correlated_features_list = list(correlated_features)

corr = cvtrain[correlated_features_list].corr()
plt.figure(figsize=(15, 15))

sns.heatmap(corr, 
            cmap='viridis', vmax=1.0, vmin=-0.1, linewidths=0.0,
            annot=True, annot_kws={"size": 8}, square=True)

n = len(correlated_features_list)//2
mean_correlations = pd.DataFrame()
mean_correlations = correlation_matrix.mean(axis = 1).sort_values(ascending = False)

correlated_features_drop = mean_correlations[mean_correlations.index.isin(correlated_features_list)]
drop_features_corr = list(correlated_features_drop.head(n).index)

#!first subset to train on, removing linearly correlated features
x_cvtrain_1 = x_cvtrain.drop(drop_features_corr, axis = 1)
x_test_1 = x_test.drop(drop_features_corr, axis = 1)

x_train_1 = x_train.drop(drop_features_corr, axis = 1)
x_val_1 = x_val.drop(drop_features_corr, axis = 1)

# %%
#*dropping non relevant features according to ANOVA test

x_cvtrain_1_num = x_cvtrain[numcolumns]

x_cvtrain_1_num = x_cvtrain_1_num.drop(drop_features_corr, axis = 1)

numcolumns_2 = list(x_cvtrain_1_num.columns)

kbest = SelectKBest(score_func=f_classif, k=25)

dfci = DataFrameColumnIdentifier()

kbest.fit_transform(x_cvtrain_1_num, y_cvtrain)
kbest_get_support_output_fcl = kbest.get_support()
best_fcl_features = list(dfci.select_columns_KBest(x_cvtrain_1_num, kbest_get_support_output_fcl, verbose=1))


drop_features_filter_fclas = [x for x in numcolumns_2 if x not in best_fcl_features]

#*dropping non relevant features according to CHI2 test

x_cvtrain_1_cat = x_cvtrain[catcolumns]

catcolumns_2 = list(x_cvtrain_1_cat.columns)

kbestchi2 = SelectKBest(score_func=f_classif, k=10)

dfci = DataFrameColumnIdentifier()

kbestchi2.fit_transform(x_cvtrain_1_cat, y_cvtrain)
kbest_get_support_output_chi2 = kbestchi2.get_support()
best_chi2_features = list(dfci.select_columns_KBest(x_cvtrain_1_cat, kbest_get_support_output_chi2, verbose=1))

drop_features_filter_chi2 = [x for x in catcolumns_2 if x not in best_chi2_features]

drop_features_filter = drop_features_filter_fclas + drop_features_filter_chi2

#!second subset to train on, removing filtered features by f-ANOVA and CHI2

x_cvtrain_2 = x_cvtrain_1.drop(drop_features_filter, axis = 1)
x_test_2 = x_test_1.drop(drop_features_filter, axis = 1)

drop_features_filter2 = [x for x in drop_features_filter if x in list(x_train.columns)]

x_train_2 = x_train_1.drop(drop_features_filter2, axis = 1)
x_val_2 = x_val_1.drop(drop_features_filter2, axis = 1)
# %%
#*dropping non relevant features according to MIC
mine = MINE(alpha=0.6, c=15) 



mic_results= pd.DataFrame(og_features, columns = ['Features'])

index = 0

for feature in og_features:
    print('pearson on:', feature)
    mic_results.loc[index, 'Pearsons R'] = stats.pearsonr(x_cvtrain[:, index], y_cvtrain)[0] #returns both Pearson's coefficient and p-value, keep the first value which is the R coefficient
    print('spearman on:', feature)
    mic_results.loc[index, 'Spearman'] = stats.spearmanr(x_cvtrain[:, index], y_cvtrain) [0]
    mine.compute_score(x_cvtrain[:, index], y_cvtrain)
    print('mic on:', feature)
    mic_results.loc[index, 'MIC'] = mine.mic()
    print('cosine on:', feature)
    mic_results.loc[index, 'Cosine Similarity']= 1-distance.cosine(x_cvtrain[:, index], y_cvtrain)
    index = index +1 

# %%
#*recursive feature elimination on models

#rfe on lgbm 
lgbm = lgb.LGBMClassifier(boosting_type = 'gbdt',
                       objective ='binary',
                       metric = 'auc',
                       num_threads = 4,
                       importance_type = 'gain') 

lgbm_rfe = RFE(estimator = lgbm, n_features_to_select=25, step=1)
print('rfe on lgbm')
lgbm_rfe = lgbm_rfe.fit(x_train_1, y_train)
best_features_rfe_lgbm = list(x_train_1.columns[lgbm_rfe.support_])
drop_features_rfe_lgbm = [x for x in list(x_train_1.columns) if x not in best_features_rfe_lgbm]
print('Chosen best 25 feature by lgbm:', x_train_1.columns[lgbm_rfe.support_])
# %%
#rfe for random forest
rf = RandomForestClassifier( random_state=666, verbose = 1, n_jobs = -1)
rfe = RFE(estimator = rf, n_features_to_select=25, step=1)
print('rfe on rf')
rfe = rfe.fit(x_train_1, y_train)

best_features_rfe_rf = list(x_train_1.columns[rfe.support_])
drop_features_rfe_rf = [x for x in list(x_train_1.columns) if x not in best_features_rfe_rf]
print('Chosen best 25 feature by rf:', x_train_1.columns[rfe.support_])

#rfe on ada boost
ada = AdaBoostClassifier( random_state=666)
ada_rfe = RFE(estimator = ada, n_features_to_select=25, step=1)
print('rfe on ada')
ada_rfe = ada_rfe.fit(x_train_1, y_train)

best_features_rfe_ada = list(x_train_1.columns[ada_rfe.support_])
drop_features_rfe_ada = [x for x in list(x_train_1.columns) if x not in best_features_rfe_ada]
print('Chosen best 25 feature by ada:', x_train_1.columns[ada_rfe.support_])

#rfe on extra
extra = ExtraTreesClassifier( random_state=666, n_jobs = -1)
extra_rfe = RFE(estimator = extra, n_features_to_select=25, step=1)
print('rfe on extra')
extra_rfe = extra_rfe.fit(x_train_1, y_train)

best_features_rfe_extra = list(x_train_1.columns[extra_rfe.support_])
drop_features_rfe_extra = [x for x in list(x_train_1.columns) if x not in best_features_rfe_extra]
print('Chosen best 25 feature by extra:', x_train_1.columns[extra_rfe.support_])



# %%
#! Saving RFE models

joblib.dump(rfe, path + '//models//rf_rfe.sav')
joblib.dump(ada_rfe, path + '//models//ada_rfe.sav')
joblib.dump(lgbm_rfe, path + '//models//lgbm_rfe.sav')
joblib.dump(extra_rfe, path + '//models//extra_rfe.sav')

# %%
#? Loading RFE models
rfe = joblib.load(path + '//models//rf_rfe.sav')
ada_rfe = joblib.load(path + '//models//ada_rfe.sav')
lgbm_rfe = joblib.load(path + '//models//lgbm_rfe.sav')
extra_rfe = joblib.load(path + '//models//extra_rfe.sav')

# %%
#todo rfe results
best_features_rfe_lgbm = ['ABCXYZ',
 'Mat_status',
 'Last_mat_status_change',
 'months_since_mat_status_change',
 'Month',
 'Cost_liter',
 'Stock_coverage_months',
 'Effective_stock_L',
 'Slow_stock_L',
 '3m_delta_effective_stock',
 'MA6_CycleStock',
 'MA6_NewStock',
 '3pm_slobs_dc_l',
 '6pm_slobs_dc_l',
 '6m_avg_slobs_dc_l',
 'Slobs_dc_cat_l',
 'Pm_slobs_dc_cat_brand_l',
 '6m_avg_slobs_dc_cat_brand_l',
 'Actuals_l',
 '1m_forecast_l',
 '5m_forecast_l',
 'Lag1_l',
 'Lag1_ts',
 'Lag1_6m_MASE',
 'MaxAE_country_cat']
 
best_features_rfe_rf = ['Mat_status',
 'months_since_mat_status_change',
 'Cost_liter',
 'Stock_coverage_months',
 'Effective_stock_L',
 'Safety_stock_L',
 'Slow_stock_L',
 '3m_delta_effective_stock',
 'Pr_slow_stock',
 'MA6_ExcessPushStock',
 'Slobs_dc_l',
 '6m_avg_slobs_dc_l',
 'Slobs_dc_cat_l',
 '3pm_slobs_dc_cat_l',
 '6pm_slobs_dc_cat_l',
 'Pm_slobs_dc_cat_brand_l',
 '6m_avg_slobs_dc_cat_brand_l',
 'Actuals_l',
 'Lag1_l',
 'Lag1_error_l',
 'Lag1_6m_MAE',
 'Lag1_6m_Sum_e',
 'Lag1_6m_MASE',
 'MaxAE_country_cat',
 'delta_pm_lag1_error']

best_features_rfe_extra = ['ABCXYZ',
 'Mat_status',
 'Last_mat_status',
 'Last_mat_status_change',
 'months_since_mat_status_change',
 'Cost_liter',
 'Stock_coverage_months',
 'Effective_stock_L',
 'Slow_stock_L',
 '3m_delta_effective_stock',
 'Pr_slow_stock',
 'MA6_CycleStock',
 'MA6_ExcessPushStock',
 'Slobs_dc_l',
 '6m_avg_slobs_dc_l',
 'Slobs_dc_cat_l',
 'Pm_slobs_dc_cat_brand_l',
 '6m_avg_slobs_dc_cat_brand_l',
 'Actuals_l',
 'Lag1_l',
 'Lag1_6m_MAE',
 'Lag1_6m_Sum_e',
 'Lag1_ts',
 'Lag1_6m_MASE',
 'MaxAE_country_cat']

best_features_rfe_ada = ['ABCXYZ',
 'Mat_status',
 'Last_mat_status',
 'Last_mat_status_change',
 'Mat_status_change_6m',
 'Produced_vol',
 'Stock_coverage_months',
 'Effective_stock_L',
 'Slow_stock_L',
 '3m_delta_effective_stock',
 'Pr_slow_stock',
 'MA6_CycleStock',
 'MA6_NewStock',
 'MA6_BlockedStock',
 'Pm_slobs_dc_l',
 '6m_avg_slobs_dc_l',
 'Slobs_dc_cat_l',
 'Actuals_l',
 '1m_forecast_l',
 'Lag1_ts',
 'Lag1_6m_MASE',
 'MaxAE_country_cat',
 'Brand_type_Flourish',
 'Brand_type_White Label',
 'Region_NWE']

drop_features_rfe_lgbm = [x for x in list(x_train_1.columns) if x not in best_features_rfe_lgbm]
drop_features_rfe_rf = [x for x in list(x_train_1.columns) if x not in best_features_rfe_rf]
drop_features_rfe_ada = [x for x in list(x_train_1.columns) if x not in best_features_rfe_ada]
drop_features_rfe_extra = [x for x in list(x_train_1.columns) if x not in best_features_rfe_extra]

# %%
#* RFECV on everything
#cvrfe on lgbm 
lgbm = lgb.LGBMClassifier(boosting_type = 'gbdt',
                       objective ='binary',
                       metric = 'accuracy',
                       num_threads = 4,
                       importance_type = 'gain')

lgbm_rfecv = RFECV(estimator=lgbm, step=1, cv=5,scoring='accuracy', n_jobs = -1, verbose = 2)   #5-fold cross-validation
lgbm_rfecv = lgbm_rfecv.fit(x_train_1, y_train)

print('Optimal number of features for lgbm:', lgbm_rfecv.n_features_)
print('Best features :', x_train_1.columns[lgbm_rfecv.support_])


best_features_rfecv_lgbm = list (x_train_1.columns[lgbm_rfecv.support_])
drop_features_rfecv_lgbm = [x for x in list(x_train_1.columns) if x not in best_features_rfecv_lgbm]


plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of Lgbm")
plt.plot(range(1, len(lgbm_rfecv.grid_scores_) + 1), lgbm_rfecv.grid_scores_)
plt.savefig('lgbm_cvrfe.png')
plt.show()
# %%
#*cvrfe for random forest
rf = RandomForestClassifier( random_state=666, verbose = 1, n_jobs = -1)
rf_rfecv = RFECV(estimator=rf, step=1, cv=5,scoring='accuracy', n_jobs = -1, verbose = 2)   #5-fold cross-validation
rf_rfecv = rf_rfecv.fit(x_train_1, y_train)

print('Optimal number of features for random forest:', rf_rfecv.n_features_)
print('Best features :', x_train_1.columns[rf_rfecv.support_])


best_features_rfecv_rf = list (x_train_1.columns[rf_rfecv.support_])
drop_features_rfecv_rf = [x for x in list(x_train_1.columns) if x not in best_features_rfecv_rf]


plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of Random Forest")
plt.plot(range(1, len(rf_rfecv.grid_scores_) + 1), rf_rfecv.grid_scores_)
plt.savefig('rf_cvrfe.png')
plt.show()

#cvrfe for adaboost
ada = AdaBoostClassifier( random_state=666)
ada_rfecv = RFECV(estimator=ada, step=1, cv=5,scoring='accuracy', n_jobs = -1, verbose = 2)   #5-fold cross-validation
ada_rfecv = ada_rfecv.fit(x_train_1, y_train)

print('Optimal number of features for ada :', ada_rfecv.n_features_)
print('Best features :', x_train_1.columns[ada_rfecv.support_])

best_features_rfecv_ada = list (x_train_1.columns[ada_rfecv.support_])
drop_features_rfecv_ada = [x for x in list(x_train_1.columns) if x not in best_features_rfecv_ada]

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of Ada Boost")
plt.plot(range(1, len(ada_rfecv.grid_scores_) + 1), ada_rfecv.grid_scores_)
plt.savefig('ada_cvrfe.png')
plt.show()


#cvrfe for extra
extra = ExtraTreesClassifier( random_state=666, n_jobs =-1)
extra_rfecv = RFECV(estimator=extra, step=1, cv=5,scoring='accuracy', n_jobs = -1, verbose = 2)   #5-fold cross-validation
extra_rfecv = extra_rfecv.fit(x_train_1, y_train)

print('Optimal number of features :', extra_rfecv.n_features_)
print('Best features :', x_train_1.columns[extra_rfecv.support_])


best_features_rfecv_extra = list (x_train_1.columns[extra_rfecv.support_])
drop_features_rfecv_extra = [x for x in list(x_train_1.columns) if x not in best_features_rfecv_extra]

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of Extra randomized trees")
plt.plot(range(1, len(extra_rfecv.grid_scores_) + 1), extra_rfecv.grid_scores_)
plt.savefig('extra_cvrfe.png')
plt.show()


# %%
#! Saving RFECV models

joblib.dump(rf_rfecv, path + '//models//rf_rfecv.sav')
joblib.dump(ada_rfecv, path + '//models//ada_rfecv.sav')
joblib.dump(lgbm_rfecv, path + '//models//lgbm_rfecv.sav')
joblib.dump(extra_rfecv, path + '//models//extra_rfecv.sav')

# %%
#? Loading RFECV models
rf_rfecv = joblib.load(path + '//models//rf_rfecv.sav')
ada_rfecv = joblib.load(path + '//models//ada_rfecv.sav')
lgbm_rfecv = joblib.load(path + '//models//lgbm_rfecv.sav')
extra_rfecv = joblib.load(path + '//models//extra_rfecv.sav')

# %%
#*getting features after loading

best_features_rfe_lgbm = list(x_train_1.columns[lgbm_rfe.support_])
drop_features_rfe_lgbm = [x for x in list(x_train_1.columns) if x not in best_features_rfe_lgbm]
print('Chosen best 25 feature by lgbm:', x_train_1.columns[lgbm_rfe.support_])

#rfe for random forest


best_features_rfe_rf = list(x_train_1.columns[rfe.support_])
drop_features_rfe_rf = [x for x in list(x_train_1.columns) if x not in best_features_rfe_rf]
print('Chosen best 25 feature by rf:', x_train_1.columns[rfe.support_])

best_features_rfe_ada = list(x_train_1.columns[ada_rfe.support_])
drop_features_rfe_ada = [x for x in list(x_train_1.columns) if x not in best_features_rfe_ada]
print('Chosen best 25 feature by ada:', x_train_1.columns[ada_rfe.support_])


best_features_rfe_extra = list(x_train_1.columns[extra_rfe.support_])
drop_features_rfe_extra = [x for x in list(x_train_1.columns) if x not in best_features_rfe_extra]
 

best_features_rfecv_lgbm = list(x_train_1.columns[lgbm_rfecv.support_])
drop_features_rfecv_lgbm = [x for x in list(x_train_1.columns) if x not in best_features_rfecv_lgbm]
print('Chosen best 25 feature by lgbm:', x_train_1.columns[lgbm_rfecv.support_])

#rfecv for random forest


best_features_rfecv_rf = list(x_train_1.columns[rf_rfecv.support_])
drop_features_rfecv_rf = [x for x in list(x_train_1.columns) if x not in best_features_rfecv_rf]
print('Chosen best 25 feature by rf:', x_train_1.columns[rf_rfecv.support_])

best_features_rfecv_ada = list(x_train_1.columns[ada_rfecv.support_])
drop_features_rfecv_ada = [x for x in list(x_train_1.columns) if x not in best_features_rfecv_ada]
print('Chosen best 25 feature by ada:', x_train_1.columns[ada_rfecv.support_])


best_features_rfecv_extra = list(x_train_1.columns[extra_rfecv.support_])
drop_features_rfecv_extra = [x for x in list(x_train_1.columns) if x not in best_features_rfecv_extra]

# %%
#*New train sets with new feature sets

x_train_1 = x_train.drop(drop_features_corr, axis = 1)
x_val_1 = x_val.drop(drop_features_corr, axis = 1)

x_train_2 = x_train_1.drop(drop_features_filter2, axis = 1)
x_val_2 = x_val_1.drop(drop_features_filter2, axis = 1)

x_train_3 = x_train_1.drop(drop_features_rfe_rf, axis = 1)
x_val_3 = x_val_1.drop(drop_features_rfe_rf, axis = 1)

x_train_4 = x_train_1.drop(drop_features_rfecv_rf, axis = 1)
x_val_4 = x_val_1.drop(drop_features_rfecv_rf, axis = 1)

x_train_5 = x_train_1.drop(drop_features_rfe_ada, axis = 1)
x_val_5 = x_val_1.drop(drop_features_rfe_ada, axis = 1)

x_train_6 = x_train_1.drop(drop_features_rfecv_ada, axis = 1)
x_val_6 = x_val_1.drop(drop_features_rfecv_ada, axis = 1)

x_train_7 = x_train_1.drop(drop_features_rfe_extra, axis = 1)
x_val_7 = x_val_1.drop(drop_features_rfe_extra, axis = 1)

x_train_8 = x_train_1.drop(drop_features_rfecv_extra, axis = 1)
x_val_8 = x_val_1.drop(drop_features_rfecv_extra, axis = 1)

x_train_9 = x_train_1.drop(drop_features_rfe_lgbm, axis = 1)
x_val_9 = x_val_1.drop(drop_features_rfe_lgbm, axis = 1)

x_train_10 = x_train_1.drop(drop_features_rfecv_lgbm, axis = 1)
x_val_10 = x_val_1.drop(drop_features_rfecv_lgbm, axis = 1)


# %%
#*lets check the importances
feature_importances_rf = pd.DataFrame()
feature_importances_rf['feature'] = x_train_3.columns
feature_importances_rf['rf_imp'] = rfe.estimator_.feature_importances_
#feature_importances_rf['model'] = 'rfe_rf'
feature_importances_rf.to_excel(path + '//output//feature_importances_rf_top25_rfe.xls', index = False)

feature_importances_rf_cv = pd.DataFrame()
feature_importances_rf_cv['feature'] = x_train_4.columns
feature_importances_rf_cv['rf_imp'] = rf_rfecv.estimator_.feature_importances_
#feature_importances_rf['model'] = 'cvrfe_rf'
feature_importances_rf_cv.to_excel(path + '//output//feature_importances_rf_cvrfe.xls', index = False)

feature_importances_ada = pd.DataFrame()
feature_importances_ada['feature'] = x_train_5.columns
feature_importances_ada['ada_imp'] = ada_rfe.estimator_.feature_importances_
#feature_importances_ada['model'] = 'rfe_ada'
feature_importances_ada.to_excel(path + '//output//feature_importances_ada_top25_rfe.xls', index = False)

feature_importances_ada_cv = pd.DataFrame()
feature_importances_ada_cv['feature'] = x_train_6.columns
feature_importances_ada_cv['ada_imp'] = ada_rfecv.estimator_.feature_importances_
#feature_importances_ada_cv['model'] = 'cvrfe_ada'
feature_importances_ada_cv.to_excel(path + '//output//feature_importances_ada_cvrfe.xls', index = False)

feature_importances_extra = pd.DataFrame()
feature_importances_extra['feature'] = x_train_7.columns
feature_importances_extra['extra_imp'] = extra_rfe.estimator_.feature_importances_
#feature_importances_extra['model'] = 'rfe_extra'
feature_importances_extra.to_excel(path + '//output//feature_importances_extra_top25_rfe.xls', index = False)

feature_importances_extra_cv = pd.DataFrame()
feature_importances_extra_cv['feature'] = x_train_8.columns
feature_importances_extra_cv['extra_imp'] = extra_rfecv.estimator_.feature_importances_
#feature_importances_extra_cv['model'] = 'cvrfe_extra'
feature_importances_extra_cv.to_excel(path + '//output//feature_importances_extra_cvrfe.xls', index = False)

feature_importances_lgbm = pd.DataFrame()
feature_importances_lgbm['feature'] = x_train_9.columns
feature_importances_lgbm['lgbm_imp'] = lgbm_rfe.estimator_.feature_importances_
#feature_importances_lgbm['model'] = 'rfe_lgbm'
feature_importances_lgbm.to_excel(path + '//output//feature_importances_lgbm_top25_rfe.xls', index = False)

feature_importances_lgbm_cv = pd.DataFrame()
feature_importances_lgbm_cv['feature'] = x_train_10.columns
feature_importances_lgbm_cv['lgbm_imp'] = lgbm_rfecv.estimator_.feature_importances_
#feature_importances_lgbm_cv['model'] = 'cvrfe_lgbm'
feature_importances_lgbm_cv.to_excel(path + '//output//feature_importances_lgbm_cvrfe.xls', index = False)

# %%
#*for rfe method

feature_importances_rf.sort_values(by=['rf_imp'], ascending=False, inplace = True)
feature_importances_ada.sort_values(by=['ada_imp'], ascending=False, inplace = True)
feature_importances_extra.sort_values(by=['extra_imp'], ascending=False, inplace = True)
feature_importances_lgbm.sort_values(by=['lgbm_imp'], ascending=False, inplace = True)

feature_importances_rf.reset_index(inplace = True)
feature_importances_ada.reset_index(inplace = True)
feature_importances_extra.reset_index(inplace = True)
feature_importances_lgbm.reset_index(inplace = True)

feature_importances_concat = pd.concat([feature_importances_rf, feature_importances_ada, feature_importances_extra, feature_importances_lgbm])
feature_importances_rf_ada = pd.merge(feature_importances_ada, feature_importances_rf, on = 'feature', how='outer')
feature_importances_rf_ada = feature_importances_rf_ada.drop (columns = [ 'index_x', 'index_y'])
feature_importances_rf_ada_extra = pd.merge(feature_importances_rf_ada, feature_importances_extra, on = 'feature', how='outer' )
feature_importances = pd.merge(feature_importances_rf_ada_extra, feature_importances_lgbm, on = 'feature', how='outer' )
feature_importances = feature_importances.drop (columns = [ 'index_x', 'index_y'])
feature_importances['lgbm_imp_n'] = feature_importances['lgbm_imp']/1000000
feature_importances_pivot = pd.melt(feature_importances, id_vars = ['feature'])

#*saving feature importances

feature_importances.to_excel(path + '//output//feature_importances_rfe.xls', index = False)
feature_importances_pivot.to_excel(path + '//output//feature_importances_pivot_rfe.xls', index = False)

# %%
#*lets check the importances for frecv method

feature_importances_rf_cv.sort_values(by=['rf_imp'], ascending=False, inplace = True)
feature_importances_ada_cv.sort_values(by=['ada_imp'], ascending=False, inplace = True)
feature_importances_extra_cv.sort_values(by=['extra_imp'], ascending=False, inplace = True)
feature_importances_lgbm_cv.sort_values(by=['lgbm_imp'], ascending=False, inplace = True)

feature_importances_rf_cv.reset_index(inplace = True)
feature_importances_ada_cv.reset_index(inplace = True)
feature_importances_extra_cv.reset_index(inplace = True)
feature_importances_lgbm_cv.reset_index(inplace = True)

feature_importances_concat_cv = pd.concat([feature_importances_rf_cv, feature_importances_ada_cv, feature_importances_extra_cv, feature_importances_lgbm_cv])
feature_importances_rf_ada_cv = pd.merge(feature_importances_ada_cv, feature_importances_rf_cv, on = 'feature', how='outer')
feature_importances_rf_ada_cv = feature_importances_rf_ada_cv.drop (columns = [ 'index_x', 'index_y'])
feature_importances_rf_ada_extra_cv = pd.merge(feature_importances_rf_ada_cv, feature_importances_extra_cv, on = 'feature', how='outer' )
feature_importances_cv = pd.merge(feature_importances_rf_ada_extra_cv, feature_importances_lgbm_cv, on = 'feature', how='outer' )
feature_importances_cv = feature_importances_cv.drop (columns = [ 'index_x', 'index_y'])
feature_importances_cv['lgbm_imp_n'] = feature_importances_cv['lgbm_imp']/1000000
feature_importances_cv_pivot = pd.melt(feature_importances_cv, id_vars = ['feature'])

#*saving feature importances

feature_importances_cv.to_excel(path + '//output//feature_importances_rfecv.xls', index = False)
feature_importances_cv_pivot.to_excel(path + '//output//feature_importances_pivot_rfecv.xls', index = False)


# %%
#*Native feature importances over all models

# Initialize an empty array to hold feature importances
feature_importances_all = np.zeros(x_train_1.shape[1])

lgbm = lgb.LGBMClassifier(boosting_type = 'gbdt',
                       objective ='binary',
                       metric = 'accuracy',
                       num_threads = 4)
rf = RandomForestClassifier( random_state=666)
ada = AdaBoostClassifier( random_state=666)
extra = ExtraTreesClassifier( random_state=666)

models = [rf, ada, extra, lgbm]

for model in models:
  
    print('training model: ', model.__class__.__name__)
    # Train using early stopping
    model.fit(x_train_1, y_train)
    y_pred = model.predict(x_val_1)
    ac = accuracy_score(y_val, y_pred)
    print('For model: ', model.__class__.__name__, ' Accuracy is: ', ac)
    if model == lgbm:
        feature_importances_all_lgbm =model.feature_importances_
        feature_importances_all_lgbm = feature_importances_all_lgbm/32
        sum_lgbm = np.sum(feature_importances_all_lgbm)
        feature_importances_all_lgbm = feature_importances_all_lgbm/sum_lgbm
        feature_importances_all += feature_importances_all_lgbm
        

        
    else:   
        # Record the feature importances
        feature_importances_all += model.feature_importances_

# %%
feature_importances_all = feature_importances_all / 4
feature_importances_all = pd.DataFrame({'feature': list(x_train_1.columns), 'importance': feature_importances_all}).sort_values('importance', ascending = False)
feature_importances_all.reset_index(drop = True, inplace = True)
#normalizing feature importance
#sum_fueature_importances_all = feature_importances_all['importance'].sum()
#feature_importances_all['importance_normalized'] = feature_importances_all['importance'] / sum_fueature_importances_all
feature_importances_all['cumulative_importance'] = np.cumsum(feature_importances_all['importance_normalized'])
feature_importances_all.to_excel(path + '//output//feature_importances_all.xls', index = False)
# %%
#*plotting feature importances
g = sns.catplot(x="feature", y="importance", data=feature_importances_all,
                height=6, aspect = 4, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("Feature importance")
g.set_xticklabels(rotation=90, horizontalalignment='right')
g.savefig(path + "//plots//Features_importance_all.png")

# %%
#*plotting Cumulative importance plot
plt.figure(figsize = (8, 6))
plt.plot(list(range(len(feature_importances_all))), feature_importances_all['cumulative_importance'], 'b-')
plt.xlabel('Number of Features'); plt.ylabel('Cumulative Importance'); 
plt.title('Cumulative Feature Importance');
plt.savefig(path + '//plots//Cumulative_feature_importance.png')
plt.show();

# %%
#*plotting something

g = sns.catplot(x="feature", y="value", hue="variable", data=feature_importances_cv_pivot,
                height=6, aspect = 4, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("feature importance")
g.set_xticklabels(rotation=90, horizontalalignment='right')

# %%
#* creating new subset of features

subset_1 = feature_importances['feature'].tolist()
subset_2 = feature_importances_all['feature'].head(40).tolist()
subset_3 = feature_importances_cv['feature'].tolist()

drop_features_rfe_top25 = [x for x in list(x_train_1.columns) if x not in subset_1]
drop_features_rfecv = [x for x in list(x_train_1.columns) if x not in subset_3]
drop_features_all = [x for x in list(x_train_1.columns) if x not in subset_2]


# %%
#*checking performance with different subset of features
rf = RandomForestClassifier( random_state=666, verbose = 1)
subsets = [drop_features_corr,
           drop_features_corr,  
           drop_features_filter2, 
           drop_features_rfe_rf, 
           drop_features_rfecv_rf, 
           drop_features_rfe_ada, 
           drop_features_rfecv_ada, 
           drop_features_rfe_extra, 
           drop_features_rfecv_extra,
           drop_features_rfe_lgbm, 
           drop_features_rfecv_lgbm,
           drop_features_rfe_top25,
           drop_features_rfecv,
           drop_features_all ]
subset_names = ['Baseline',
                'No_collinearity', 
                'Filtered', 
                'Rfe_rf', 
                'Rfecv_rf', 
                'Rfe_ada', 
                'Rfecv_ada', 
                'Rfe_extra', 
                'Rfecv_extra',
                'Rfe_lgbm', 
                'Rfecv_lgbm',
                'Rfe_top25',
                'Rfecv',
                'top40' ]
    


subsets_metrics = pd.DataFrame()
index = 0
for subset in subsets:
    x_trainx = x_train.copy(deep = True)
    x_valx = x_val.copy(deep = True)
    if index > 0:
        x_trainx = x_train.drop(subset, axis = 1)
        x_valx = x_val.drop(subset, axis =1)
     
    print('training random forest on subset: ', index)   
    rf.fit(x_trainx, y_train)
    y_pred = rf.predict(x_valx)
    
    report = classification_report(y_val, y_pred, output_dict = True)
    pd_report = pd.DataFrame.from_dict(report)
    
    subsets_metrics.loc[index, 'Subset'] = subset_names[index]
    subsets_metrics.loc[index, 'Accuracy'] = pd_report.loc['precision', 'accuracy']
    subsets_metrics.loc[index, 'Precision_0'] = pd_report.loc['precision', '0']
    subsets_metrics.loc[index, 'Precision_1'] = pd_report.loc['precision', '1']
    subsets_metrics.loc[index, 'Recall_0'] = pd_report.loc['recall', '0']
    subsets_metrics.loc[index, 'Recall_1'] = pd_report.loc['recall', '1']
    subsets_metrics.loc[index, 'F1_0'] = pd_report.loc['f1-score', '0']
    subsets_metrics.loc[index, 'F1_1'] = pd_report.loc['f1-score', '1']
    subsets_metrics.loc[index, 'Support_0'] = pd_report.loc['support', '0']
    subsets_metrics.loc[index, 'Support_1'] = pd_report.loc['support', '1']
    subsets_metrics.loc[index, 'MacroAVG_precision'] = pd_report.loc['precision', 'macro avg']
    subsets_metrics.loc[index, 'MacroAVG_recall'] = pd_report.loc['recall', 'macro avg']
    subsets_metrics.loc[index, 'WeightedAVG_precision'] = pd_report.loc['precision', 'weighted avg']
    subsets_metrics.loc[index, 'WeightedAVG_recall'] = pd_report.loc['recall', 'weighted avg']
    
    index = index + 1


subsets_metrics.to_excel(path + '//output//Feature_subsets_metrics.xlsx', index = False)






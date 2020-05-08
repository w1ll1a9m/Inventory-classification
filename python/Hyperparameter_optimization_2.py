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
from sklearn.model_selection import train_test_split,  cross_val_score
from sklearn.ensemble import RandomForestClassifier
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

from skopt.space import Integer, Categorical, Real
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt import callbacks
import skopt
import skopt.plots
import lightgbm as lgb
from sklearn.externals import joblib

from sklearn.metrics import roc_auc_score
from skopt.callbacks import DeltaXStopper
from skopt.callbacks import EarlyStopper
from skopt import dump, load
from skopt.callbacks import CheckpointSaver

from Utilities import Preprocessing as preprocessing

#%%
#* Loaing and formatting data 
path = os.getcwd()
path = 'c:\\William\\Slobs\\new'

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
#*sampling the huge ass dataset if you want

x_cvtrain70,  x_cvtrain30, y_cvtrain70, y_cvtrain30 = train_test_split(x_cvtrain_unbalanced, y_cvtrain_unbalanced, test_size = 0.3, stratify = y_cvtrain_unbalanced)



# %%
#models
space = []
rf = ensemble.RandomForestClassifier(n_estimators = 100, n_jobs =-1)
ada = ensemble.AdaBoostClassifier(n_estimators = 100)
extra = ensemble.ExtraTreesClassifier(n_estimators = 100, n_jobs =-1)
knn = neighbors.KNeighborsClassifier()

num_boost_round = 300
early_stopping_rounds = 30
d_train = lgb.Dataset(x_cvtrain30, label=y_cvtrain30)


def train_evaluate(d_train, params):
    
    cv_results = lgb.cv(params, d_train, num_boost_round=300, nfold=5, 
                    verbose_eval=20, early_stopping_rounds=30)
    
    score = cv_results['auc-mean'][-1]

    
    
    return score

static_params = {'boosting': 'gbdt',
                'objective':'binary',
                'metric': 'auc',
                'num_threads': 4,
                }

hpo_params = {'n_calls':100,
              'n_random_starts':10,
              'base_estimator':'ET',
              'acq_func':'EI',
              'xi':0.02,
              'kappa':1.96,
              'n_points':10000,
             }
rf_space = [Categorical([10, 100, 500], name = 'n_estimators'),
            Categorical(['auto', 'log2'], name = 'max_features'),
            Categorical([2, 5, 10, 20, None], name = 'max_depth'),
            Real(0.0001, 1, name = 'min_samples_split'),
            Integer (1, 5, name = 'min_samples_leaf'),
            Categorical([None, 50, 100, 150, 200], name = 'max_leaf_nodes')
            #Integer(1, 37, name = 'max_features')
            ]

ada_space = [#Integer(200, 500, name = 'n_estimators'),
             Real(0.01, 1, prior = "log-uniform", name = 'learning_rate')
             ]

gbc_space = [Real(0.01, 1, prior = "log-uniform", name = 'learning_rate'),
             Integer(200, 500, name = 'n_estimators'),
             Integer(1, 10, name = 'max_depth'),
             Real (0.1, 1, name = 'min_samples_split'),
             Real (0.1, 0.5, name = 'min_samples_leaf'),
             Integer(1, 10, name = 'max_features')
             ]

extra_space = [#Integer(200, 500, name = 'n_estimators'),
               Categorical(['auto', 'log2'], name = 'max_features'),
               Categorical([2, 5, 10, 20, None], name = 'max_depth'),
               Real(0.1, 1, name = 'min_samples_split'),
               Integer (1, 5, name = 'min_samples_leaf'),
               Categorical([None, 50, 100, 150, 200], name = 'max_leaf_nodes')
               ]

knn_space = [Categorical([3, 5, 7, 9, 11, 15], name = 'n_neighbors'),
             Integer(1, 4, name = 'p')
             ]

xgb_space = [Real(0.6, 0.7, name="colsample_bylevel"),
             Real(0.6, 0.7, name="colsample_bytree"),
             Real(0.01, 1, name="gamma"),
             Real(0.0001, 1, name="learning_rate"),
             Real(0.1, 10, name="max_delta_step"),
             Integer(6, 15, name="max_depth"),
             Real(10, 500, name="min_child_weight"),
             Integer(10, 100, name="n_estimators"),
             Real(0.1, 100, name="reg_alpha"),
             Real(0.1, 100, name="reg_lambda"),
             Real(0.4, 0.7, name="subsample"),
]

lgbm_space = [skopt.space.Real(0.01, 0.5, name='learning_rate', prior='log-uniform'),
         skopt.space.Integer(1, 30, name='max_depth'),
         skopt.space.Integer(2, 100, name='num_leaves'),
         skopt.space.Integer(10, 1000, name='min_data_in_leaf'),
         skopt.space.Real(0.1, 1.0, name='feature_fraction', prior='uniform'),
         skopt.space.Real(0.1, 1.0, name='subsample', prior='uniform'),
         ]

space = lgbm_space

@skopt.utils.use_named_args(space)
def objective(**params):
    all_params = {**params, **static_params}
    #model.set_params(**params)
    print(params)
    #score = -np.mean(cross_val_score(model, x_cvtrain30, y_cvtrain30, cv=5, n_jobs=-1, scoring = 'roc_auc'))
    lgbm_score = -1.0 * train_evaluate(d_train, all_params) 
    return lgbm_score





delta = 0.001
early_stop = skopt.callbacks.DeltaXStopper(delta)
#checkpoint_saver = skopt.callbacks.CheckpointSaver("rf_results_check.pkl", compress=9)

#print('\n optimizing RF \n')
#model = rf
#space = rf_space
#
##, x0 =[500, 37, 110, 2, 1]
#
#rf_results =  skopt.gp_minimize(objective, space,  verbose = True, random_state=0)
#skopt.dump(rf_results, path + '//models//hpo//rf_results_gp_objective.pkl')
#skopt.dump(rf_results, path + '//models//hpo//rf_results_gp_no_objective.pkl', store_objective = False)
#
#print('\n optimizing ADA \n')
#space = ada_space
#model = ada
#ada_results =  skopt.gp_minimize(objective,  ada_space,  verbose = True, random_state=0)
#skopt.dump(ada_results, path + '//models//ada_results_gp.pkl')
#
#
#print('\n optimizing GBC \n')
#space = gbc_space
#model = gbc
#gbc_results =  skopt.gp_minimize(objective,  space, callback = [early_stop], verbose = True, random_state=0)
#skopt.dump(gbc_results, path + '//models//gbc_results.pkl')
#print('\n optimizing extra \n')
#space = extra_space
#model = extra
#extra_results =  skopt.gp_minimize(objective, space, verbose = True, random_state=0)
#skopt.dump(extra_results, path + '//models//extra_results_gp.pkl')
#print('\n optimizing KNN \n')
#space = knn_space
#model = knn
model = lgb
lgbm_results =  skopt.gp_minimize(objective,  space,  verbose = True, random_state=0)
#skopt.dump(lgbm_results, path + '//models//lgbm_results_gp.pkl')

# %%
#*loading results

lgbm_results = load(path + '//models//lgbm_hpo_results_gp.pkl')
rf_results = load(path + '//models//rf_hpo_results_gp.pkl')
ada_results = load(path + '//models//ada_hpo_results_gp.pkl')
extra_results = load(path + '//models//extra_hpo_results_gp.pkl')

results = [('Lgbm_results', lgbm_results),
           ('Random_Forest_results', rf_results),
           ('Ada_boost_results', ada_results),
           ('Extra_Trees_results', extra_results)]

skopt.plots.plot_convergence(*results)

# %%
#* retrieving the best parameters
print('Best parameters for Random forest')
print("Best score=%.4f" % rf_results.fun)

print("""Best parameters:
-max_features = %s
- max_depth=%.6f
- min_samples_split=%.6f
- min_samples_leaf=%.6f
- max_leaf_nodes=%s""" % (rf_results.x[0], rf_results.x[1], 
                            rf_results.x[2], rf_results.x[3], 
                            rf_results.x[4]))



# %%
#model = ensemble.RandomForestClassifier(n_estimators = 100)
modelx  = ensemble.RandomForestClassifier(n_estimators = 100, n_jobs =-1)

#print('cross validating')
#result = np.mean(cross_val_score(modelx, x_cvtrain, y_cvtrain, cv=5, n_jobs=-1,
#                                    scoring = 'roc_auc'))
print('fitting')
modelx.fit(x_train, y_train,)

y_pred = modelx.predict(x_val)

report = classification_report(y_val, y_pred , output_dict = True)

pd_report = pd.DataFrame.from_dict(report)

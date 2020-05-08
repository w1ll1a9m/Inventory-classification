# %%
import sys

sys.path.append("C:\\William\\Slobs\\new\\Python\\")
import os
from sklearn.tree import export_graphviz
import six
import pydot
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

from Utilities import Preprocessing as preprocessing
# %%
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
rf = joblib.load(path + '//models//cv//RandomForestClassifier_cal_iso.sav')
# %%
#* explore the tree
col = list(x_cvtrain.columns)
dotfile = six.StringIO()
i_tree = 0
for tree_in_forest in rf.estimators_:
    print('iteration: ', i_tree)
    print('exporting tree')
    export_graphviz(tree_in_forest,out_file='tree.dot',
    feature_names=col,
    filled=True,
    rounded=True)
    print('graph from pydot')
    (graph,) = pydot.graph_from_dot_file('tree.dot')
    name = 'tree' + str(i_tree)
    print('writing png')
    graph.write_png(path + '//Trees//'+name+  '.png')
    print('os command')
    os.system('dot -Tpng tree.dot -o tree.png')
    i_tree +=1
# %%
rf55 =  ensemble.RandomForestClassifier(n_estimators = 500, n_jobs =-1)
print('fitting rf')
# fit and calibrate model on training data
rf_calibrator_iso = CalibratedClassifierCV(rf, method ='isotonic', cv=5)
rf_calibrator_iso.fit(x_cvtrain_unbalanced, y_cvtrain_unbalanced)
joblib.dump(rf_calibrator_iso, path +'//models//cv//RandomForestClassifier_cal_iso.sav')
# %%
cols = list(x_cvtrain.columns)
X = x_cvtrain.values
explainer = lime.lime_tabular.LimeTabularExplainer(X, feature_names=x_cvtrain.columns, class_names=['NO SLOB', ' SLOB'], discretize_continuous=True)

# %%
choosen_instance = x_test.loc[[541775]].values[0]
exp = explainer.explain_instance(choosen_instance, rf_calibrator_iso.predict_proba, num_features=10)

# %%

explist = exp.as_list()
# %%

indexes = list(x_test.index.values)

l1, l2, l3, l4, l5, l6, l7, l8, l9, l10 = np.array_split(indexes, 10)
l1 =l1.tolist()
l2 =l2.tolist()
l3 =l3.tolist()
l4 =l4.tolist()
l5 =l5.tolist()
l6 =l6.tolist()
l7 =l7.tolist()
l8 =l8.tolist()
l9 =l9.tolist()
l10 =l10.tolist()

# %%
rf_exp = pd.DataFrame()
i = 0
for index in tqdm.tqdm(l10):
   
    
    choosen_instance = x_test.loc[[index]].values[0]
    exp = explainer.explain_instance(choosen_instance, rf.predict_proba, num_features=10)
    explist = exp.as_list()
    tempdf = pd.DataFrame(explist)
    tempdf['order'] = i
    tempdf['oldindex'] = index
    rf_exp = rf_exp.append(tempdf)
    
    if i % 10 == 0:
    
        rf_exp.to_excel(path + '//output//rf_explanations_test_l10.xlsx')
        
    
    i=i+1
    
rf_exp.to_excel(path + '//output//rf_explanations_test_l10.xlsx')   


# %%
for index in tqdm.tqdm(l1):
    print(index)
# %%
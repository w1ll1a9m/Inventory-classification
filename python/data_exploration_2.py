#%%
import sys

#sys.path.append("C:\\William\\Slobs\\new\\Python\\")
import numpy as np
import pandas as pd
import seaborn as sns
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing, tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler, KBinsDiscretizer, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from dataframe_column_identifier import DataFrameColumnIdentifier
from sklearn.feature_selection import RFE, RFECV
from Utilities import Preprocessing as preprocessing
import os
plt.style.use('default')

#%%
#* Loading and formatting data 
path = os.getcwd()
path = 'c:\\William\\Slobs\\new'

test = pd.read_csv(path + '\\data\\slobs_2019_06_test_all.csv')
cvtrain = pd.read_csv(path + '\\data\\slobs_2019_05_cvtrain_all.csv')
train = pd.read_csv(path + '\\data\\slobs_2019_04_train_all.csv')
val = pd.read_csv(path + '\\data\\slobs_2019_05_val.csv')

prepro = preprocessing()

x_train, x_val, x_test, y_train, y_val, y_test, x_cvtrain, y_cvtrain, x_train_unbalanced, y_train_unbalanced, x_cvtrain_unbalanced, y_cvtrain_unbalanced, og_features, og_features_train, catcolumns, numcolumns = prepro.Get_sets(train, val, test, cvtrain)


#%%
#* Scaling and binning the train data to review
scaler = RobustScaler()
data_flo = cvtrain.select_dtypes(include=['float'])
data_int = cvtrain.select_dtypes(include=['int'])
data_str = cvtrain.select_dtypes(include='object')
numcolumns = data_flo.columns

binner = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
cvtrain_binned = cvtrain.copy(deep = True)
cvtrain_binned[numcolumns] = binner.fit_transform(cvtrain[numcolumns])
#%%

#*checking correlations using the binning

for x in cvtrain_binned:
    if  x != 'Target_slow3cm' :

        print('SLob Correlation by:', x)
        print(cvtrain_binned[[x, 'Target_slow3cm']].groupby(x, as_index=False).mean().sort_values(by='Target_slow3cm', ascending=False))
        print('-'*10, '\n')

# %%
#checking numerical variables distributions
plt.style.use('bmh')
cvtrain_binned['Log_test'] = np.log10(cvtrain['Lag1_6m_MAE']+1)
plt.figure(figsize=(9, 8))
sns.distplot(cvtrain['Lag1_6m_MAE'], bins = 10, kde = True, label='Lag1_6m_MAE')



# %%
#some pair plots
plt.style.use('bmh')
g = sns.pairplot(cvtrain.sample(1000), vars=[ 'Effective_stock_L', 'Safety_stock_L', 'Cycle_stock_L', 'New_stock_L', 'Obsolete_stock_L', 'Excess_push_stock_L'], hue="Target_slow3cm", palette="Set2", diag_kind="hist", height=2.5)
# %%
#plt.style.use('bmh')
#graph individual features by target
fig, saxis = plt.subplots(2, 3,figsize=(10,10))

sns.barplot(x = 'Mat_status_change', y = 'Target_slow3cm', data=cvtrain_binned, ax = saxis[0,0])
sns.barplot(x = 'Mat_status_change_3m', y = 'Target_slow3cm', data=cvtrain_binned, ax = saxis[0,1])
sns.barplot(x = 'Mat_status_change_6m', y = 'Target_slow3cm', data=cvtrain_binned, ax = saxis[0,2])

sns.pointplot(x = 'Lag1_6m_MASE', y = 'Target_slow3cm',  data=cvtrain_binned, ax = saxis[1,0])
sns.pointplot(x = 'Lag1_6m_MAE', y = 'Target_slow3cm',  data=cvtrain_binned, ax = saxis[1,1])
sns.pointplot(x = 'Lag1_error_l', y = 'Target_slow3cm', data=cvtrain_binned, ax = saxis[1,2])


# %%
#plotting feature interactions on numerical features

train_smpl = cvtrain.sample(1000)
train_smply = train_smpl['Target_slow3cm']
train_smpl.drop('Target_slow3cm', axis = 1, inplace = True)

plt.style.use('default')
trainx_float = train_smpl[numcolumns]
trainx_float_std = (trainx_float - trainx_float.mean()) / (trainx_float.std())              # standardization
data = pd.concat([train_smply, trainx_float_std.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="Target_slow3cm",
                    var_name="features",
                    value_name='value')
#For boxplot
plt.figure(figsize=(10,10))
sns.boxplot(x="features", y="value", hue="Target_slow3cm", data=data)
plt.xticks(rotation=90)

#for violinplot
#plt.figure(figsize=(10,10))
#sns.violinplot(x="features", y="value", hue="Target_slow", data=data,split=True, inner="quart")
#plt.xticks(rotation=90)

#for swarmplot

#plt.figure(figsize=(10,10))
#tic = time.time()
#sns.swarmplot(x="features", y="value", hue="Target_slow", data=data)
#plt.xticks(rotation=90)

# %%
#correlation matrix
plt.style.use('default')

corr = cvtrain[numcolumns].corr()
plt.figure(figsize=(15, 15))

sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 5}, square=True)
#%%
import numpy as np
import pandas as pd
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
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN
import os

path = os.getcwd()
path = 'c:\\William\\Slobs\\new'



convert_dict = {'RecordID' : int,
                'Product_category' : str,
                 'Brand_type' : str, 
                 'Region' : str, 
                 'SG' : int, 
                 'ABCXYZ' : int, 
                 'Mat_status' : int, 
                 'Mat_status_change' : int, 
                 'Last_mat_status' : int, 
                 'Last_mat_status_change' : int, 
                 'months_since_mat_status_change' : int,
                 'Mat_status_change_3m' : int,
                 'Mat_status_change_6m' : int,
                 'Month' : int, 
                 'Cost_liter' : 'float32', 
                 'Produced_vol' : 'float32', 
                 'Stock_coverage_months' : 'float32', 
                 'Effective_stock_L' : 'float32', 
                 'Safety_stock_L' : 'float32', 
                 'Cycle_stock_L' : 'float32', 
                 'New_stock_L' : 'float32', 
                 'Slow_stock_L' : 'float32', 
                 'Obsolete_stock_L' : 'float32', 
                 'Excess_push_stock_L' : 'float32', 
                 'Pr_effective_stock' : 'float32', 
                 '3m_delta_effective_stock' : 'float32', 
                 'Pr_slow_stock' : 'float32', 
                 'Pr_obsolete_stock' : 'float32', 
                 'Pr_CycleStock' : 'float32', 
                 'Pr_ExcessPushStock' : 'float32', 
                 'Pr_NewStock' : 'float32', 
                 'Pr_BlockedStock' : 'float32', 
                 'MA6_CycleStock' : 'float32', 
                 'MA6_ExcessPushStock' : 'float32', 
                 'MA6_NewStock' : 'float32', 
                 'MA6_BlockedStock' : 'float32', 
                 'Slobs_dc_l' : 'float32', 
                 'Pm_slobs_dc_l' : 'float32', 
                 '3pm_slobs_dc_l' : 'float32', 
                 '6pm_slobs_dc_l' : 'float32', 
                 '6m_avg_slobs_dc_l' : 'float32', 
                 'Slobs_dc_cat_l' : 'float32', 
                 'Pm_slobs_dc_cat_l' : 'float32', 
                 '3pm_slobs_dc_cat_l' : 'float32', 
                 '6pm_slobs_dc_cat_l' : 'float32', 
                 '6m_avg_slobs_dc_cat_l' : 'float32', 
                 'Slobs_dc_cat_brand_l' : 'float32', 
                 'Pm_slobs_dc_cat_brand_l' : 'float32', 
                 '3pm_slobs_dc_cat_brand_l' : 'float32', 
                 '6pm_slobs_dc_cat_brand_l' : 'float32', 
                 '6m_avg_slobs_dc_cat_brand_l' : 'float32', 
                 'Actuals_l' : 'float32', 
                 '1m_forecast_l' : 'float32', 
                 '5m_forecast_l' : 'float32', 
                 'sum_6m_forecast_l' : 'float32', 
                 'Lag1_l' : 'float32', 
                 'Lag1_error_l' : 'float32', 
                 'Lag1_6m_MAE' : 'float32', 
                 'Lag1_6m_Sum_e' : 'float32', 
                 'Lag1_ts' : 'float32', 
                 'Lag1_6m_MASE' : 'float32', 
                 'MaxAE_country_cat' : 'float32', 
                 'Delta_maxAE_country_cat' : 'float32', 
                 'delta_pm_lag1_error' : 'float32', 
                 'Target_slow' : int,
                 'Target_slow3cm' : int}

test = pd.read_csv(path + '\\data\\slobs_2019_06_test_all.csv')
cvtrain = pd.read_csv(path + '\\data\\slobs_2019_05_cvtrain_all.csv')
train = pd.read_csv(path + '\\data\\slobs_2019_04_train_all.csv')
val = pd.read_csv(path + '\\data\\slobs_2019_05_val_all.csv')
test.dropna(how='all',inplace=True)
cvtrain.dropna(how='all',inplace=True)
train.dropna(how='all',inplace=True)
val.dropna(how='all',inplace=True)

test.infer_objects()
cvtrain.infer_objects()
train.infer_objects()
val.infer_objects()

columns = list(cvtrain.columns)

test = test.astype(convert_dict)
cvtrain = cvtrain.astype(convert_dict)
train = train.astype(convert_dict)
val = val.astype(convert_dict)

#cut

train['set'] = 1
val['set'] = 0

merged_data = train.append(val, ignore_index = True)

merged_data =  pd.get_dummies(merged_data)

x_train = merged_data[merged_data['set'] == 1]
x_val = merged_data[merged_data['set'] == 0]

y_train = x_train['Target_slow3cm']
y_val = x_val['Target_slow3cm']


x_train.drop(['Target_slow3cm', 'Target_slow', 'set'], axis = 1, inplace = True)
x_val.drop(['Target_slow3cm', 'Target_slow', 'set'], axis = 1, inplace = True)
#%%
print('creating smote')
x_2, y_2 = SMOTE( n_jobs = 4).fit_resample(x_train, y_train)
print('creating adasyn')
x_3, y_3 = ADASYN(n_jobs = 4).fit_resample(x_train, y_train )
print('creating borderlinesmote')
x_4, y_4 = BorderlineSMOTE( n_jobs = 4).fit_resample(x_train, y_train)
print('creating smoteenn')
x_5, y_5 = SMOTEENN( n_jobs = 4).fit_resample(x_train, y_train)

#%%

#cut
rf = RandomForestClassifier(n_estimators=100, verbose = 1, n_jobs = -1, class_weight ='balanced')
print('Training forest')
rf.fit(x_train, y_train)

rf2 = RandomForestClassifier(n_estimators=100, verbose = 1, n_jobs = 4)
print('Training forest resampled by smote')
rf2.fit(x_2, y_2 )

rf3 = RandomForestClassifier(n_estimators=100, verbose = 1, n_jobs = 4)
print('Training forest resampled by adasyn')
rf3.fit(x_3, y_3)


rf4 = RandomForestClassifier(n_estimators=100, verbose = 1, n_jobs = 4)
print('Training forest resampled by borderlinesmote')
rf4.fit(x_4, y_4)

rf5 = RandomForestClassifier(n_estimators=100, verbose = 1, n_jobs = 4)
print('Training forest resampled by  smoteenn')
rf5.fit(x_5, y_5)


rf0 = RandomForestClassifier(n_estimators=100, verbose = 1, n_jobs = 4)
print('Training unbalanced')
rf0.fit(x_train, y_train)


#%%



x = x_val
y = y_val

preds = pd.DataFrame()


y_pred0 = rf0.predict(x)
y_pred1 = rf.predict(x)
y_pred2 = rf2.predict(x)
y_pred3 = rf3.predict(x)
y_pred4 = rf4.predict(x)
y_pred5 = rf5.predict(x)



preds['Unbalanced'] = y_pred0
preds['Class_weighted'] = y_pred1
preds['SMOTE'] = y_pred2
preds['ADASYN'] = y_pred3
preds['BorderLine_SMOTE'] = y_pred4
preds['SMOTEENN'] = y_pred5

#accuracy = accuracy_score(y,y_pred)
#print('Accuracy:', accuracy)

#classification_report = pd.DataFrame()
#target_names = ['Slob', 'no Slob']

classifiers = [ 'Unbalanced', 'Class_weighted', 'SMOTE', 'ADASYN', 'Borderline_SMOTE', 'SMOTEENN']


report0 = classification_report(y_val, preds['Unbalanced'], output_dict = True)
report1 = classification_report(y_val, preds['Class_weighted'], output_dict = True)
report2 = classification_report(y_val, preds['SMOTE'], output_dict = True)
report3 = classification_report(y_val, preds['ADASYN'], output_dict = True)
report4 = classification_report(y_val, preds['BorderLine_SMOTE'], output_dict = True)
report5 = classification_report(y_val, preds['SMOTEENN'], output_dict = True)

pd_report0 = pd.DataFrame.from_dict(report0)
pd_report1 = pd.DataFrame.from_dict(report1)
pd_report2 = pd.DataFrame.from_dict(report2)
pd_report3 = pd.DataFrame.from_dict(report3)
pd_report4 = pd.DataFrame.from_dict(report4)
pd_report5 = pd.DataFrame.from_dict(report5)


sampling_metrics = pd.DataFrame()

reports = [pd_report0, pd_report1, pd_report2, pd_report3, pd_report4, pd_report5 ]


index = 0

for clf in classifiers:
    
    sampling_metrics.loc[index, 'Approach'] = clf
    sampling_metrics.loc[index, 'Accuracy'] = reports[index].loc['precision', 'accuracy']
    sampling_metrics.loc[index, 'Precision_0'] = reports[index].loc['precision', '0']
    sampling_metrics.loc[index, 'Precision_1'] = reports[index].loc['precision', '1']
    sampling_metrics.loc[index, 'Recall_0'] = reports[index].loc['recall', '0']
    sampling_metrics.loc[index, 'Recall_1'] = reports[index].loc['recall', '1']
    sampling_metrics.loc[index, 'F1_0'] = reports[index].loc['f1-score', '0']
    sampling_metrics.loc[index, 'F1_1'] = reports[index].loc['f1-score', '1']
    sampling_metrics.loc[index, 'Support_0'] = reports[index].loc['support', '0']
    sampling_metrics.loc[index, 'Support_1'] = reports[index].loc['support', '1']
    sampling_metrics.loc[index, 'MacroAVG_precision'] = reports[index].loc['precision', 'macro avg']
    sampling_metrics.loc[index, 'MacroAVG_recall'] = reports[index].loc['recall', 'macro avg']
    sampling_metrics.loc[index, 'WeightedAVG_precision'] = reports[index].loc['precision', 'weighted avg']
    sampling_metrics.loc[index, 'WeightedAVG_recall'] = reports[index].loc['recall', 'weighted avg']
    
    index = index + 1
    
#%%

sampling_metrics.loc[0, 'Approach'] = 'Unbalanced_all'
sampling_metrics.loc[1, 'Approach'] = 'Class_weighted_all'
sampling_metrics.loc[2, 'Approach'] = 'SMOTE_all'
sampling_metrics.loc[3, 'Approach'] = 'ADASYN_all'
sampling_metrics.loc[4, 'Approach'] = 'Borderline_SMOTE_all'
sampling_metrics.loc[5, 'Approach'] = 'SMOTEENN_all'

#%%

sampling_metrics.to_excel(path + '//output//final_sampling_metrics.xlsx', index = False)

#%%
""" 
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, columns=np.unique(y_test), index = np.unique(y_test))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size """

#%%
from sklearn.metrics import classification_report
rf_feature = rf.fit(x_train,y_train).feature_importances_

cols = x_train.columns.values

feature_dataframe = pd.DataFrame( {'features': cols,
     'Random Forest feature importances': rf_feature,

    })

trace = go.Scatter(
    y = feature_dataframe['Random Forest feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['Random Forest feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Random Forest Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')


#%%
# Getting the no. of instances with Label 0
n_class_0 = test[test['Target_slow3cm'] == 0].shape[0]
# Getting the no. of instances with label 1
n_class_1 = test[test['Target_slow3cm']== 1].shape[0]
# Bar Visualization of Class Distribution
import matplotlib.pyplot as plt # required library
x = ['0', '1']
y = np.array([n_class_0, n_class_1])
plt.bar(x, y)
plt.xlabel('Labels/Classes')
plt.ylabel('Number of Instances')
plt.title('Distribution of Labels/Classes in the Dataset')
#%%
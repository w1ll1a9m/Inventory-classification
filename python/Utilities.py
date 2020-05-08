import os

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn import tree
import joblib
from imblearn.over_sampling import SMOTE

class Preprocessing:
    
    def __init__( self):
        pass
        
        
    
    def Get_sets( self, train, val, test, cvtrain):
        self.convert_dict = {'RecordID' : int,
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
        self.train = train
        self.test = test
        self.val = val
        self.cvtrain = cvtrain
        self.path = os.getcwd()
        
        self.test.dropna(how='all',inplace=True)
        self.cvtrain.dropna(how='all',inplace=True)
        self.train.dropna(how='all',inplace=True)
        self.val.dropna(how='all',inplace=True)

        self.test.infer_objects()
        self.cvtrain.infer_objects()
        self.train.infer_objects()
        self.val.infer_objects()

        self.columns = list(self.cvtrain.columns)

        self.test = self.test.astype(self.convert_dict)
        self.cvtrain = self.cvtrain.astype(self.convert_dict)
        self.train = self.train.astype(self.convert_dict)
        self.val = self.val.astype(self.convert_dict)
        
        self.cvtrain['set'] = 1
        self.test['set'] = 0

        self.merged_data = self.cvtrain.append(self.test, ignore_index = True)

        self.merged_data =  pd.get_dummies(self.merged_data)

        self.x_cvtrain = self.merged_data[self.merged_data['set'] == 1]
        self.x_test = self.merged_data[self.merged_data['set'] == 0]

        self.y_cvtrain = self.x_cvtrain['Target_slow3cm']
        self.y_test = self.x_test['Target_slow3cm']
        
        

        self.x_cvtrain.drop(['Target_slow3cm', 'Target_slow', 'set','RecordID'], axis = 1, inplace = True)
        self.x_test.drop(['Target_slow3cm', 'Target_slow', 'set','RecordID'], axis = 1, inplace = True)

        self.x_cvtrain_cols = self.x_cvtrain.columns

        self.x_cvtrain_smote, self.y_cvtrain_smote = SMOTE(random_state = 3, n_jobs =-1).fit_resample(self.x_cvtrain, self.y_cvtrain)

        self.x_cvtrain_unbalanced = self.x_cvtrain.copy(deep = True)
        self.y_cvtrain_unbalanced = self.y_cvtrain.copy(deep = True)

        #* getting train and test for holdout on Train data

        self.train['set'] = 1
        self.val['set'] = 0

        self.merged_data2 = self.train.append(self.val, ignore_index = True)

        self.merged_data2 =  pd.get_dummies(self.merged_data2)

        self.x_train = self.merged_data2[self.merged_data2['set'] == 1]
        self.x_val = self.merged_data2[self.merged_data2['set'] == 0]

        self.y_train = self.x_train['Target_slow3cm']
        self.y_val = self.x_val['Target_slow3cm']


        self.x_train.drop(['Target_slow3cm', 'Target_slow', 'set', 'RecordID'], axis = 1, inplace = True)
        self.x_val.drop(['Target_slow3cm', 'Target_slow', 'set','RecordID'], axis = 1, inplace = True)

        self.x_train_cols = self.x_train.columns

        self.x_train_smote, self.y_train_smote = SMOTE(random_state = 3, n_jobs =-1).fit_resample(self.x_train, self.y_train)

        self.x_train_unbalanced = self.x_train.copy(deep = True)
        self.y_train_unbalanced = self.y_train.copy(deep = True)

        #*converting to dataframes

        self.x_cvtrain = pd.DataFrame(data = self.x_cvtrain_smote, columns = self.x_cvtrain_cols )
        self.y_cvtrain = pd.DataFrame(data = self.y_cvtrain_smote, columns = ['Target_slow3cm'])

        self.x_train = pd.DataFrame(data = self.x_train_smote, columns = self.x_train_cols )
        self.y_train = pd.DataFrame(data = self.y_train_smote, columns = ['Target_slow3cm'])

        #*shuffling

        self.merged_cvtrain = pd.concat([self.x_cvtrain, self.y_cvtrain], axis=1)
        self.merged_cvtrain = self.merged_cvtrain.sample(frac=1).reset_index(drop=True)
        self.merged_train = pd.concat([self.x_train, self.y_train], axis=1)
        self.merged_train = self.merged_train.sample(frac=1).reset_index(drop=True)


        self.y_cvtrain = self.merged_cvtrain['Target_slow3cm']
        self.x_cvtrain = self.merged_cvtrain.copy(deep = True)
        self.x_cvtrain.drop(columns = ['Target_slow3cm'], inplace = True)

        self.y_train = self.merged_train['Target_slow3cm']
        self.x_train = self.merged_train.copy(deep = True)
        self.x_train.drop(columns = ['Target_slow3cm'], inplace = True)

        #*columntypes

        self.data_flo = self.merged_data.select_dtypes(include=['float32'])
        self.data_int = self.merged_data.select_dtypes(exclude=['float32'])
        self.data_int.drop(['Target_slow3cm', 'Target_slow', 'set', 'RecordID'], axis = 1, inplace = True)
        self.data_str = self.merged_data.select_dtypes(include='object')
        self.numcolumns = list(self.data_flo.columns)
        self.catcolumns = list(self.data_int.columns)
        
        self.og_features = list(self.x_cvtrain.columns)
        self.og_features_train = list(self.x_train.columns)
        
        #* getting dtypes under control

        self.x_train_dummys = {'Product_category_COLORANTS' : int,
                                           'Product_category_CONCRETE FLOOR COATI' : int,
                                           'Product_category_EXTERIOR WALL PAINTS' : int,
                                           'Product_category_INTERIOR WALL PAINTS' : int,
                                           'Product_category_METAL COATINGS' : int, 
                                           'Product_category_PREDECO' : int,
                                           'Product_category_TRIM PAINTS' : int, 
                                           'Product_category_WOODCARE' : int,
                                           'Brand_type_AN DECO Brand' : int, 
                                           'Brand_type_AN non-DECO Brand' : int,
                                           'Brand_type_Flourish' : int, 
                                           'Brand_type_KUS Consumer' : int, 
                                           'Brand_type_KUS Pro' : int,
                                           'Brand_type_Metalcare' : int, 
                                           'Brand_type_Other Deco' : int,
                                           'Brand_type_PreDeco Consumer' : int, 
                                           'Brand_type_PreDeco Pro' : int,
                                           'Brand_type_Premium Pro' : int, 
                                           'Brand_type_Private Label' : int,
                                           'Brand_type_Specialist' : int, 
                                           'Brand_type_White Label' : int,
                                           'Brand_type_Woodcare' : int, 
                                           'Region_ESEA' : int, 
                                           'Region_NWE' : int, 
                                           'Region_UKI' : int}

        self.x_cvtrain_dummys = {'Product_category_PLASTERS/MORTARS' : int,
                            'Brand_type_Building Adhesives' : int, 
                            'Brand_type_Non-AN 3rd Party' : int
                            }

        self.train_convert_dict = {**self.x_train_dummys, **self.convert_dict}
        self.train_convert_dict.pop('Target_slow')
        self.train_convert_dict.pop('Target_slow3cm')
        self.train_convert_dict.pop('Product_category')
        self.train_convert_dict.pop('Brand_type')
        self.train_convert_dict.pop('RecordID')
        self.train_convert_dict.pop( 'Region')
        self.cvtrain_convert_dict = {**self.train_convert_dict, **self.x_cvtrain_dummys}
        self.x_train = self.x_train.astype(self.train_convert_dict)
        self.x_cvtrain = self.x_cvtrain.astype(self.cvtrain_convert_dict)
        
        del self.merged_cvtrain, self.merged_data, self.merged_data2, self.merged_train, self.cvtrain, self.train, self.test, self.val, self.x_cvtrain_smote, self.x_train_smote, self.y_cvtrain_smote, self.y_train_smote
        
        return self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test, self.x_cvtrain, self.y_cvtrain, self.x_train_unbalanced, self.y_train_unbalanced, self.x_cvtrain_unbalanced, self.y_cvtrain_unbalanced, self.og_features, self.og_features_train, self.catcolumns, self.numcolumns
    
    def FeatureSelection(self,og_features, og_features_train, x_cvtrain, x_cvtrain_unbalanced, x_train, x_train_unbalanced, x_val, x_test):
        self.path =  '/Users/williamlopez/Documents/Maastricht University/Internship paper/William/Slobs/new'
        #self.path = self.path + '/Documents/Maastricht University/Internship paper/William/Slobs/new/'

        self.feature_importances_all = pd.read_excel(self.path + '/output/feature_importances_all.xls')
        self.best_all = self.feature_importances_all['feature'].head(42).tolist()
        self.drop_features_all = [x for x in self.og_features if x not in self.best_all]
        self.drop_features_train = [x for x in self.og_features_train if x not in self.best_all]

        self.x_cvtrain = self.x_cvtrain.drop(self.drop_features_all, axis = 1)
        self.x_cvtrain_unbalanced = self.x_cvtrain_unbalanced.drop(self.drop_features_all, axis = 1)
        self.x_train = self.x_train.drop(self.drop_features_train, axis = 1)
        self.x_train_unbalanced = self.x_train_unbalanced.drop(self.drop_features_train, axis = 1)
        self.x_val = self.x_val.drop(self.drop_features_train, axis = 1)
        self.x_test = self.x_test.drop(self.drop_features_all, axis = 1)
        
        return self.x_cvtrain, self.x_cvtrain_unbalanced, self.x_train, self.x_train_unbalanced, self.x_val, self.x_test

        
        
        


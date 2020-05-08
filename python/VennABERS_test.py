
import numpy as np
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

# Some elementary functions to speak the same language as the paper
# (at some point we'll just replace the occurrence of the calls with the function body itself)
def push(x,stack):
    stack.append(x)

def pop(stack):
    return stack.pop()

def top(stack):
    return stack[-1]

def nextToTop(stack):
    return stack[-2]


# perhaps inefficient but clear implementation
def nonleftTurn(a,b,c):   
    d1 = b-a
    d2 = c-b
    return np.cross(d1,d2)<=0

def nonrightTurn(a,b,c):   
    d1 = b-a
    d2 = c-b
    return np.cross(d1,d2)>=0


def slope(a,b):
    ax,ay = a
    bx,by = b
    return (by-ay)/(bx-ax)

def notBelow(t,p1,p2):
    p1x,p1y = p1
    p2x,p2y = p2
    tx,ty = t
    m = (p2y-p1y)/(p2x-p1x)
    b = (p2x*p1y - p1x*p2y)/(p2x-p1x)
    return (ty >= tx*m+b)

kPrime = None

# Because we cannot have negative indices in Python (they have another meaning), I use a dictionary

def algorithm1(P):
    global kPrime
    
    S = []
    P[-1] = np.array((-1,-1))
    push(P[-1],S)
    push(P[0],S)
    for i in range(1,kPrime+1):
        while len(S)>1 and nonleftTurn(nextToTop(S),top(S),P[i]):
            pop(S)
        push(P[i],S)
    return S

def algorithm2(P,S):
    global kPrime
    
    Sprime = S[::-1]     # reverse the stack

    F1 = np.zeros((kPrime+1,))
    for i in range(1,kPrime+1):
        F1[i] = slope(top(Sprime),nextToTop(Sprime))
        P[i-1] = P[i-2]+P[i]-P[i-1]
        if notBelow(P[i-1],top(Sprime),nextToTop(Sprime)):
            continue
        pop(Sprime)
        while len(Sprime)>1 and nonleftTurn(P[i-1],top(Sprime),nextToTop(Sprime)):
            pop(Sprime)
        push(P[i-1],Sprime)
    return F1

def algorithm3(P):
    global kPrime
    
    P[kPrime+1] = P[kPrime]+np.array((1.0,0.0))

    S = []
    push(P[kPrime+1],S)
    push(P[kPrime],S)
    for i in range(kPrime-1,0-1,-1):  # k'-1,k'-2,...,0
        while len(S)>1 and nonrightTurn(nextToTop(S),top(S),P[i]):
            pop(S)
        push(P[i],S)
    return S

def algorithm4(P,S):
    global kPrime
    
    Sprime = S[::-1]     # reverse the stack
    
    F0 = np.zeros((kPrime+1,))
    for i in range(kPrime,1-1,-1):   # k',k'-1,...,1
        F0[i] = slope(top(Sprime),nextToTop(Sprime))
        P[i] = P[i-1]+P[i+1]-P[i]
        if notBelow(P[i],top(Sprime),nextToTop(Sprime)):
            continue
        pop(Sprime)
        while len(Sprime)>1 and nonrightTurn(P[i],top(Sprime),nextToTop(Sprime)):
            pop(Sprime)
        push(P[i],Sprime)
    return F0[1:]

def prepareData(calibrPoints):
    global kPrime
    
    ptsSorted = sorted(calibrPoints)
    
    xs = np.fromiter((p[0] for p in ptsSorted),float)
    ys = np.fromiter((p[1] for p in ptsSorted),float)
    ptsUnique,ptsIndex,ptsInverse,ptsCounts = np.unique(xs, 
                                                        return_index=True,
                                                        return_counts=True,
                                                        return_inverse=True)
    a = np.zeros(ptsUnique.shape)
    np.add.at(a,ptsInverse,ys)
    # now a contains the sums of ys for each unique value of the objects
    
    w = ptsCounts
    yPrime = a/w
    yCsd = np.cumsum(w*yPrime)   # Might as well do just np.cumsum(a)
    xPrime = np.cumsum(w)
    kPrime = len(xPrime)
    
    return yPrime,yCsd,xPrime,ptsUnique

def computeF(xPrime,yCsd):    
    P = {0:np.array((0,0))}
    P.update({i+1:np.array((k,v)) for i,(k,v) in enumerate(zip(xPrime,yCsd))})
    
    S = algorithm1(P)
    F1 = algorithm2(P,S)
    
    # P = {}
    # P.update({i+1:np.array((k,v)) for i,(k,v) in enumerate(zip(xPrime,yCsd))})    
    
    S = algorithm3(P)
    F0 = algorithm4(P,S)
    
    return F0,F1

def getFVal(F0,F1,ptsUnique,testObjects):
    pos0 = np.searchsorted(ptsUnique[1:],testObjects,side='right')
    pos1 = np.searchsorted(ptsUnique[:-1],testObjects,side='left')+1
    return F0[pos0],F1[pos1]

def ScoresToMultiProbs(calibrPoints,testObjects):
    # sort the points, transform into unique objects, with weights and updated values
    yPrime,yCsd,xPrime,ptsUnique = prepareData(calibrPoints)
    
    # compute the F0 and F1 functions from the CSD
    F0,F1 = computeF(xPrime,yCsd)
    
    # compute the values for the given test objects
    p0,p1 = getFVal(F0,F1,ptsUnique,testObjects)
                    
    return p0,p1

def computeF1(yCsd,xPrime):
    global kPrime
    
    P = {0:np.array((0,0))}
    P.update({i+1:np.array((k,v)) for i,(k,v) in enumerate(zip(xPrime,yCsd))})
    
    S = algorithm1(P)
    F1 = algorithm2(P,S)
    
    return F1

def ScoresToMultiProbsV2(calibrPoints,testObjects):
    # sort the points, transform into unique objects, with weights and updated values
    yPrime,yCsd,xPrime,ptsUnique = prepareData(calibrPoints)
   
    # compute the F0 and F1 functions from the CSD
    F1 = computeF1(yCsd,xPrime)
    pos1 = np.searchsorted(ptsUnique[:-1],testObjects,side='left')+1
    p1 = F1[pos1]
    
    yPrime,yCsd,xPrime,ptsUnique = prepareData((-x,1-y) for x,y in calibrPoints)    
    F0 = 1 - computeF1(yCsd,xPrime)
    pos0 = np.searchsorted(ptsUnique[:-1],testObjects,side='left')+1
    p0 = F0[pos0]
    
    return p0,p1





#path = os.getcwd()
path =  '/Users/williamlopez/Documents/Maastricht University/Internship paper/William/Slobs/new'

model_predict_val = pd.read_excel(path+'/output/final_model_predictions_val.xlsx', index_col = 0)
model_predict_test = pd.read_excel(path+'/output/final_model_predictions_testv2.xlsx', index_col = 0)

# %%
subset1 = model_predict_val[['ExtraTreesClassifier','Target_slow3cm']]


#getting the list of pairs

extra_cal_list = list(zip(*map(subset1.get, subset1)))
extra_test_list = model_predict_test['ExtraTreesClassifier_pr']

p0,p1 = ScoresToMultiProbs(extra_cal_list,extra_test_list)
#%%
p0 += 0.
# %%


arr = np.array(p0)
p0_df = pd.DataFrame(data = arr, index = model_predict_test.index, columns = ['p0'])
arr2 = np.array(p1)
p1_df = pd.DataFrame(data = arr2, index = model_predict_test.index, columns = ['p1'] )

p0_p1_df  = pd.concat([p0_df,p1_df], axis=1)

p0_p1_df['final_prob'] = p0_p1_df['p1']/(1-p0_p1_df['p0']+p0_p1_df['p1'])

aaa = p0_p1_df.copy(deep = True)
# %%


# plot perfectly calibrated
plt.plot([0, 1], [0, 1], linestyle='--')
# plot model reliability
fop_extra, mpv_extra = calibration_curve(model_predict_test['Target_slow3cm'], model_predict_test['ExtraTreesClassifier_pr'], n_bins=10)
# plot model reliability
plt.plot(mpv_extra, fop_extra, marker='.')

fop_rf, mpv_rf = calibration_curve(model_predict_test['Target_slow3cm'], p0_p1_df['p0'], n_bins=10)
# plot model reliability
plt.plot(mpv_rf, fop_rf, marker='.')



plt.title('Calibration plots (reliability diagram)', fontweight='bold', fontsize=12)
plt.legend(['Perfect', 'ExtraTrees', 'VENNABBERS'], loc='upper left')

plt.show()


# %%

#for loop to do it for all classifiers


classifiers = [ 'LGBMClassifier',  'AdaBoostClassifier', 'ExtraTreesClassifier', 'RandomForestClassifier']

for cl in classifiers:
    cl2 = cl + '_pr'
    cl3 = cl2 + '_VennABERS'
    
    indices = model_predict_test.index.values
    cal_list = list(zip(*map(model_predict_val.get, [cl, 'Target_slow3cm'])))
    test_list = model_predict_test[cl+'_pr']
    p00,p11 = ScoresToMultiProbs(cal_list,test_list)
    p00 += 0.
    
    
    arr0 = np.array(p00)
    p00_df = pd.DataFrame(data = arr0, index = model_predict_test.index, columns = ['p0'])
    arr1 = np.array(p11)
    p11_df = pd.DataFrame(data = arr1, index = model_predict_test.index, columns = ['p1'] )
    p00_p11_df  = pd.concat([p00_df,p11_df], axis=1)
    p00_p11_df['final_prob'] = p00_p11_df['p1']/(1-p00_p11_df['p0']+p00_p11_df['p1'])
  
    model_predict_test.loc[indices, cl3+'p0'] = p00_p11_df['p0']
    model_predict_test.loc[indices, cl3+'p1'] = p00_p11_df['p1']
    model_predict_test.loc[indices, cl3+'final'] = p00_p11_df['p1']/(1-p00_p11_df['p0']+p00_p11_df['p1'])
    
    
    
    

    
# %%
#*lets check how bad the classifiers are predicting probs

# reliability diagram
fop, mpv = calibration_curve(model_predict_test['Target_slow3cm'], model_predict_test['AdaBoostClassifier_pr_VennABERSfinal'], n_bins=10)

# plot model reliability
plt.plot(mpv, fop, marker='.')

fop_extra, mpv_extra = calibration_curve(model_predict_test['Target_slow3cm'], model_predict_test['ExtraTreesClassifier_pr_VennABERSfinal'], n_bins=10)
# plot model reliability
plt.plot(mpv_extra, fop_extra, marker='.')

fop_rf, mpv_rf = calibration_curve(model_predict_test['Target_slow3cm'], model_predict_test['RandomForestClassifier_pr_VennABERSfinal'], n_bins=10)
# plot model reliability
plt.plot(mpv_rf, fop_rf, marker='.')


fop_lgbm, mpv_lgbm = calibration_curve(model_predict_test['Target_slow3cm'], model_predict_test['LGBMClassifier_pr_VennABERSfinal'], n_bins=10)
# plot model reliability
plt.plot(mpv_lgbm, fop_lgbm, marker='.')


# plot perfectly calibrated
#plt.plot([0, 1], [0, 1], linestyle='--')

lims = [np.min([plt.get_xlim(), plt.get_ylim()]),  # min of both axes 
        np.max([plt.get_xlim(), plt.get_ylim()]),  # max of both axes
        ]
plt.plot(lims, lims, ls="--",  alpha=0.75, zorder=0) #ax1.set_aspect('equal')
plt.set_xlim(lims)
plt.set_ylim(lims)

plt.title('Calibration plots Venn ABERS calibrated models', fontweight='bold', fontsize=12)
plt.legend(['Perfect', 'AdaBoost', 'ExtraTrees', 'RandomForest', 'LGBM'], loc='upper left')

plt.show()


# %%
#*lets check how bad the classifiers are predicting probs
f, ax = plt.subplots(figsize=(6, 6))
# reliability diagram
fop, mpv = calibration_curve(model_predict_test['Target_slow3cm'], model_predict_test['AdaBoostClassifier_pr_VennABERSfinal'], n_bins=10)
# lims = [np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes 
#         np.max([ax.get_xlim(),ax.get_ylim()]),  # max of both axes
#         ]
# ax.plot(lims, lims, ls="--",  alpha=0.75, zorder=0) #ax1.set_aspect('equal')
# ax.set_xlim(lims)
# ax.set_ylim(lims)



# plot model reliability
ax.plot(mpv, fop, marker='.')

fop_extra, mpv_extra = calibration_curve(model_predict_test['Target_slow3cm'], model_predict_test['ExtraTreesClassifier_pr_VennABERSfinal'], n_bins=10)
# plot model reliability
ax.plot(mpv_extra, fop_extra, marker='.')

fop_rf, mpv_rf = calibration_curve(model_predict_test['Target_slow3cm'], model_predict_test['RandomForestClassifier_pr_VennABERSfinal'], n_bins=10)
# plot model reliability
plt.plot(mpv_rf, fop_rf, marker='.')


fop_lgbm, mpv_lgbm = calibration_curve(model_predict_test['Target_slow3cm'], model_predict_test['LGBMClassifier_pr_VennABERSfinal'], n_bins=10)
# plot model reliability
ax.plot(mpv_lgbm, fop_lgbm, marker='.')


# plot perfectly calibrated
ax.plot([0, 1], [0, 1], linestyle='--')

plt.title('Calibration plots Venn ABERS calibrated models', fontweight='bold', fontsize=12)
plt.legend(['Perfect', 'AdaBoost', 'ExtraTrees', 'RandomForest', 'LGBM'], loc='upper left')

plt.show()




################################
# posterior probability applied for xgboost, etc

# generate clf_probs_XXX where XXX is 0.05 .. 1.00 . Defaul for xgb is == 1 - the weight of the +ve examples 

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.isotonic import IsotonicRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#import os; mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-8.1.0-posix-seh-rt_v6-rev0\\mingw64\\bin'; os.environ['PATH'] = mingw_path + ';' + os.environ['PATH'] # needed for xgboost
import xgboost # on Dell G5 installed with: !pip install xgboost
# xgboost loading problems solved: https://www.ibm.com/developerworks/community/blogs/jfp/entry/Installing_XGBoost_For_Anaconda_on_Windows?lang=en
from sklearn import metrics
from sklearn import preprocessing
from sklearn import decomposition

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc # for Latex Interpretation
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib auto
from IPython import get_ipython
get_ipython().magic('matplotlib auto')


# PREPARE DATA

    # plt.close("all")
data_opt = 2 # set manually
if data_opt == 1:
    # option I - use Marine DB
    dfDataScaled = pickle.load(open('C:/reader/models/data/data/DataScaled.pkl','rb'))
    dfDataScaled = dfDataScaled.iloc[:,1:]
    dfDataScaledTrain = dfDataScaled.iloc[200000:300000,1:] # 100000 examples
    dfDataScaledTest  = dfDataScaled.iloc[1:100000,1:] # 99999 examples
    del dfDataScaled
elif data_opt == 2: # load UCI dataset
    use_mesh = 0 # set manually # if == 1, then predictions will be made on top of a mesh (the mesh will be the test dataset)
    use_pca = 0 # set manually # convert the UCI data into 2D data using PCA
    
#        # try the implied posterior probability estimation on weka dataset: e.g. german credit
    UCIdata = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric',header=None,delim_whitespace=True)
#        # UCIdata = pickle.load(open(r'C:\g\r\weka\gcredit.pkl','rb'))
##    UCIdata = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',header=None,delim_whitespace=True)
#    UCIdata = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat',header=None,delim_whitespace=True)

    UCIclasses = UCIdata.iloc[:,-1].unique() # the last column is the class attribute
    if len(UCIclasses) != 2:
        raise Exception('UCIclasses should contain exactly 2 classes')
    else: # convert the classes to "0" and "1"
        class_0 = UCIdata.iloc[:,-1]==UCIclasses[0]
        class_1 = UCIdata.iloc[:,-1]==UCIclasses[1]
        UCIdata.iloc[np.where(class_0)[0],-1] = 1 # note: the classes seem "inversed"
        UCIdata.iloc[np.where(class_1)[0],-1] = 0 # note: the classes seem "inversed"
        del class_0, class_1, UCIclasses
        ### depricated: UCIdata.iloc[:,-1] = -(UCIdata.iloc[:,-1]-2) # so that we have 0-1 classes
    if use_pca == 1:
        ############## PCA (use raw data)
        X = UCIdata.copy()
        y = X.iloc[:,-1];
        X = X.iloc[:,1:-1];
        pca = decomposition.PCA(n_components=2)
        pca.fit(X);
        print(pca.explained_variance_ratio_) # Percentage of variance explained by each of the selected components
        X = pca.transform(X)
        plt.figure(3); plt.scatter(X[:, 0],X[:, 1],c=y)
        #        # plot a "value name" for each color 
        #        #fig, ax = plt.subplots()
        #        x1=list(X[:, 0])
        #        x2 = list(X[:, 1])
        #        classes = list(y)
        #        unique = list(set(classes))
        #        unique2 = list(np.random.rand(len(unique)))
        #        colors = [plt.cm.jet(float(i)/max(unique2)) for i in unique2]
        #        for i, u in enumerate(unique):
        #            xi = [x1[j] for j  in range(len(x1)) if classes[j] == u]
        #            yi = [x2[j] for j  in range(len(x1)) if classes[j] == u]
        #            plt.scatter(xi, yi, c=colors[i], label=str(u))
        #        plt.legend()
        #        figManager = plt.get_current_fig_manager()
        #        figManager.window.showMaximized()
        #        plt.show()
        UCIdata = pd.DataFrame(np.column_stack((X,y)))
    if use_mesh == 0:
        # opt 1 : test set = training set
        # dfDataScaledTest = dfDataScaledTrain.copy()
        # opt 2: 0-50% training, 50-100% test
        dfDataScaledTrain = UCIdata.iloc[0:int(UCIdata.shape[0]/2),:] # UCIdata.copy() #UCIdata.iloc[0:501,:] # 501 for training
        dfDataScaledTest = UCIdata.iloc[int(UCIdata.shape[0]/2):,:] # UCIdata.copy() #UCIdata.iloc[501:,:] # 499 for testing
        dfDataScaledTrain.shape
    elif use_mesh == 1:
        dfDataScaledTrain = UCIdata.copy()
        nx, ny = (100, 120)
        z1 = np.linspace(np.min(X[:,0])-0.5, np.max(X[:,0])+0.5, nx)
        z2 = np.linspace(np.min(X[:,1])-0.5, np.max(X[:,1])+0.5, ny)
        z1v, z2v = np.meshgrid(z1, z2)
        z1vr = z1v.reshape((ny*nx, 1))
        z2vr = z2v.reshape((ny*nx, 1))
        dfDataScaledTest = pd.DataFrame(np.hstack([z1vr,z2vr,np.array([[0]*z1vr.shape[0]]).T]))
        x1_pos = X[y[:]==1,0]; x2_pos = X[y[:]==1,1]; x1_neg = X[y[:]==0,0]; x2_neg = X[y[:]==0,1];
elif data_opt == 3:
    use_mesh = 1 # set manually (if == 0, then  test set = training set; imposed == 1 if use_hastie == 1)
    use_hastie = 0 # set manually
    if use_hastie == 1:
        import rpy2.robjects as robjects
        from rpy2.robjects import numpy2ri
        robjects.r['load'](r'D:\g\py\hot\postprob\ESL.mixture.rda') # https://web.stanford.edu/~hastie/ElemStatLearn/datasets/ESL.mixture.rda
        X = numpy2ri.ri2py(robjects.r['ESL.mixture'][0])
        y = numpy2ri.ri2py(robjects.r['ESL.mixture'][1])
        dfDataScaledTrain = pd.DataFrame(np.vstack((X.T,y)).T)
        # note: test_error = sum(marginal_hastie*(postprob_hastie*I(pred <0)+(1-postprob_hastie)*I(pred>=0)))
        xnew = numpy2ri.ri2py(robjects.r['ESL.mixture'][2])
        postprob_hastie = numpy2ri.ri2py(robjects.r['ESL.mixture'][3])
        dfDataScaledTest = pd.DataFrame(np.vstack([xnew[:,0],xnew[:,1],np.sign(postprob_hastie-0.5)]).T)
        marginal_hastie = numpy2ri.ri2py(robjects.r['ESL.mixture'][4]) # marginal probability at each lattice point
        z1 = numpy2ri.ri2py(robjects.r['ESL.mixture'][5])
        z2 = numpy2ri.ri2py(robjects.r['ESL.mixture'][6])
        nx = len(z1)
        ny = len(z2)
        z1v, z2v = np.meshgrid(z1, z2)
        use_mesh = 1 # redefined, if not already  == 1
        x1_pos = X[0:100,0]; x2_pos = X[0:100,1]; x1_neg = X[100:,0]; x2_neg = X[100:,1]
        # means_hastie = numpy2ri.ri2py(robjects.r['ESL.mixture'][7])
    else:
        # option III: artificially generated 2d data:
        mean_pos = [3,4]
        cov_pos = [[2, 0.5], [0.5, 3]] # [[1, 0.5], [0.5, 0.4]]
        num_pos = 10 # 100
        x1_pos,x2_pos = np.random.multivariate_normal(mean_pos, cov_pos,num_pos).T
        mean_neg = [5,4]
        cov_neg =  [[1, -0.1], [-0.1, 0.4]]    # cov_pos.copy() ,  [[1, -0.1], [-0.1, 0.4]] 
        num_neg = num_pos
        x1_neg,x2_neg = np.random.multivariate_normal(mean_neg, cov_neg,num_neg).T
        X = np.vstack( [ np.vstack([x1_pos,x2_pos]).T , np.vstack([x1_neg,x2_neg]).T])
        y = np.hstack(np.array([[0]*num_pos, [1]*num_neg]))
        dfDataScaledTrain = pd.DataFrame(np.vstack((X.T,y)).T)
        if use_mesh == 0:
        # opt 1 : test set = training set
            dfDataScaledTest = pd.DataFrame(dfDataScaledTrain.copy())
        elif use_mesh == 1:
        # opt 2: make grid predictions
            nx, ny = (100, 120)
            z1 = np.linspace(1, 8, nx) # z1 = np.linspace(1, 8, nx)  # z1 = np.linspace(3, 6, nx)
            z2 = np.linspace(0, 9, ny) # z2 = np.linspace(0, 9, ny)  # z2 = np.linspace(2, 7, ny)
            z1v, z2v = np.meshgrid(z1, z2)
            z1vr = z1v.reshape((ny*nx, 1))
            z2vr = z2v.reshape((ny*nx, 1))
            dfDataScaledTest = pd.DataFrame(np.hstack([z1vr,z2vr,np.array([[0]*z1vr.shape[0]]).T]))
        # for poster: cov_pos = [[2, 0.5], [0.5, 3]] , cov_neg =  [[1, -0.1], [-0.1, 0.4]], use_rbf = 1, C_SVC = 10, gamma_SVC = 0.1
    
    
# START LEARNING:
    
    # set manually
myclf = 'SVC' # 'xgboost', 'RF' , 'LinearSVC', 'SVC', 'Logistic', 'GaussianNB', 'Ada', 'lda' , 'qda'
bootstr = 1 # set manually # number of bootstrap samples - needed for CI computation for the posterior probability

# step 1: produce baseline probablity scores using the "predict_proba" sklearn method (or raw LinearSVC scores)
BSidx = np.empty((dfDataScaledTrain.shape[0],bootstr))
clf_probsBase = [None]*bootstr
if myclf == 'SVC':
    clf_SVC = [None]*bootstr
if (myclf == 'SVC') or (myclf == 'LinearSVC'):
    SVs = [None]*bootstr # needed for SVM
for b in range(0,bootstr):
    print('bootstrap sample ' + str(b))
    if b == bootstr-1: # b = bootstr-1
        BSidx[:,b] = np.arange(0,dfDataScaledTrain.shape[0],1)
    else:
        # stratified bootsrtap - the numer of instances in each class is preserved
        cl0idx = np.array(dfDataScaledTrain.loc[dfDataScaledTrain.iloc[:,-1]==0,dfDataScaledTrain.columns[-1]].index.tolist())
        cl1idx = np.array(dfDataScaledTrain.loc[dfDataScaledTrain.iloc[:,-1]==1,dfDataScaledTrain.columns[-1]].index.tolist())
        a0 = cl0idx[np.random.randint(low=0, high=len(cl0idx), size=len(cl0idx))]
        a1 = cl1idx[np.random.randint(low=0, high=len(cl1idx), size=len(cl1idx))]
        BSidx[:,b] = np.concatenate([a0,a1])
        # BSidx[:,b] = np.random.randint(low=0, high=dfDataScaledTrain.shape[0], size=dfDataScaledTrain.shape[0])  # boostrstap sample
    if myclf == 'xgboost':
        clfBase = xgboost.XGBClassifier(max_depth=5,
                                        n_estimators=10,
                                        reg_alpha = 0,
                                        gamma = 0,
                                        nthred = 3,
                                        subsample = 0.9,
                                        scale_pos_weight = 1
                                       )
    elif myclf == 'RF': # random forest
        n_estimators_RF = 100
        clfBase = RandomForestClassifier(n_estimators = n_estimators_RF, bootstrap=True, class_weight={0: 0.5, 1:0.5}, criterion='gini',
                    max_depth = 4, max_features='auto', max_leaf_nodes=None,
                    min_impurity_decrease=0.0, min_impurity_split=None,
                    min_samples_leaf=1, min_samples_split=2,
                    min_weight_fraction_leaf=0.0, n_jobs=5,
                    oob_score=False, random_state=0, verbose=0, warm_start=False)
    elif myclf == 'LinearSVC': # LinearSVC
        C_LinearSVC = 1*(2**-4) + 0*20
        loss_LinearSVC = 'hinge'
        clfBase = svm.LinearSVC(penalty='l2', # penalty='l2', penalty='l1'
                          loss=loss_LinearSVC, # loss='squared_hinge', loss='hinge'
                          dual=True, 
                          tol=0.0001, 
                          C=C_LinearSVC,
                          multi_class='ovr', 
                          fit_intercept=True, 
                          intercept_scaling=1, 
                          class_weight={0: 0.5, 1:0.5},
                          verbose=0, 
                          random_state=None, 
                          max_iter=10000000)
    elif myclf == 'GaussianNB':
        clfBase = GaussianNB(priors=[0.5,0.5]) # not supported: class_weight={0: 0.5, 1:0.5}
    elif myclf == 'lda':
        clfBase = LinearDiscriminantAnalysis(priors=[0.5,0.5]) # not supported: class_weight={0: 0.5, 1:0.5}
    elif myclf == 'qda':
        clfBase = QuadraticDiscriminantAnalysis(priors=[0.5,0.5]) # not supported: class_weight={0: 0.5, 1:0.5}
    elif myclf == 'Ada': # Adaboost
        clfBase = AdaBoostClassifier(n_estimators=100, learning_rate=0.01)
    elif myclf == 'SVC': # Linear amd RBF SVM
        use_rbf = 1 # set manually
        if use_rbf == 0:
            C_SVC = 2**-4 + 2**2 + 10
            clfBase = svm.SVC(tol=0.0001, 
                              C=C_SVC, 
                              decision_function_shape='ovr', 
                              class_weight={0: 0.5, 1:0.5},
                              verbose=0, 
                              random_state=None, 
                              max_iter=10000000,
                              kernel='linear',
                              probability = True)
        else: # use RBF kernel
            C_SVC = 10 # 10 (hastie = 20) # for the SVMtutorial paper use: 10
            gamma_SVC = 0.001 # 1 , 0.001   (hastie = 0.3) # for the SVMtutorial paper use: 0.001
            clfBase = svm.SVC(tol=0.0001, 
                              C=C_SVC,
                              gamma=gamma_SVC,
                              decision_function_shape='ovr', 
                              class_weight={0: 0.5, 1:0.5},
                              verbose=0, 
                              random_state=None, 
                              max_iter=10000000,
                              kernel='rbf',
                              probability = True)
            ### fit and test on training data:
            # clfBase.fit(dfDataScaledTrain.iloc[:,:-1], dfDataScaledTrain.iloc[:,-1]);
            # preds = clfBase.decision_function(dfDataScaledTrain.iloc[:,:-1])
    elif myclf == 'Logistic': # logistic regression
        C_Logistic = 1000
        penalty_Logistic = 'l2'
        clfBase = LogisticRegression(penalty = penalty_Logistic, C = C_Logistic, solver = 'liblinear', fit_intercept = True,class_weight={0: 0.5, 1:0.5})
        #clfBase = LogisticRegression(penalty = 'l2', C = 10, solver = 'liblinear', fit_intercept = True,class_weight={0: 1, 1:1})
    #
    clfBase.fit(dfDataScaledTrain.iloc[BSidx[:,b],:-1], dfDataScaledTrain.iloc[BSidx[:,b],-1]);
    if (myclf == 'LinearSVC'): # or (myclf == 'SVC'):
        clf_probsBase[b] = clfBase.decision_function(dfDataScaledTest.iloc[:,:-1]) # this is raw score
        a = (dfDataScaledTrain.iloc[BSidx[:,b],-1]*2-1)*clfBase.decision_function(dfDataScaledTrain.iloc[BSidx[:,b],:-1])<=1
        aa = np.arange(0,dfDataScaledTrain.shape[0],1)*a; aa = aa[aa!=0]; 
        SVs[b] = aa.values
    elif (myclf == 'SVC'):
        clf_SVC[b] = clfBase.decision_function(dfDataScaledTest.iloc[:,:-1])
        clf_probsBase[b] = clfBase.predict_proba(dfDataScaledTest.iloc[:,:-1])[:,1] # Platt
        SVs[b] = np.sort(clfBase.support_)
        # SVs[b] = (dfDataScaledTrain.iloc[BSidx[:,b],-1]*2-1)*clfBase.decision_function(dfDataScaledTrain.iloc[BSidx[:,b],:-1])<=1
    else:
        clf_probsBase[b] = clfBase.predict_proba(dfDataScaledTest.iloc[:,:-1])[:,1]

# step 2. Implied probability step 1: produce probablity scores for each threshold. Afterwards, the implied probability will be calculated based on the threshold which produces posterior prob = 0.5
    # thres_pos = np.arange(0.05,10.05,0.15)
if myclf == 'xgboost':
    thres_pos = np.exp(np.arange(-5,12,0.25))
    thres_neg = np.ones(len(thres_pos))
else:
    # option 1. Assign thres_pos and thres_neg - refined grid
####    thres_pos = np.concatenate([np.array([2**x for x in np.arange(-26,-1,0.5)]),np.array([1-2**x for x in np.arange(-1,-26,-0.5)])])
    thres_pos = np.concatenate([[0.99999],np.arange(0.995,0.0,0.995-1),[0.00001]])
    # thres_pos = np.concatenate([np.array([2**x for x in np.arange(-15,-1,0.2)]),np.array([1-2**x for x in np.arange(-1,-15,-0.2)])])
    thres_neg = 1-thres_pos # np.arange(9.95,0.0,-0.05)
#    thres_pos = np.array([2**-26])
#    thres_neg = 1-thres_pos # np.arange(9.95,0.0,-0.05)
#    # option 2. Assign thres_pos and thres_neg - rough grid
#    thres_pos = np.arange(0.9,0.0,-0.1)
#    thres_neg = 1-thres_pos # np.arange(9.95,0.0,-0.05)
clf_probs_BS = [None]*bootstr
clf = [None] * len(thres_pos)
for b in range(0,bootstr): # range(0,bootstr):
    clf_probs = np.zeros((dfDataScaledTest.shape[0],len(thres_pos)))
    for i in range(0,len(thres_pos)):
        print('bs= ' + str(b) + ', thres_idx= ' + str(i))
        # xgboost
        if myclf == 'xgboost': # note: only weight on positive examples is supported
            clf[i] = xgboost.XGBClassifier(max_depth=5,
                                        n_estimators=10,
                                        reg_alpha = 0,
                                        gamma = 0,
                                        nthred = 3,
                                        subsample = 0.9,
                                        scale_pos_weight = thres_pos[i]  # thres_pos[i]  ,  1
                                       )
        elif myclf == 'RF': # random forest
            clf[i] = RandomForestClassifier(n_estimators=n_estimators_RF, bootstrap=True, class_weight={0: thres_neg[i], 1:thres_pos[i]}, criterion='gini',
                        max_depth = 4, max_features='auto', max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=1, min_samples_split=2,
                        min_weight_fraction_leaf=0.0, n_jobs=5,
                        oob_score=False, random_state=0, verbose=0, warm_start=False)
        elif myclf == 'LinearSVC': # LinearSVC
            clf[i] = svm.LinearSVC(penalty='l2', # penalty='l2', penalty='l1'
                              loss=loss_LinearSVC, # loss='squared_hinge', loss='hinge'
                              dual=True, 
                              tol=0.0001, 
                              C=C_LinearSVC, 
                              multi_class='ovr', 
                              fit_intercept=True, 
                              intercept_scaling=1, 
                              class_weight={0: thres_neg[i], 1:thres_pos[i]},  # {0: thres_neg[i], 1:thres_pos[i]}, class_weight={0: 1, 1:1},
                              verbose=0, 
                              random_state=None, 
                              max_iter=10000000)
        elif myclf == 'GaussianNB':
            clf[i] = GaussianNB(priors=[thres_neg[i],thres_pos[i]])
        elif myclf == 'lda':
            clf[i] = LinearDiscriminantAnalysis(priors=[thres_neg[i],thres_pos[i]])
        elif myclf == 'qda':
            clf[i] = QuadraticDiscriminantAnalysis(priors=[thres_neg[i],thres_pos[i]])
        elif myclf == 'Ada': # Adaboost
            clf[i] = AdaBoostClassifier(n_estimators=100, learning_rate=0.01)
        elif myclf == 'SVC': # Linear and RBF SVM. NOTE: only the SVs from clf_probsBase should enter here !
            if use_rbf == 0:
                clf[i] = svm.SVC(tol=0.0001, 
                                  C=C_SVC, 
                                  decision_function_shape='ovr', 
                                  class_weight={0: thres_neg[i], 1:thres_pos[i]},  # {0: thres_neg[i], 1:thres_pos[i]}, class_weight={0: 1, 1:1},
                                  verbose=0, 
                                  random_state=None, 
                                  max_iter=10000000,
                                  kernel='linear',
                                  probability = True)
            else:  # use RBF kernel
                clf[i] = svm.SVC(tol=0.0001, 
                                  C=C_SVC,
                                  gamma=gamma_SVC,
                                  decision_function_shape='ovr', 
                                  class_weight={0: thres_neg[i], 1:thres_pos[i]},  # {0: thres_neg[i], 1:thres_pos[i]}, class_weight={0: 1, 1:1},
                                  verbose=0, 
                                  random_state=None, 
                                  max_iter=10000000,
                                  kernel='rbf',
                                  probability = True)
        elif myclf == 'Logistic': # logistic regression
            clf[i] = LogisticRegression(penalty = penalty_Logistic, C = C_Logistic, solver = 'liblinear', fit_intercept = True,class_weight={0: thres_neg[i], 1:thres_pos[i]}) # {0: 1, 1:1}  ,   {0: thres_neg[i], 1:thres_pos[i]}

        if (myclf == 'Ada'):
            sample_weight = np.ones(dfDataScaledTrain.shape[0])
            sample_weight[np.where(dfDataScaledTrain.iloc[BSidx[:,b],-1]==1)] = thres_pos[i]
            sample_weight[np.where(dfDataScaledTrain.iloc[BSidx[:,b],-1]==0)] = thres_neg[i]
            clf[i].fit(dfDataScaledTrain.iloc[BSidx[:,b],:-1], dfDataScaledTrain.iloc[BSidx[:,b],-1],sample_weight=sample_weight);
#        elif (myclf == 'LinearSVC') or (myclf == 'SVC'): # (myclf == 'LinearSVC---') or (myclf == 'SVC---'):
#            sample_weight = np.ones(len(SVs[b]))
##            sample_weight[np.where(dfDataScaledTrain.iloc[BSidx[SVs[b],b],-1]==1)] = thres_pos[i]
##            sample_weight[np.where(dfDataScaledTrain.iloc[BSidx[SVs[b],b],-1]==0)] = thres_neg[i]
#            clf[i].fit(dfDataScaledTrain.iloc[BSidx[SVs[b],b],:-1], dfDataScaledTrain.iloc[BSidx[SVs[b],b],-1],sample_weight=sample_weight);
        else:
            clf[i].fit(dfDataScaledTrain.iloc[BSidx[:,b],:-1], dfDataScaledTrain.iloc[BSidx[:,b],-1]);
        
        if (myclf == 'LinearSVC') or (myclf == 'SVC'):
            # NOTE: predict_proba is used when SVC is a base classifier
            clf_probs[:,i] = clf[i].decision_function(dfDataScaledTest.iloc[:,:-1])
        else:
            clf_probs[:,i] = clf[i].predict_proba(dfDataScaledTest.iloc[:,:-1])[:,1] # note: the sign of predict_proba - 0.5 is enough
    clf_probs_BS[b] = clf_probs


# step 3. Implied probability step 2: the implied probability (newPoint_prob_i_all) will be calculated based on the threshold which produces posterior prob = 0.5
    # example plot
i = -1
i=i+1 # i=i+25
plt.figure(1); plt.clf()
clf_probsBase_i_all = np.zeros((bootstr,1))
newPoint_prob_i_all = np.zeros((bootstr,1))
for b in range(0,bootstr): # range(0,bootstr):  range(bootstr-1,bootstr):
#    if (myclf == 'LinearSVC') or (myclf == 'SVC'):
#        numpos = sum(dfDataScaledTrain.iloc[BSidx[SVs[b],b],-1]) # the last column is the target
#        numneg = len(dfDataScaledTrain.iloc[BSidx[SVs[b],b],-1]) - numpos
#    else:
    # b+=1
    numpos = sum(dfDataScaledTrain.iloc[BSidx[:,b],-1]) # the last column is the target
    numneg = dfDataScaledTrain.shape[0] - numpos
    PPratioORG = numpos/numneg # Pnew(+)/Pnew(-)
    postprob = 1/(1+((thres_neg*numneg)/(thres_pos*numpos))*PPratioORG) # == thres_pos when tp+tn=1 !!
    plt.plot(0.5*np.ones(len(thres_pos)),postprob); plt.xlabel('P(+\\xi) at various thres_pos'); plt.ylabel('Estimated P(+\\xi)'); plt.title('Estimated posterior prob per point i (occurs at the crossing of x=0.5 with curve i, i=1..n)')
    plt.plot(clf_probs_BS[b][i,:]+0.5*int([(myclf == 'LinearSVC') or (myclf == 'SVC')][0]),postprob) # look where this curve crosses the vertical 0.5 line
    plt.scatter(0.5,1-clf_probsBase[b][i]) # plot prediction from the base model
    clf_probsBase_i_all[b] = 1-clf_probsBase[b][i] # save prediction(s) from the base model
    t = clf_probs_BS[b][i,:]+0.5*int([(myclf == 'LinearSVC') or (myclf == 'SVC')][0]) # prediction via implied posterior probability
    if (min(t) > 0.5) or (max(t)<0.5):
        print("warning: 0.5 is not among the t values for i = {0}".format(i))
        newPoint_prob = 0
    else:
        # option 1: use the expectation of "post prob > 0.5" overall possible weights
        # NOTE: this works only if postprob are equally spaced in the interval (0,1)
        tp = 1 - (sum(sum([t>0.5]))*1 + sum(sum([t==0.5]))*0.5 + sum(sum([t<0.5]))*0)/len(t)
        tn = 1 - tp
#        # NOTE: better idea than the code below: use sign change (as the values near prob 0 and 1 may be jumping a lot)
#        #        tp = thres_pos[np.where(np.abs(t-0.5) == np.amin(np.abs(t-0.5)))[0][0]] # note: this may not be crossing point (e.g. [...0.49,0.48,0.53...] will give 0.49 as a result)
#        #        tn = thres_neg[np.where(np.abs(t-0.5) == np.amin(np.abs(t-0.5)))[0][0]]
#        tt = np.where(np.diff(np.sign(t-0.5)) != 0)[0] # biased towards prob==1
#        if len(tt) > 0:
#            tp = 0.5*(thres_pos[tt[0]] + thres_pos[tt[0]+1])
#            tn = 0.5*(thres_neg[tt[0]] + thres_neg[tt[0]+1])
#        else: # legacy...
#            tp = thres_pos[np.where(np.abs(t-0.5) == np.amin(np.abs(t-0.5)))[0][0]] # note: this may not be crossing point (e.g. [...0.49,0.48,0.53...] will give 0.49 as a result)
#            tn = thres_neg[np.where(np.abs(t-0.5) == np.amin(np.abs(t-0.5)))[0][0]]
        PPratio = (tn*numneg)/(tp*numpos) # P(-)/P(+)
        newPoint_prob = 1/(1+PPratio*PPratioORG) # == tp (!!!) Since: tp+tn=1 , 1/(1+((1-k)*numneg / k*numpos)  * numpos/numneg ) = k
    newPoint_prob_i_all[b] = newPoint_prob
plt.figure(2); plt.clf(); plt.scatter(clf_probsBase_i_all,newPoint_prob_i_all); plt.xlabel('clf_probsBase_i_all: ' + myclf); plt.ylabel('Implied posterior probability'); 
plt.plot([0,1],[0,1],color='black',linestyle='dashed'); plt.axis((0,1,0,1));
try:
    clf_probsBase_i_mean = np.mean(clf_probsBase_i_all); plt.figure(); plt.hist(clf_probsBase_i_all); plt.xlim([0,1]); plt.title('Base'); plt.xlabel('Estimated posterior probability')
    clf_probsBase_i_std = np.std(clf_probsBase_i_all) # should be percentile , as we cannot go outside [0,1] # np.percentile(clf_probsBase_i_all,90) , np.percentile(clf_probsBase_i_all,10) 
    impliedPProb_i_mean = np.mean(newPoint_prob_i_all); plt.figure(); plt.hist(newPoint_prob_i_all); plt.xlim([0,1]); plt.title('Implied'); plt.xlabel('Estimated posterior probability')
    impliedPProb_i_std = np.std(newPoint_prob_i_all) # should be percentile , as we cannot go outside [0,1]
except:
    pass
skipme = 0 # plot for a given i the "step 2" surface (the separation line after re-weighting)
if skipme == 0:
    if use_mesh == 1:
        b = 0
        t = clf_probs_BS[b][i,:]+0.5*int([(myclf == 'LinearSVC') or (myclf == 'SVC')][0])
        t_idx_candidate1 = np.where(np.diff(np.sign(t-0.5)) != 0)[0]
        t_idx_candidate2 = t_idx_candidate1 + 1
        if len(t_idx_candidate1) == 1:
            if np.abs(t[t_idx_candidate1] - 0.5) < np.abs(t[t_idx_candidate2] - 0.5):
                t_idx = t_idx_candidate1
            else:
                t_idx = t_idx_candidate2
        else:
            t_idx = np.where(np.abs(t-0.5) == np.amin(np.abs(t-0.5)))[0][0] # legacy. e.g. t_idx = 129 for i=7856
        clf_preds = clf_probs_BS[b][:,t_idx].reshape((ny,nx))
        fig = plt.figure(10); plt.clf();
        ax = fig.gca(projection='3d')
        ax.plot_surface(z1v, z2v, clf_preds, cmap=cm.coolwarm, linewidth=0, antialiased=False,alpha = 0.1)
        ax.contour(z1v, z2v, clf_preds, levels=[0.5-0.5*int([(myclf == 'LinearSVC') or (myclf == 'SVC')][0])],linewidths = [5],colors = ['red'],linestyles =['dotted'])
        ax.plot(x1_pos, x2_pos, 'x',color = 'black')
        ax.plot(x1_neg, x2_neg, 'o',markeredgewidth=1,markeredgecolor='red',markerfacecolor='None' )
        ax.plot([dfDataScaledTest.iloc[i,0]], [dfDataScaledTest.iloc[i,1]], 'x',color = 'blue',markersize=15)
        ax.view_init(90, -90); plt.draw()
        #        # plot the same in 2D:
        #        plt.figure(); plt.clf()
        #        plt.plot(x1_pos, x2_pos, 'x',color = 'black')
        #        plt.plot(x1_neg, x2_neg, 'o',markeredgewidth=1,markeredgecolor='red',markerfacecolor='None' )
        #        plt.axis('equal')
        #        plt.contour(z1v, z2v, clf_preds, levels=[0.5-0.5*int([(myclf == 'LinearSVC') or (myclf == 'SVC')][0])],colors = ['red'],linestyles =['dotted'])
        #        plt.plot(dfDataScaledTest.iloc[i,0], dfDataScaledTest.iloc[i,1], 'x',color = 'blue',markersize=15)
        #        plt.title('Prob(black\current_point) = ' + str(np.round(newPoint_prob,3)))
        #        plt.show()
    elif use_pca == 1: # note: use_mesh == 0
        fig = plt.figure(3); plt.clf(); 
        ax = fig.gca()
        ax.scatter(X[:, 0],X[:, 1],c=y)
        ax.plot([dfDataScaledTest.iloc[i,0]], [dfDataScaledTest.iloc[i,1]], 'x',color = 'red',markersize=20)
        
skipme = 1 # same as above, but for many t_idx values (show 50-50 lines for various P(+) values) 
if skipme == 0: # tested on hastie data
    if use_mesh == 1:
        b = 0
        i_plot = -1 # 3600 # -1
        t_idx_50 = [1*(np.unique(np.sign(clf_probs_BS[b][:,x]-0.5+0.5*int([(myclf == 'LinearSVC') or (myclf == 'SVC')][0]))).shape[0] == 2) for x in np.arange(0,clf_probs_BS[b].shape[1])] # points for which there are predictiions for >50 and < 50 % probability for class A 
        t_idx_50 = pd.Series(t_idx_50); t_idx_50 = np.array(t_idx_50[t_idx_50==1].index.tolist())
        numpos = sum(dfDataScaledTrain.iloc[BSidx[:,b],-1]) # the last column is the target
        numneg = dfDataScaledTrain.shape[0] - numpos
        PPratioORG = numpos/numneg # Pnew(+)/Pnew(-)
        fig = plt.figure(10); plt.clf();
        ax = fig.gca(projection='3d')
        plt.show(block=False)
        ax.view_init(90, -90); plt.draw()
        for t_idx in t_idx_50:
            try:
                clf_preds = clf_probs_BS[b][:,t_idx].reshape((ny,nx))
                tp = thres_pos[t_idx]
                tn = thres_neg[t_idx]
                PPratio = (tn*numneg)/(tp*numpos) # P(-)/P(+)
                newPoint_prob = 1/(1+PPratio*PPratioORG) # == tp (!!!) Since: tp+tn=1 , 1/(1+((1-k)*numneg / k*numpos)  * numpos/numneg ) = k
                plt.clf();
                ax = fig.gca(projection='3d')
                ax.plot_surface(z1v, z2v, clf_preds, cmap=cm.coolwarm, linewidth=0, antialiased=False,alpha = 0.1)
                ax.contour(z1v, z2v, clf_preds, levels=[0.5-0.5*int([(myclf == 'LinearSVC') or (myclf == 'SVC')][0])],linewidths = [5],colors = ['red'],linestyles =['dotted'])
                ax.plot(x1_pos, x2_pos, 'x',color = 'black')
                ax.plot(x1_neg, x2_neg, 'o',markeredgewidth=1,markeredgecolor='red',markerfacecolor='None' )
                if i_plot != -1:
                    ax.plot([dfDataScaledTest.iloc[i_plot,0]], [dfDataScaledTest.iloc[i_plot,1]], 'x',color = 'blue',markersize=15)
                ax.view_init(90, -90)
                plt.title('P(-)/P(+) = ' + str(np.round(PPratio,2)) + ', P(+|redDots) = ' + str(np.round(newPoint_prob,2)))
                plt.draw()
                plt.pause(0.5)
                #input("Press the <ENTER> key to continue...")
            except KeyboardInterrupt:
                break

#skipme = 1
#    PPratioORG = sum(dfDataScaledTrain.iloc[:,-1])/(dfDataScaledTrain.shape[0]-sum(dfDataScaledTrain.iloc[:,-1])) # Pnew(+)/Pnew(-)
#    numpos = sum(dfDataScaledTrain.iloc[:,-1]) # the last column is the target
#    numneg = dfDataScaledTrain.shape[0] - numpos
#    
#    # example plot
#    #i = 0; plt.plot(thres_pos,0.5*np.ones(len(thres_pos))); plt.ylabel('1/(1+(f_i(+)/f_i(-))*(Pdata(-)/Pdata(+))) per point i'); plt.xlabel('P_i(+)'); plt.title('Estimated posterior prob per point i (occurs at the crossing of x=0.5 with curve i, i=1..n)')
#    #PPratios = (thres_neg*numneg)/(thres_pos*numpos)
#    i = 0; plt.plot(np.log(thres_pos*numpos/(thres_neg*numneg)),0.5*np.ones(len(thres_pos))); plt.ylabel('P(+\\xi)'); plt.xlabel('log(P_i(+)/P_i(-))'); plt.title('Estimated posterior prob per point i)')
#    # a = 1/(1+PPratios*PPratioORG) ; plt.scatter(thres_pos,a) # note that mathematically thres_pos == a when tp+tn=1 !
#    plt.scatter(np.log(thres_pos*numpos/(thres_neg*numneg)),clf_probs_BS[0][i,:]);
#    i=i+1
#    i = 0; plt.figure(); plt.plot(thres_pos,0.5*np.ones(len(thres_pos))); plt.ylabel('P(+\\xi)'); plt.xlabel('thres_pos'); plt.title('Estimated posterior prob per point i)')
#    # a = 1/(1+PPratios*PPratioORG) ; plt.scatter(thres_pos,a) # note that mathematically thres_pos == a when tp+tn=1 !
#    plt.scatter(thres_pos,clf_probs_BS[0][i,:]);
#    i=i+1
#    
#    postprob =  1/(1+((thres_neg*numneg)/(thres_pos*numpos))*PPratioORG) # == thres_pos when tp+tn=1 !!
#    # postprob =  1/(1+((thres_neg*numneg)/(thres_pos*numpos))*(thres_pos*PPratioORG)) # may be relevant for xgboost ???
#    #i = 0; plt.plot(thres_pos,0.5*np.ones(len(thres_pos))); plt.ylabel('1/(1+(P_i(-)/P_i(+))*(Pdata(-)/Pdata(+))) per point i'); plt.xlabel('P_i(+)'); plt.title('Estimated posterior prob per point i (occurs at the crossing of x=0.5 with curve i, i=1..n)')
#    i = 0; plt.plot(postprob,0.5*np.ones(len(thres_pos))); plt.xlabel('postprob'); plt.ylabel('P(+\\xi) at various thres_pos'); plt.title('Estimated posterior prob per point i (occurs at the crossing of x=0.5 with curve i, i=1..n)')
#    plt.plot(postprob,clf_probs_BS[0][i,:]); i=i+1
#    # same, but rotated
#    i = 0; plt.clf(); plt.plot(0.5*np.ones(len(thres_pos)),postprob); plt.xlabel('P(+\\xi) at various thres_pos'); plt.ylabel('Estimated P(+\\xi)'); plt.title('Estimated posterior prob per point i (occurs at the crossing of x=0.5 with curve i, i=1..n)')
#    plt.plot(clf_probs_BS[0][i,:],postprob); plt.scatter(0.5,1-clf_probsBase[0][i]); i=i+1;


# Step 3: predict the posterior probability for all new points (newPoint_prob_all)
if (myclf == 'LinearSVC') or (myclf == 'SVC'):
    ref_thres = 0 # decision function "score"
else:
    ref_thres = 0.5 # probability "score"
newPoint_prob_all = np.zeros(dfDataScaledTest.shape[0])
for p in range(len(newPoint_prob_all)):
#    print(p)
    #p = 4; # newpoint index
#    # option 1 - general case : per point
#    t = np.zeros(len(thres_pos))
#    newPoint = dfDataScaledTrain.iloc[p:p+1,:-1]
#    for i in range(len(thres_pos)):
#        t[i] = clf[i].predict_proba(newPoint)[0,1]
    # option 2:
    b = 0
    numpos = sum(dfDataScaledTrain.iloc[BSidx[:,b],-1]) # the last column is the target
    numneg = dfDataScaledTrain.shape[0] - numpos
    PPratioORG = numpos/numneg
    t = clf_probs_BS[b][p,:]
    if max(t)<0.5:
        print("warning: 0.5 is not among the t values for i = {0}".format(p))
        newPoint_prob = 0
    elif min(t) > 0.5:
        print("warning: 0.5 is not among the t values for i = {0}".format(p))
        newPoint_prob = 1
    else:
#        # option 1: use the expectation of "post prob > 0.5" overall possible weights
        tp = 1 - (sum(sum([t>0.5]))*1 + sum(sum([t==0.5]))*0.5 + sum(sum([t<0.5]))*0)/len(t)
        tn = 1 - tp
#        # find prob, which is closest to 0.5
#    #            # heuristic: if two or more predictions are equally close to 0.5, then take the average of the min and max probability estimate
#    #        tp_left = thres_pos[np.where(np.abs(t-0.5) == np.amin(np.abs(t-0.5)))[0][0]] # Get the indices of minimum element in numpy array
#    #        tp_right = thres_pos[np.where(np.abs(t-0.5) == np.amin(np.abs(t-0.5)))[0][-1]] # Get the indices of minimum element in numpy array
#    #        tp = (tp_left + tp_right)/2
#        tt = np.where(np.diff(np.sign(t-ref_thres)) != 0)[0] # biased towards prob==1
#        if len(tt) > 0:
#            tp = 0.5*(thres_pos[tt[0]] + thres_pos[tt[0]+1])
#            tn = 0.5*(thres_neg[tt[0]] + thres_neg[tt[0]+1])
#        else: # legacy...
#            tp = thres_pos[np.where(np.abs(t-ref_thres) == np.amin(np.abs(t-ref_thres)))[0][0]]
#            tn = thres_neg[np.where(np.abs(t-ref_thres) == np.amin(np.abs(t-ref_thres)))[0][0]]
        PPratio = (tn*numneg)/(tp*numpos) # P(-)/P(+)
        newPoint_prob = 1 - 1/(1+PPratio*PPratioORG) # == tn (!!!) Since: tp+tn=1 , 1/(1+((1-k)*numneg / k*numpos)  * numpos/numpeg ) = k
        #print(newPoint_prob)
    newPoint_prob_all[p] = newPoint_prob
plt.figure(); plt.hist(newPoint_prob_all,bins=10); plt.title('Distribution of point according to postptob - implied'); plt.xlim([0, 1])
plt.figure(); plt.hist(clf_probsBase[0],bins=10); plt.title('Distribution of point according to postptob - BASE'); plt.xlim([0, 1])


# compute the test error:
if (data_opt == 3) and (use_hastie == 1):
    pred = np.sign(newPoint_prob_all-0.5)
    #test_error = sum(marginal_hastie*(postprob_hastie*I(pred <0)+(1-postprob_hastie)*I(pred>=0)))
    test_error_hastie = sum((marginal_hastie*(postprob_hastie*[pred==-1]+(1-postprob_hastie)*[pred!=-1])).T)[0]
    pred = np.sign(postprob_hastie-0.5)
    test_error_hastie_Bayes = sum((marginal_hastie*(postprob_hastie*[pred==-1]+(1-postprob_hastie)*[pred!=-1])).T)[0]
    del pred

    # Model error per posterior-probability bin
    wrk_test_error = np.array([])
    for l in np.arange(0,1,0.05):
        r = l + 0.05
        wrk_true = postprob_hastie[(postprob_hastie>l) & (postprob_hastie<=r)]
        wrk_pred = np.sign(newPoint_prob_all[(postprob_hastie>l) & (postprob_hastie<=r)]-0.5)
        wrk_marginal = marginal_hastie[(postprob_hastie>l) & (postprob_hastie<=r)]
        wrk_test_error = np.append(wrk_test_error,sum((wrk_marginal*(wrk_true*[wrk_pred==-1]+(1-wrk_true)*[wrk_pred!=-1])).T)[0])
    wrk_test_cumerror = np.cumsum(wrk_test_error)
    # note: wrk_test_cumerror[-1] ==  the overall Model error
    
    # Bayes error per posterior-probability bin
    wrk_test_error_Bayes = np.array([])
    for l in np.arange(0,1,0.05):
        r = l + 0.05
        wrk_true = postprob_hastie[(postprob_hastie>l) & (postprob_hastie<=r)]
        wrk_pred = np.sign(postprob_hastie[(postprob_hastie>l) & (postprob_hastie<=r)]-0.5)
        wrk_marginal = marginal_hastie[(postprob_hastie>l) & (postprob_hastie<=r)]
        wrk_test_error_Bayes = np.append(wrk_test_error_Bayes,sum((wrk_marginal*(wrk_true*[wrk_pred==-1]+(1-wrk_true)*[wrk_pred!=-1])).T)[0])
    wrk_test_cumerror_Bayes = np.cumsum(wrk_test_error_Bayes)
    # note: wrk_test_cumerror_Bayes[-1] ==  the overall Bayes error
    
#    plt.figure()
#    plt.plot(np.arange(0,1,0.05),wrk_test_cumerror/wrk_test_cumerror[-1]) # model error, scaled
#    plt.plot(np.arange(0,1,0.05),wrk_test_cumerror_Bayes/wrk_test_cumerror_Bayes[-1]) # Bayes error, scaled
    plt.figure()
    plt.plot(np.arange(0,1,0.05),wrk_test_cumerror) # model error per poterior-prob bin, cimulative
    plt.plot(np.arange(0,1,0.05),wrk_test_cumerror_Bayes) # Bayes error per poterior-prob bin, cimulative
    
    plt.figure()
    plt.scatter(wrk_test_cumerror_Bayes/wrk_test_cumerror_Bayes[-1],wrk_test_cumerror/wrk_test_cumerror[-1])
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Cumulative contribution to total error per bin - Model')
    plt.xlabel('Cumulative contribution to total error per bin - Bayes classifier')    
    
    plt.figure()
    plt.scatter(np.arange(0,1,0.05)+0.05/2,wrk_test_error)
    plt.scatter(np.arange(0,1,0.05)+0.05/2,wrk_test_error_Bayes)
    plt.xlabel('Bins'); plt.ylabel('Expected test error within (poterior-probability) bin'); plt.grid(); 
    plt.gca().legend(('Model error','Expected (Bayes) error'))
    
    plt.figure()
    plt.scatter(wrk_test_error_Bayes,wrk_test_error)
    plt.plot(np.array([0,max(wrk_test_error_Bayes)]),np.array([0,max(wrk_test_error)]))
    plt.xlabel('Bayes error per (poterior-probability) bin'); plt.ylabel('Model error per (poterior-probability) bin'); plt.grid()
    plt.axes().set_aspect('equal', 'box')
    
    # draw reliability diagram (see "D:\mypapers\PostProb\refs\10.1.1.13.7457.pdf")
    wrk_pred = np.array([])
    wrk_true = np.array([])
    for l in np.arange(0,1,0.1):
        r = l + 0.1
        #wrk_pred = np.append(wrk_pred,np.mean(newPoint_prob_all[(newPoint_prob_all>l) & (newPoint_prob_all<=r)]))
        wrk_pred = np.append(wrk_pred,(l+r)/2)
        wrk_true = np.append(wrk_true,np.mean(postprob_hastie[(newPoint_prob_all>l) & (newPoint_prob_all<=r)]))
    plt.figure()
    plt.scatter(wrk_pred,wrk_true)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Implied posterior probability (binned)')
    plt.ylabel('Expected class membership probability')
    plt.title('UNWEIGHTED by construction')
    
    # reliability diagrams from raw scores (hastie)
    clf_probsBase_scaled = (clf_probsBase[0] - np.min(clf_probsBase[0]))/(np.max(clf_probsBase[0]) - np.min(clf_probsBase[0]))
    wrk_pred = np.array([])
    wrk_true = np.array([])
    for l in np.arange(0,1,0.1):
        r = l + 0.1
        wrk_pred = np.append(wrk_pred,(l+r)/2)
        wrk_true = np.append(wrk_true,np.mean(postprob_hastie[(clf_probsBase_scaled>l) & (clf_probsBase_scaled<=r)]))
    plt.figure()
    plt.scatter(wrk_pred,wrk_true)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Base classifier scores (binned)')
    plt.ylabel('Expected class membership probability')
    plt.title('UNWEIGHTED by construction')

# ISOTONIC - calibration #
# reliability diagrams from raw scores (Base classifier)  (NON-hastie)
    # clf_probsBase[0] = clf_SVC[0] # in order to use raw scores for SVC
### clf_probsBase_scaled = (clf_probsBase[0] - np.min(clf_probsBase[0]))/(np.max(clf_probsBase[0]) - np.min(clf_probsBase[0]))
if myclf == 'SVC':
    clf_probsBase_scaled = (clf_SVC[0] - np.min(clf_SVC[0]))/(np.max(clf_SVC[0]) - np.min(clf_SVC[0]))
else:
    clf_probsBase_scaled = clf_probsBase
wrk_pred = np.array([])
wrk_true = np.array([])
bin_size = np.array([])
isotonic_x_BASE = np.array([])
isotonic_y_BASE = np.array([])
for l in np.arange(0,1,0.1):
    r = l + 0.1
    a = dfDataScaledTest.iloc[(clf_probsBase_scaled>=l) & (clf_probsBase_scaled<=r),-1]
    isotonic_x_BASE = np.append(isotonic_x_BASE,clf_probsBase_scaled[(clf_probsBase_scaled>=l) & (clf_probsBase_scaled<=r)] )
    isotonic_y_BASE = np.append(isotonic_y_BASE,np.mean(a)*np.ones((len(a),1)))
    bin_size = np.append(bin_size,len(a))
    wrk_pred = np.append(wrk_pred,(l+r)/2)
    wrk_true = np.append(wrk_true,np.mean(a))
plt.figure(7); plt.clf()
plt.scatter(wrk_pred,wrk_true)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Base classifier scores') # 'Base classifier scores (binned)'
plt.ylabel('Empirical class membership probability')
plt.title('Unequal number of points per bin by construction: base')
# weighted (by bin size) average diff b/n expected and realized posprob over bins
# avg_diff_posprob_BASE = sum(((abs(wrk_pred - np.nan_to_num(wrk_true)))*bin_size)/sum(bin_size)) # bin-based
ir = IsotonicRegression()
#y_ = ir.fit_transform(isotonic_x_BASE, isotonic_y_BASE) # isotonic regression using bins as input
#plt.figure(8); plt.clf()
#plt.scatter(isotonic_x_BASE,isotonic_y_BASE,color='red')
#plt.scatter(isotonic_x_BASE,y_,color='green')
#plt.title('Isotonic regression (green)'); plt.grid(); plt.xlim([0, 1]); plt.ylim([0, 1])
#plt.figure(7); plt.plot(isotonic_x_BASE,y_,color='green'); plt.grid()
###avg_diff_posprob_BASE_iso = sum((abs(isotonic_y_BASE - isotonic_x_BASE )))/dfDataScaledTest.shape[0]
    # base, no prior binning - isotonic regression
y_ = ir.fit_transform(clf_probsBase_scaled, dfDataScaledTest.iloc[:,-1]  ) # isotonic without prior setting of bins
plt.scatter(clf_probsBase_scaled, y_,color='black')
plt.plot(np.sort(clf_probsBase_scaled), np.sort(y_),color='black') # assumption : reordering of clf_probsBase_scaled and y_ is with the same index
avg_diff_posprob_BASE_iso = sum((abs(y_ - clf_probsBase_scaled )))/dfDataScaledTest.shape[0]
    # base - platt isotonic
if myclf == 'SVC':
    clf_probsPlatt_scaled = clf_probsBase[0]
    wrk_pred = np.array([])
    wrk_true = np.array([])
    for l in np.arange(0,1,0.1):
        r = l + 0.1
        a = dfDataScaledTest.iloc[(clf_probsPlatt_scaled>=l) & (clf_probsPlatt_scaled<=r),-1]
        wrk_pred = np.append(wrk_pred,(l+r)/2)
        wrk_true = np.append(wrk_true,np.mean(a))
    y_ = ir.fit_transform(clf_probsPlatt_scaled, dfDataScaledTest.iloc[:,-1]  ) # isotonic without prior setting of bins
    plt.figure(9); plt.clf();
    plt.scatter(wrk_pred,wrk_true)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.scatter(clf_probsPlatt_scaled, y_,color='black')
    plt.plot(np.sort(clf_probsPlatt_scaled), np.sort(y_),color='black') # assumption : reordering of clf_probsPlatt_scaled and y_ is with the same index
    plt.xlabel('Platt classifier scores') # 'Platt classifier scores (binned)'
    plt.ylabel('Empirical class membership probability')
    plt.title('Unequal number of points per bin by construction: Platt')
    avg_diff_posprob_Platt_iso = sum((abs(y_ - clf_probsPlatt_scaled )))/dfDataScaledTest.shape[0]

#y_bin_BASE = np.array([])
#x_bin = np.arange(0,1,0.1)
#for l in x_bin:
#    r = l + 0.1
#    a = y_[(clf_probsBase_scaled>=l) & (clf_probsBase_scaled<=r)]
#    y_bin_BASE = np.append(y_bin_BASE,a[0])
#y_bin_BASE = np.sort(y_bin_BASE)
#avg_diff_posprob_BASE_iso = sum(((abs(y_bin_BASE - np.nan_to_num(wrk_true)))*bin_size)/sum(bin_size))
#plt.figure()
#plt.scatter(x_bin,wrk_true,color='red')
#plt.plot(x_bin,y_bin_BASE,color='green')
#plt.title('Isotonic regression (green)'); plt.grid(); plt.xlim([0, 1]); plt.ylim([0, 1])


# reliability diagrams from IMPLIED probability scores (NON-hastie)
wrk_pred = np.array([])
wrk_true = np.array([])
bin_size = np.array([])
isotonic_x_implied = np.array([])
isotonic_y_implied = np.array([])
for l in np.arange(0,1,0.1):
    r = l + 0.1
    a = dfDataScaledTest.iloc[(newPoint_prob_all>l) & (newPoint_prob_all<=r),-1]
    isotonic_x_implied = np.append(isotonic_x_implied,newPoint_prob_all[(newPoint_prob_all>=l) & (newPoint_prob_all<=r)] )
    isotonic_y_implied = np.append(isotonic_y_implied,np.mean(a)*np.ones((len(a),1)))
    bin_size = np.append(bin_size,len(a))
    wrk_pred = np.append(wrk_pred,(l+r)/2)
    wrk_true = np.append(wrk_true,np.nan_to_num(np.mean(a)))
plt.figure(10); plt.clf()
plt.scatter(wrk_pred[(wrk_true>0) & (wrk_true<1)],wrk_true[(wrk_true>0) & (wrk_true<1)],color = 'green')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Implied posterior probability', fontsize = 22) # Implied posterior probability (binned)
plt.ylabel('Empirical class membership probability', fontsize = 22)
plt.title('Unequal number of points per bin by construction: implied')
# weighted (by bin size) average diff b/n expected and realized posprob over bins
# avg_diff_posprob_implied = sum(((abs(wrk_pred - np.nan_to_num(wrk_true)))*bin_size)/sum(bin_size)) # bin-based
ir = IsotonicRegression()
#y_ = ir.fit_transform(isotonic_x_implied, isotonic_y_implied)
####y_ = ir.fit_transform(wrk_pred, np.nan_to_num(wrk_true))
#plt.figure(8); plt.clf()
#plt.scatter(isotonic_x_implied,isotonic_y_implied,color='red')
#plt.scatter(isotonic_x_implied,y_,color='green')
#plt.plot(isotonic_x_implied,y_,color='green')
#plt.title('Isotonic regression (green)'); plt.grid(); plt.xlim([0, 1]); plt.ylim([0, 1])
#plt.figure(7); plt.plot(isotonic_x_implied,y_,color='green'); plt.grid()
    # implied, no prior binning - isotonic regression
y_ = ir.fit_transform(newPoint_prob_all, dfDataScaledTest.iloc[:,-1]  ) # isotonic without prior setting of bins
plt.scatter(newPoint_prob_all, y_,color='black')
plt.plot(np.sort(newPoint_prob_all), np.sort(y_),color='black') # assumption : reordering of clf_probsPlatt_scaled and y_ is with the same index
plt.scatter(newPoint_prob_all, dfDataScaledTest.iloc[:,-1] + np.random.rand(500)*0.2 - 0.1,marker='x',color='blue'); plt.ylim([-0.25, 1.25])
avg_diff_posprob_implied_iso = sum((abs(y_ - newPoint_prob_all )))/dfDataScaledTest.shape[0]
#y_bin_implied = np.array([])
#x_bin = np.arange(0,1,0.1)
#for l in x_bin:
#    r = l + 0.1
#    a = y_[(clf_probsBase_scaled>=l) & (clf_probsBase_scaled<=r)]
#    y_bin_implied = np.append(y_bin_implied,np.mean(a))
#y_bin_implied = np.sort(y_bin_implied)
#avg_diff_posprob_implied_iso = sum(((abs(y_bin_implied - np.nan_to_num(wrk_true)))*bin_size)/sum(bin_size))
#plt.figure()
#plt.scatter(x_bin,wrk_true,color='red')
#plt.plot(x_bin,y_bin_implied,color='green')
#plt.title('Isotonic regression (green)'); plt.grid(); plt.xlim([0, 1]); plt.ylim([0, 1])


# compare BASE to implied
###avg_diff_posprob_implied - avg_diff_posprob_BASE # should be LESS THAN zero for implied to be better
###avg_diff_posprob_implied_iso - avg_diff_posprob_BASE_iso  # should be LESS THAN zero for implied to be better
if myclf == 'SVC':
    avg_diff_posprob_BASE_iso - avg_diff_posprob_Platt_iso # if >0, then BASE is worse than Platt
    avg_diff_posprob_Platt_iso - avg_diff_posprob_implied_iso # if >0, then Platt is worse than implied
avg_diff_posprob_BASE_iso - avg_diff_posprob_implied_iso # if >0, then BASE is worse than implied

plt.figure()
if myclf != 'LinearSVC':
    plt.scatter(newPoint_prob_all,clf_probsBase[0]); plt.xlabel('Implied posterior probability'); plt.ylabel('clf_probsBase ' + myclf); plt.grid(); # plt.xlabel('newPoint_prob_all');
    if myclf != 'SVC':
        plt.axis((0,1,0,1))
else:
    plt.scatter(newPoint_prob_all,clf_probsBase[0]); plt.xlabel('Implied posterior probability'); plt.ylabel('raw_SVM_score'); plt.grid() # plt.xlabel('newPoint_prob_all');
#plt.scatter(newPoint_prob_all[0:210],clf_probsBase[0:210])

if myclf == 'SVC':
    plt.figure()
    plt.scatter(newPoint_prob_all,clf_SVC[0]); plt.xlabel('Implied posterior probability'); plt.ylabel('raw_SVM_score'); plt.grid(); plt.xlim([0, 1]) # plt.xlabel('newPoint_prob_all');
    plt.figure()
    plt.scatter(clf_probsBase[0],clf_SVC[0]); plt.xlabel('Platt probability'); plt.ylabel('raw_SVM_score'); plt.grid(); plt.xlim([0, 1])

# ROC
if len(np.unique(dfDataScaledTest.iloc[:,-1].values))>1:
    plt.figure()
    # ROC curve for clf_probsBase  (use xgboost original scores):
    fpr, tpr, threshold = metrics.roc_curve(dfDataScaledTest.iloc[:,-1].values, clf_probsBase[0])
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('ROC curve')
    plt.plot(fpr, tpr, 'b', label = '(base) AUC = %0.2f' % roc_auc)
    plt.axes().set_aspect('equal', 'box')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    # ROC curve for newPoint_prob_all (use estimated implied probability - new method)::
    fpr, tpr, threshold = metrics.roc_curve(dfDataScaledTest.iloc[:,-1].values, newPoint_prob_all)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, 'k--', label = '(implied) AUC = %0.2f' % roc_auc)
    plt.axes().set_aspect('equal', 'box')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# in case the dfDataScaledTest is 2D
if dfDataScaledTest.shape[1] == 3:
    plt.figure()
    plt.plot(x1_pos, x2_pos, 'x',color = 'black')
    plt.plot(x1_neg, x2_neg, 'o',markeredgewidth=1,markeredgecolor='red',markerfacecolor='None' )
    plt.axis('equal')
    plt.show()
    clf_preds = clf_probsBase[0].reshape((ny,nx))
    if myclf != 'LinearSVC':
        # plot isoline of the predictions - base
            # plt.contour(z1v, z2v, clf_preds, levels=[0.9],colors = ['red'],linestyles =['dotted'])
            # plt.plot(dfDataScaledTest.iloc[i,0], dfDataScaledTest.iloc[i,1], 'd',color = 'blue')
        plt.contour(z1v, z2v, clf_preds, levels=[0.5],colors = ['red'],linestyles =['dotted'])
        plt.contour(z1v, z2v, clf_preds, levels=[0.7],colors = ['red'])
        plt.contour(z1v, z2v, clf_preds, levels=[0.95],colors = ['red'],linewidths = [3])
    else:
        plt.contour(z1v, z2v, clf_preds, levels=[0],colors = ['red'],linestyles =['dotted'])        
    #
    clf_preds = newPoint_prob_all.reshape((ny,nx))
    # plot isoline of the predictions - implied posterior probability
    plt.contour(z1v, z2v, clf_preds, levels=[0.5],colors = ['blue'],linestyles =['dotted'])
    plt.contour(z1v, z2v, clf_preds, levels=[0.7],colors = ['blue'])
    plt.contour(z1v, z2v, clf_preds, levels=[0.95],colors = ['blue'],linewidths = [3])

# note: choose manually whether clf_preds are from clf_probsBase or newPoint_prob_all
    clf_preds_base = clf_probsBase[0].reshape((ny,nx))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(z1v, z2v, clf_preds_base, cmap=cm.coolwarm, linewidth=0, antialiased=False,alpha = 0.1)
    ax.contour(z1v, z2v, clf_preds_base, levels=[0.5 - 0.5*int([myclf == 'LinearSVC'][0])],linewidths = [5],colors = ['green'],linestyles =['dotted'])
    ax.plot(x1_pos, x2_pos, 'x',color = 'black')
    ax.plot(x1_neg, x2_neg, 'o',markeredgewidth=1,markeredgecolor='red',markerfacecolor='None' )
    plt.title('Posterior probablity base classifier: ' + myclf) # clf_probsBase
    #
    clf_preds = newPoint_prob_all.reshape((ny,nx))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(z1v, z2v, clf_preds, cmap=cm.coolwarm, linewidth=0, antialiased=False,alpha = 0.1)
    ax.contour(z1v, z2v, clf_preds, levels=[0.5],linewidths = [5],colors = ['green'],linestyles =['dotted'])
    ax.plot(x1_pos, x2_pos, 'x',color = 'black')
    ax.plot(x1_neg, x2_neg, 'o',markeredgewidth=1,markeredgecolor='red',markerfacecolor='None' )
    plt.title('Implied posterior probability: ' + myclf) # plt.title('Posterior probablity surface using newPoint_prob_all')

# plot in 3D the posterior probability for class 1:
    if (data_opt == 3) and (use_hastie == 0):
        from scipy.stats import multivariate_normal
        Xdata = np.empty(z1v.shape + (2,))
        Xdata[:, :, 0] = z1v; Xdata[:, :, 1] = z2v
        rv_pos = multivariate_normal(mean=mean_pos, cov=cov_pos)
        rv_neg = multivariate_normal(mean=mean_neg, cov=cov_neg);
        postprob_cl1 = (rv_neg.pdf(Xdata)*(num_neg/(num_pos+num_pos)))/(rv_pos.pdf(Xdata)*(num_pos/(num_pos+num_pos)) + rv_neg.pdf(Xdata)*(num_neg/(num_pos+num_pos)))
        #plt.figure(); plt.contourf(z1v, z2v, postprob_cl1)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(z1v, z2v, postprob_cl1, cmap=cm.coolwarm, linewidth=0, antialiased=False,alpha = 0.1)
        ax.contour(z1v, z2v, postprob_cl1, levels=[0.5],linewidths = [5],colors = ['green'],linestyles =['dotted'])
        ax.plot(x1_pos, x2_pos, 'x',color = 'black')
        ax.plot(x1_neg, x2_neg, 'o',markeredgewidth=1,markeredgecolor='red',markerfacecolor='None' )
        plt.title('True posterior probability') # plt.title('Posterior probablity surface using newPoint_prob_all')
        # plt.contour(z1v, z2v, postprob_cl1, levels=[0.5],colors = ['black'],linestyles =['dotted'])
    elif (data_opt == 3) and (use_hastie == 1):
        postprob_cl1 = postprob_hastie.reshape((z1v.shape[0], z1v.shape[1]))
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(z1v, z2v, postprob_cl1, cmap=cm.coolwarm, linewidth=0, antialiased=False,alpha = 0.1)        
        ax.contour(z1v, z2v, postprob_cl1, levels=[0.5],linewidths = [5],colors = ['green'],linestyles =['dotted'])
        ax.plot(x1_pos, x2_pos, 'x',color = 'black')
        ax.plot(x1_neg, x2_neg, 'o',markeredgewidth=1,markeredgecolor='red',markerfacecolor='None' )
        plt.title('True posterior probability') # plt.title('Posterior probablity surface using newPoint_prob_all')
        ax.view_init(90, -90); plt.draw()

#        fig = plt.figure()
#        ax = fig.gca(projection='3d')
#        marginal_mesh = marginal_hastie.reshape((ny,nx))
#        ax.plot_surface(z1v, z2v, (postprob_cl1-clf_preds)*marginal_mesh, cmap=cm.coolwarm, linewidth=0, antialiased=False,alpha = 0.1)        
#        plt.title('True minus Model posterior probability') # plt.title('Posterior probablity surface using newPoint_prob_all')
#        ax.view_init(90, -90); plt.draw()

# for SVC, compute implied posterior probability assuming the heuristic that the separation line will move parallelly with change in P(+)/P(-)
#skipme = 1
#if (skipme == 0) and (myclf == 'SVC'):
    # 1. compute the current sum of errors on the training set for SVs only (+ve and -ve instances separately)
    # 2. as we move the sep line parallelly, we assume that this is equivalent to building a new model, which results
    #    in this sep line. Tne change in P(+) (and P(-)) is linearly proportional to the change in C*sum_errors_positive (C*sum_errors_negative)
    # eg: For the base solution we have C*sum_errors_pos = 23.5, C*sum_errors_neg = 14.2, (implying P(+)/P(-) = 23.5/14.2)
    #   when we move the sep line (parallelly, keeing ONLY the original SVs in the set), then
    #   C*sum_errors_pos = 30 and C*sum_errors_neg = 8 (implying P(+)/P(-) = 30/8)
    #   ... dead end: at some point C*sum_errors_neg = 0. Moreover the "8" maybe not in line, as we can compute it (due to lower degress of freedom imposed by w1+w2 = 1)



### for the SVM tutorial paper
    # load spydata: "D:\mypapers\PostProb\SVMtutorial\data\2d_toy_data.spydata" (2D, n+=n-=10)
# We consider LinearSVM here
# from matplotlib import rc
rc('text', usetex=True) # use LaTex interpreter
i = 2940 # i=340 is an example of degeneracy, i==2940 is an example non-degeneracy
numpos = sum(dfDataScaledTrain.iloc[BSidx[:,b],-1]) # the last column is the target
numneg = dfDataScaledTrain.shape[0] - numpos
PPratioORG = numpos/numneg # Pnew(+)/Pnew(-)
postprob = 1/(1+((thres_neg*numneg)/(thres_pos*numpos))*PPratioORG) # == thres_pos when tp+tn=1 !!
t = clf_probs_BS[b][i,:]+0*0.5*int([(myclf == 'LinearSVC') or (myclf == 'SVC')][0]) # prediction via implied posterior probability
plt.figure();plt.scatter(postprob, t, color = 'black') # for each hyperplane among the hyperplanes associated wit
plt.title('Raw classification scores from 201 hyperplanes')
plt.xlabel('Levels of posterior probability estimates $\widehat{P(+|x)}$ for 201 hyperplanes', fontsize=18); 
plt.ylabel('Raw SVM scores', fontsize=18); plt.grid()
plt.figure();plt.scatter(postprob, np.sign(t), color = 'black') # for each hyperplane among the hyperplanes associated wit
plt.title('Classification label $\in {-1,+1}$')
plt.xlabel('Levels of posterior probability estimates $\widehat{P(+|x)}$ for 201 hyperplanes', fontsize=18); 
plt.ylabel('Predicted class label from 201 hyperplanes', fontsize=18); plt.grid()
# estimated posterior probabilities 0.1..0.9, there is an SVM line, which classifies the current point in some way (>0 or < 0)
# the estimated posterior porbability is the average of the sign of those predictions:
newPoint_prob_i_all[b] = sum(clf_probs_BS[b][i,:] > 0)/len(clf_probs_BS[b][i,:]) # in the LinearSVC and SVC cases the condition is ">0" rather than ">0.5"
if use_mesh == 1:
    b = 0
    i_plot = i # 3600 # -1
    t_idx_50 = [1*(np.unique(np.sign(clf_probs_BS[b][:,x]-0.5+0.5*int([(myclf == 'LinearSVC') or (myclf == 'SVC')][0]))).shape[0] == 2) for x in np.arange(0,clf_probs_BS[b].shape[1])] # points for which there are predictiions for >50 and < 50 % probability for class A 
    t_idx_50 = pd.Series(t_idx_50); t_idx_50 = np.array(t_idx_50[t_idx_50==1].index.tolist())
    numpos = sum(dfDataScaledTrain.iloc[BSidx[:,b],-1]) # the last column is the target
    numneg = dfDataScaledTrain.shape[0] - numpos
    PPratioORG = numpos/numneg # Pnew(+)/Pnew(-)
    fig = plt.figure(10); plt.clf();
    plt.clf();
    plt.plot(x1_pos, x2_pos, '+',color = 'black',markersize=10)
    plt.plot(x1_neg, x2_neg, 'o',markeredgewidth=1,markeredgecolor='black',markerfacecolor='None',markersize=10)
    for t_idx in t_idx_50:
        try:
            # ii = -1
            # ii +=1; t_idx = t_idx_50[ii]
            clf_preds = clf_probs_BS[b][:,t_idx].reshape((ny,nx))
            tp = thres_pos[t_idx]
            tn = thres_neg[t_idx]
            PPratio = (tn*numneg)/(tp*numpos) # P(-)/P(+)
            newPoint_prob = 1/(1+PPratio*PPratioORG) # == tp (!!!) Since: tp+tn=1 , 1/(1+((1-k)*numneg / k*numpos)  * numpos/numneg ) = k
            if i_plot > 0:
                plt.plot([dfDataScaledTest.iloc[i_plot,0]], [dfDataScaledTest.iloc[i_plot,1]], 'x',color = 'black',markersize=20)
#                plt.xlim([-3, 10.8])
#                plt.ylim([-1.6, 9.7])
            plt.axis('equal')
            plt.show()
            clf_preds = clf_probs_BS[b][:,t_idx].reshape((ny,nx))
            if myclf != 'LinearSVC':
                # plot isoline of the predictions - base
                    # plt.contour(z1v, z2v, clf_preds, levels=[0.9],colors = ['red'],linestyles =['dotted'])
                    # plt.plot(dfDataScaledTest.iloc[i,0], dfDataScaledTest.iloc[i,1], 'd',color = 'blue')
                plt.contour(z1v, z2v, clf_preds, levels=[0.5],colors = ['red'],linestyles =['dotted'])
                plt.contour(z1v, z2v, clf_preds, levels=[0.7],colors = ['red'])
                plt.contour(z1v, z2v, clf_preds, levels=[0.95],colors = ['red'],linewidths = [3])
            else:
                plt.contour(z1v, z2v, clf_preds, levels=[0],colors = ['black'],linestyles =['dotted'])        
            plt.title('P(-)/P(+) = ' + str(np.round(PPratio,2)) + ', $\widehat{P(+|dots)}$ = ' + str(np.round(newPoint_prob,2)))
            plt.draw()
            plt.pause(1)
            #input("Press the <ENTER> key to continue...")
        except KeyboardInterrupt:
            break
rc('text', usetex=False) # use LaTex interpreter






































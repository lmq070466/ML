# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 21:13:14 2019

@author: limingqi
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
from sklearn import model_selection
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score




d = pd.read_excel("D:\\limingqi20190902\\imblean_data.xlsx",encoding='utf-8')
#print(d.shape)
'''
colormap = plt.cm.viridis
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(d.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
plt.show()
'''

#d.drop(['AGE'],inplace=True)
df = copy.deepcopy(d)
df.drop(['label','AGE'],axis=1,inplace=True)
d.drop(['AGE'],axis=1,inplace=True)
col=df.columns
#print(col)
X =d.iloc[:,:-1]
Y = d.iloc[:,-1]
std = StandardScaler()
X = std.fit_transform(X)

def get_top_n_features(X, Y, top_n_features,col):
    # randomforest
    rf_est = RandomForestClassifier(random_state=0)
    rf_param_grid = {'n_estimators': [500], 'min_samples_split': [2, 3], 'max_depth': [20]}
    rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs=25, cv=10, verbose=1,scoring="recall")
    rf_grid.fit(X, Y)
    print('Top N Features Best RF Params:' + str(rf_grid.best_params_))
    print('Top N Features Best RF Score:' + str(rf_grid.best_score_))
    print('Top N Features RF Train Score:' + str(rf_grid.score(X, Y)))
    feature_imp_sorted_rf = pd.DataFrame({'feature': col,
                                          'importance': rf_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    
    rf_1=feature_imp_sorted_rf[:10]
    rf_2= 100*feature_imp_sorted_rf[:10]['importance']
    #print(rf_2)
    features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']
    print('Sample 10 Feeatures from RF Classifier')
    print(str(features_top_n_rf[:10]))
    '''
    pos = np.arange(rf_2.shape[0]) + 0.5

    plt.figure(1, figsize = (18, 8))

    plt.subplot(121)
    plt.barh(pos, rf_1['importance'][::-1])
    plt.yticks(pos, rf_1['feature'][::-1])
    plt.xlabel('Relative Importance')
    plt.title('RandomForest Feature Importance')
    '''


    # AdaBoost
    ada_est = AdaBoostClassifier(random_state=0)
    ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1]}
    ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs=25, cv=10, verbose=1,scoring="recall")
    ada_grid.fit(X, Y)
    print('Top N Features Best Ada Params:' + str(ada_grid.best_params_))
    print('Top N Features Best Ada Score:' + str(ada_grid.best_score_))
    print('Top N Features Ada Train Score:' + str(ada_grid.score(X, Y)))
    feature_imp_sorted_ada = pd.DataFrame({'feature': col,
                                           'importance': ada_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    '''
    ada_1=feature_imp_sorted_ada[:10]
    ada_2= 100*feature_imp_sorted_ada[:10]['importance']
    #plt.figure(1, figsize = (18, 8))
    plt.subplot(122)
    plt.barh(pos, ada_1['importance'][::-1])
    plt.yticks(pos, ada_1['feature'][::-1])
    plt.xlabel('Relative Importance')
    plt.title('Adaboost Feature Importance')
    plt.show()'''
    features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']
    print('Sample 10 Features from Ada Classifier:')
    print(str(features_top_n_ada[:10]))

   
    # ExtraTree
    et_est = ExtraTreesClassifier(random_state=0)
    et_param_grid = {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [20]}
    et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=25, cv=10, verbose=1,scoring="recall")
    et_grid.fit(X, Y)
    print('Top N Features Best ET Params:' + str(et_grid.best_params_))
    print('Top N Features Best DT Score:' + str(et_grid.best_score_))
    print('Top N Features ET Train Score:' + str(et_grid.score(X, Y)))
    feature_imp_sorted_et = pd.DataFrame({'feature': col,
                                          'importance': et_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    '''
    et_1=feature_imp_sorted_et[:10]
    et_2= 100*feature_imp_sorted_et[:10]['importance']
    
    plt.figure(1, figsize = (18, 8))
    pos = np.arange(et_2.shape[0]) + 0.5
    plt.subplot(121)
    plt.barh(pos, et_1['importance'][::-1])
    plt.yticks(pos, et_1['feature'][::-1])
    plt.xlabel('Relative Importance')
    plt.title('ExtraTrees Feature Importance')
    '''
    features_top_n_et = feature_imp_sorted_et.head(top_n_features)['feature']
    print('Sample 10 Features from ET Classifier:')
    print(str(features_top_n_et[:10]))

    # GradientBoosting
    gb_est = GradientBoostingClassifier(random_state=0)
    gb_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1], 'max_depth': [20]}
    gb_grid = model_selection.GridSearchCV(gb_est, gb_param_grid, n_jobs=25, cv=10, verbose=1,scoring="recall")
    gb_grid.fit(X, Y)
    print('Top N Features Best GB Params:' + str(gb_grid.best_params_))
    print('Top N Features Best GB Score:' + str(gb_grid.best_score_))
    print('Top N Features GB Train Score:' + str(gb_grid.score(X, Y)))
    feature_imp_sorted_gb = pd.DataFrame({'feature': col,
                                          'importance': gb_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    '''
    gb_1=feature_imp_sorted_gb[:10]
    gb_2= 100*feature_imp_sorted_gb[:10]['importance']
    
    #plt.figure(1, figsize = (18, 8))
    
    plt.subplot(122)
    pos = np.arange(gb_2.shape[0]) + 0.5
    plt.barh(pos, et_1['importance'][::-1])
    plt.yticks(pos, et_1['feature'][::-1])
    plt.xlabel('Relative Importance')
    plt.title('GradientBoosting Feature Importance')
    plt.show()'''
    features_top_n_gb = feature_imp_sorted_gb.head(top_n_features)['feature']
    print('Sample 10 Feature from GB Classifier:')
    print(str(features_top_n_gb[:10]))

    # DecisionTree
    dt_est = DecisionTreeClassifier(random_state=0)
    dt_param_grid = {'min_samples_split': [2, 4], 'max_depth': [20]}
    dt_grid = model_selection.GridSearchCV(dt_est, dt_param_grid, n_jobs=25, cv=10, verbose=1,scoring="recall")
    dt_grid.fit(X,Y)
    print('Top N Features Bset DT Params:' + str(dt_grid.best_params_))
    print('Top N Features Best DT Score:' + str(dt_grid.best_score_))
    print('Top N Features DT Train Score:' + str(dt_grid.score(X, Y)))
    feature_imp_sorted_dt = pd.DataFrame({'feature': col,
                                          'importance': dt_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
     
    dt_1=feature_imp_sorted_gb[:10]
    dt_2= 100*feature_imp_sorted_gb[:10]['importance']
    
    plt.figure(1, figsize = (18, 8))
    pos = np.arange(dt_2.shape[0]) + 0.5
    #plt.subplot(121)
    plt.barh(pos, dt_1['importance'][::-1])
    plt.yticks(pos, dt_1['feature'][::-1])
    plt.xlabel('Relative Importance')
    plt.title('DecisionTree Feature Importance')
    features_top_n_dt = feature_imp_sorted_dt.head(top_n_features)['feature']
    print('Sample 10 Features from DT Classifier:')
    print(str(features_top_n_dt[:10]))

    # merge the three models
    features_top_n = pd.concat(
        [features_top_n_rf, features_top_n_ada, features_top_n_et, features_top_n_gb, features_top_n_dt],
        ignore_index=True).drop_duplicates()
    features_importance = pd.concat([feature_imp_sorted_rf, feature_imp_sorted_ada, feature_imp_sorted_et,
                                     feature_imp_sorted_gb, feature_imp_sorted_dt], ignore_index=True)
           
    plt.show()

    return features_top_n, features_importance

feature_to_pick = 30
feature_top_n,feature_importance = get_top_n_features(X,Y,feature_to_pick,col)
#X_new = pd.DataFrame(X[feature_top_n])
#titanic_test_data_X = pd.DataFrame(X[feature_top_n])
dk=d[feature_top_n]
dc=pd.concat([dk,d['label']],axis=1)
dc.to_excel('D:\\limingqi20190902\\extract_feature.xlsx')
print(feature_top_n)
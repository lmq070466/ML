# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:56:56 2019

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
from sklearn.svm import SVC
from sklearn.exceptions import NotFittedError



d = pd.read_excel("D:\\limingqi20190902\\extract_feature.xlsx",encoding='utf-8')
x,y = d.ix[:,:-1],d.ix[:,-1]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

rf_est = RandomForestClassifier(random_state=0)
rf_param_grid = {'n_estimators': [500], 'min_samples_split': [2,3], 'max_depth': [20]}

rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs=25, cv=10, verbose=1,scoring='recall')
rf_grid.fit(x_train, y_train)
print('Top N Features Best RF Params:' + str(rf_grid.best_params_))
print('Top N Features Best RF Score:' + str(rf_grid.best_score_))
print('Top N Features RF Train Score:' + str(rf_grid.score(x_train, y_train)))
y_true, y_pred = y_test, rf_grid.predict(x_test)
print(classification_report(y_true, y_pred))
    
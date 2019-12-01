# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 14:54:45 2019

@author: 李明琦
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost
from sklearn.metrics import accuracy_score,classification_report
#import sklearn.model_selection as model_select
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection

t1 = time.time()



# ---------------------------------- get data -------------------------------
df_tmp = pd.read_csv("F:\\pythoncode\\analysis_and_prediction_for_diabetes-master\\data\\cleaned_pima_data.csv")
x = df_tmp[["Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction", "Age"]]
y = df_tmp["Outcome"]
 
#x = StandardScaler().fit_transform(x)

# ---- split train test ------
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=7, test_size=0.25)

model = xgboost.XGBClassifier()

#print(classification_report(y_true, y_pred))
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
#y_pred.shape
#y_pred = pd.DataFrame(y_pred)
#predictions = [round(value) for value in y_pred]
# 显示准确率
accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
#print(accuracy)
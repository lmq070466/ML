"""
选择随机森林模型  二元分类
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.ensemble as se
import sklearn.metrics as sm
import sklearn.model_selection as model_select
import numpy as np
import time

t1 = time.time()

pd.set_option("display.max_columns", 1000)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 1000)

# ---------------------------------- get data -------------------------------
df_tmp = pd.read_csv("../data/cleaned_pima_data.csv")
x = df_tmp[["Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction", "Age"]]
y = df_tmp["Outcome"]

# ---- split train test ------
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=7, test_size=0.25)

# ----------------------fit and select hyper-parameter 选择最优超参数训练--------------------------
# ---基于网格搜索，获取最优超参数-----

max_depth_lst = np.arange(x.shape[1] - 1, x.shape[1] + 2)
print(max_depth_lst, "--------")
n_estimators_lst = np.arange(500, 1000, 100)

model = model_select.GridSearchCV(estimator=se.RandomForestClassifier(),
                                  param_grid=[{"max_depth": max_depth_lst,
                                               "n_estimators": n_estimators_lst,
                                               "min_samples_split": [2, 3]}],
                                  scoring="accuracy",
                                  cv=5)

# 训练模型（1.选最优模型   2.使用最优模型训练）
model.fit(x_train, y_train)

# 拿到网格搜索模型训练后的副产品
print(model.best_params_)  # {"max_depth": 6, "min_samples_split": 2, "n_estimators": 800}
print(model.best_score_)  # 0.7743055555555556
print(model.best_estimator_)
"""
RandomForestClassifier(bootstrap=True, class_weight=None, criterion="gini",
                       max_depth=6, max_features="auto", max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=800,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
"""

print("到这里总共执行 %d 秒" % int(time.time() - t1))
# 到这里总共执行 156 秒

# best_pa = model.best_params_
# forest = se.RandomForestClassifier(max_depth=best_pa["max_depth"], n_estimators=best_pa["n_estimators"],
#                                    min_samples_split=best_pa["min_samples_split"])
y_pred = model.predict(x_test)

res = sm.classification_report(y_test, y_pred)
print("-" * 50)
print(res)
"""
              precision    recall  f1-score   support

           0       0.81      0.86      0.84       122
           1       0.73      0.66      0.69        70

    accuracy                           0.79       192
   macro avg       0.77      0.76      0.76       192
weighted avg       0.78      0.79      0.78       192
"""

t2 = time.time()
print("总共执行 %d 秒" % int(t2 - t1))
# 总共执行 156 秒

# ------混淆矩阵-------------
confusion = sm.confusion_matrix(y_test, y_pred)
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
TP = confusion[1, 1]

accuracy = round(sm.accuracy_score(y_test, y_pred) * 100, 5)
print(accuracy, "---accuracy")

recall = round(TP / (FN + TP) * 100, 5)
print(recall, "---recall")

precision = round(TP / (TP + FP) * 100, 5)
print(precision, "---precision")

f1_score = round(2 * precision * recall / (precision + recall), 5)
print(f1_score, "---F1分数")
"""
79.16667 ---accuracy
65.71429 ---recall
74.19355 ---precision
69.69697 ---F1分数
"""

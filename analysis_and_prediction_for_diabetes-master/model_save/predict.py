"""
选择模型预测结果
"""
import pandas as pd
import numpy as np
import sklearn.metrics as skmetric
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

pd.set_option("display.max_columns", 1000)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 1000)

# ---------------------------------- get data -------------------------------
df = pd.read_csv("../data/cleaned_pima_data.csv")
x = df[["Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction", "Age"]]
x.columns = ["Pregnance", "Glucose", "BlPressure", "BMI", "DiabetesFunc", "Age"]
y = df["Outcome"]

# ---- split train test ------
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=7, test_size=0.25)

# ----------------------- 加载-->已经训练好的k邻近算法模型 -----------------------
with open("../model_save/knn.pkl", "rb") as f:
    knn = pickle.load(f)

# -------------------预测-----------------
y_pred = knn.predict(x_test)

# --------获取准确率--------
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
# 0.8020833333333334

# --------获取置信概率--------
prob_y = knn.predict_proba(x_test)

# 测试样本,添加预测的标签,和置信概率
test_df = pd.DataFrame(x_test)
test_df["Outcome"] = y_test  # 实际的结果
test_df["Predict"] = y_pred  # 预测的结果
test_df["Prob_0"] = np.round(prob_y[:, 0], 2)  # 置信概率,预测为0的概率
test_df["Prob_1"] = np.round(prob_y[:, 1], 2)  # 置信概率,预测为1的概率

print("=" * 50)
print(test_df.head())
"""
     Pregnance  Glucose  BlPressure   BMI  DiabetesFunc  Age  Outcome  Predict  Prob_0  Prob_1
353          1     90.0          62  27.2         0.580   24        0        0    1.00    0.00
236          7    181.0          84  35.9         0.586   51        1        1    0.21    0.79
323         13    152.0          90  26.8         0.731   43        1        1    0.36    0.64
98           6     93.0          50  28.7         0.356   23        0        0    1.00    0.00
701          6    125.0          78  27.6         0.565   49        1        0    0.50    0.50
"""
# -----------------------------------预测结果-分类报告--------------------------------
print("=" * 50)

report = skmetric.classification_report(y_test, y_pred)
print(report)
"""
              precision    recall  f1-score   support

           0       0.80      0.93      0.86       122
           1       0.82      0.59      0.68        70

    accuracy                           0.80       192
   macro avg       0.81      0.76      0.77       192
weighted avg       0.80      0.80      0.79       192
"""

# -----------------------------------分析--预测错误的样本--------------------------------
print("=" * 50)

mask_series = y_test != y_pred
check_df = pd.DataFrame(mask_series)

check_df = check_df[check_df["Outcome"] == True]
print("预测错误的样本有%d个" % check_df.shape[0])
# 预测错误的样本有38个
# check_df 数据结构,index是样本行数,val是bool

# 预测错误的样本的行索引
ind_lst = check_df.index.values
print(ind_lst)
"""
[701 242 744 549 608 296 766 493 125 667 179 755  66   9 336 261 387 706
 270 618 419 128 109  25 228  99 153 308  70 328 321 326 485 630 294 470
 683 327]
"""

print("=" * 50)
# 查看预测错误的样本, 预测的结果, 0表示非糖尿病, 1是糖尿病,
# Outcome:实际结果, Predict:预测结果, Prob_0:置信概率为0类别的百分比, Prob_1:置信概率为1类别的百分比
print(test_df.loc[ind_lst].head())
"""
     Pregnance  Glucose  BlPressure   BMI  DiabetesFunc  Age  Outcome  Predict  Prob_0  Prob_1
701          6    125.0          78  27.6         0.565   49        1        0    0.50    0.50
242          3    139.0          54  25.6         0.402   22        1        0    0.71    0.29
744         13    153.0          88  40.6         1.174   39        0        1    0.29    0.71
549          4    189.0         110  28.5         0.680   37        0        1    0.11    0.89
608          0    152.0          82  41.5         0.270   27        0        1    0.36    0.64

例如: 701行的样本,预测为是糖尿病和非糖尿病的概率都是50%, 这个错误的预测是可以接受的范围之内
例如: 549行的样本,预测为是糖尿病和非糖尿病的概率都是11%,89%, 差距太大
"""

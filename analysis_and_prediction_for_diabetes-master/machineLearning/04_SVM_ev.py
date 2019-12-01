"""
选择逻辑分类回归模型  二元分类
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.svm as svm
import sklearn.metrics as sm
import sklearn.model_selection as model_select
import time

pd.set_option("display.max_columns", 1000)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 1000)

t1 = time.time()
# ---------------------------------- get data -------------------------------
df_tmp = pd.read_csv("../data/cleaned_pima_data.csv")
x = df_tmp[["Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction", "Age"]]
y = df_tmp["Outcome"]

# ---- split train test ------
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=7, test_size=0.25)

# ------------------------------  svm分类器  选择最优超参数--------------------------

# 训练svm分类器
model = svm.SVC()
# 基于网格搜索，获取最优模型
params = [
    {"kernel": ["linear"], "C": [1, 10, 100, 1000]},  # 线性核函数
    {"kernel": ["poly"], "degree": [2, 3]},  # 多项式核函数
    {"kernel": ["rbf"], "C": [1, 10, 100, 1000],  # 径向基核函数
     "gamma": [1, 0.1, 0.01, 0.001]}]

model = model_select.GridSearchCV(model, params, cv=5)

# --------训练模型（1.选最优模型   2.使用最优模型训练）--------
model.fit(x_train, y_train)

# --------拿到网格搜索模型训练后的副产品--------
print(model.best_params_)
# {"C": 1, "kernel": "linear"}
print(model.best_score_)  # 是f1 score 分散
# 0.7708333333333334
print(model.best_estimator_)
"""
SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape="ovr", degree=3, gamma="auto_deprecated",
    kernel="linear", max_iter=-1, probability=False, random_state=None,
    shrinking=True, tol=0.001, verbose=False)
"""

# -------- 输出每组超参数组合的交叉验证得分 --------
for param, score in zip(
        model.cv_results_["params"],
        model.cv_results_["mean_test_score"]):
    print(param, "->", score)

# 预测
y_pred = model.predict(x_test)
# 输出分类结果
cr = sm.classification_report(y_test, y_pred)
print(cr)

t2 = time.time()
print("总共执行 %d 秒" % int(t2 - t1))
"""
{"C": 1, "kernel": "linear"} -> 0.7708333333333334
{"C": 10, "kernel": "linear"} -> 0.7690972222222222
{"C": 100, "kernel": "linear"} -> 0.7638888888888888
{"C": 1000, "kernel": "linear"} -> 0.7586805555555556
{"degree": 2, "kernel": "poly"} -> 0.7569444444444444
{"degree": 3, "kernel": "poly"} -> 0.7326388888888888
{"C": 1, "gamma": 1, "kernel": "rbf"} -> 0.65625
{"C": 1, "gamma": 0.1, "kernel": "rbf"} -> 0.6527777777777778
{"C": 1, "gamma": 0.01, "kernel": "rbf"} -> 0.7326388888888888
{"C": 1, "gamma": 0.001, "kernel": "rbf"} -> 0.7621527777777778
{"C": 10, "gamma": 1, "kernel": "rbf"} -> 0.65625
{"C": 10, "gamma": 0.1, "kernel": "rbf"} -> 0.6475694444444444
{"C": 10, "gamma": 0.01, "kernel": "rbf"} -> 0.6753472222222222
{"C": 10, "gamma": 0.001, "kernel": "rbf"} -> 0.7326388888888888
{"C": 100, "gamma": 1, "kernel": "rbf"} -> 0.65625
{"C": 100, "gamma": 0.1, "kernel": "rbf"} -> 0.6475694444444444
{"C": 100, "gamma": 0.01, "kernel": "rbf"} -> 0.6649305555555556
{"C": 100, "gamma": 0.001, "kernel": "rbf"} -> 0.71875
{"C": 1000, "gamma": 1, "kernel": "rbf"} -> 0.65625
{"C": 1000, "gamma": 0.1, "kernel": "rbf"} -> 0.6475694444444444
{"C": 1000, "gamma": 0.01, "kernel": "rbf"} -> 0.6649305555555556
{"C": 1000, "gamma": 0.001, "kernel": "rbf"} -> 0.703125
              precision    recall  f1-score   support

           0       0.77      0.90      0.83       122
           1       0.76      0.53      0.62        70

    accuracy                           0.77       192
   macro avg       0.76      0.72      0.73       192
weighted avg       0.76      0.77      0.75       192

总共执行 3751 秒
"""
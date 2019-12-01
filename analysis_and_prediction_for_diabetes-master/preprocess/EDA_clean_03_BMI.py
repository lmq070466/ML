import pandas as pd
import matplotlib.pyplot as mp
import seaborn as sns
import numpy as np

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

mp.rcParams['font.sans-serif'] = ['SimHei']
mp.rcParams['axes.unicode_minus'] = False

# ---------------------------------- get data -------------------------------
pima_data = pd.read_csv("../data/cleaned_pima_data.csv")

# --------------------------------  BMI shengao --------------------------------

#   ------------------------- 1.数据分布  可视化 -------------------------
print(pd.DataFrame(pima_data["BMI"]).describe())
"""
              BMI
count  768.000000
mean    31.992578
std      7.884160
min      0.000000
25%     27.300000
50%     32.000000
75%     36.600000
max     67.100000
"""

# draw  boxplot -- 图1
mp.figure("BMI数据情况")
mp.title("BMI数据情况")
sns.boxplot(data=pima_data["BMI"])
mp.xlabel("BMI")
mp.ylabel("values")
mp.show()

# draw histogram -- 图2
mp.figure("BMI histogram ", facecolor="lightgray")
mp.title("BMI histogram")
pima_data["BMI"].hist()
mp.xlabel("BMI值")
mp.ylabel("人数")
mp.show()

# ---------  离群点 情况 ---------------
BMI_anomaly = pima_data.loc[(pima_data["BMI"] > 50) | (pima_data["BMI"] == 0)]

print("数量:", BMI_anomaly.shape[0])  # 数量: 19
print("数量:", pima_data.loc[pima_data["BMI"] > 50].shape[0])  # 数量: 8
print("数量:", pima_data.loc[pima_data["BMI"] == 0].shape[0])  # 数量: 11

print("-" * 50)
print(pima_data.loc[pima_data["BMI"] > 50])
print("-" * 50)
print(pima_data.loc[pima_data["BMI"] == 0])

#   ------------------------- 2.数据处理 -------------------------
# 确认 BMI ==0 为异常值
#  异常值 标签分布
print(pima_data.loc[pima_data["BMI"] == 0, "Outcome"].value_counts())
"""
0    9
1    2
"""
# ----------------------正常样本中--------------------
# 糖尿病人 的 BMI
diabetes_bmi = pima_data.loc[(pima_data["BMI"] > 0) & (pima_data["Outcome"] == 1), "BMI"]

# 非糖尿病人 的 BMI
diabetes_not_bmi = pima_data.loc[(pima_data["BMI"] > 0) & (pima_data["Outcome"] == 0), "BMI"]

# 非糖尿病人bmi的平均值
mean = diabetes_not_bmi.median()
print(mean, "mean")  # 30.1 mean
# 糖尿病人bmi的标准差
std = diabetes_not_bmi.std()

# 糖尿病人bmi的平均值
d_mean = diabetes_bmi.median()
print(d_mean, "d_mean")  # 34.3 d_mean
# 非糖尿病人bmi的标准差
d_std = diabetes_bmi.std()

# --------------  插值 --------------

# --------- 异常样本中 ---------

# BMI异常值中,糖尿病人的样本
anomaly_true_out = pima_data.loc[(pima_data["BMI"] == 0) & (pima_data["Outcome"] == 1)]

# BMI异常值中,非糖尿病人的样本
anomaly_false_out = pima_data.loc[(pima_data["BMI"] == 0) & (pima_data["Outcome"] == 0)]

#  糖尿病插值
lst = list(anomaly_true_out.index)
for index_ in lst:
    pima_data.loc[index_, "BMI"] = round(np.random.normal(mean, std), 1)

#  非糖尿病插值
lst = list(anomaly_false_out.index)
for index_ in lst:
    pima_data.loc[index_, "BMI"] = round(np.random.normal(d_mean, d_std), 1)

#   ------------------------- 3.数据处理后 展示 -------------------------
print(pd.DataFrame(pima_data["BMI"]).describe())

# draw  boxplot -- 图1
mp.figure("BMI数据情况")
mp.title("BMI数据情况")
sns.boxplot(data=pima_data["BMI"])
mp.xlabel("BMI")
mp.ylabel("values")
mp.show()

# draw histogram -- 图2
mp.figure("BMI histogram ", facecolor="lightgray")
mp.title("BMI histogram")
pima_data["BMI"].hist()
mp.xlabel("BMI值")
mp.ylabel("人数")
mp.show()

# 写入新文件
pima_data["DiabetesPedigreeFunction"] = round(pima_data["DiabetesPedigreeFunction"], 3)
pima_data.to_csv("../data/cleaned_pima_data.csv", index=None)

print(pima_data.describe())
"""
       Pregnancies     Glucose  BloodPressure         BMI  DiabetesPedigreeFunction         Age     Outcome
count   768.000000  768.000000     768.000000  768.000000                768.000000  768.000000  768.000000
mean      3.845052  121.697358      72.644531   32.487891                  0.471876   33.240885    0.348958
std       3.369578   30.462008      12.139023    6.900346                  0.331329   11.760232    0.476951
min       0.000000   44.000000      40.000000   18.200000                  0.078000   21.000000    0.000000
25%       1.000000   99.750000      64.000000   27.500000                  0.243750   24.000000    0.000000
50%       3.000000  117.000000      72.000000   32.400000                  0.372500   29.000000    0.000000
75%       6.000000  141.000000      80.000000   36.600000                  0.626250   41.000000    1.000000
max      17.000000  199.000000     122.000000   67.100000                  2.420000   81.000000    1.000000
"""

# -- 图2
mp.figure("数据清理后的展示")
mp.title("数据清理后的展示")
sns.boxplot(data=pima_data)
mp.gcf().autofmt_xdate()
mp.xlabel("属性")
mp.ylabel("值")
mp.show()

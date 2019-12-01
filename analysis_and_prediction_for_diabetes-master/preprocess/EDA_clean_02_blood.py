from collections import Counter

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

# --------------------------------  BloodPressure --------------------------------
#   ------------------------- 1.数据分布  可视化 -------------------------

# boxplot -- 图1
mp.figure("BloodPressure boxplot")
mp.title("BloodPressure boxplot")
sns.boxplot(data=pima_data["BloodPressure"])
mp.grid(linestyle=":")
mp.xlabel("BloodPressure")
mp.ylabel("values")
mp.show()

# draw histogram -- 图2
mp.figure("BloodPressure histogram ", facecolor="lightgray")
mp.title("BloodPressure histogram")
pima_data["Glucose"].hist()
mp.xlabel("血压值")
mp.ylabel("人数")
mp.show()

# draw barplot -- 图3
c = Counter(pima_data["BloodPressure"])
x = [key for key in dict(c)]
y = [val for val in dict(c).values()]

mp.figure("血压值_人数分布", facecolor="lightgray")
mp.title("血压值_人数分布")
ax = mp.gca()
ax.yaxis.set_major_locator(mp.MultipleLocator(5))
ax.yaxis.set_minor_locator(mp.MultipleLocator(1))
mp.grid(":", alpha=0.8)
sns.barplot(x, y)
mp.gcf().autofmt_xdate(rotation=60)
mp.xlabel("血压")
mp.ylabel("人数")
mp.show()

print(pd.DataFrame(pima_data["BloodPressure"]).describe())
"""
       BloodPressure
count     768.000000
mean       69.105469
std        19.355807
min         0.000000
25%        62.000000
50%        72.000000
75%        80.000000
max       122.000000
"""

#  异常值 标签分布-1
print(pima_data.loc[pima_data["BloodPressure"] == 0, "Outcome"].value_counts())
"""
0    19
1    16
"""

#  异常值 标签分布-2
print(pima_data.loc[pima_data["BloodPressure"] < 40, "Outcome"].value_counts())
"""
0    22
1    17
"""

print("-" * 50)
#   ------------------------- 2.数据处理 -------------------------
# 确认 BloodPressure < 40 为异常值

# 获取异常值索引
blood_anomaly_index_lst = list(pima_data.loc[pima_data["BloodPressure"] < 40].index)
print(blood_anomaly_index_lst)
# [7, 15, 18, 49, 60, 78, 81, 125, 172, 193, 222, 261, 266, 269, 300, 332, 336, 347, 357, ..., 706]

# 获取异常值的  数量 , 占比
print(len(blood_anomaly_index_lst), str(round(len(blood_anomaly_index_lst) / pima_data.shape[0] * 100, 2)) + "%")
# 39 5.08%

# 有糖尿病的人,  BloodPressure_series
out_true = pima_data.loc[(pima_data["BloodPressure"] >= 40) & (pima_data["Outcome"] == 1), "BloodPressure"]

# 没有糖尿病的人,  BloodPressure_series
out_false = pima_data.loc[(pima_data["BloodPressure"] >= 40) & (pima_data["Outcome"] == 0), "BloodPressure"]

# 有糖尿病的人, 血压的中位数
blood_true_out_median = out_true.median()
# 没有有糖尿病的人, 血压的中位数
blood_false_out_median = out_false.median()

# 有糖尿病的人, 血压的平均数
blood_true_out_mean = out_true.mean()
# 没有糖尿病的人, 血压的平均数
blood_false_out_mean = out_false.mean()

# 有糖尿病的人, 血压的标准差
blood_true_out_std = out_true.std()
# 没有糖尿病的人, 血压的标准差
blood_false_out_std = out_false.std()

print("out_true : ", out_true.mean())
print("out_false : ", out_false.mean())
print("out_true--median : ", blood_true_out_median)
print("out_false--median  : ", blood_false_out_median)
print("blood_true_out_std  : ", blood_true_out_std)
print("blood_false_out_std : ", blood_false_out_std)
"""
out_true :  75.5019920318725
out_false :  71.1297071129707
out_true--median :  75.0
out_false--median  :  70.0
blood_true_out_std  :  11.98511560294419
blood_false_out_std :  11.763743303804743
"""

# --------------------  插值    -----------------------------

#  获取 糖尿病人异常值的index
anomaly_true_out_index_lst = list(pima_data.loc[(pima_data["BloodPressure"] < 40) &
                                                (pima_data["Outcome"] == 1),
                                                "BloodPressure"].index)
#  获取 非糖尿病人异常值的index
anomaly_false_out_index_lst = list(pima_data.loc[(pima_data["BloodPressure"] < 40) &
                                                 (pima_data["Outcome"] == 0),
                                                 "BloodPressure"].index)

print("-" * 50)
print(len(anomaly_true_out_index_lst), anomaly_true_out_index_lst)
# 17 [15, 78, 125, 193, 261, 266, 269, 300, 332, 357, 435, 468, 484, 535, 604, 619, 706]
print(len(anomaly_false_out_index_lst), anomaly_false_out_index_lst)
# 22 [7, 18, 49, 60, 81, 172, 222, 336, 347, 426, 430, 453, 494, 522, 533, 589, 597, 599, 601, 643, 697, 703]


# 糖尿病的人 BloodPressure 插值
for index_ in anomaly_true_out_index_lst:
    pima_data.loc[index_, "BloodPressure"] = int(np.random.normal(blood_true_out_median, blood_true_out_std))

# 非糖尿病的人 BloodPressure 插值
for index_ in anomaly_false_out_index_lst:
    pima_data.loc[index_, "BloodPressure"] = int(np.random.normal(blood_false_out_median, blood_false_out_std))

#   ------------------------- 3.数据处理后 展示 -------------------------
print("-" * 50)
print(pd.DataFrame(pima_data["BloodPressure"]).describe())
"""
       BloodPressure
count     768.000000
mean       72.640625
std        12.038552
min        40.000000
25%        64.000000
50%        72.000000
75%        80.000000
max       122.000000
"""

"""  对比 数据处理之前  describe
       BloodPressure
count     768.000000
mean       69.105469
std        19.355807
min         0.000000
25%        62.000000
50%        72.000000
75%        80.000000
max       122.000000
"""

# boxplot -- 图1
mp.figure("BloodPressure boxplot")
mp.title("BloodPressure boxplot")
sns.boxplot(data=pima_data["BloodPressure"])
mp.grid(linestyle=":")
mp.xlabel("BloodPressure")
mp.ylabel("values")
mp.show()

# draw barplot -- 图2
c = Counter(pima_data["BloodPressure"])
x = [key for key in dict(c)]
y = [val for val in dict(c).values()]

mp.figure("血压值_人数分布", facecolor="lightgray")
mp.title("血压值_人数分布")
ax = mp.gca()
ax.yaxis.set_major_locator(mp.MultipleLocator(5))
ax.yaxis.set_minor_locator(mp.MultipleLocator(1))
mp.grid(":", alpha=0.8)
sns.barplot(x, y)
mp.gcf().autofmt_xdate(rotation=60)
mp.xlabel("血压")
mp.ylabel("人数")
mp.show()

# 写入新文件
pima_data["DiabetesPedigreeFunction"] = round(pima_data["DiabetesPedigreeFunction"], 3)
pima_data.to_csv("../data/cleaned_pima_data.csv", index=None)

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
pima_data = pd.read_csv("../data/PimaIndianDiabetes.csv")

# -------------------------------------  删除列--------------------------------

del pima_data["SkinThickness"]
del pima_data["Insulin"]

# -------------------------------------  Glucose 葡萄糖 putao--------------------------------
#   ------------------------- 1.数据分布  可视化 -------------------------

# boxplot -- 图1
mp.figure("Glucose boxplot", facecolor="lightgray")
mp.title("Glucose boxplot")
sns.boxplot(data=pima_data["Glucose"])
mp.gcf().autofmt_xdate(rotation=0)
mp.grid(linestyle=":")
mp.ylabel("葡萄糖值")
mp.show()

# draw histogram -- 图2
mp.figure("Glucose histogram ", facecolor="lightgray")
mp.title("Glucose histogram")
pima_data["Glucose"].hist()
mp.xlabel("葡萄糖值")
mp.ylabel("人数")
mp.show()

# ----------------  describe  ----------------
print(pd.DataFrame(pima_data["Glucose"]).describe())
"""
          Glucose
count  768.000000
mean   120.894531
std     31.972618
min      0.000000
25%     99.000000
50%    117.000000
75%    140.250000
max    199.000000
"""

#   ------------------------- 2.数据处理 -------------------------
#  异常值 标签分布
print(pima_data.loc[pima_data["Glucose"] == 0, "Outcome"].value_counts())
"""
0    3
1    2
"""


def get_mean_groupby_outcome(data, col, n):
    """

    :param data:
    :param col: column
    :param n: Outcome n=0是第一行 非糖尿病标签
              Outcome n=1是第二行 是糖尿病标签
    :return:
    """
    df = data.loc[data[col] != 0, [col, "Outcome"]].groupby("Outcome"). \
        agg([np.mean, np.median, np.std])

    mean = df.loc[n, (col, "mean")]
    median = df.loc[n, (col, "median")]
    std = df.loc[n, (col, "std")]
    return mean, median, std


#  去除Glucose为0的样本, 糖尿病人的Glucose 各个值
d_mean, d_median, d_std = get_mean_groupby_outcome(pima_data, "Glucose", 1)
print(d_median, "d_median")  # 140.0 d_median

#  去除Glucose为0的样本, 非糖尿病人的Glucose  各个值
mean, median, std = get_mean_groupby_outcome(pima_data, "Glucose", 0)
print(mean, "median")  # 110.64386317907444 median

#  --------------------- 修改异常值 ---------------------
# 糖尿病人的Glucose
pima_data.loc[(pima_data["Glucose"] == 0) & (pima_data["Outcome"] == 1), "Glucose"] = d_mean
# 非糖尿病人的Glucose
pima_data.loc[(pima_data["Glucose"] == 0) & (pima_data["Outcome"] == 0), "Glucose"] = mean

#   ------------------------- 3.数据处理后 展示 -------------------------
print(pd.DataFrame(pima_data["Glucose"]).describe())
""" 数据处理后
          Glucose
count  768.000000
mean   121.697358
std     30.462008
min     44.000000
25%     99.750000
50%    117.000000
75%    141.000000
max    199.000000
"""

""""   数据清理之前 describe--------
          Glucose
count  768.000000
mean   120.894531
std     31.972618
min      0.000000
25%     99.000000
50%    117.000000
75%    140.250000
max    199.000000
"""

# boxplot -- 图1
mp.figure("Glucose boxplot", facecolor="lightgray")
mp.title("Glucose boxplot")
sns.boxplot(data=pima_data["Glucose"])
mp.gcf().autofmt_xdate(rotation=0)
mp.grid(linestyle=":")
mp.ylabel("葡萄糖值")
mp.show()

# draw histogram -- 图2
mp.figure("Glucose histogram ", facecolor="lightgray")
mp.title("Glucose histogram")
pima_data["Glucose"].hist()
mp.xlabel("葡萄糖值")
mp.ylabel("人数")
mp.show()

# 写入新文件
pima_data["DiabetesPedigreeFunction"] = round(pima_data["DiabetesPedigreeFunction"], 3)
pima_data.to_csv("../data/cleaned_pima_data.csv", index=None)

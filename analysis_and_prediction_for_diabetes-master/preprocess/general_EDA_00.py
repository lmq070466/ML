import pandas as pd
import numpy as np
import matplotlib.pyplot as mp
import seaborn as sns
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

# ---------------------------------- get data -------------------------------
pima_data = pd.read_csv("../data/PimaIndianDiabetes.csv")

# 展示部分数据
print(pima_data.head())
"""
   Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age  Outcome
0            6      148             72             35        0  33.6                     0.627   50        1
1            1       85             66             29        0  26.6                     0.351   31        0
2            8      183             64              0        0  23.3                     0.672   32        1
3            1       89             66             23       94  28.1                     0.167   21        0
4            0      137             40             35      168  43.1                     2.288   33        1

"""

# --------------------------  查看整体情况  ----------------------------
print("-" * 50)
# 看数据的形状
print(pima_data.shape)  # (768, 9)

# 标签分布
print(pima_data["Outcome"].value_counts())
"""
0    500
1    268
"""

print(pima_data.info())
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 768 entries, 0 to 767
Data columns (total 9 columns):
Pregnancies                 768 non-null int64
Glucose                     768 non-null int64
BloodPressure               768 non-null int64
SkinThickness               768 non-null int64
Insulin                     768 non-null int64
BMI                         768 non-null float64
DiabetesPedigreeFunction    768 non-null float64
Age                         768 non-null int64
Outcome                     768 non-null int64
dtypes: float64(2), int64(7)
memory usage: 54.1 KB
None
"""

print("-" * 50)
print(pima_data.describe())
"""
       Pregnancies     Glucose  BloodPressure  SkinThickness     Insulin         BMI  DiabetesPedigreeFunction         Age     Outcome
count   768.000000  768.000000     768.000000     768.000000  768.000000  768.000000                768.000000  768.000000  768.000000
mean      3.845052  120.894531      69.105469      20.536458   79.799479   31.992578                  0.471876   33.240885    0.348958
std       3.369578   31.972618      19.355807      15.952218  115.244002    7.884160                  0.331329   11.760232    0.476951
min       0.000000    0.000000       0.000000       0.000000    0.000000    0.000000                  0.078000   21.000000    0.000000
25%       1.000000   99.000000      62.000000       0.000000    0.000000   27.300000                  0.243750   24.000000    0.000000
50%       3.000000  117.000000      72.000000      23.000000   30.500000   32.000000                  0.372500   29.000000    0.000000
75%       6.000000  140.250000      80.000000      32.000000  127.250000   36.600000                  0.626250   41.000000    1.000000
max      17.000000  199.000000     122.000000      99.000000  846.000000   67.100000                  2.420000   81.000000    1.000000

SkinThickness ---  normal : 14.9～18.1mm
Insulin(胰岛素) ---  normal : 
BloodPressure  --- normal : 高血压值就是高压高于140mmHg，低压高于90mmHg，
Glucose --- normal :  70～140
BMI --- normal : １８至２５之间为健康的标准体重
"""

#  ---------------------- 查看各个列 0值的情况 ----------------------
print("-" * 50)

col_lst = pima_data.columns
rows = pima_data.shape[0]
for col in col_lst:
    zeros_index_lst = list(pima_data.loc[pima_data[col] == 0].index)
    size_ = len(zeros_index_lst)
    print(col, "-", size_, "-", "{}{}".format(round((size_ / rows) * 100, 3), "%"))

"""
Pregnancies - 111 - 14.453%
Glucose - 5 - 0.651%
BloodPressure - 35 - 4.557%
SkinThickness - 227 - 29.557%
Insulin - 374 - 48.698%
BMI - 11 - 1.432%
DiabetesPedigreeFunction - 0 - 0.0%
Age - 0 - 0.0%
Outcome - 500 - 65.104%
"""

# --------------------------------主成分分析-------------------------
print("=" * 50, "主成分分析")
pca = PCA()
pca.fit(pima_data)
# 返回特征向量
print(np.round(pca.components_, 2))
"""
[[-0.    0.1   0.02  0.06  0.99  0.01  0.   -0.    0.  ]
 [-0.02 -0.97 -0.14  0.06  0.09 -0.05 -0.   -0.14 -0.01]
 [-0.02  0.14 -0.92 -0.31  0.02 -0.13 -0.   -0.13  0.  ]
 [-0.05  0.12 -0.26  0.88 -0.07  0.19  0.   -0.3   0.  ]
 [ 0.15 -0.09 -0.23  0.26 -0.    0.02  0.    0.92  0.01]
 [-0.01  0.05  0.08  0.22 -0.01 -0.97 -0.   -0.01 -0.01]
 [ 0.99  0.   -0.   -0.    0.   -0.   -0.01 -0.16  0.02]
 [-0.02 -0.01  0.   -0.    0.   -0.01  0.24 -0.    0.97]
 [ 0.01  0.   -0.   -0.   -0.    0.    0.97 -0.   -0.24]]
"""
print("-------")
# 返回各个成分各自的方差百分比
print(np.round(pca.explained_variance_ratio_, 2))
# [0.89 0.06 0.03 0.01 0.01 0.   0.   0.   0.  ]

# ---------------------------- 可视化数据分布 ----------------------------

# draw histogram
pima_data.hist(figsize=(20, 15))
mp.xlabel("features")
mp.ylabel("values")
mp.show()

#  draw pairplot
sns.pairplot(pima_data)
mp.show()

#  draw  boxplot
mp.figure("features_show")
mp.title("features_show")
sns.boxplot(data=pima_data)
mp.gcf().autofmt_xdate()
mp.xlabel("features")
mp.ylabel("values")
mp.show()

# draw heatmap  各个特征数值分布
mp.figure("heatmap_show")
mp.title("heatmap_show")
sns.heatmap(data=pima_data)
mp.gcf().autofmt_xdate()
mp.show()

# draw heatmap  计算数据：　皮尔逊积矩相关系数　　　各个特征的相关系数
cols = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
chinese_features = ["怀孕次数", "葡萄糖", "血压", "BMI", "家族系数", "年龄", "标签"]
df_corr = pima_data[cols].corr()

mp.rcParams['font.sans-serif'] = ['SimHei']
mp.rcParams['axes.unicode_minus'] = False
mp.figure("各个特征的相关系数", facecolor="lightgray")
mp.title("各个特征的相关系数")
sns.heatmap(df_corr, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15},
            yticklabels=chinese_features, xticklabels=chinese_features)
mp.tight_layout()
mp.xlim(0, 7)
mp.ylim(0, 7)
mp.show()

print(df_corr)
"""
                          Pregnancies   Glucose  BloodPressure       BMI  DiabetesPedigreeFunction       Age   Outcome
Pregnancies                  1.000000  0.129459       0.141282  0.017683                 -0.033523  0.544341  0.221898
Glucose                      0.129459  1.000000       0.152590  0.221071                  0.137337  0.263514  0.466581
BloodPressure                0.141282  0.152590       1.000000  0.281805                  0.041265  0.239528  0.065068
BMI                          0.017683  0.221071       0.281805  1.000000                  0.140647  0.036242  0.292695
DiabetesPedigreeFunction    -0.033523  0.137337       0.041265  0.140647                  1.000000  0.033561  0.173844
Age                          0.544341  0.263514       0.239528  0.036242                  0.033561  1.000000  0.238356
Outcome                      0.221898  0.466581       0.065068  0.292695                  0.173844  0.238356  1.000000
"""

"""
由此可见:
标签(即糖尿病) 与葡萄糖的关系最大,高达0.47, 呈现正相关,第二是BMI是0.29, 第三是年龄是0.24
年龄和怀孕数关系很大,相关系数值高达0.54,呈现正相关关系
DiabetesPedigreeFunction(家族系数) 和各个特征的值关系不大
"""

print("=" * 50)
#   draw bar  糖尿病对应各个特征的相关性
mp.figure("糖尿病与各个特征的相关性", facecolor="lightgray")
mp.title("糖尿病与各个特征的相关性")
df_barplot = df_corr["Outcome"].sort_values()[::-1].drop("Outcome")
sns.barplot(list(df_barplot.index), df_barplot)
mp.gcf().autofmt_xdate()
mp.xlabel("特征")
mp.ylabel("特征相关系数")
mp.show()

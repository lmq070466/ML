"""
清洗数据后------数据分析---总体分析 + 葡萄糖特征分析
"""

import pandas as pd
import matplotlib.pyplot as mp
import seaborn as sns

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

mp.rcParams['font.sans-serif'] = ['SimHei']
mp.rcParams['axes.unicode_minus'] = False

# ---------------------------------- get data -------------------------------
pima_data = pd.read_csv("../data/cleaned_pima_data.csv")

# ------------------------------------- 糖尿病人数据集 -------------------------------------
diabetes_data = pima_data.loc[pima_data["Outcome"] == 1]
print(diabetes_data.head())
"""
   Pregnancies  Glucose  BloodPressure   BMI  DiabetesPedigreeFunction  Age  Outcome
0            6    148.0             72  33.6                     0.627   50        1
2            8    183.0             64  23.3                     0.672   32        1
4            0    137.0             40  43.1                     2.288   33        1
6            3     78.0             50  31.0                     0.248   26        1
8            2    197.0             70  30.5                     0.158   53        1
"""

# 打印糖尿病人当中,各个特征值的平均值,
# 由此可见:糖尿病人当中:怀孕平均次数是4.8次,葡萄糖的平均值142,
# 血压平均是76,BMI平均是35,家族系数平均是0.55,年龄平均是37岁
print(diabetes_data.mean(axis=0))
"""
Pregnancies                   4.865672
Glucose                     142.319549
BloodPressure                75.988806
BMI                          35.409701
DiabetesPedigreeFunction      0.550500
Age                          37.067164
Outcome                       1.000000
"""
# ------------------------------------- 非糖尿病人数据集 -------------------------------------
no_diabetes_data = pima_data.loc[pima_data["Outcome"] == 0]
print(no_diabetes_data.head())
"""
    Pregnancies  Glucose  BloodPressure   BMI  DiabetesPedigreeFunction  Age  Outcome
1             1     85.0             66  26.6                     0.351   31        0
3             1     89.0             66  28.1                     0.167   21        0
5             5    116.0             74  25.6                     0.201   30        0
7            10    115.0             72  35.3                     0.134   29        0
10            4    110.0             92  37.6                     0.191   30        0
"""

# 打印非糖尿病人当中,各个特征值的平均值,
# 由此可见:非糖尿病人当中:怀孕平均次数是3.3次,葡萄糖的平均值110,
# 血压平均是71,BMI平均是30,家族系数平均是0.42,年龄平均是31岁
print(no_diabetes_data.mean(axis=0))
"""
Pregnancies                   3.298000
Glucose                     110.643863
BloodPressure                71.114000
BMI                          30.960000
DiabetesPedigreeFunction      0.429734
Age                          31.190000
Outcome                       0.000000
dtype: float64
"""

# ------------------------------------- 非糖尿病和糖尿病的对比 -------------------------------------

no_d = no_diabetes_data.mean(axis=0)
d = diabetes_data.mean(axis=0)
cols = no_d.index

df_no_d = pd.DataFrame([no_d], columns=cols, index=["非糖尿病"])
df_d = pd.DataFrame([d], columns=cols, index=["糖尿病"])
df = df_no_d.append(df_d)
df.columns = ["Pregnance", "Glucose", "BlPressure", "BMI", "DiabetesFunc", "Age", "Outcome"]

print("=" * 20, "非糖尿病和糖尿病的对比", "=" * 20)
print(df)  # 对比糖尿病和非糖尿病人士的各个特征的平均值
"""
         Pregnance     Glucose  BlPressure        BMI  DiabetesFunc        Age  Outcome
非糖尿病   3.298000   110.643863   71.114000  30.960000     0.429734  31.190000      0.0
糖尿病     4.865672  142.319549   75.988806  35.409701      0.550500  37.067164      1.0
"""

# 对比糖尿病和非糖尿病人士, 他们明显的差别是,葡萄糖的差值比较高,患糖尿病的人比不患糖尿病的人高出32的值,
# 其次是年龄和BMI 相对来说,年轻的人比较少患糖尿病, 身体肥胖的人比较容易得糖尿病
# 再其次是血压

# ------------------------葡萄糖-特征值分析------------------------------------

# ------------计算数据------------
statistics = pima_data["Glucose"].describe()

statistics['range'] = statistics['max'] - statistics['min']  # 极差
statistics['var'] = statistics['std'] / statistics['mean']  # 方差
statistics['dis'] = statistics['75%'] - statistics['25%']  # 四分距
print(statistics)
"""
count    768.000000
mean     121.697358
std       30.462008
min       44.000000
25%       99.750000
50%      117.000000
75%      141.000000
max      199.000000
range    155.000000
var        0.250310
dis       41.250000
"""

# -------查看葡萄糖值高,但不患糖尿病的人-------
check = pima_data.loc[(pima_data["Outcome"] == 0) & (pima_data["Glucose"] > 190)]
print("-------查看葡萄糖值高,但不患糖尿病的人-------")
print("总共有%d人" % check.shape[0])
# 总共有4人

print(check.head())
"""
     Pregnancies  Glucose  BloodPressure   BMI  DiabetesPedigreeFunction  Age  Outcome
228            4    197.0             70  36.7                     2.329   31        0
258            1    193.0             50  25.9                     0.655   24        0
260            3    191.0             68  30.9                     0.299   34        0
489            8    194.0             80  26.1                     0.551   67        0
"""

print("-" * 50)
print(check.mean(axis=0))
"""                             特征
Pregnancies                   4.0000  糖尿病该特征均值4.86,非糖均值:3.29
Glucose                     193.7500
BloodPressure                67.0000  糖尿病该特征均值75,非糖均值:71
BMI                          29.9000  糖尿病该特征均值35,非糖均值:30
DiabetesPedigreeFunction      0.9585  糖尿病该特征均值0.55,非糖均值:0.42
Age                          39.0000  糖尿病该特征均值37,非糖均值:31
Outcome                       0.000000
dtype: float64

由此可见: 葡萄糖值高,但不患糖尿病的人当中,他们的血压和BMI比较低(比不患糖尿病人的均值还低)

葡萄糖值高,但不患糖尿病人, 跟身体瘦和血压低有很大的关系,
葡萄糖值高又不想得糖尿病人的,可以通过减肥和降血压来预防糖尿病

葡萄糖值高,和家族遗传有很大的关系高达:0.9585
葡萄糖值高,但不患糖尿病的人,跟年轻是有一些关系,4人中,有3人的年龄是24,31,34是非常年轻的
"""

# -------查看葡萄糖值低,但患糖尿病的人-------
check = pima_data.loc[(pima_data["Outcome"] == 1) & (pima_data["Glucose"] < 100)]
print("-------查看葡萄糖值低,但患糖尿病的人-------")
print("总共有%d人" % check.shape[0])
# 总共有14人

print(check.head())
"""
     Pregnancies  Glucose  BloodPressure   BMI  DiabetesPedigreeFunction  Age  Outcome
6              3     78.0             50  31.0                     0.248   26        1
38             2     90.0             68  38.2                     0.503   27        1
109            0     95.0             85  37.4                     0.247   24        1
125            1     88.0             80  55.0                     0.496   26        1
218            5     85.0             74  29.0                     1.224   32        1
"""

print("-" * 50)
print(check.mean(axis=0))
"""
Pregnancies                  4.785714 糖尿病该特征均值4.86,非糖均值:3.29
Glucose                     89.928571 
BloodPressure               72.857143 糖尿病该特征均值75,非糖均值:71
BMI                         35.607143 糖尿病该特征均值35,非糖均值:30
DiabetesPedigreeFunction     0.598214 糖尿病该特征均值0.55,非糖均值:0.42
Age                         34.928571 糖尿病该特征均值37,非糖均值:31
Outcome                      1.000000
dtype: float64

由此可见在葡萄糖值低的人士,但得病的人当中, 
患糖尿病的人,其中一个是32岁较年轻,血压和BMI都比较低,但是它的家族遗传系数很高很高,由此可见是遗传导致的
总体来说,这些人的的家族遗传,肥胖,怀孕次数都有挺大的关系,次三个均值都超出了患此病人的均值
"""

# ------------------------------------数据可视化展示------------------------------
# -------------draw 1 --------------
# 由此可见,不患糖尿病的人士的葡萄糖呈现正态分布,高峰是100-120,超过120和低于100的人数逐渐减少
mp.figure("总样本中的葡萄糖频率分布", facecolor="lightgray")
mp.title("总样本中的葡萄糖频率分布")
pima_data["Glucose"].hist(label="葡萄糖值-总体样本")
no_diabetes_data["Glucose"].hist(label="没糖尿病人的葡萄糖")
mp.xlabel("葡萄糖值")
mp.ylabel("人数")
mp.grid()
mp.legend()
mp.grid(":", alpha=0.7)
mp.show()

# -------------draw 2 --------------
# 由此可见,患糖尿病人士的葡萄糖分布值, 当中超过7成的人是在120-200之间的
mp.figure("总样本中--患糖尿病人士的---葡萄糖频率分布", facecolor="lightgray")
mp.title("总样本中--患糖尿病人士的---葡萄糖频率分布")
diabetes_data["Glucose"].hist(label="得糖尿病人的葡萄糖")
mp.xlabel("葡萄糖值")
mp.ylabel("人数")
mp.legend()
mp.show()

# -------------draw 3 --------------
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
sns.barplot(df_barplot.index, df_barplot)
mp.gcf().autofmt_xdate()
mp.xlabel("特征")
mp.ylabel("特征相关系数")
mp.show()

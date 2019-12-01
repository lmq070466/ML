"""
简单使用集中模型进行预测:
逻辑回归,线性判别分析,决策树,高斯贝叶斯分类器,支持向量机SVM分类器,k邻近算法
                     LR      LDA    DTree       NB      SVM      KNN
train_accuracy  0.75369  0.77278  0.68917  0.76410  0.65451  0.72931
train_f1_score  0.74339  0.76426  0.68281  0.75828  0.52142  0.72357
test_accuracy   0.77604  0.77604  0.68229  0.75521  0.63542  0.72917
"""

import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
import matplotlib.pyplot as mp
from sklearn import metrics

pd.set_option("display.max_columns", 1000)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 1000)

# ---------------------------------- get data -------------------------------
df = pd.read_csv("../data/cleaned_pima_data.csv")

# ---------------------------------- x, y 赋值 -------------------------------
x = df[["Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction", "Age"]]
y = df["Outcome"]

# ---------------------------------- split train test -------------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=7, test_size=0.25)

# ---------------------------------- models -------------------------------
models = []
models.append(("LR", LogisticRegression()))  # 逻辑回归
models.append(("LDA", LinearDiscriminantAnalysis()))  # 线性判别分析
models.append(("DTree", DecisionTreeClassifier()))  # 决策树
models.append(("NB", GaussianNB()))  # 高斯贝叶斯分类器
models.append(("SVM", SVC()))  # 支持向量机SVM
models.append(("KNN", KNeighborsClassifier()))  # k邻近算法

model_names = []
train_accuracy = []
train_f1_score = []
test_accuracy = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=7)

    # train
    train_acc = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring="accuracy")

    train_f1 = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring="f1_weighted")

    # test
    y_test_pre = cross_val_predict(model, x_test, y_test, cv=kfold)
    test_acc = metrics.accuracy_score(y_test, y_test_pre)

    model_names.append(name)
    train_accuracy.append(round(train_acc.mean(), 5))
    train_f1_score.append(round(train_f1.mean(), 5))
    test_accuracy.append(round(test_acc, 5))

columns = model_names
df = pd.DataFrame(columns=columns)
df.loc["train_accuracy"] = train_accuracy
df.loc["train_f1_score"] = train_f1_score
df.loc["test_accuracy"] = test_accuracy

print("=" * 50)
print(df)

"""
                     LR      LDA    DTree       NB      SVM      KNN
train_accuracy  0.75369  0.77278  0.68917  0.76410  0.65451  0.72931
train_f1_score  0.74339  0.76426  0.68281  0.75828  0.52142  0.72357
test_accuracy   0.77604  0.77604  0.68229  0.75521  0.63542  0.72917
"""

# ---------------------------------- plot -------------------------------

mp.figure("accuracy_score", facecolor="lightgray")
mp.title("accuracy_score")

ax = mp.gca()
ax.yaxis.set_minor_locator(mp.MultipleLocator(0.01))
ax.yaxis.set_major_locator(mp.MultipleLocator(0.05))

mp.xlabel("models")
mp.ylabel("accuracy")

size_ = np.arange(len(columns))
mp.bar(size_ - 0.2, train_accuracy, width=0.4, color="dodgerblue", zorder=3, label="train accuracy")
mp.bar(size_ + 0.2, test_accuracy, width=0.4, color="orange", zorder=3, label="test accuracy")
mp.plot(size_, train_f1_score, "o-", color="red", zorder=5, label="train f1_score")

mp.ylim(0.4, 0.9)
mp.xticks(size_, model_names)

mp.grid(":")
mp.legend()
mp.show()

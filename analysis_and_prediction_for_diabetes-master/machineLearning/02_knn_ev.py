import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as mp
import pickle

# knn邻近算法, 选择k值是28的时候,准确率达80.2%, 召回率达58.57%, 准确率达82%, F1分数达68%

# ---------------------------------- get data -------------------------------
df = pd.read_csv("../data/cleaned_pima_data.csv")
x = df[["Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction", "Age"]]
y = df["Outcome"]

# ---- split train test ------
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=7, test_size=0.25)

k_range = range(1, 50)
accuracy_lst = []
precison_lst = []
recall_lst = []

#  ---- fit ----
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)

    accuracy = cross_val_score(knn, x_train, y_train, cv=5, scoring="accuracy")
    precision = cross_val_score(knn, x_train, y_train, cv=5, scoring="precision")
    recall = cross_val_score(knn, x_train, y_train, cv=5, scoring="recall")

    accuracy_lst.append(round(accuracy.mean() * 100, 5))
    precison_lst.append(round(precision.mean() * 100, 5))
    recall_lst.append(round(recall.mean() * 100, 5))

mp.rcParams["font.sans-serif"] = ["SimHei"]
mp.rcParams["axes.unicode_minus"] = False

mp.figure("knn evaluation")
mp.title("knn evaluation")
mp.plot(k_range, accuracy_lst, "o-", color="orangered", label="train_accuracy")
mp.plot(k_range, precison_lst, "o-", color="dodgerblue", label="train_precison")
mp.plot(k_range, recall_lst, "o-", color="green", label="train_recall")

ax = mp.gca()
ax.yaxis.set_major_locator(mp.MultipleLocator(5))
ax.yaxis.set_minor_locator(mp.MultipleLocator(1))

ax.xaxis.set_major_locator(mp.MultipleLocator(5))
ax.xaxis.set_minor_locator(mp.MultipleLocator(1))

mp.legend()
mp.grid(":", alpha=0.8)
mp.xlabel("k值")
mp.ylabel("准确度")

#  -------------------test 预测  -------------------
k = 28
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)
y_test_predict = knn.predict(x_test)

confusion = metrics.confusion_matrix(y_test, y_test_predict)
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
TP = confusion[1, 1]

accuracy = round(metrics.accuracy_score(y_test, y_test_predict) * 100, 5)
print(accuracy, "---accuracy")

recall = round(TP / (FN + TP) * 100, 5)
print(recall, "---recall")

precision = round(TP / (TP + FP) * 100, 5)
print(precision, "---precision")

f1_score = round(2 * precision * recall / (precision + recall), 5)
print(f1_score, "---F1分数")

""" 
knn邻近算法, 选择k值是28的时候,准确率达80.2%, 召回率达58.57%, 准确率达82%, F1分数达68%

80.20833 ---accuracy
58.57143 ---recall
82.0 ---precision
68.33333 ---F1分数
"""

#  draw
mp.scatter(k, accuracy, marker="o", edgecolor="black", facecolor="violet", s=100, zorder=3)

mp.scatter(k, precision, marker="o", edgecolor="black", facecolor="yellow", s=100, zorder=3)

mp.scatter(k, recall, marker="o", edgecolor="black", facecolor="pink", s=100, zorder=3)

mp.annotate("test_precision", xycoords="data", xy=(k, precision),
            textcoords="offset points", xytext=(10, 6), fontsize=13,
            arrowprops=dict(arrowstyle="->", connectionstyle="angle3"))

mp.annotate("test_accuracy", xycoords="data", xy=(k, accuracy),
            textcoords="offset points", xytext=(-110, 5), fontsize=13,
            arrowprops=dict(arrowstyle="->", connectionstyle="angle3"))

mp.annotate("test_recall", xycoords="data", xy=(k, recall),
            textcoords="offset points", xytext=(10, 15), fontsize=13,
            arrowprops=dict(arrowstyle="->", connectionstyle="angle3"))
mp.legend()
mp.show()


# ------------- 混淆矩阵可视化  --------------
def cm_plot(y, yp):
    cm = metrics.confusion_matrix(y, yp)
    mp.matshow(cm, cmap="jet")
    mp.colorbar()

    for x in range(len(cm)):
        for y in range(len(cm)):
            mp.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')

    mp.ylabel('原始样本标签')
    mp.xlabel('预测的标签')
    return mp


cm_plot(y_test, y_test_predict)
mp.legend()
mp.show()

# 保存-->训练好的k邻近算法模型
with open('../model_save/knn.pkl', 'wb') as f:
    pickle.dump(knn, f)

#  ----------------
# lst = list(range(5, 18))
# lst = lst + list(range(20, 26))
#
# testing = {}
# for k_val in lst:
#     knn = KNeighborsClassifier(n_neighbors=k_val)
#     knn.fit(x_train, y_train)
#     y_test_predict = knn.predict(x_test)
#
#     confusion = metrics.confusion_matrix(y_test, y_test_predict)
#     TN = confusion[0, 0]
#     FP = confusion[0, 1]
#     FN = confusion[1, 0]
#     TP = confusion[1, 1]
#
#     accuracy = round(metrics.accuracy_score(y_test, y_test_predict) * 100, 5)
#     recall = round(TP / (FN + TP) * 100, 5)
#     precision = round(TP / (TP + FP) * 100, 5)
#     f1_score = round(2 * precision * recall / (precision + recall), 5)
#
#     testing[k_val] = accuracy, precision, recall, f1_score
#
# values_lst = []
# for key, values in testing.items():
#     values_lst.append(values[-1])
#     print(key, values)

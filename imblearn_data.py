import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE


d= pd.read_excel("D:\\limingqi20190902\\feature.xlsx",encoding='utf-8')
smo = SMOTE(ratio={1:1400},random_state=42)  #采样的过程因为样本不平衡采样按照2:1的比例1357
X = pd.concat([d.iloc[:,0:30],d.iloc[:,31:]],axis=1)
#print(X.head(5))
Y = d.iloc[:,30]
#print(Y.head(10))
X_smo, y_smo = smo.fit_sample(X,Y)
#print(y_smo)
#print(X_smo.shape)
x=pd.DataFrame(X_smo)
y=pd.DataFrame(y_smo)
x.columns=X.columns
y.columns=['label']
#print(y['label'].value_counts())
#print(y.head(20))
#print(y.head(50))
df = pd.concat([x,y],axis=1)
#df.to_excel('D:\\limingqi20190902\\imblean_data.xlsx')
#print(df.head(50))
#df.columns=d.columns
#print(df.head(5))
#print(df.head(10))
#print(df.shape)
'''
#print(corr)
X = df.iloc[:,0:28]
#d.iloc[:,29:]
#print(X.head(5))
Y = df.iloc[:,28]
#print(Y.head(5))
'''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,roc_curve,auc,accuracy_score,confusion_matrix
from sklearn.model_selection import GridSearchCV

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2,random_state=6)
std = StandardScaler()
x_train = std.fit_transform(X_train)
x_test = std.transform(X_test)
 # 逻辑回归
lg = LogisticRegression()
lg.fit(x_train,Y_train)
train_score=lg.score(x_train,Y_train)#训练的得分
test_score=lg.score(x_test,Y_test)#测试得分
#print(lg.coef_)
print(train_score)
print(test_score)
y_predict = lg.predict(x_test)
y_pred_proba=lg.predict_proba(x_test)#计算每个样本的预测概率
#print(y_predict)
print("准确率：", lg.score(x_test,Y_test))
#y_predict_log = lg.predict(x_test)

# 调用accuracy_score计算分类准确度
acc=accuracy_score(Y_test,y_predict)
print(acc)
print("召回率：", classification_report(Y_test,y_predict,target_names=["正常", "患病"]))
cnf_matrix = confusion_matrix(Y_test,y_predict)

param_grid = [
    {
        'C':[0.01,0.1,1,10,100],
        'penalty':['l2','l1'],
        'class_weight':['balanced',None]
    }
]
grid_search = GridSearchCV(lg,param_grid,cv=10,n_jobs=-1)
grid_search.fit(X_train,Y_train)
print(grid_search.best_estimator_)
print(grid_search.best_score_)
print(grid_search.best_params_)

decision_scores = lg.decision_function(x_test)

from sklearn.metrics import precision_recall_curve

precisions,recalls,thresholds = precision_recall_curve(y_test,decision_scores)
plt.plot(thresholds,precisions[:-1])
plt.plot(thresholds,recalls[:-1])
plt.grid()
plt.show()

def plot_cnf_matirx(cnf_matrix, description):
    class_names = [0, 1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    # create a heat map
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='OrRd',
                fmt='g')
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.title(description, y=1.1, fontsize=16)
    plt.ylabel('true0/1', fontsize=12)
    plt.xlabel('predict0/1', fontsize=12)
    plt.show()
plot_cnf_matirx(cnf_matrix, 'Confusion matrix -- Logistic Regression')

fprs,tprs,thresholds = roc_curve(y_test,decision_scores)


def plot_roc_curve(fprs, tprs):
    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(fprs, tprs)
    plt.plot([0, 1], linestyle='--')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.ylabel('TP rate', fontsize=15)
    plt.xlabel('FP rate', fontsize=15)
    plt.title('ROC曲线', fontsize=17)
    plt.show()


plot_roc_curve(fprs, tprs)
from sklearn.metrics import roc_auc_score  #auc:area under curve

roc_auc_score(Y_test,decision_scores)


#print(roc_curve)


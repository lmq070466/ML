# 首先导入探索分析所需的库
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import pandas as pd 
import random
import math
import time
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
from sklearn.preprocessing import StandardScaler


confirmed_df = pd.read_csv('time_series_19-covid-Confirmed.csv')
deaths_df = pd.read_csv('time_series_19-covid-Deaths.csv')
recoveries_df = pd.read_csv('time_series_19-covid-Recovered.csv')

confirmed_df.head(3)
deaths_df.head(3)
recoveries_df.head(3)

#更改数据格式
confirmed_df.rename(columns={'2002/1/20':'2/1/20','2002/2/20':'2/2/20','2002/3/20':'2/3/20','2002/4/20':'2/4/20',
                             '2002/5/20':'2/5/20','2002/6/20':'2/6/20','2002/7/20':'2/7/20','2002/8/20':'2/8/20',
                             '2002/9/20':'2/9/20','2002/10/20':'2/10/20','2002/11/20':'2/11/20','2002/12/20':'2/12/20','2003/11/20':'3/11/20','2003/1/20':'3/1/20','2003/2/20':'3/2/20','2003/3/20':'3/3/20',
                             '2003/4/20':'3/4/20','2003/5/20':'3/5/20','2003/6/20':'3/6/20','2003/7/20':'3/7/20','2003/8/20':'3/8/20'
                             ,'2003/9/20':'3/9/20','2003/10/20':'3/10/20','2003/11/20':'3/11/20'},inplace=True)
deaths_df.rename(columns={'2002/1/20':'2/1/20','2002/2/20':'2/2/20','2002/3/20':'2/3/20','2002/4/20':'2/4/20',
                             '2002/5/20':'2/5/20','2002/6/20':'2/6/20','2002/7/20':'2/7/20','2002/8/20':'2/8/20',
                             '2002/9/20':'2/9/20','2002/10/20':'2/10/20','2002/11/20':'2/11/20','2002/12/20':'2/12/20','2003/11/20':'3/11/20','2003/1/20':'3/1/20','2003/2/20':'3/2/20','2003/3/20':'3/3/20',
                             '2003/4/20':'3/4/20','2003/5/20':'3/5/20','2003/6/20':'3/6/20','2003/7/20':'3/7/20','2003/8/20':'3/8/20'
                             ,'2003/9/20':'3/9/20','2003/10/20':'3/10/20','2003/11/20':'3/11/20'},inplace=True)
recoveries_df.rename(columns={'2002/1/20':'2/1/20','2002/2/20':'2/2/20','2002/3/20':'2/3/20','2002/4/20':'2/4/20',
                             '2002/5/20':'2/5/20','2002/6/20':'2/6/20','2002/7/20':'2/7/20','2002/8/20':'2/8/20',
                             '2002/9/20':'2/9/20','2002/10/20':'2/10/20','2002/11/20':'2/11/20','2002/12/20':'2/12/20','2003/11/20':'3/11/20','2003/1/20':'3/1/20','2003/2/20':'3/2/20','2003/3/20':'3/3/20',
                             '2003/4/20':'3/4/20','2003/5/20':'3/5/20','2003/6/20':'3/6/20','2003/7/20':'3/7/20','2003/8/20':'3/8/20'
                             ,'2003/9/20':'3/9/20','2003/10/20':'3/10/20','2003/11/20':'3/11/20'},inplace=True)
cols = confirmed_df.keys()#输出列名
#print(cols)


confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]#提取日期以内的病人数
deaths = deaths_df.loc[:, cols[4]:cols[-1]]
recoveries = recoveries_df.loc[:, cols[4]:cols[-1]]
# 建立对应日期列表
dates = confirmed.keys()
#print(dates[45:50])
world_cases = []
total_deaths = [] 
mortality_rate = []
total_recovered = [] 
# 分别对对应日期的确诊病例数、死亡人数、康复人数记录求和
for i in dates:
    confirmed_sum = confirmed[i].sum()
    death_sum = deaths[i].sum()
    recovered_sum = recoveries[i].sum()
    world_cases.append(confirmed_sum)
    total_deaths.append(death_sum)
    mortality_rate.append(death_sum/confirmed_sum)
    total_recovered.append(recovered_sum)
#print("截至10日总计确诊:",int(world_cases[-2]),"人")
#print("截至10日总计死亡:",int(total_deaths[-2]),"人")
#print("截至10日总计治愈:",int(total_recovered[-2]),"人")
#print("截至11日总计确诊:",int(world_cases[-1]),"人")
#print("截至11日总计死亡:",int(total_deaths[-1]),"人")
#print("截至11日总计治愈:",int(total_recovered[-1]),"人")
#print(confirmed[dates[-1]])
#探究不同国家内确诊病例数
for i in confirmed_df.index:
    if np.isnan(confirmed_df['3/11/20'][i]) or confirmed_df['3/11/20'][i]==0.0:
        confirmed_df['3/11/20'][i] = confirmed_df['3/10/20'][i]
    else:
        continue
#print(confirmed[dates[-1]].isnull())
df = confirmed_df.groupby(['Country/Region'])['3/11/20'].sum().reset_index()
df.drop([129],axis = 0,inplace=True)
#提取出中国的人数
dk = df[(df['Country/Region']!='China')&(df['Country/Region']!='Mainland China')]
diff = df[df['Country/Region']=='China']
#print(np.sum(dk['3/11/20']))
#探究除中国之外的其他国家的确诊人数
ds = diff.append([{'Country/Region':'China outside','3/11/20':np.sum(dk['3/11/20'])}])
di = confirmed_df[(confirmed_df['Country/Region']=='China')|(confirmed_df['Country/Region']=='Mainland China')]
#di = df[(df['Country/Region']=='China')|(df['Country/Region']=='Mainland China')]
#print(di.head(10))

#中国不同省份的确诊人数的分布
plt.figure(figsize=(20,20))
plt.pie(di['3/11/20'])
plt.legend(di['Province/State'].unique(), loc='upper left')
plt.show()
#不同国家的确诊人数对比
plt.figure(figsize=(10, 8))
plt.barh(ds['Country/Region'], ds['3/11/20'])
plt.title('# of Coronavirus Confirmed Cases')
plt.show()

#转换成log就会特征明显的显示出来
df['log'] = np.log2(df['3/11/20'])
plt.figure(figsize=(20,16))
plt.barh(df['Country/Region'][100:], df['log'][100:])
plt.title('# of Coronavirus Confirmed Cases in Countries/Regions')
plt.xlabel('# of Covid19 Confirmed Cases')
plt.tight_layout
plt.show()
#死亡和恢复人数对比图
plt.figure(figsize=(20,6))
plt.plot(dates, total_deaths, color='red')
plt.plot(dates, total_recovered, color='green')
plt.legend(['death', 'recoveries'], loc='best', fontsize=20)
plt.title('# Coronavirus Cases', size=30)
plt.xlabel('Time', size=30)
plt.ylabel('# Cases', size=30)
plt.xticks(rotation=50, size=15)
plt.show()

#总共恢复人数按照时间的分布情况
plt.figure(figsize=(20,7))
plt.plot(dates, total_recovered, color='green')
plt.title('# Coronavirus Cases Recovered Over Time', size=30)
plt.xlabel('Time', size=30)
plt.ylabel('# Recovered Cases', size=30)
plt.xticks(rotation=50, size=15)
plt.show()
#病死率
mean_mortality_rate = np.mean(mortality_rate)
plt.figure(figsize=(18, 8))
plt.plot(dates, mortality_rate, color='orange')
plt.axhline(y = mean_mortality_rate,linestyle='--', color='black')
plt.title('# Mortality Rate of Coronavirus Over Time', size=30)
plt.legend(['mortality rate', 'y='+str(mean_mortality_rate)])
plt.xlabel('Time', size=30)
plt.ylabel('# Mortality Rate', size=30)
plt.xticks(rotation=50, size=15)
plt.show()

#总共的死亡率随着日期的变化
plt.figure(figsize=(18,8))
plt.plot(dates, total_deaths, color='red')
plt.title('# Coronavirus Deaths Over Time', size=30)
plt.xlabel('Time', size=30)
plt.ylabel('# Deaths', size=30)
plt.xticks(rotation=50, size=15)
plt.show()


X = np.array([x for x in range(len(dates))]).reshape(-1,1)
world_cases = np.array(world_cases).reshape(-1,1)

X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(X, world_cases, test_size=0.1, shuffle=False) 
#print(X_test_confirmed)

# SVR的核函数，它必须是'linear'，'poly'，'rbf'，'sigmoid'，'precomputed'或者callable之一。如果没有给出，将使用'rbf'。
kernel = ['linear', 'rbf']
# c是错误的惩罚参数C.默认1
c = [0.01, 0.1, 1, 10]
# gamma是'rbf'，'poly'和'sigmoid'的核系数。默认是'auto'
gamma = [0.01, 0.1, 1]
# Epsilon在epsilon-SVR模型中。它指定了epsilon-tube，其中训练损失函数中没有惩罚与在实际值的距离epsilon内预测的点。默认值是0.1
epsilon = [0.01, 0.1, 1]
# shrinking指明是否使用收缩启发式。默认为True
shrinking = [True, False]
svm_grid = {'kernel': kernel, 'C': c, 'gamma' : gamma, 'epsilon': epsilon, 'shrinking' : shrinking}
# 建立支持向量回归模型
svm = SVR()
# 使用随机搜索进行超参优化
svm_search = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=30, verbose=1)
svm_search.fit(X_train_confirmed, y_train_confirmed)
svm_confirmed = svm_search.best_estimator_
# check against testing data
svm_test_pred = svm_confirmed.predict(X_test_confirmed)
plt.plot(svm_test_pred,'r')
plt.plot(y_test_confirmed,'b')
plt.show()

#随机森林回归曲线
ensemble_grid =  {'n_estimators': [(i+1)*10 for i in range(20)],
                 'criterion': ['mse', 'mae'],
                 'bootstrap': [True, False],
                 }
ensemble = RandomForestRegressor()
ensemble_search = RandomizedSearchCV(ensemble, ensemble_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=10, verbose=1)
ensemble_search.fit(X_train_confirmed, y_train_confirmed)
ensemble_confirmed = ensemble_search.best_estimator_
# check against testing data
ensemble_test_pred = ensemble_confirmed.predict(X_test_confirmed)
plt.plot(ensemble_test_pred,'r')
plt.plot(y_test_confirmed,'b')
plt.show()

#线性函数预测
linear_model = LinearRegression(fit_intercept=False, normalize=True)
linear_model.fit(X_train_confirmed, y_train_confirmed)
test_linear_pred = linear_model.predict(X_test_confirmed)
print('MAE:', mean_absolute_error(test_linear_pred, y_test_confirmed))
print('MSE:',mean_squared_error(test_linear_pred, y_test_confirmed))
print (linear_model.coef_)
plt.plot(y_test_confirmed,'b')
plt.plot(test_linear_pred,'r')
plt.show()

#三种算法结果对比图
plt.figure(figsize=(16,8))
plt.plot(dates[45:50], y_test_confirmed)
plt.plot(dates[45:50], svm_test_pred, linestyle='dashed')
plt.plot(dates[45:50], ensemble_test_pred, linestyle='dashed')
plt.plot(dates[45:50], test_linear_pred, linestyle='dashed')
plt.title('#confirmed Coronavirus Cases Over Time', size=30)
plt.xlabel('Time in Days', size=30)
plt.ylabel('# confirmed Cases', size=30)
plt.legend(['Confirmed Cases', 'SVM predictions', 'Random Forest predictions', 'Linear Regression'])
plt.xticks(rotation=50, size=15)
plt.show()

#确诊病例与时间的关系
plt.figure(figsize=(16,8))
plt.plot(dates, world_cases)
plt.xlabel('Time in Days', size=30)
plt.ylabel('# confirmed Cases', size=30)
plt.xticks(rotation=50, size=15)
plt.show()
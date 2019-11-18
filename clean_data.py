import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

df = pd.read_excel("D:\\limingqi20190902\\副本前列腺癌数据3000例.xlsx",encoding='utf-8')
dk = df
#print(df.head(5))
dk.columns=['Case_ID','LABEL','AGE','BODY_HEIGHT','BODY_WEIGHT',
            '快速微量尿白蛋白/肌酐测定','骨钙素','载脂蛋白AⅡ',
            '载脂蛋白C2','载脂蛋白C3','载脂蛋白E','磷脂','脑利钠肽前体',
            'α1球蛋白','α2球蛋白','γ球蛋白','血清白蛋白','尿尿酸','碱性磷酸酶',
            '肌酸激酶同工酶','PSA（游离）','PSA（总）','钠','尿钠','钙','尿钙',
            '氯化物','尿氯化物','无机磷','尿磷','游离钙','乳酸脱氢酶','肌酸激酶',
            '尿钾','肌酐','尿肌酐','血清尿酸','肌钙蛋白T','甘油三酯',
            '高密度脂蛋白胆固醇','低密度脂蛋白胆固醇','载脂蛋白A1','载脂蛋白B',
            '钾','β1球蛋白','β2球蛋白']  #便于指标的观察
#print(dk.head(5))
#dk.to_excel('D:\\limingqi20190902\\test.xlsx')
def bin(x):     #有前列腺癌的为1    没有的为0
    if x==1:
        return 0
    else:
        return 1

dk['label']=dk['LABEL'].apply(bin)
#print(dk.head(5))
#print(dk.info())
#print(dk.head(5))
#dk.to_excel('D:\\limingqi20190902\\test1.xlsx')
#print(dk['label'].value_counts())
#0    2771    由此可见得癌症的人数229    没有的癌症的人数2771     样本不平衡
#1     229
#print(dk.info())
#快速微量尿白蛋白/肌酐测定    92 non-null object
#骨钙素              641 non-null object
#脑利钠肽前体           271 non-null object
#α1球蛋白            151 non-null float64
#α2球蛋白            151 non-null float64
#γ球蛋白             151 non-null float64
#尿尿酸              111 non-null object
#尿钠               73 non-null object
#尿钙               55 non-null float64
#尿氯化物             71 non-null object
#尿磷               55 non-null float64
#尿钾               76 non-null object
#尿肌酐              113 non-null object
#肌钙蛋白T            625 non-null object
#β1球蛋白            151 non-null float64
#β2球蛋白            151 non-null float64
#d= pd.read_excel("D:\\limingqi20190902\\test1.xlsx",encoding='utf-8')
#d.drop(['β2球蛋白','β1球蛋白','肌钙蛋白T','尿肌酐','尿钾','尿磷','尿氯化物','尿钙',
          # '尿钠','尿尿酸','γ球蛋白','α2球蛋白','α1球蛋白','脑利钠肽前体',
           # '骨钙素','快速微量尿白蛋白/肌酐测定'],axis=1,inplace=True)
#print(dk.info())
#dk.to_excel('D:\\limingqi20190902\\test2.xlsx')

d= pd.read_excel("D:\\limingqi20190902\\test1.xlsx",encoding='utf-8')
def func1(tmp):
    tmp = re.findall("-?\d+\.?\d*e?-?\d*?", tmp)
    return tmp
d['甘油三酯_1']=d['甘油三酯'].apply(func1)

res = []
for i in d['甘油三酯_1']:
    if i:
        res.append(i.pop())
    else:
        res.append(0)
#print(len(res))  3000
d.insert(8,'甘油三酯_2',res)

d.drop(['甘油三酯'],axis=1,inplace=True)
d.drop(['甘油三酯_1'],axis=1,inplace=True)
#print(d.info())
#print(d['甘油三脂_2'])
d.to_excel('D:\\limingqi20190902\\clean.xlsx')
#d=d[d['label']==1]
#print(d.head(10))
#print(dk_1)
#print(d.info())
d.drop(['β2球蛋白','β1球蛋白','肌钙蛋白T','尿肌酐','尿钾','尿磷','尿氯化物','尿钙',
         '尿钠','尿尿酸','γ球蛋白','α2球蛋白','α1球蛋白','脑利钠肽前体',
           '骨钙素','快速微量尿白蛋白/肌酐测定'],axis=1,inplace=True)
d.drop(['Case_ID','LABEL'],axis=1,inplace=True)
#print(d.info())
#print(d.describe())
#磷脂             2678 non-null float64      这三个是缺失值比较多的  磷脂mean   2.287057  max 2.900000   min 1.710000  不存在拖尾现象所以可以直接均值填充
#PSA（游离）        2963 non-null object      mean值填充
#PSA（总）         2986 non-null object      mean值填充
d['PSA（游离）']=d['PSA（游离）'].astype(float)
d['PSA（总）']=d['PSA（总）'].astype(float)
d['磷脂'][d['磷脂'].isnull()] = d['磷脂'].dropna().mean()
d['PSA（游离）'][d['PSA（游离）'].isnull()] = d['PSA（游离）'].dropna().mean()
d['PSA（总）'][d['PSA（总）'].isnull()] = d['PSA（总）'].dropna().mean()
d['高密度脂蛋白胆固醇'][d['高密度脂蛋白胆固醇'].isnull()] = d['高密度脂蛋白胆固醇'].dropna().mean()
d['低密度脂蛋白胆固醇'][d['低密度脂蛋白胆固醇'].isnull()] = d['低密度脂蛋白胆固醇'].dropna().mean()
d['BODY_HEIGHT'][d['BODY_HEIGHT'].isnull()] = d['BODY_HEIGHT'].dropna().mean()
d['BODY_WEIGHT'][d['BODY_WEIGHT'].isnull()] = d['BODY_WEIGHT'].dropna().mean()

d['label'].value_counts().plot.pie(labeldistance = 1.1,autopct = '%1.2f%%',
                                               shadow = False,startangle = 90,pctdistance = 0.6)

#前列腺癌的样本明显很不平衡   患癌症的人7.63%     正常的人92.37%
#plt.show()
#print(d.describe())   #载脂蛋白C3  乳酸脱氢酶 肌酸激酶 存在数据拖尾的现象

#初始特征的探索（数据预处理）分析特征与患癌的关联性1：按照特征顺序 2：按照连续，离散，非结构化特征的顺序
#print(d.groupby(['AGE','label'])['AGE'].count())
#print(d[['AGE','label']].groupby(['AGE']).mean())
#d[['AGE','label']].groupby(['AGE']).mean().plot.bar()
plt.figure(figsize=(15, 5))
plt.subplot(121)
d['AGE'].hist(bins=100)
plt.xlabel('AGE')
plt.ylabel('Num')
plt.subplot(122)
d.boxplot(column='AGE',showfliers=False)
#可以看出来45-60岁人最多   53岁左右是高发期



facet = sns.FacetGrid(d,hue="label",aspect=4)
facet.map(sns.kdeplot,'AGE',shade=True)
facet.set(xlim=(0,d['AGE'].max()))
facet.add_legend()
plt.show()

#print(d.info())
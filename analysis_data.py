import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

d = pd.read_excel("D:\\limingqi20190902\\clean.xlsx",encoding='utf-8')

d.rename(columns={'β2球蛋白':'β2globulin','β1球蛋白':'β1globulin','肌钙蛋白T':'Troponin T',
                '尿肌酐':'Urinary creatinine', '尿钾':'Urine potassium','尿磷':'Urine phosphorus',
                 '尿氯化物':'Urine chloride','尿钙':'Urinary calcium','尿钠':'Urine sodium',
                  '尿尿酸':'Uric acid','γ球蛋白':'Gamma globulin','α2球蛋白':'Alpha 2 globulin',
                 'α1球蛋白':'Alpha 1 globulin','脑利钠肽前体':'Brain natriuretic peptide precursor',
                '骨钙素':'Osteocalcin' ,'快速微量尿白蛋白/肌酐测定':'Rapid microalbuminuria/creatinine assay'},inplace=True)

def func2(x):
    if pd.isnull(x):
        return 0
    else:
        return 1
d['β2globulin']=d['β2globulin'].apply(func2)
#print(d['β2球蛋白'].value_counts())
#sns.factorplot('β2globulin','label',data = d, size=3, aspect=2)
#plt.title('β2globulin and label rate')

def func3(x):
    if pd.isnull(x):
        return 0
    else:
        return 1
d['β1globulin']=d['β1globulin'].apply(func3)
#print(d['β2球蛋白'].value_counts())
#sns.factorplot('β1globulin','label',data = d, size=3, aspect=2)
#plt.title('β1globulin and label rate')

def func4(x):
    if pd.isnull(x):
        return 0
    else:
        return 1
d['Troponin T']=d['Troponin T'].apply(func4)
#print(d['β2球蛋白'].value_counts())
#sns.factorplot('Troponin T','label',data =d, size=3, aspect=2)
#plt.title('Troponin T and label rate')


def func5(x):
    if pd.isnull(x):
        return 0
    else:
        return 1
d['Urinary creatinine']=d['Urinary creatinine'].apply(func5)
#print(d['β2球蛋白'].value_counts())
#sns.factorplot('Urinary creatinine','label',data =d, size=3, aspect=2)
#plt.title('Urinary creatinine and label rate')

def func6(x):
    if pd.isnull(x):
        return 0
    else:
        return 1
d['Urine potassium']=d['Urine potassium'].apply(func6)
#print(d['β2球蛋白'].value_counts())
#sns.factorplot('Urine potassium','label',data =d, size=3, aspect=2)
#plt.title('Urine potassium and label rate')

def func7(x):
    if pd.isnull(x):
        return 0
    else:
        return 1
d['Urine phosphorus']=d['Urine phosphorus'].apply(func7)
#print(d['β2球蛋白'].value_counts())
#sns.factorplot('Urine phosphorus','label',data =d, size=3, aspect=2)
#plt.title('Urine phosphorus and label rate')

def func8(x):
    if pd.isnull(x):
        return 0
    else:
        return 1
d['Urine chloride']=d['Urine chloride'].apply(func8)
#print(d['β2球蛋白'].value_counts())
#sns.factorplot('Urine chloride','label',data =d, size=3, aspect=2)
#plt.title('Urine chloride and label rate')

def func9(x):
    if pd.isnull(x):
        return 0
    else:
        return 1
d['Urinary calcium']=d['Urinary calcium'].apply(func9)
#print(d['β2球蛋白'].value_counts())
#sns.factorplot('Urinary calcium','label',data =d, size=3, aspect=2)
#plt.title('Urinary calcium and label rate')

def func10(x):
    if pd.isnull(x):
        return 0
    else:
        return 1
d['Urine sodium']=d['Urine sodium'].apply(func10)
#print(d['β2球蛋白'].value_counts())
#sns.factorplot('Urine sodium','label',data = d, size=3, aspect=2)
#plt.title('Urine sodium and label rate')

def func11(x):
    if pd.isnull(x):
        return 0
    else:
        return 1
d['Uric acid']=d['Uric acid'].apply(func11)
#print(d['β2球蛋白'].value_counts())
#sns.factorplot('Uric acid','label',data = d, size=3, aspect=2)
#plt.title('Uric acid and label rate')

def func12(x):
    if pd.isnull(x):
        return 0
    else:
        return 1
d['Gamma globulin']=d['Gamma globulin'].apply(func12)
#print(d['β2球蛋白'].value_counts())
#sns.factorplot('Gamma globulin','label',data = d, size=3, aspect=2)
#plt.title('Gamma globulin and label rate')

def func13(x):
    if pd.isnull(x):
        return 0
    else:
        return 1
d['Alpha 2 globulin']=d['Alpha 2 globulin'].apply(func13)
#print(d['β2球蛋白'].value_counts())
#sns.factorplot('Alpha 2 globulin','label',data = d, size=3, aspect=2)
#plt.title('Alpha 2 globulin and label rate')

def func14(x):
    if pd.isnull(x):
        return 0
    else:
        return 1
d['Alpha 1 globulin']=d['Alpha 1 globulin'].apply(func14)
#print(d['β2球蛋白'].value_counts())
#sns.factorplot('Alpha 1 globulin','label',data = d, size=3, aspect=2)
#plt.title('Alpha 1 globulin and label rate')

def func15(x):
    if pd.isnull(x):
        return 0
    else:
        return 1
d['Brain natriuretic peptide precursor']=d['Brain natriuretic peptide precursor'].apply(func15)
#print(d['β2球蛋白'].value_counts())
#sns.factorplot('Brain natriuretic peptide precursor','label',data = d, size=3, aspect=2)
#plt.title('Brain natriuretic peptide precursor and label rate')

def func16(x):
    if pd.isnull(x):
        return 0
    else:
        return 1
d['Osteocalcin']=d['Osteocalcin'].apply(func16)
#print(d['β2球蛋白'].value_counts())
#sns.factorplot('Osteocalcin','label',data = d, size=3, aspect=2)
#plt.title('Osteocalcin and label rate')

def func17(x):
    if pd.isnull(x):
        return 0
    else:
        return 1
d['Rapid microalbuminuria/creatinine assay']=d['Rapid microalbuminuria/creatinine assay'].apply(func17)
#print(d['β2球蛋白'].value_counts())
#sns.factorplot('Rapid microalbuminuria/creatinine assay','label',data = d, size=3, aspect=2)
#plt.title('Rapid microalbuminuria/creatinine assay and label rate')
#plt.show()
#print(d.info())
d.drop(['β2globulin','β1globulin','Urinary creatinine','Urine potassium','Urine phosphorus','Urine chloride','Urinary calcium',
          'Urine sodium','Uric acid','Gamma globulin','Alpha 2 globulin','Alpha 1 globulin',
         'Osteocalcin','Rapid microalbuminuria/creatinine assay'],axis=1,inplace=True)
d.drop(['Case_ID','LABEL'],axis=1,inplace=True)

#print(d.columns)
d['PSA（游离）']=d['PSA（游离）'].astype(float)
d['PSA（总）']=d['PSA（总）'].astype(float)
d['磷脂'][d['磷脂'].isnull()] = d['磷脂'].dropna().mean()
#d['PSA（游离）'][d['PSA（游离）'].isnull()] = d['PSA（游离）'].dropna().mean()
d['PSA（总）'][d['PSA（总）'].isnull()] = d['PSA（总）'].dropna().mean()
d['高密度脂蛋白胆固醇'][d['高密度脂蛋白胆固醇'].isnull()] = d['高密度脂蛋白胆固醇'].dropna().mean()
d['低密度脂蛋白胆固醇'][d['低密度脂蛋白胆固醇'].isnull()] = d['低密度脂蛋白胆固醇'].dropna().mean()
d['BODY_HEIGHT'][d['BODY_HEIGHT'].isnull()] = d['BODY_HEIGHT'].dropna().mean()
d['BODY_WEIGHT'][d['BODY_WEIGHT'].isnull()] = d['BODY_WEIGHT'].dropna().mean()

from sklearn.ensemble import RandomForestRegressor

psa_df = d[['PSA（游离）','AGE','BODY_HEIGHT','BODY_WEIGHT', '载脂蛋白AⅡ',
            '甘油三酯_2', '载脂蛋白C2','载脂蛋白C3',
            '血清白蛋白','碱性磷酸酶','肌酸激酶同工酶','PSA（总）','钙',
            '氯化物','无机磷','游离钙','乳酸脱氢酶','肌酸激酶','肌酐',
            '血清尿酸','高密度脂蛋白胆固醇','低密度脂蛋白胆固醇',
            '载脂蛋白A1','载脂蛋白B','钾','label']]
psa_df_notnull = psa_df.loc[(d['PSA（游离）'].notnull())]
psa_df_isnull = psa_df.loc[(d['PSA（游离）'].isnull())]
X = psa_df_notnull.values[:,1:]
Y = psa_df_notnull.values[:,0]
RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
RFR.fit(X,Y)
predictPsa = RFR.predict(psa_df_isnull.values[:,1:])
d.loc[d['PSA（游离）'].isnull(), ['PSA（游离）']]= predictPsa

#空值填充的过程'''
#print(d.info())

d['label'].value_counts().plot.pie(labeldistance = 1.1,autopct = '%1.2f%%',
                                               shadow = False,startangle = 90,pctdistance = 0.6)
#plt.show()
#print(d['label'].value_counts())
#d.to_excel('D:\\limingqi20190902\\spass.xlsx')

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

fig,axis1 = plt.subplots(1,1,figsize=(18,4))
d['AGE'] = d['AGE'].astype(int)
average_age = d[["AGE", "label"]].groupby(['AGE'],as_index=False).mean()
sns.barplot(x='AGE',y='label',data=average_age)
plt.xticks(rotation=70)
#print(d['AGE'].describe())

#bins = [0,20,40,60,80,100]   #分析不同年龄段的患癌症情况
#d = d[(d['BODY_HEIGHT']<250)&(d['BODY_HEIGHT']>100)]    #  发现异常值过滤掉(d['BODY_HEIGHT']>100)
#d = d[(d['BODY_WEIGHT']>0)&(d['PSA（free）']<500)]
#print(d['BODY_HEIGHT'].describe())
def func1(x):
    if x>0 and x<=20:
        return 1
    elif x>20 and x<=40:
        return 2
    elif x>40 and x<=60:
        return 3
    elif x>60 and x<=80:
        return 4
    else:
        return 5
d['age_group'] = d['AGE'].apply(func1)

#print(d.head(5))
#by_age = d.groupby('age_group')['label'].mean()
#print(by_age)    #不同年龄段的患癌症的概率20岁之前患癌症的概率为0,20-40岁患癌症概率wei
                 #0.020690     40-60 患癌症的概率为0.022947       60-80 患癌症的概率人数明显增加
                #可以看出患癌症的高峰期    80-100 患癌症的概率明显降低

#age_group
#(0, 20]           NaN
#(20, 40]     0.020690
#(40, 60]     0.022947
#(60, 80]     0.209275
#(80, 100]    0.094340


#by_age.plot(kind='bar')
#plt.show()
'''
smo = SMOTE(ratio={1:1350},random_state=42)  #采样的过程因为样本不平衡采样按照2:1的比例
X = d.iloc[:,:-1]
Y = d['label']
X_smo, y_smo = smo.fit_sample(X,Y)
#print(X_smo.shape)
x=pd.DataFrame(X_smo)
y=pd.DataFrame(y_smo)
df = pd.concat([x,y],axis=1)
df.columns=d.columns
print(df['label'].value_counts())
#df.to_excel('D:\\limingqi20190902\\test1.xlsx')'''
print(d.columns)
d.columns=['AGE', 'BODY_HEIGHT', 'BODY_WEIGHT',
           'Apolipoprotein AⅡ', 'Triglyceride_2', 'Apolipoprotein C2',
       'Apolipoprotein C3', 'Apolipoprotein E', 'Lecithin','Brain natriuretic peptide precursor',
           'Serum albumin', 'Alkaline phosphatase', 'Creatine kinase isoenzyme',
        'PSA（free）','PSA（total）', 'sodium',
           'calcium', 'chloride', 'Inorganic phosphorus',
           'Free calcium', 'Lactate dehydrogenase','Creatine kinase',
           'Creatinine', 'Serum uric acid','Troponin T', 'High density lipoprotein cholesterol',
           'Low density lipoprotein cholesterol', 'ApolipoproteinA1',
           'ApolipoproteinB','Potassium', 'label','age_group']
#print(d.columns)
d = d[(d['BODY_HEIGHT']<250)&(d['BODY_HEIGHT']>100)&(d['BODY_WEIGHT']>0)]    #  发现异常值过滤掉(d['BODY_HEIGHT']>100)
#d.to_excel('D:\\limingqi20190902\\analysis.xlsx')
#(d['PSA（free）']<5  (d['PSA（total）']<20)
#print(d.info())转换成英文的便于处理分析
#print(d['BODY_HEIGHT'].describe())
#height的分析
'''
plt.figure(figsize=(15, 5))
plt.subplot(121)
d['BODY_HEIGHT'].hist(bins=100)
plt.xlabel('BODY_HEIGHT')
plt.ylabel('Num')

plt.subplot(122)
d.boxplot(column='BODY_HEIGHT', showfliers=False)

facet = sns.FacetGrid(d,hue="label",aspect=4)
facet.map(sns.kdeplot,'BODY_HEIGHT',shade=True)
facet.set(xlim=(80,d['BODY_HEIGHT'].max()))
facet.add_legend()


fig,axis1 = plt.subplots(1,1,figsize=(100,4))
d['BODY_HEIGHT'] = d['BODY_HEIGHT'].astype(float)
average_height = d[["BODY_HEIGHT", "label"]].groupby(['BODY_HEIGHT'],as_index=False).mean()
sns.barplot(x='BODY_HEIGHT',y='label',data=average_height)
plt.xticks(rotation=70)



#wieght的分析
plt.figure(figsize=(15, 5))
plt.subplot(121)
d['BODY_WEIGHT'].hist(bins=100)
plt.xlabel('BODY_WEIGHT')
plt.ylabel('Num')

plt.subplot(122)
d.boxplot(column='BODY_WEIGHT', showfliers=False)

facet = sns.FacetGrid(d,hue="label",aspect=4)
facet.map(sns.kdeplot,'BODY_WEIGHT',shade=True)
facet.set(xlim=(0,d['BODY_WEIGHT'].max()))
facet.add_legend()


#cols = [ 'AGE', 'BODY_HEIGHT', 'BODY_WEIGHT']
#sns.pairplot(d[cols],height = 2)
#print(d[['Apolipoprotein AⅡ', 'riglyceride_2T', 'Apolipoprotein C2',
      # 'Apolipoprotein C3', 'Apolipoprotein E']].describe())

f, ax = plt.subplots(1,3,figsize=(20,6))
facet = sns.FacetGrid(d,hue="label",aspect=4)
facet.map(sns.kdeplot,'Apolipoprotein AⅡ',shade=True,ax=ax[0])
facet.map(sns.kdeplot,'Triglyceride_2',shade=True,ax=ax[1])
facet.map(sns.kdeplot,'Apolipoprotein C2',shade=True,ax=ax[2])
facet.add_legend()
print(d['Apolipoprotein AⅡ'].describe())
print(d['Triglyceride_2'].describe())
print(d['Apolipoprotein C2'].describe())
plt.show()

f, ax = plt.subplots(1,3,figsize=(20,6))
facet = sns.FacetGrid(d,hue="label",aspect=4)
facet.map(sns.kdeplot,'Apolipoprotein C3',shade=True,ax=ax[0])
facet.map(sns.kdeplot,'Apolipoprotein E',shade=True,ax=ax[1])
facet.map(sns.kdeplot,'Lecithin',shade=True,ax=ax[2])
facet.add_legend()
print(d['Apolipoprotein C3'].describe())
print(d['Apolipoprotein E'].describe())
print(d['Lecithin'].describe())
plt.show()


f, ax = plt.subplots(1,3,figsize=(20,6))#'Serum albumin', 'Alkaline phosphatase', 'Creatine kinase isoenzyme',
facet = sns.FacetGrid(d,hue="label",aspect=4)
facet.map(sns.kdeplot,'Serum albumin',shade=True,ax=ax[0])
facet.map(sns.kdeplot,'Alkaline phosphatase',shade=True,ax=ax[1])
facet.map(sns.kdeplot,'Creatine kinase isoenzyme',shade=True,ax=ax[2])
facet.add_legend()
print(d['Serum albumin'].describe())
print(d['Alkaline phosphatase'].describe())
print(d['Creatine kinase isoenzyme'].describe())
plt.show()


f, ax = plt.subplots(1,3,figsize=(20,6))
facet = sns.FacetGrid(d,hue="label",aspect=4)#'PSA（free）','PSA（total）', 'sodium',
facet.map(sns.kdeplot,'PSA（free）',shade=True,ax=ax[0])
facet.map(sns.kdeplot,'PSA（total）',shade=True,ax=ax[1])
facet.map(sns.kdeplot,'sodium',shade=True,ax=ax[2])
facet.add_legend()
print(d['PSA（free）'].describe())
print(d['PSA（total）'].describe())
print(d['sodium'].describe())
plt.show()


f, ax = plt.subplots(1,3,figsize=(20,6))
facet = sns.FacetGrid(d,hue="label",aspect=4)#'calcium', 'chloride', 'Inorganic phosphorus',
facet.map(sns.kdeplot,'calcium',shade=True,ax=ax[0])
facet.map(sns.kdeplot,'chloride',shade=True,ax=ax[1])
facet.map(sns.kdeplot,'Inorganic phosphorus',shade=True,ax=ax[2])
facet.add_legend()
print(d['calcium'].describe())
print(d['chloride'].describe())
print(d['Inorganic phosphorus'].describe())
plt.show()

f, ax = plt.subplots(1,3,figsize=(20,6))
facet = sns.FacetGrid(d,hue="label",aspect=4)#'Free calcium', 'Lactate dehydrogenase','Creatine kinase',
facet.map(sns.kdeplot,'Free calcium',shade=True,ax=ax[0])
facet.map(sns.kdeplot,'Lactate dehydrogenase',shade=True,ax=ax[1])
facet.map(sns.kdeplot,'Creatine kinase',shade=True,ax=ax[2])
facet.add_legend()
print(d['Free calcium'].describe())
print(d['Lactate dehydrogenase'].describe())
print(d['Creatine kinase'].describe())
plt.show()


f, ax = plt.subplots(1,3,figsize=(20,6))
facet = sns.FacetGrid(d,hue="label",aspect=4)# 'Creatinine', 'Serum uric acid', 'High density lipoprotein cholesterol',
facet.map(sns.kdeplot,'Creatinine',shade=True,ax=ax[0])
facet.map(sns.kdeplot,'Serum uric acid',shade=True,ax=ax[1])
facet.map(sns.kdeplot,'High density lipoprotein cholesterol',shade=True,ax=ax[2])
facet.add_legend()
print(d['Creatinine'].describe())
print(d['Serum uric acid'].describe())
print(d['High density lipoprotein cholesterol'].describe())
plt.show()


f, ax = plt.subplots(1,4,figsize=(20,6))
facet = sns.FacetGrid(d,hue="label",aspect=4)# 最后四个指标
facet.map(sns.kdeplot,'Low density lipoprotein cholesterol',shade=True,ax=ax[0])
facet.map(sns.kdeplot,'ApolipoproteinA1',shade=True,ax=ax[1])
facet.map(sns.kdeplot,'ApolipoproteinB',shade=True,ax=ax[2])
facet.map(sns.kdeplot,'Potassium',shade=True,ax=ax[3])
facet.add_legend()
print(d['Low density lipoprotein cholesterol'].describe())
print(d['ApolipoproteinA1'].describe())
print(d['Potassium'].describe())
plt.show()
#print(d['PSA（free）'].describe())

#print(d['BODY_WEIGHT'].describe())
'''
'''
fig, ax = plt.subplots(1,3, figsize=(18, 5))
ax[0].set_yticks(range(0, 60, 10))
sns.violinplot("age_group", "Apolipoprotein AⅡ", hue="label", data=d, split=True,ax=ax[0])
ax[0].set_title('age and Apolipoprotein AⅡ vs label')

ax[1].set_yticks(range(0, 21, 1))
sns.violinplot("age_group", "Triglyceride_2", hue="label", data=d, split=True,ax=ax[1])
ax[1].set_title('age and Triglyceride_2 vs label')

ax[2].set_yticks(range(0, 44, 5))
sns.violinplot("age_group", "Apolipoprotein C2", hue="label", data=d, split=True,ax=ax[2])
ax[2].set_title('age and Apolipoprotein C2 vs label')

fig, ax = plt.subplots(1,3, figsize=(18, 5))  #'Apolipoprotein C3', 'Apolipoprotein E', 'Lecithin',
ax[0].set_yticks(range(0, 140, 10))
sns.violinplot("age_group", "Apolipoprotein C3", hue="label", data=d, split=True,ax=ax[0])
ax[0].set_title('age and Apolipoprotein C3ein AⅡ vs label')

ax[1].set_yticks(range(0, 30, 5))
sns.violinplot("age_group", "Apolipoprotein E", hue="label", data=d, split=True,ax=ax[1])
ax[1].set_title('age and Apolipoprotein E vs label')

ax[2].set_yticks(range(0, 12, 5))
sns.violinplot("age_group", "Lecithin", hue="label", data=d, split=True,ax=ax[2])
ax[2].set_title('age and Lecithin vs label')


fig, ax = plt.subplots(1,3, figsize=(18, 5))#'Serum albumin', 'Alkaline phosphatase', 'Creatine kinase isoenzyme',
ax[0].set_yticks(range(0, 60, 10))
sns.violinplot("age_group", "Serum albumin", hue="label", data=d, split=True,ax=ax[0])
ax[0].set_title('age and Serum albumin vs label')

ax[1].set_yticks(range(0, 570, 20))
sns.violinplot("age_group", "Alkaline phosphatase", hue="label", data=d, split=True,ax=ax[1])
ax[1].set_title('age and Alkaline phosphatase vs label')

ax[2].set_yticks(range(0, 170, 10))
sns.violinplot("age_group", "Creatine kinase isoenzyme", hue="label", data=d, split=True,ax=ax[2])
ax[2].set_title('age and Creatine kinase isoenzyme vs label')


fig, ax = plt.subplots(1,3, figsize=(18, 5))#'PSA（free）','PSA（total）', 'sodium'
ax[0].set_yticks(range(0, 10, 1))
sns.violinplot("age_group", "PSA（free）", hue="label", data=d, split=True,ax=ax[0])
ax[0].set_title('age and PSA（free） vs label')

ax[1].set_yticks(range(0, 30, 10))
sns.violinplot("age_group", "PSA（total）", hue="label", data=d, split=True,ax=ax[1])
ax[1].set_title('age and PSA（total） vs label')

ax[2].set_yticks(range(120, 152, 5))
sns.violinplot("age_group", "sodium", hue="label", data=d, split=True,ax=ax[2])
ax[2].set_title('age and sodium vs label')


fig, ax = plt.subplots(1,3, figsize=(18, 5))#'calcium', 'chloride', 'Inorganic phosphorus',
ax[0].set_yticks(range(0, 5, 1))
sns.violinplot("age_group", "calcium", hue="label", data=d, split=True,ax=ax[0])
ax[0].set_title('age and calcium vs label')

ax[1].set_yticks(range(80, 115, 2))
sns.violinplot("age_group", "chloride", hue="label", data=d, split=True,ax=ax[1])
ax[1].set_title('age and chloride vs label')

ax[2].set_yticks(range(0, 5, 1))
sns.violinplot("age_group", "Inorganic phosphorus", hue="label", data=d, split=True,ax=ax[2])
ax[2].set_title('age and Inorganic phosphorus vs label')


fig, ax = plt.subplots(1,3, figsize=(18, 5))#'Free calcium', 'Lactate dehydrogenase','Creatine kinase',
ax[0].set_yticks(range(0, 2, 1))
sns.violinplot("age_group", "Free calcium", hue="label", data=d, split=True,ax=ax[0])
ax[0].set_title('age and Free calcium vs label')

ax[1].set_yticks(range(60, 500, 20))
sns.violinplot("age_group", "Lactate dehydrogenase", hue="label", data=d, split=True,ax=ax[1])
ax[1].set_title('age and Lactate dehydrogenase vs label')

ax[2].set_yticks(range(9, 4400, 120))
sns.violinplot("age_group", "Creatine kinase", hue="label", data=d, split=True,ax=ax[2])
ax[2].set_title('age and Creatine kinase vs label')


fig, ax = plt.subplots(1,3, figsize=(18, 5))#'Creatinine', 'Serum uric acid', 'High density lipoprotein cholesterol'
ax[0].set_yticks(range(0, 940, 100))
sns.violinplot("age_group", "Creatinine", hue="label", data=d, split=True,ax=ax[0])
ax[0].set_title('age and Creatinine vs label')

ax[1].set_yticks(range(100, 900, 100))
sns.violinplot("age_group", "Serum uric acid", hue="label", data=d, split=True,ax=ax[1])
ax[1].set_title('age and Serum uric acid vs label')

ax[2].set_yticks(range(0, 3, 1))
sns.violinplot("age_group", "High density lipoprotein cholesterol", hue="label", data=d, split=True,ax=ax[2])
ax[2].set_title('age and High density lipoprotein cholesterol vs label')


fig, ax = plt.subplots(1,3, figsize=(18, 5))#'Low density lipoprotein cholesterol', 'ApolipoproteinA1',
ax[0].set_yticks(range(0, 10, 1))   #'ApolipoproteinB','Potassium'
sns.violinplot("age_group", "Low density lipoprotein cholesterol", hue="label", data=d, split=True,ax=ax[0])
ax[0].set_title('age and Low density lipoprotein cholesterol vs label')

ax[1].set_yticks(range(0, 10, 1))
sns.violinplot("age_group", "ApolipoproteinA1", hue="label", data=d, split=True,ax=ax[1])
ax[1].set_title('age and ApolipoproteinA1 vs label')

ax[2].set_yticks(range(0, 10, 1))
sns.violinplot("age_group", "ApolipoproteinB", hue="label", data=d, split=True,ax=ax[2])
ax[2].set_title('age and ApolipoproteinB vs label')


fig, ax = plt.subplots(1,1, figsize=(18, 5))#'Low density lipoprotein cholesterol', 'ApolipoproteinA1',
ax.set_yticks(range(0, 10, 1))   #'ApolipoproteinB','Potassium'
sns.violinplot("age_group", "Potassium", hue="label", data=d, split=True)
ax.set_title('age and Potassium vs label')

plt.show()


corr = d.corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, linewidths=0.05,vmax=1, vmin=0 ,annot=True,annot_kws={'size':6,'weight':'bold'})
#plt.show()
#print(d.shape)
#d = d.dropna()
#print(d.shape)
print(d.head(5))


X = d.iloc[:,0:28]
#X =pd.concat([d.iloc[:,0:28],d.iloc[:,29:]],axis=1)
#d.iloc[:,29:]
#print(X.shape)
Y = d.iloc[:,28]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
std = StandardScaler()
x_train = std.fit_transform(X_train)
x_test = std.transform(X_test)
 # 逻辑回归
rf = RandomForestClassifier()
rf.fit(x_train,Y_train)
#print(lg.coef_)
y_predict = rf.predict(x_test)
print("准确率：", rf.score(x_test,Y_test))
print(accuracy_score((y_predict,Y_test)))
print("11111111111111111111111")
conf_mat = confusion_matrix(y_predict,Y_test)
print(conf_mat)
print("召回率：", classification_report(Y_test, y_predict,target_names=["正常", "患病"]))
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
#K-Means聚类分割灰度图像
#读取原始图像灰度颜色
img = cv2.imread('F:\\pythoncode\\lianxi\\tupian\\kmeans.png',0)

#获取图像高度、宽度
rows, cols = img.shape[:]

#图像二维像素转换为一维
data = img.reshape((rows * cols, 1))
data = np.float32(data)

#定义中心 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

#设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

#K-Means聚类 聚集成4类
compactness, labels4, centers4 = cv2.kmeans(data, 4, None, criteria, 10, flags)
compactness, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10, flags)
compactness, labels6, centers6 = cv2.kmeans(data, 6, None, criteria, 10, flags)
compactness, labels8, centers8 = cv2.kmeans(data, 8, None, criteria, 10, flags)
compactness, labels10, centers10 = cv2.kmeans(data, 10, None, criteria, 10, flags)
#生成最终图像
dst4 = labels4.reshape((img.shape[0], img.shape[1]))
dst2 = labels2.reshape((img.shape[0], img.shape[1]))
dst6 = labels6.reshape((img.shape[0], img.shape[1]))
dst8 = labels8.reshape((img.shape[0], img.shape[1]))
dst10 = labels10.reshape((img.shape[0], img.shape[1]))

#dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
#dst4 = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)
#dst6 = cv2.cvtColor(dst6, cv2.COLOR_BGR2RGB)
#dst8 = cv2.cvtColor(dst8, cv2.COLOR_BGR2RGB)
#dst10 = cv2.cvtColor(dst10, cv2.COLOR_BGR2RGB)
#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

#显示图像
titles = [u'原始图像', u'聚类4',u'聚类2',u'聚类6',u'聚类8',u'聚类10']
images = [img, dst2,dst4,dst6,dst8,dst10]
for i in range(6):
   plt.subplot(2,3,i+1), plt.imshow(images[i], 'gray'),
   plt.title(titles[i])
   plt.xticks([]),plt.yticks([])
plt.show()

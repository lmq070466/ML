import numpy as np
import cv2
#OpenCV的k - means聚类 -对图片进行颜色量化
img = cv2.imread('F:\\pythoncode\\lianxi\\tupian\\junheng.png')
Z = img.reshape((-1, 3))
# convert to np.float32
Z = np.float32(Z)
j = 0
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
Klist = [2, 4, 6, 8, 10]
for i in Klist:
    ret, label, center = cv2.kmeans(Z, i, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    j += 2
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    cv2.imshow(str(("spaceship K=", i)), res2)
    cv2.waitKey(0)
cv2.imshow('quondam image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


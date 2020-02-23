
import cv2
import numpy as np
from matplotlib import pyplot as plt

img_file = 'F:\\pythoncode\\lianxi\\tupian\\clean.png'     #读取图片
img = cv2.imread(img_file)
#cv2.imshow("Origin",img)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#cv2.imshow("HSV",img_hsv)
# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img_hsv)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))
img_h = result[..., 0]
img_s = result[..., 1]
img_v = result[..., 2]


def calcAndDrawHist(image, color):
    hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    histImg = np.zeros([256, 256, 3], np.uint8)
    hpt = int(0.9 * 256);

    for h in range(256):
        intensity = int(hist[h] * hpt / maxVal)
        cv2.line(histImg, (h, 256), (h, 256 - intensity), color)
    return histImg

histImgH = calcAndDrawHist(img_h, [255, 0, 0])
histImgS = calcAndDrawHist(img_s, [0, 255, 0])
histImgV = calcAndDrawHist(img_v, [0, 0, 255])

cv2.imshow("histImgH", histImgH)
cv2.imshow("histImgS", histImgS)
cv2.imshow("histImgV", histImgV)
cv2.imshow("Img", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("Histogram equalization", result)
cv2.imwrite('F:\\pythoncode\\lianxi\\tupian\\junheng.jpg', result)

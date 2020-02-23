
import cv2
import numpy as np
from matplotlib import pyplot as plt

img_file = 'F:\\pythoncode\\lianxi\\tupian\\clean.png'     #读取图片
img = cv2.imread(img_file)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

img_h = img_hsv[..., 0]
img_s = img_hsv[..., 1]
img_v = img_hsv[..., 2]
'''
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
cv2.imshow("Img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
fig = plt.gcf()                      # 分通道显示图片
fig.set_size_inches(10, 15)

plt.subplot(221)
plt.imshow(img_hsv)
plt.axis('off')
plt.title('HSV')

plt.subplot(222)
plt.imshow(img_h, cmap='gray')
plt.axis('off')
plt.title('H')

plt.subplot(223)
plt.imshow(img_s, cmap='gray')
plt.axis('off')
plt.title('S')

plt.subplot(224)
plt.imshow(img_v, cmap='gray')
plt.axis('off')
plt.title('V')

plt.show()

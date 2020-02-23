import cv2
import numpy as np
from matplotlib import pyplot as plt

img_file = 'F:\\pythoncode\\lianxi\\tupian\\9.jpg'     #读取图片
img = cv2.imread(img_file)

img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
img_ls = img_lab[..., 0]
img_as = img_lab[..., 1]
img_bs = img_lab[..., 2]

# 分通道显示图片
fig = plt.gcf()
fig.set_size_inches(10, 15)

plt.subplot(221)
plt.imshow(img_lab)
plt.axis('off')
plt.title('L*a*b*')

plt.subplot(222)
plt.imshow(img_ls, cmap='gray')
plt.axis('off')
plt.title('L*')

plt.subplot(223)
plt.imshow(img_as, cmap='gray')
plt.axis('off')
plt.title('a*')

plt.subplot(224)
plt.imshow(img_bs, cmap='gray')
plt.axis('off')
plt.title('b*')

plt.show()

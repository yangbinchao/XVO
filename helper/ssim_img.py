'''
# Author: yangbinchao
# Date:   2021-11-27
# Email:  heroybc@qq.com
# Describe: 对比两张图的结构相似度，给出误差图和ssim值，同时进行边缘检测
'''


import cv2
import imutils
from skimage.metrics import structural_similarity
import time
from skimage import filters, img_as_ubyte
import numpy as np

start = time.time()
# 读入图像，转为灰度图像
src = cv2.imread('../test_img/1.jpg')
img = cv2.imread('../test_img/2.jpg')

grayA = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 计算两个灰度图像之间的结构相似度
(score, diff) = structural_similarity(grayA, grayB, win_size=101, full=True)
diff = (diff * 255).astype("uint8")
cv2.namedWindow("diff", cv2.WINDOW_NORMAL)
cv2.imshow("diff", diff)
print("SSIM:{}".format(score))
cv2.imwrite('../test_img/1_2_diff.jpg',diff)
# 找到不同的轮廓以致于可以在表示为 '不同'的区域放置矩形
# 全局自适应阈值分割（二值化），返回值有两个，第一个是阈值，第二个是二值图像
dst = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cv2.namedWindow("threshold", cv2.WINDOW_NORMAL)
cv2.imshow('threshold', dst)
# findContours找轮廓，返回值有两个，第一个是轮廓信息，第二个是轮廓的层次信息（“树”状拓扑结构）
# cv2.RETR_EXTERNAL：只检测最外层轮廓
# cv2.CHAIN_APPROX_SIMPLE：压缩水平方向、垂直方向和对角线方向的元素，保留该方向的终点坐标，如矩形的轮廓可用4个角点表示
contours, hierarchy = cv2.findContours(dst.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#print(contours)
newimg = np.zeros(dst.shape, np.uint8)  # 定义一个和图像分割处理后相同大小的黑色图
# drawContours画轮廓，将找到的轮廓信息画出来
cv2.drawContours(newimg, contours, -1, (255, 255, 255), 1)
cv2.namedWindow("contours", cv2.WINDOW_NORMAL)
cv2.imshow('contours', newimg)
# cnts = cnts[0] if imutils.is_cv3() else cnts[0]    取findContours函数的第一个返回值，即取轮廓信息

# 找到一系列区域，在区域周围放置矩形
for c in contours:
    (x, y, w, h) = cv2.boundingRect(c)  # boundingRect函数：计算轮廓的垂直边界最小矩形，矩形是与图像上下边界平行的
    cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)  # rectangle函数：使用对角线的两点pt1，pt2画一个矩形轮廓
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 画矩形的图, pt1, pt2,(对角线两点的坐标), 矩形边框的颜色，矩形边框的粗细

end = time.time()
print(end - start)
# 用cv2.imshow 展现最终对比之后的图片
cv2.namedWindow("right", cv2.WINDOW_NORMAL)
cv2.imshow('right', src)
cv2.namedWindow("left", cv2.WINDOW_NORMAL)
cv2.imshow('left', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


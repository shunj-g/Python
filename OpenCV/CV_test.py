#导入cv模块
import cv2 as cv
#导入numpy
import numpy as np
img = np.mat(np.zeros((300,300)))
cv.imshow("test",img)
cv.waitKey(0)

#读取图像，支持 bmp、jpg、png、tiff 等常用格式
#img = cv.imread("im.jpg")
#创建窗口并显示图像
#cv.namedWindow("Image")
#cv.imshow("Image",img)
#cv.waitKey(0)
#释放窗口
#cv.destroyAllWindows()
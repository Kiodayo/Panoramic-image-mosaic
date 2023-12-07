from Stitcher import Stitcher
import cv2
import numpy as np

# 读取拼接图片
imageA = cv2.imread(r'D:\Panoramic-image-mosaic-main\image1-1.jpg')
imageB = cv2.imread(r'D:\Panoramic-image-mosaic-main\image1-2.jpg')
imageA= cv2.resize(imageA, (750,900), fx=0.5, fy=0.5)
imageB= cv2.resize(imageB,(750,900), fx=0.5, fy=0.5)
# 把图片拼接成全景图
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
result = cv2.resize(result,(720,720),interpolation=cv2.INTER_LANCZOS4)
# #局部滤波
# # 选择区域
# x,y,w,h = 330,0,380,720
# roi = result[y:y+h, x: x+w]
# # 应用滤波器
# filtered_roi = cv2.bilateralFilter(roi,0,40,20)
#
# result[y:y+h, x: x+w] = filtered_roi

# # 四周填充黑色像素，再得到阈值图
# result = cv2.copyMakeBorder(
#     result, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
#
# gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 128, 255,cv2.THRESH_BINARY)
# cv2.imshow('gray',thresh)
# print(gray)
# cnts, hierarchy = cv2.findContours(
#     thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnt = max(cnts, key=cv2.contourArea)
#
# mask = np.zeros(thresh.shape, dtype="uint8")
# x, y, w, h = cv2.boundingRect(cnt)
# cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
#
# minRect = mask.copy()
# sub = mask.copy()
#
# # 开始while循环，直到sub中不再有前景像素
# while cv2.countNonZero(sub) > 0:
#     minRect = cv2.erode(minRect, None)
#     sub = cv2.subtract(minRect, thresh)
#
# cnts, hierarchy = cv2.findContours(
#     minRect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnt = max(cnts, key=cv2.contourArea)
# x, y, w, h = cv2.boundingRect(cnt)
#
# # 使用边界框坐标提取最终的全景图
# result = result[y:y + h, x:x + w]

# # 显示所有图片
# cv2.imshow("Image A", imageA)
# cv2.imshow("Image B", imageB)
# cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result PINTIE", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('images/final'+str(15)+'.jpg', result)

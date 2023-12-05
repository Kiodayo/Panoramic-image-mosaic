from Stitcher import Stitcher
import cv2

# 读取拼接图片
imageA = cv2.imread(r'C:\Users\DELL\Desktop\Panoramic-image-mosaic-main\Panoramic-image-mosaic-main\image1-3.jpg')
imageB = cv2.imread(r'C:\Users\DELL\Desktop\Panoramic-image-mosaic-main\Panoramic-image-mosaic-main\image1-4.jpg')
imageA= cv2.resize(imageA,(480,1020))
imageB= cv2.resize(imageB,(480,1020))
# 把图片拼接成全景图
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

# 显示所有图片
# cv2.imshow("Image A", imageA)
# cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

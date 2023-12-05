import cv2
import numpy as np


def get_homo(img1, img2):
    # 1.创建特征转换对象 这里用的是sift 因为sift虽然慢 但是精确度高 效果较好
    sift = cv2.SIFT_create()
    # 2.通过特征转换对象获得特征点和描述子
    k1, d1 = sift.detectAndCompute(img1, mask=None)
    k2, d2 = sift.detectAndCompute(img2, mask=None)
    # 3.创建特征匹配器 暴力特征匹配
    bf = cv2.BFMatcher()
    verify_matches = []
    # 4.进行特征匹配 K=2
    matches = bf.knnMatch(d1, d2, 2)
    verify_ratio = 0.8
    for m1, m2 in matches:
        if m1.distance < 0.8 * m2.distance:
            verify_matches.append(m1)

    min_matches = 8
    if len(verify_matches) > min_matches:
        img1_pts = []
        img2_pts = []
        for m in verify_matches:
            img1_pts.append(k1[m.queryIdx].pt)
            img2_pts.append(k2[m.queryIdx].pt)
            # [(x1,y1),(x2,y2)……]
        img1_pts = np.float_(img1_pts).reshape(-1, 1, 2)
        img2_pts = np.float_(img2_pts).reshape(-1, 1, 2)
        # 5.过滤特征找出有效的特征匹配点
        H, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
        return H
    else:
        print('erro: not enough matches')




def stitch_image(img1,img2,H):
    #获得每张图片的四个角点 获得原始图像的宽高
    h1,w1 = img1.shape[0:2]
    h2,w2 = img2.shape[0:2]
    img1_dims = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    img2_dims = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    #对图片进行变换 单应性矩阵使图进行旋转与平移 方便下一步进行拼贴
    img1_transform = cv2.perspectiveTransform(img1_dims,H)
    # print(img1_dims)
    # print(img1_transform)
    # 计算大图的范围 使用差值的最大值来做
    result_dims = np.concatenate((img2_dims,img1_transform),axis=0)
    #通过ravel将xy的二维值变成一维的
    [x_min,y_min] = np.int32(result_dims.min(axis=0).ravel()-0.5)
    [x_max,y_max] = np.int32(result_dims.max(axis=0).ravel()+0.5)
    #平移的距离
    transform_dist = [-x_min, -y_min]
    # 乘以齐次坐标 完成拼贴
    # [1,0,dx]
    # [0,1,dy]
    # [0,0,1]
    transform_array = np.array([[1,0,transform_dist[0]],
                                [0,1,transform_dist[1]],
                                [0,0,1]])
    #变换
    result_image = cv2.warpPerspective(img1,H,(x_max-x_min,y_max-y_min))
    print(result_image)
    result_image[transform_dist[1]:transform_dist[1] + h2,
    transform_dist[0]:transform_dist[0] + w2] = img2
    # result_image[transform_dist[1]:transform_dist[1]+h2,
    #              transform_dist[0]:transform_dist[0]+w2] = img2

    return  result_image

# # 将图片A进行视角变换，result是变换后图片
#         result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
#         #self.cv_show('result', result)
#         # 将图片B传入result图片最左端
#         result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
#         #self.cv_show('result', result)



# 1.读取文件
img1 = cv2.imread(r'C:\Users\DELL\Desktop\Panoramic-image-mosaic-main\Panoramic-image-mosaic-main\image1-1.jpg')
img2 = cv2.imread(r'C:\Users\DELL\Desktop\Panoramic-image-mosaic-main\Panoramic-image-mosaic-main\image1-2.jpg')

# 2.将两张图片设置成同一大小640×480
img1 = cv2.resize(img1, (360, 680))
img2 = cv2.resize(img2, (360, 680))

inputs = np.hstack((img1, img2))

# 获得单应性矩阵
H = get_homo(img1, img2)
#进行图片拼贴
result_image = stitch_image(img1,img2,H)

cv2.imshow('input image', result_image)
cv2.waitKey()

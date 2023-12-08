import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def remove_the_blackborder(image):
    img = image
    # img = cv.medianBlur(image, 5) #中值滤波，去除黑色边际中可能含有的噪声干扰
    b = cv.threshold(img, 127, 255, cv.THRESH_BINARY)  # 调整裁剪效果
    binary_image = b[1]  # 二值图--具有三通道
    binary_image = cv.cvtColor(binary_image, cv.COLOR_BGR2GRAY)
    # ret,binary_image=cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # 显示图像
    # cv_show("binary_image",binary_image)

    edges_y, edges_x = np.where(binary_image == 255)  ##h, w
    bottom = min(edges_y)
    top = max(edges_y)
    height = top - bottom

    left = min(edges_x)
    right = max(edges_x)
    height = top - bottom
    width = right - left

    res_image = image[bottom:bottom + height, left:left + width]


    return res_image


if __name__ == '__main__':
    top, bot, left, right = 500, 1500, 0, 1500  # ！！！！！
    img1 = cv.imread('betterSIFT/1-1.jpg')
    img2 = cv.imread('betterSIFT/1-2.jpg')
    # img1=cv.resize(img1,(0,0),fx=0.5,fy=0.5)#按比例缩放图像A
    # img2=cv.resize(img2,(0,0),fx=0.5,fy=0.5)#按比例缩放图像A
    srcImg = cv.copyMakeBorder(img1, top, bot, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
    testImg = cv.copyMakeBorder(img2, top, bot, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
    img1gray = cv.cvtColor(srcImg, cv.COLOR_BGR2GRAY)
    img2gray = cv.cvtColor(testImg, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    # 检测A、B图片的SIFT关键特征点，并计算特征描述子
    kp1, des1 = sift.detectAndCompute(img1gray, None)
    kp2, des2 = sift.detectAndCompute(img2gray, None)
    # FLANN匹配算法
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # 指定递归次数
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)  # 建立快速近似最近邻搜索库

    matchesMask = [[0, 0] for i in range(len(matches))]
    # 准备一个空的掩膜来绘制好的匹配
    good = []
    pts1 = []
    pts2 = []
    # 向掩膜中添加数据
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)

    img1key = img1
    cv.drawKeypoints(img1, kp1, img1key, (255, 0, 0), flags=1)
    plt.imshow(img1key, ), plt.show()
    cv.imwrite('keypoint1.jpg', img1key)

    img2key = img2
    cv.drawKeypoints(img2, kp2, img2key, (255, 0, 0), flags=1)
    plt.imshow(img2key, ), plt.show()
    cv.imwrite('keypoint2.jpg', img2key)

    img3 = cv.drawMatchesKnn(img1gray, kp1, img2gray, kp2, matches, None, **draw_params)
    plt.imshow(img3, ), plt.show()
    cv.imwrite('images/matche.jpg', img3)
    rows, cols = srcImg.shape[:2]

    # 配准拼接（当合格掩膜数量大于10时）
    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        # 获取匹配对的点坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # 计算视角变换矩阵
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        # 对图2进行透视变换
        warpImg = cv.warpPerspective(testImg, np.array(M), (testImg.shape[1], testImg.shape[0]),
                                     flags=cv.WARP_INVERSE_MAP)
        plt.imshow(warpImg, ), plt.show()
        cv.imwrite('images/warpImg.jpg', warpImg)
        for col in range(0, cols):
            if srcImg[:, col].any() and warpImg[:, col].any():
                left = col
                break
        for col in range(cols - 1, 0, -1):
            if srcImg[:, col].any() and warpImg[:, col].any():
                right = col
                break

        res = np.zeros([rows, cols, 3], np.uint8)
        # 填充图像
        for row in range(0, rows):
            for col in range(0, cols):
                if not srcImg[row, col].any():
                    res[row, col] = warpImg[row, col]  # 如果没有图1，用变换后的图2的填充
                elif not warpImg[row, col].any():
                    res[row, col] = srcImg[row, col]  # 如果没有图2，用图1的填充
                else:
                    srcImgLen = float(abs(col - left))
                    testImgLen = float(abs(col - right))
                    alpha = srcImgLen / (srcImgLen + testImgLen)
                    res[row, col] = np.clip(srcImg[row, col] * (1 - alpha) + warpImg[row, col] * alpha, 0,
                                            255)  # 重合部分以加权叠加方式填充

        # opencv是bgr显示的, matplotlib是rgb显示的
        res = remove_the_blackborder(res)
        # cv.imwrite('images/result.jpg', res)
        res = cv.cvtColor(res, cv.COLOR_BGR2RGB)
        # result = remove_all_blackborder(res,img1)
        # result = cv.cvtColor(result, cv.COLOR_BGR2RGB)
        # 展示结果
        plt.figure()
        plt.imshow(res)

        plt.show()
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
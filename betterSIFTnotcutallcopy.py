import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os


def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# def remove_all_blackborder(image, img1):
#     # Convert to gray scale
#     gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#
#     # Find the topmost non-black pixel along the leftmost column
#     topmost = 0
#     for row in range(gray.shape[0]):
#         if gray[row, 0] != 0:  # Found a non-black pixel
#             topmost = row
#             break
#
#     # Define the size of the mask based on img1 and the width of the original image
#     w = image.shape[1]
#     h = img1.shape[0]
#
#     # Create a mask using the height of img1 and full width, so it has the same width as image
#     mask = np.zeros(image.shape[:2], dtype=np.uint8)
#
#     # Draw the white rectangle, ensuring not to exceed the image boundaries
#     # Note: If the calculated bottom exceeds the image height, take the max possible without exceeding
#     bottom_y = min(topmost + h, image.shape[0])
#
#     # Draw the rectangle
#     mask[topmost:bottom_y, 0:w] = 255
#
#     # Apply the mask to the image with bitwise_and
#     result = cv.bitwise_and(image, image, mask=mask)
#
#
#     contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#     if contours:
#         cnt = contours[0]
#         x, y, w, h = cv.boundingRect(cnt)
#         # print("Bounding box coordinates (x, y, w, h):", x, y, w, h)
#     else:
#         print("No contours found.")
#
#     result = result[y:y + h, x:x + w]
#
#     # Return the result
#     return result


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


def stitch_images_not_cut(image_path1, image_path2):
    # The given padding values
    top, bot, left, right = 500, 1500, 0, 1500

    # Reading the images
    img1 = cv.imread(image_path1)
    img2 = cv.imread(image_path2)

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
        # cv.imwrite('images/warpImg.jpg', warpImg)
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
        res = cv.cvtColor(res, cv.COLOR_BGR2RGB)
        # result = remove_all_blackborder(res, img1)

        return res


# Example of running the function directly
# if __name__ == '__main__':
#     result_image = stitch_images('images/image1-1.jpg', 'images/image1-2.jpg')
#     if result_image is not None:
#         plt.figure()
#         plt.imshow(result_image)
#         plt.show()
#         # 如果您还想保存拼接后的图像，可以取消注释以下行：
#         cv.imwrite('stitched_image2.jpg', cv.cvtColor(result_image, cv.COLOR_RGB2BGR))
#         print("Stitched image is displayed.")
#     else:
#         print("Unable to stitch images due to insufficient matches.")


img_dir = 'input'
names = os.listdir(img_dir)
images = []
for name in names:
    img_path = os.path.join(img_dir, name)
    image = cv.imread(img_path)
    if image is None:
        print(f"读取图片失败: {name}")
        continue
    # print(img_path)
    images.append(img_path)

def run(output_num):
    stitched_img = stitch_images_not_cut('images[0]', 'images[1]')
    stitched_img = cv.cvtColor(stitched_img, cv.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(stitched_img)
    plt.show()
    cv.imwrite('outputs/final' + str(output_num) + '.jpg', stitched_img)



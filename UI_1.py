import tkinter as tk
from tkinter import filedialog

import cv2
from PIL import Image, ImageTk

import main
import runpy


class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Viewer")
        self.root.geometry("1000x800")  # 设置窗口的大小

        # 图片框架，用于显示图片
        self.frame_images = tk.Frame(self.root)
        self.frame_images.pack(fill=tk.BOTH, expand=True)

        # 用于显示图像一和图像二的标签
        self.label_image1 = tk.Label(self.frame_images)
        self.label_image1.pack(side=tk.LEFT, padx=10)

        self.label_image2 = tk.Label(self.frame_images)
        self.label_image2.pack(side=tk.LEFT, padx=10)

        self.label_image3 = tk.Label(self.frame_images)
        self.label_image3.pack(side=tk.LEFT, padx=10)

        # 图片对象，防止被垃圾收集器回收
        self.photo_image1 = None
        self.photo_image2 = None
        self.photo_image3 = None

        # 按钮框架，用于放置按钮
        self.frame_buttons = tk.Frame(self.root)
        self.frame_buttons.pack(fill=tk.X)

        # 加载第一张图片的按钮
        self.btn_load_image1 = tk.Button(
            self.frame_buttons,
            text="第一张图片",
            command=lambda: self.load_image(1))
        self.btn_load_image1.pack(side=tk.LEFT, anchor=tk.SW)

        # 加载第二张图片的按钮
        self.btn_load_image2 = tk.Button(
            self.frame_buttons,
            text="第二张图片",
            command=lambda: self.load_image(2))
        self.btn_load_image2.pack(side=tk.BOTTOM, anchor=tk.S)

        # 全景图片拼接的按钮
        self.btn_merge_images = tk.Button(
            self.frame_buttons,
            text="全景图片拼接",
            command=self.merge_images)
        self.btn_merge_images.pack(side=tk.RIGHT, anchor=tk.SE)

        # 重置输入
        self.reset_button = tk.Button(
            self.frame_buttons,
            text="重置",
            command=lambda: self.image_input_reset())
        self.reset_button.pack(side=tk.TOP)

    # 加载图片

    def load_image(self, image_num):
        file_path = filedialog.askopenfilename()
        print(file_path)
        if file_path:
            # 加载图片并调整大小
            image = Image.open(file_path)

            # 导入到图像库中
            image_read = cv2.imread(file_path)
            main.images.append(image_read)

            image = self.resize_image(image, 0.2)
            photo = ImageTk.PhotoImage(image, 0.2)

            # 显示图片
            if image_num == 1:
                self.label_image1.config(image=photo)
                self.label_image1.image = photo  # 保存引用
                self.photo_image1 = photo  # 引用保存在实例变量中
            elif image_num == 2:
                self.label_image2.config(image=photo)
                self.label_image2.image = photo  # 保存引用
                self.photo_image2 = photo  # 引用保存在实例变量中

    # 调整图片大小
    def resize_image(self, image, scale):
        # 按窗口大小比例缩放图片（40%大小）
        base_width = int(self.root.winfo_width() * scale)

        # 计算缩放比例
        w_percent = base_width / float(image.size[0])
        h_size = int(float(image.size[1]) * float(w_percent))

        return image.resize((base_width, h_size), Image.BICUBIC)

    # 全景图片拼接处理
    def merge_images(self):
        # 此处的实现将根据你的具体需求而定
        # 确保两张图片都已被加载
        if self.photo_image1 and self.photo_image2:
            # 假设 final.jpg 存在于当前文件夹中
            merged_image_path = "final.jpg"
            image = Image.open(merged_image_path)
            image = self.resize_image(image, 0.5)
            photo = ImageTk.PhotoImage(image)

            # 更新标签以显示图片
            self.label_image3.config(image=photo)
            self.label_image3.image = photo
            # 重设图片路径
            self.image_input_reset()

    # 重置输入

    def image_input_reset(self):
        self.photo_image1 = None
        self.photo_image2 = None
        # 清空images库，以便于第二次拼接
        main.images = []


# 创建主窗口
root = tk.Tk()
app = ImageApp(root)
root.mainloop()

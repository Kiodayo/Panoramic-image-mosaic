import random
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
        self.has_output = False

        # 图片框架，用于显示图片
        self.frame_images = tk.Frame(self.root)
        self.frame_images.pack(fill=tk.BOTH, expand=True)

        # 用于显示图像一和图像二的标签
        self.label_image = []
        self.output_image_label = tk.Label(self.frame_images)

        # 图片对象，防止被垃圾收集器回收
        self.photo_image = []

        # 按钮框架，用于放置按钮
        self.frame_buttons = tk.Frame(self.root)
        self.frame_buttons.pack(fill=tk.X)

        # 加载图片的按钮
        self.btn_load_image1 = tk.Button(
            self.frame_buttons,
            text="添加图片",
            command=lambda: self.load_image())
        self.btn_load_image1.pack(side=tk.LEFT, anchor=tk.SW)

        # 全景图片拼接的按钮
        self.btn_merge_images = tk.Button(
            self.frame_buttons,
            text="全景图片拼接",
            command=lambda: self.merge_images())
        self.btn_merge_images.pack(side=tk.RIGHT, anchor=tk.SW)

        # 重置输入
        self.reset_button = tk.Button(
            self.frame_buttons,
            text="重置",
            command=lambda: self.image_input_reset())
        self.reset_button.pack(side=tk.TOP)

    # 加载图片

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if self.has_output:
            self.image_output_reset()
            self.image_input_reset()

        if file_path:
            image = image = Image.open(file_path)
            photo = ImageTk.PhotoImage(image)

            # 读取图片列表
            image_read = cv2.imread(file_path)
            main.images.append(image_read)

            # 重置输出显示
            self.image_output_reset()

            # 创建并添加新图片的标签
            self.label_image.append(tk.Label(self.frame_images))
            self.label_image[-1].pack(side=tk.LEFT, padx=5, pady=5)
            self.label_image[-1].config(image=photo)

            # 添加相应的图片对象占位符
            self.photo_image.append(photo)

            # 调整所有图片大小以适应界面
            self.resize_images()

    def resize_images(self):
        total_images = len(self.label_image)
        # print(total_images)
        # 根据想要摆放的图片数量可以改变缩放因子
        scale = 0.3 if total_images < 4 else ( 0.9 / total_images)
        print(scale)
        for index, label in enumerate(self.label_image):
            image = main.images[index]
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image = self.resize_image(image, scale)
            photo = ImageTk.PhotoImage(image)

            self.photo_image[index] = photo  # 更新引用

            label.config(image=photo)
            label.image = photo  # 保留引用

    # 调整图片大小
    def resize_image(self, image, scale):
        # 获取当前宽度用于缩放
        base_width = int(self.root.winfo_width() * scale)
        w_percent = (base_width / float(image.size[0]))
        h_size = int((float(image.size[1]) * float(w_percent)))

        return image.resize((base_width, h_size), Image.BICUBIC)

    # 全景图片拼接处理

    def merge_images(self):
        # 此处的实现将根据你的具体需求而定
        # 确保两张图片都已被加载
        if self.photo_image:
            index = random.randint(1, 10000)
            print(index)
            main.run(index)
            try:
                merged_image_path = 'outputs/final' + str(index) + '.jpg'
                image = Image.open(merged_image_path)
                image = self.resize_image(image, 0.5)
                photo = ImageTk.PhotoImage(image)


                self.image_input_reset()

                self.photo_image.append(photo)

                self.output_image_label.pack(
                    side=tk.LEFT, padx=5, anchor='center')
                self.output_image_label.config(image=photo)

                self.has_output = True

                self.display_messageSuccess()

            except FileNotFoundError as ex:
                print("图片不符合拼接要求")
                self.display_messageError()

    # 重置输入

    def image_input_reset(self):
        for label in self.label_image:
            print("delete")
            label.config(image='')
            label.image = None
            label.pack_forget()
        self.photo_image = []
        # 清空images库，以便于第二次拼接
        self.label_image = []
        main.images = []

    def image_output_reset(self):
        self.output_image_label.pack_forget()
        self.has_output = False

    def display_messageError(self):
        tk.messagebox.showerror(title='出现错误',
                                message='您选择的图片不适合全景拼接')  # 消息提醒弹窗，点击确定返回值为 ok

    def display_messageSuccess(self):
        tk.messagebox.showinfo(title='成功！',
                                message='已生成全景拼接后的图像')  # 消息提醒弹窗，点击确定返回值为 ok


# 创建主窗口
root = tk.Tk()
app = ImageApp(root)
root.mainloop()

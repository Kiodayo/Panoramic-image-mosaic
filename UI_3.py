import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tkinter.font as tkFont
import tkutils as tku



def load_images(label_images, frame_images,photo_images):
    filenames = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.gif *.png")])
    # for widget in frame_images.winfo_children():
    #     widget.destroy()
    for filename in filenames:
        img = Image.open(filename)
        print(filename)
        img.thumbnail((300, 300))  # Resize the image to 100x100
        photo = ImageTk.PhotoImage(img)
        photo_images.append(photo)


        label_images.append(tk.Label(frame_images,image=photo))
        label_images[-1].pack(side=tk.LEFT, padx=10)








def reset_images():
    for widget in frame_images.winfo_children():
        widget.destroy()


def merge_image():
    # Select a single image to output
    first_page.pack_forget()
    second_page.pack_forget()
    third_page.pack(fill='both', expand=True)



def show_second_page():
    first_page.pack_forget()
    second_page.pack(fill='both', expand=True)


def show_first_page():
    second_page.pack_forget()
    first_page.pack(fill='both', expand=True)


# Create the main window
window = tk.Tk()
window.title("Image UI")
window.geometry("1000x800")

# First page with background image
first_page = tk.Frame(window)
background_image = Image.open("background/background.jpg")  # Replace with your image path
background_photo = ImageTk.PhotoImage(background_image)
background_label = tk.Label(first_page, image=background_photo)
background_label.place(relwidth=1, relheight=1)
button_frame = tk.Frame(first_page)
button_frame.pack(side="right", padx=10, pady=5)  # Use a frame to hold buttons

# Adjust the pack parameters to stack the buttons vertically
button1 = tk.Button(button_frame, text="基于SIFT实现的两图拼接", command=show_second_page)
button1.pack(side="top", pady=(0, 0), expand=tk.NO, anchor=tk.CENTER, padx=5)
button2 = tk.Button(button_frame, text="基于ORB实现的图拼接  ", command=show_second_page)
button2.pack(side="bottom", pady=(10, 0), expand=tk.NO, anchor=tk.CENTER, padx=5)


# Second page
window.title("加载图片")
second_page = tk.Frame(window)

# Frame for image display
frame_images = tk.Frame(second_page)
frame_images.pack(fill=tk.BOTH, expand=True)

label_images = []
photo_images = []

# Buttons on the second page
frame_buttons = tk.Frame(second_page)
frame_buttons.pack(side="bottom", anchor="se", pady=5)
load_button = tk.Button(frame_buttons, text="载入图片", command=lambda: load_images(label_images, frame_images,photo_images))
load_button.pack(side="left", padx=5)
reset_button = tk.Button(frame_buttons, text="重置", command=reset_images)
reset_button.pack(side="left", padx=5)
output_button = tk.Button(frame_buttons, text="输出图片", command=merge_image)
output_button.pack(side="left", padx=5)

first_page.pack(fill='both', expand=True)  # Show the first page

# Third page
window.title("输出图片")
third_page = tk.Frame(window)


# Start the GUI event loop
window.mainloop()

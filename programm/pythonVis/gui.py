import math
import tkinter
from tkinter import *
from tkinter import ttk, filedialog
import numpy as np
import time
from threading import Thread
from queue import Queue
from PIL import ImageTk, Image

import stl_converter
import pc2img
import imageAnalyser

from PIL import ImageTk, Image, ImageDraw

pointclouds = ["", ""]
images = ["", ""]


def get_file_path_pc(index):
    global pointclouds
    file = filedialog.askopenfilename()
    pointclouds[index] = file
    pc_file_labels[index].config(text=file.split("/")[-1])


def show_message(message: str):
    info_label.config(text=message)


def save_stitched():
    if stitched_image:
        files = [('Pictures', '*.png*')]
        name_suggestion = imageAnalyser.get_matching_name(pointclouds[0], pointclouds[1])
        if len(name_suggestion) == 0:
            name_suggestion = imageAnalyser.get_matching_name(images[0], images[1], ending=".ply")
        filename = filedialog.asksaveasfile(mode='w', filetypes=files, defaultextension='.png',
                                            initialfile=name_suggestion+"_stitched.png")
        if not filename:
            return
        image = stitched_image[0]
        stitched_image.clear()
        imageAnalyser.save_image(image, filename.name)
    else:
        show_message("Stitching not done, nothing to save")


def save_pointcloud_image(index):
    if len(pc2img_results) > index:
        files = [('Pictures', '*.png*')]
        filename = filedialog.asksaveasfile(mode='w', filetypes=files, defaultextension='.png')
        if not filename:
            return
        image = pc2img_results[index]
        imageAnalyser.showAndSaveImage(image)
        imageAnalyser.save_image(image, filename.name)
    else:
        show_message("Conversion not done!, nothing to save")


def get_image_file(col, row, index):
    global images
    file = filedialog.askopenfilename()
    images[index] = file
    image_labels[index].config(text=file.split("/")[-1])


threads = [Thread] * 2
queues = [Queue()] * 2

conversion_progress = [0]
pc2img_results = [None, None]
last_update_time = time.time()


def update_progress():
    global last_update_time
    if time.time() - last_update_time > 0.2:
        global progress_pc_conversion_var
        last_update_time = time.time()

        image_progress_var.set(image_progress[0])
        progress_pc_conversion_var.set(conversion_progress[0])

        #treshold_max[0] = math.floor(tresh_max_slider.get())
        #treshold_min[0] = math.floor(tresh_min_slider.get())
        # check if image stitching done
        if stitched_image:
            show_message("Stitching done!")
            save_stitched_button.state(["!disabled"])
            display_image(stitched_image[0], 2)

        else:
            save_stitched_button.state(["disabled"])
        # check if converting clouds done
        if conversion_progress[0] == 100:
            show_message("Converting clouds done!")
            display_image(pc2img_results[0], 0)
            display_image(pc2img_results[1], 1)
            for save_button in save_buttons:
                save_button.state(["!disabled"])
        else:
            for save_button in save_buttons:
                save_button.state(["disabled"])

def start_pcd_2_image():
    show_message("Starting Point cloud conversion...")
    show_message("Loading files....")
    if len(pointclouds) < 1:
        show_message("Please select a file first")
    else:
        thread_pc2img = Thread(target=pc2img.convert_point_cloud,
                               args=(pointclouds, conversion_progress, pc2img_results, show_pointclouds.get()))
        thread_pc2img.start()


image_progress = [0]
stitched_image = []
convert_result = [[0], [0]]


def image_from_pc(index):
    thread_st = Thread(target=pc2img.get_2d_array_from_file,
                       args=(pointclouds[index], convert_result[index]))
    thread_st.start()


def start_stitching():
    show_message("Starting stitching...")
    stitching_thread = Thread(target=imageAnalyser.stitch_images,
                              args=(images[0], images[1], image_progress, stitched_image))
    stitching_thread.start()


def stitch_2_pcs():
    show_message("Starting Point cloud conversion...")
    show_message("Loading files....")
    if len(pointclouds) < 1:
        show_message("Please select a file first")
    else:
        thread_st = Thread(target=pc2img.stitch_pcs,
                           args=(pointclouds[0], pointclouds[1]))
        thread_st.start()


result_pc_array = []


def stitch_2_pcs_arrays():
    show_message("comparing 2d arrays")
    if len(pointclouds) < 1:
        show_message("Please select a file first")
    else:
        thread_st = Thread(target=pc2img.compare_2_arrays,
                           args=(convert_result[0][0], convert_result[1][0], result_pc_array))
        thread_st.start()


def display_image(image, index):
    img_s = Image.fromarray(image)
    img_s.thumbnail(image_size, Image.Resampling.LANCZOS)
    image_s = ImageTk.PhotoImage(image=img_s)
    image_displays[index].configure(image=image_s)
    image_displays[index].image = image_s

def stitching_with_converted():
    show_message("Stitching converted images...")
    stitching_thread = Thread(target=imageAnalyser.stitch_images,
                              args=(pc2img_results[0], pc2img_results[1], image_progress, stitched_image))
    stitching_thread.start()


treshold_max = [255]
treshold_min = [0]

def show_image_loop():
    image_loop_t = Thread(target=imageAnalyser.show_image_tresh,
                          args=(images[0], images[1], treshold_min, treshold_max))
    image_loop_t.start()


def load_stl_file():
    file = filedialog.askopenfilename()
    stl_converter.convert_stl(file)

def set_image_loop():
    ttk.Button(frm, text="Start loop", command=lambda: show_image_loop()).grid(column=2, row=7)

    # Treshhold slider
    tresh_min_slider = ttk.Scale(
        frm,
        from_=-500,
        to=500,
        orient='horizontal',  # horizontal
    )
    tresh_min_slider.grid(column=0, row=8)
    tresh_max_slider = ttk.Scale(
        frm,
        from_=-500,
        to=500,
        orient='horizontal',  # vertical
    )
    tresh_max_slider.grid(column=1, row=8)


root = Tk()
root.geometry("700x500")
frm = ttk.Frame(root, padding=10, )
frm.grid()

# Info label
info_label = ttk.Label(frm, text="Hello :) ")
info_label.grid(column=0, row=7)

# point clouds
AMOUNT_POINTCLOUDS = 2
pc_file_labels = []
save_buttons = []
for i in range(AMOUNT_POINTCLOUDS):
    pc_file_labels.append(ttk.Label(frm, text="No file"))
    pc_file_labels[i].grid(column=1, row=i)
    ttk.Button(frm, text="Select Point Cloud",
               command=lambda i=i: get_file_path_pc(index=i)).grid(column=0, row=i)
    button = ttk.Button(frm, text="Save image", command=lambda i=i: save_pointcloud_image(i))
    button.grid(column=2, row=i)
    save_buttons.append(button)

ttk.Button(frm, text="Start PC conversion", command=lambda: start_pcd_2_image()).grid(column=0, row=3)
progress_pc_conversion_var = tkinter.IntVar()
progress_pc_conversion_bar = ttk.Progressbar(frm, maximum=100, variable=progress_pc_conversion_var)
progress_pc_conversion_bar.grid(column=1, row=3, padx=10, pady=10)

show_pointclouds = tkinter.BooleanVar()
ttk.Checkbutton(frm, text="Show pointclouds", variable=show_pointclouds).grid(column=0, row=2)

stitch_converted_button = ttk.Button(frm, text="Stitch converted", command=lambda: stitching_with_converted())
stitch_converted_button.grid(column=2, row=3)
save_buttons.append(stitch_converted_button)

# images
image_labels = [ttk.Label(frm, text="No file selected"), ttk.Label(frm, text="No file selected")]
image_labels[0].grid(column=1, row=4)
image_labels[1].grid(column=1, row=5)

ttk.Button(frm, text="Select top image",
           command=lambda: get_image_file(0, 4, 0)).grid(column=0, row=4)
ttk.Button(frm, text="Select bot image",
           command=lambda: get_image_file(0, 5, 1)).grid(column=0, row=5)
start_stitching_button = ttk.Button(frm, text="Start stitching", command=lambda: start_stitching())
start_stitching_button.grid(column=0, row=6)

image_progress_var = tkinter.IntVar()
image_progress_bar = ttk.Progressbar(frm, maximum=100, variable=image_progress_var)
image_progress_bar.grid(column=1, row=6, padx=10, pady=10)

save_stitched_button = ttk.Button(frm, text="Save stitched image", command=lambda: save_stitched())
save_stitched_button.grid(column=2, row=6)

stl_button = ttk.Button(frm, text="Convert .stl", command=lambda: load_stl_file())
stl_button.grid(column=2, row=8)


# 3 Blank images
image_size = 128,128
blank_image = np.zeros(image_size)
image_displays = []
for i in range(3):
    img_t = Image.fromarray(blank_image)
    image_t = ImageTk.PhotoImage(image=img_t)
    label_image = ttk.Label(frm, image=image_t, justify="left", anchor="nw")
    label_image.image = image_t
    label_image.grid(column=i, row=10)
    image_displays.append(label_image)

while True:
    root.update_idletasks()
    root.update()
    update_progress()

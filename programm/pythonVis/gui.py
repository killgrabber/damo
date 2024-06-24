import tkinter
from tkinter import *
from tkinter import ttk, filedialog
import numpy as np
import time
from threading import Thread
from queue import Queue

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
        filename = filedialog.asksaveasfile(mode='w', filetypes=files, defaultextension='.png')
        if not filename:
            return
        image = stitched_image[0]
        stitched_image.clear()
        imageAnalyser.save_image(image, filename.name)
    else:
        show_message("Stitching not done, nothing to save")


def save_pointcloud_image(index):
    if pc2img_results[index]:
        files = [('Pictures', '*.png*')]
        filename = filedialog.asksaveasfile(mode='w', filetypes=files, defaultextension='.png')
        if not filename:
            return
        image = pc2img_results[index][0]
        pc2img_results[index].clear()
        imageAnalyser.showAndSaveImage(image)
        imageAnalyser.save_image(image, filename.name)
    else:
        show_message("Conversion not done!, nothing to save")


def get_image_file(col, row, index):
    global images
    file = filedialog.askopenfilename()
    images[index] = file
    ttk.Label(frm, text=file.split("/")[-1]).grid(column=col + 1, row=row)


threads = [Thread] * 2
queues = [Queue()] * 2

status = [""]
pc2img_results = [[], []]
last_update_time = time.time()


def update_progress():
    global last_update_time
    if time.time() - last_update_time > 0.2:
        global progress_pc_conversion_var
        last_update_time = time.time()

        image_progress_var.set(image_progress[0] * 100)

        # check if image stitching done
        global stitched_image
        if stitched_image:
            show_message("Stitching done!")
        if all(pc2img_results):
            show_message("Converting clouds done!")


def start_pcd_2_image():
    show_message("Starting Point cloud conversion...")
    show_message("Loading files....")
    if len(pointclouds) < 1:
        show_message("Please select a file first")
    else:
        for i in range(len(pointclouds)):
            threads[i] = Thread(target=pc2img.convert_point_cloud,
                                args=(pointclouds[i], status[0], pc2img_results[i], show_pointclouds.get()))
            threads[i].start()


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

def stitching_with_converted():
    show_message("Stitching converted images...")
    stitching_thread = Thread(target=imageAnalyser.stitch_images,
                              args=(pc2img_results[0][0], pc2img_results[1][0], image_progress, stitched_image))
    stitching_thread.start()


root = Tk()
root.geometry("1000x300")
frm = ttk.Frame(root, padding=10, )
frm.grid()

# Info label
info_label = ttk.Label(frm, text="Hello :) ")
info_label.grid(column=0, row=7)

# point clouds
pc_file_labels = [ttk.Label(frm, text="No file"), ttk.Label(frm, text="No file")]
for i in range(len(pc_file_labels)):
    pc_file_labels[i].grid(column=1, row=i)
    ttk.Button(frm, text="Select Point Cloud",
               command=lambda i=i: get_file_path_pc(index=i)).grid(column=0, row=i)

ttk.Button(frm, text="Start PC conversion", command=lambda: start_pcd_2_image()).grid(column=0, row=3)
progress_pc_conversion_var = tkinter.IntVar()
progress_pc_conversion_bar = ttk.Progressbar(frm, maximum=200, variable=progress_pc_conversion_var)
progress_pc_conversion_bar.grid(column=1, row=3, padx=10, pady=10)

show_pointclouds = tkinter.BooleanVar()
ttk.Checkbutton(frm, text="Show pointclouds", variable=show_pointclouds).grid(column=3, row=0)
ttk.Button(frm, text="Save pc top", command=lambda: save_pointcloud_image(0)).grid(column=2, row=3)
ttk.Button(frm, text="save pc bottom", command=lambda: save_pointcloud_image(1)).grid(column=3, row=3)

ttk.Button(frm, text="Stitch converted", command=lambda: stitching_with_converted()).grid(column=2, row=2)

# images
ttk.Label(frm, text="No file selected").grid(column=1, row=4)
ttk.Label(frm, text="No file selected").grid(column=1, row=5)
ttk.Button(frm, text="Select top image",
           command=lambda: get_image_file(0, 4, 0)).grid(column=0, row=4)
ttk.Button(frm, text="Select bot image",
           command=lambda: get_image_file(0, 5, 1)).grid(column=0, row=5)
ttk.Button(frm, text="Start stitching", command=lambda: start_stitching()).grid(column=0, row=6)
image_progress_var = tkinter.IntVar()
image_progress_bar = ttk.Progressbar(frm, maximum=100, variable=image_progress_var)
image_progress_bar.grid(column=1, row=6, padx=10, pady=10)
ttk.Button(frm, text="Save stitched image", command=lambda: save_stitched()).grid(column=2, row=6)

ttk.Button(frm, text="Convert pc1", command=lambda: image_from_pc(0)).grid(column=4, row=0)
ttk.Button(frm, text="Convert pc2", command=lambda: image_from_pc(1)).grid(column=4, row=1)
ttk.Button(frm, text="compare the 2", command=lambda: stitch_2_pcs_arrays()).grid(column=4, row=3)


while True:
    root.update_idletasks()
    root.update()
    update_progress()

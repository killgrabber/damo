import tkinter
from tkinter import *
from tkinter import ttk, filedialog

import time
from threading import Thread
from queue import Queue

import pc2img

files = []


def get_folder_path(col, row):
    global files
    file = filedialog.askopenfilename()
    files.append(file)
    splitted = file.split("/")
    fileName = splitted[len(splitted) - 1]
    ttk.Label(frm, text=fileName).grid(column=col + 1, row=row)
    # do what you want with file


threads = [Thread] * 2
queues = [Queue()] * 2

progresses = [[0],[0]]
last_update_time = time.time()
def update_progress():
    global last_update_time
    if time.time() - last_update_time > 0.2:
        global progress1, progress2
        last_update_time = time.time()
        if (progresses[0][0]) > 0:
            ttk.Label(frm, text="Converting clouds....").grid(column=1, row=3)
        progress1.set(progresses[0][0] * 100)
        progress2.set(progresses[1][0] * 100)
        #ttk.Label(frm, text="{:.1f}%".format(round(progresses[0][0] * 100, 4))).grid(column=4, row=0)
        #ttk.Label(frm, text="{:.1f}%".format(round(progresses[1][0] * 100, 4))).grid(column=4, row=1)


def start_pcd_2_image():
    print("Starting Point cloud conversion...")
    ttk.Label(frm, text="Loading files....").grid(column=1, row=3)
    if len(files) != 2:
        ttk.Label(frm, text="Please select 2 files first").grid(column=4, row=0)
    else:
        for i in range(len(files)):
            threads[i] = Thread(target=pc2img.convert_point_cloud, args=(files[i],progresses[i]))
            threads[i].start()


root = Tk()
root.geometry("500x300")
frm = ttk.Frame(root, padding=10)
frm.grid()
ttk.Label(frm, text="Select Top File:").grid(column=0, row=0)
ttk.Label(frm, text="Select Bottom File:").grid(column=0, row=1)
ttk.Label(frm, text="No file selected").grid(column=2, row=0)
ttk.Label(frm, text="No file selected").grid(column=2, row=1)
ttk.Button(frm, text="Select file", command=lambda: get_folder_path(1, 0)).grid(column=1, row=0)
ttk.Button(frm, text="Select file", command=lambda: get_folder_path(1, 1)).grid(column=1, row=1)

ttk.Button(frm, text="Start PCD Conversion", command=lambda: start_pcd_2_image()).grid(column=0, row=3)
progress1 = tkinter.IntVar()
progress2 = tkinter.IntVar()
progressbar1 = ttk.Progressbar(frm, maximum=100, variable=progress1)
progressbar2 = ttk.Progressbar(frm, maximum=100, variable=progress2)
progressbar1.grid(column=4, row=0)
progressbar2.grid(column=4, row=1)

while True:
    root.update_idletasks()
    root.update()
    update_progress()

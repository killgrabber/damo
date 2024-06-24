import sys

import numpy as np
import numpy.ma as ma
import open3d as o3d
import numba as nb
import math
import copy
from numba.experimental import jitclass
import cv2


@nb.njit(parallel=True, fastmath=True)
def get_2d_array(point_cloud: np.array):  # 2-D Array that stores the height values
    data = np.array
    width = 0
    height = 0
    max_height = 0
    min_height = sys.maxsize

    length = len(point_cloud)
    scale = 100
    x_min = math.floor(np.min(point_cloud[:, 0]) * scale)
    abs_x_min = abs(x_min)
    y_min = math.floor(np.min(point_cloud[:, 1]) * scale)
    width = math.ceil(np.amax(point_cloud[:, 0]) * scale) + 1
    height = math.ceil(np.amax(point_cloud[:, 1]) * scale) + 1
    data = np.zeros((width + abs_x_min, height, 1))
    for i in nb.prange(length):
        min_height = min(point_cloud[i][2], min_height)
        max_height = max(point_cloud[i][2], max_height)
        x = int(point_cloud[i][0] * scale) + abs_x_min
        y = int(point_cloud[i][1] * scale)
        #print(f"Setting {x},{y} to {point_cloud[i][2]}")
        data[x, y] = point_cloud[i][2]
    data = data
    width = height
    height = width + abs_x_min
    print("Creating pointcloud array done!")
    return data


@nb.njit(parallel=False, fastmath=True)
def conversion(old_value, old_min, old_max, new_min=10, new_max=255):
    return ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min


@nb.njit(parallel=True, fastmath=True)
def get_image_fast(array: np.array, rgbs: []):
    width = len(array)
    height = len(array[0])
    length_rgbs = len(rgbs)
    color_image = np.zeros((width, height, 1), dtype=np.uint8)
    for x in nb.prange(width):
        for y in nb.prange(height):
            if array[x, y] != 0:
                #print(array[x, y])
                #rgb_index = conversion((array[x, y]), min_h, max_h, 0, length_rgbs)
                #print(rgb_index)
                color_image[x, y] = (255, 255, 255)
    #print(f"Size: {len(array)},{len(array[0])}")
    #print(f"Done creating image")
    return color_image


@nb.njit(parallel=True, fastmath=True)
def compare_2_2d_images(image_a: np.array, image_b: np.array):
    min_width = min(len(image_b), len(image_a))
    min_height = min(len(image_b[0]), len(image_a[0]))
    distances = np.zeros(shape=(min_width, min_height, 1), dtype=np.uint64)
    for x in nb.prange(min_width):
        #print(f"Done {x/min_width* 100:.2f}%")
        for y in nb.prange(min_height):
            distance = image_a[x, y] - image_b[x, y]
            distances[x, y] = distance
    return distances


@nb.njit(parallel=True, fastmath=True)
def resize_array(original: np.array, new_width, new_height, x, y):
    old_width, old_height = original.shape[:2]
    new_array = np.zeros((new_width, new_height, 1))
    new_array[x:x + old_width, y:y + old_height] = original
    return new_array


def combine_arrays(arr1, arr2):
    print("Combining image...")
    if arr1.shape == arr2.shape:
        new_arr = np.zeros(arr1.shape)
        for x in nb.prange(arr1.shape[0]):
            for y in nb.prange(arr1.shape[1]):
                new_arr[x, y] = max(arr1[x, y], arr2[x, y])
        return new_arr

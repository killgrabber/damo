import sys
import time

import open3d as o3d
import numpy as np
import cv2 as cv
import collections
import numba as nb
import copy
import math

class imageStructure:
    x: int
    y: int
    depth: float
    color: int


DATA_DIR = "../../daten/240405_Messungen_LL/"

FILES = {
    DATA_DIR + "FDM_1.ply",
    DATA_DIR + "FDM_2.ply",
}


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


def loadPointClouds(vis=False):
    pcds = []
    for file in FILES:
        pcd = o3d.io.read_point_cloud(file)
        pcds.append(pcd)
        print("Loaded file: ", file + " with " + str(len(pcd.points)) + " points.")
    if (vis):
        visPcds = []
        i = 0
        for pcd in pcds:
            aabb = pcd.get_axis_aligned_bounding_box()
            maxX = aabb.get_max_bound()[0]
            #pcd.translate((maxX*i, 0, 0))
            #aabb.translate((maxX*i, 0, 0))
            i = i + 1
            aabb.color = (1, 0, 0)
            visPcds.append(aabb)
            visPcds.append(pcd)
        o3d.visualization.draw_geometries(visPcds)
    return pcds


def show_pointcloud(pc):
    aabb = pc.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)
    o3d.visualization.draw_geometries([aabb, pc])


def remove_outliers(pc):
    print("Statistical oulier removal")
    oldAmount = len(pc.points)
    cl, ind = pc.remove_statistical_outlier(nb_neighbors=50,
                                            std_ratio=0.6)
    newAmount = len(pc.select_by_index(ind).points)
    print("Reduced from: " + str(oldAmount) + " to " + str(newAmount) + " points")
    cleared = pc.select_by_index(ind)
    return cleared


def anaylseZ(pcd):
    MIN_AMOUNT = 5
    counter = collections.Counter(pcd[:, 2])
    mostCommon = counter.most_common(500)
    minFound = sys.maxsize
    maxFound = 0
    for key, val in mostCommon:
        if (val > MIN_AMOUNT):
            maxFound = max(key, maxFound)
            minFound = min(key, minFound)
    return maxFound, minFound


@nb.njit(parallel=False, fastmath=True)
def conversion(old_value, old_min, old_max, new_min=10, new_max=255):
    return ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min


def pcNeighbours(pcd):
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    print("Find its 200 nearest neighbors, and paint them blue.")
    [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[5000], 10)
    np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
    print("Visualize the point cloud.")
    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.5599,
                                      front=[-0.4958, 0.8229, 0.2773],
                                      lookat=[2.1126, 1.0163, -1.8543],
                                      up=[0.1007, -0.2626, 0.9596])


def divide_pointcloud(original, splits: int) -> []:
    chunk_size = round(len(original) / splits)
    chunked = []
    for i in range(0, len(original), chunk_size):
        single_chunk = original[i:i + chunk_size]
        chunked.append(single_chunk)
    return chunked


@nb.njit(parallel=False, fastmath=True)
def convert_chunk(array: np.array):
    scale = 100
    x_min = math.floor(np.min(array[:, 0]) * scale)
    abs_x_min = abs(x_min)
    y_min = math.floor(np.min(array[:, 1]) * scale)
    width = math.ceil(np.amax(array[:, 0])*scale)+1
    height = math.ceil(np.amax(array[:, 1])*scale)+1
    image = np.zeros((width+abs_x_min, height, 1), np.uint8)
    length = len(array)
    #maxFound, minFound = anaylseZ(array)
    for i in nb.prange(length):
        point = array[i]
        #print(f"Checking {point[0]},{point[1]},{point[2]}")
        x = int(point[0] * scale) + abs_x_min
        y = int(point[1] * scale)
        #print(f"Setting {x},{y} to {255}....")
        image[x, y] = 255
    return image


# returns image
def convert_point_cloud(file, progress: [str], result: [], show_pc=False):
    pcd = o3d.io.read_point_cloud(file)
    if len(pcd.points) == 0:
        print("Loading point cloud failed")
        return
    print("Removing outliers...")
    pcd = remove_outliers(pcd)
    if show_pc:
        show_pointcloud(pcd)
    np_points = np.asarray(pcd.points)
    print("Converting image....")
    image = convert_chunk(np_points)
    result.append(image)
    print(f"Done with {file.split('/')[-1]}")

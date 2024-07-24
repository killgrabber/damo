import sys
import time

import cv2
import open3d as o3d
import numpy as np
import cv2 as cv
import collections
import numba as nb
import copy
import math
import imageAnalyser
from threading import Thread


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
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([aabb, pc, mesh_frame])


def remove_outliers(pc):
    print("Statistical outlier removal")
    oldAmount = len(pc.points)
    cl, ind = pc.remove_statistical_outlier(nb_neighbors=250,
                                            std_ratio=0.2)
    newAmount = len(pc.select_by_index(ind).points)
    diff = oldAmount - newAmount
    print(f"Reduced from: {oldAmount} to {newAmount}, diff {diff}, percentage: {diff / oldAmount:.2f}")
    cleared = pc.select_by_index(ind)
    return cleared


def anaylseZ(pcd):
    MIN_AMOUNT = -1
    counter = collections.Counter(pcd[:, 2])
    mostCommon = counter.most_common(round(len(counter) * 0.8))
    minFound = sys.maxsize
    maxFound = 0
    for key, val in mostCommon:
        if (val > MIN_AMOUNT):
            maxFound = max(key, maxFound)
            minFound = min(key, minFound)
    return maxFound, minFound


@nb.njit(parallel=False, fastmath=True)
def conversion(old_value, old_min, old_max, new_min=10, new_max=255):
    result = ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    #print(f"Converting {int(old_value*1000)} to {math.floor(result)}")
    return result


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


#@nb.njit(parallel=True, fastmath=True)
def convert_chunk(array: np.array, max_h, min_h):
    print(f"converting with max: {int(max_h * 1000)} min: {int(min_h * 1000)}...")
    scale = 100
    x_min = math.floor(np.min(array[:, 0]) * scale)
    abs_x_min = abs(x_min)
    y_min = math.floor(np.min(array[:, 1]) * scale)
    abs_y_min = abs(y_min)
    width = math.ceil(np.amax(array[:, 0]) * scale) + 1
    height = math.ceil(np.amax(array[:, 1]) * scale) + 1
    image = np.zeros((width + abs_x_min, height + abs_y_min, 3), np.uint8)
    print(f"Mins are: x:{abs_x_min},y:{abs_y_min}")
    length = len(array)
    for i in nb.prange(length):
        #print(f"Checking {array[i]}....")
        if min_h <= array[i][2] <= max_h:
            x = int(array[i][0] * scale) + abs_x_min
            y = int(array[i][1] * scale) + abs_y_min
            #print(f"Setting {x},{y} to {255}....")
            brightness = 255  #math.floor(conversion(array[i][2], min_h, max_h, 0, 255))
            image[x, y] = brightness
    return image


def get_points_from_pc(file, progress, show_pc, pcds_points):
    pcd = o3d.io.read_point_cloud(file)
    progress[0] += 10
    if len(pcd.points) == 0:
        print("Loading point cloud failed")
        return

    print("Removing outliers...")
    pcd = remove_outliers(pcd)
    progress[0] += 10

    if show_pc:
        show_pointcloud(pcd)

    np_points = np.asarray(pcd.points)
    pcds_points.append(np_points)


# returns image
def convert_point_cloud(files: [], progress: [], result: [], show_pc=False):
    start = time.time()
    progress[0] += 10
    pcds_points = []
    converter_threads = []
    for file in files:
        converter_thread = Thread(target=get_points_from_pc,
                                  args=(file, progress, show_pc, pcds_points))
        converter_thread.start()
        converter_threads.append(converter_thread)

    for t in converter_threads:
        t.join()

    max_hs = []
    min_hs = []
    for np_points in pcds_points:
        print("Getting height boundaries....")
        max_h, min_h = anaylseZ(np_points)
        progress[0] += 10
        max_hs.append(max_h)
        min_hs.append(min_h)
    for i in range(len(pcds_points)):
        image = convert_chunk(pcds_points[i], np.mean(max_hs), np.mean(min_hs))
        #image = cv2.blur(image, (4, 4))
        #ret, thresh = cv2.threshold(image, 100, 255, 0)
        progress[0] += 10
        imageAnalyser.showAndSaveImage(image)
        result[i] = image
        print(f"Done with {files[i].split('/')[-1]}")

    end = time.time()
    progress[0] = 100
    print(f"Converted 2 clouds in: {end - start:.2f}s")


import pc_stitcher


def stitch_pcs(top_file, bot_file):
    top_pc = o3d.io.read_point_cloud(top_file)
    bot_pc = o3d.io.read_point_cloud(bot_file)
    pc_stitcher.stitch_2_pcs(top_pc, bot_pc)


import pointcloud_array


def crop_array(array, top, size=1000, offset=50):
    height, width = array.shape[:2]
    if top:
        crop_a = array[height - size - offset:height - offset, offset:width - offset]
    else:
        crop_a = array[offset:size, offset:width - offset]
    return crop_a


def compare_2_arrays(pc_array_1: np.array, pc_array_2, result: []):
    pc_array_1 = crop_array(pc_array_1, True, 500)
    pc_array_2 = crop_array(pc_array_2, False, 500)

    print(f"Shape1: {pc_array_1.shape}, shape2: {pc_array_2.shape}")
    # Resize image 1
    max_size = (max(pc_array_1.shape[0], pc_array_2.shape[0]),
                max(pc_array_1.shape[1], pc_array_2.shape[1]),)
    pc_array_1 = pointcloud_array.resize_array(pc_array_1, max_size[0] * 2, max_size[1] + 100, 0, 50)
    pc_array_2 = pointcloud_array.resize_array(pc_array_2, max_size[0] * 2, max_size[1] + 100, 0, 0)
    rgbs = [(255, 255, 0)]
    image_a = pointcloud_array.get_image_fast(pc_array_1, rgbs)
    imageAnalyser.showAndSaveImage(image_a, 0)
    image_b = pointcloud_array.get_image_fast(pc_array_2, rgbs)
    imageAnalyser.showAndSaveImage(image_b, 0)
    for x in range(0, round(max_size[0] / 2), 10):
        for y in range(0, round(max_size[1] / 2), 10):
            pc_array_2 = np.roll(pc_array_2, 10, axis=0)
            distances = pointcloud_array.compare_2_2d_images(pc_array_1, pc_array_2)
            unique, counts = np.unique(distances, return_counts=True)
            #print(f"Occurrences: {dict(zip(unique, counts / (len(distances) * len(distances[0]))))}")
            percentage = counts[0] / (len(distances) * len(distances[0]))
            print(f"Overlap: {percentage * 100:.2f}%, sum: {np.sum(distances)}")
            image_c = pointcloud_array.get_image_fast(pc_array_1 + pc_array_2, rgbs)
            imageAnalyser.showAndSaveImage(image_c, 1)
        print("Shifting axis 1...")
        pc_array_2 = np.roll(pc_array_2, 10, axis=1)


def get_2d_array_from_file(file: str, result: []):
    pointcloud = o3d.io.read_point_cloud(file)
    np_points = np.asarray(pointcloud.points)
    pc_array = pointcloud_array.get_2d_array(np_points)
    result[0] = pc_array
    print(f"Done with {file.split('/')[-1]}")

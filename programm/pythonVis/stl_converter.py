import math
import sys

import open3d as o3d
import pc2img
import numpy as np
import numba as nb
import imageAnalyser
import contourMatcher
import cv2
from threading import Thread


def convert_stl(filepath: str, progress: [], result: []):
    progress[0] = 1
    mesh = o3d.io.read_triangle_mesh(filepath)
    print(f"Loading stl done")
    rotation_m = mesh.get_rotation_matrix_from_xyz((0, 0, np.pi / 2))
    mesh.rotate(rotation_m, center=(0, 0, 0))
    pcd = mesh.sample_points_poisson_disk(number_of_points=500000, init_factor=5)
    #pc2img.show_pointcloud(pcd)
    print(f"Sampling pointcloud done")
    stl_points = np.asarray(pcd.points)
    max_h, min_h = pc2img.anaylseZ(stl_points)
    image = pc2img.convert_chunk(stl_points, max_h, min_h)
    #Invert the image because otherwise 2 contours are found on a line
    image_inv = cv2.bitwise_not(image)
    progress[0] = 100
    result[0] = image_inv


def find_contours_in_image(image, progress: [], result: []):
    image_blur = cv2.blur(image, (2, 2))
    contours = imageAnalyser.getContours(image_blur, 0)
    contours = contours[:4]
    #contourMatcher.display_contours_in_image(contours, image, name="with image", wait=1)
    #contourMatcher.display_contours(contours, name="without", wait=0)

    print(f"Getting contours done")
    result[0] = contours
    progress[0] = 200


@nb.njit(parallel=False, fastmath=True)
def get_center_of_mass(contour: []):
    all_x = 0
    all_y = 0
    for point in contour:
        all_x += point[0]
        all_y += point[1]
    mean_x = all_x / len(contour)
    mean_y = all_y / len(contour)
    return mean_x, mean_y


#@nb.njit(parallel=False, fastmath=True)
def check_2_contours(arr_1, arr_2, progress, search_radius):
    print(f"Comparing contour arrays...")
    distance = 0
    len1 = len(arr_1)
    len2 = len(arr_1[-1])
    for i in range(len1):
        for j in range(len2):
            if not arr_1[i, j]:
                continue
            # Find the closest '1' in arr2
            closest_match = (i, j)
            smallest_dist = sys.maxsize
            x_range_min = max(-search_radius, i-search_radius)
            x_range_max = min(search_radius, len(arr_2)-i)
            y_range_min = max(-search_radius, j - search_radius)
            y_range_max = min(search_radius, len(arr_2) - j)
            for x in range(x_range_min, x_range_max):
                for y in range(y_range_min, y_range_max):
                    if arr_2[i + x, j + y]:
                        point_dist = contourMatcher.compare_point((i + x, j + y), (i, j))
                        if point_dist < smallest_dist:
                            closest_match = i + x, j + y
                            smallest_dist = point_dist
            if smallest_dist != sys.maxsize:
                distance += contourMatcher.compare_point(closest_match, (i, j))
            #print(f"Progress: {i/len1:.3f}, distance: {distance}")

    return distance


def convert_to_2d_array(contour: []):
    x, y, w, h = contourMatcher.get_boundaries([contour])
    array = np.zeros(shape=(w + 1, h + 1), dtype=bool)
    for point in contour:
        x = point[0]
        y = point[1]
        array[x, y] = True
    return array


def compare_images(image_paths: [], progress: []):
    if len(image_paths) != 2:
        print(f"Please select 2 instead of {len(image_paths)} images")
        return
    print(f"Starting to compare {image_paths[0].split('/')[-1]} "
          f"with {image_paths[1].split('/')[-1]}")
    image_1 = cv2.imread(image_paths[0])
    image_2 = cv2.imread(image_paths[1])

    contours_1 = [[]]
    progress_1 = [0]
    contours_2 = [[]]
    progress_2 = [0]

    print(f"Starting threads....")
    get_c1_t = Thread(target=find_contours_in_image,
                      args=(image_1, progress_1, contours_1))
    get_c1_t.start()
    get_c2_t = Thread(target=find_contours_in_image,
                      args=(image_2, progress_2, contours_2))
    get_c2_t.start()

    get_c1_t.join()
    get_c2_t.join()
    print(f"Threads done, found {len(contours_1[0])} and {len(contours_2[0])} contours")

    x1, y1 = get_center_of_mass(contours_1[0][0])
    print(f"Average is: {x1:.2f};{y1:.2f}")
    x2, y2 = get_center_of_mass(contours_2[0][0])
    print(f"Average is: {x2:.2f};{y2:.2f}")
    all_c = contours_1[0] + contours_2[0]

    for i in range(-5, 5):
        for j in range(-5, 5):
            all_c += [[(int(x1 + i), int(y1 + j))]]
            all_c += [[(int(x2 + i), int(y2 + j))]]
    colors_1 = [(255, 255, 0)] * len(contours_1[0])
    colors_2 = [(0, 255, 255)] * len(contours_2[0])

    #contourMatcher.display_contours(all_c, name="without1", wait=1,
    #                                colors=colors_1 + colors_2)

    translation = (int(x1 - x2), int(y1 - y2))
    print(f"Translation is: {translation}")

    contour_a1 = convert_to_2d_array(contours_1[0][0])

    #contourMatcher.display_contours([contours_1[0][0], moved_c], name="without", wait=0,
    #                                colors=[(255, 255, 0), (255, 0, 255)])
    for i in range(0, 100, 10):
        for ii in range(0, 100, 10):
            moved_c = contourMatcher.move_contour(contours_2[0][0], (i, ii))
            contour_a2 = convert_to_2d_array(moved_c)
            distance = check_2_contours(contour_a1, contour_a2, progress, search_radius=50)
            contourMatcher.display_contours([contours_1[0][0], moved_c], name="without", wait=1,
                                            colors=[(255, 255, 0), (255, 0, 255)])
            print(f"Distance is: {distance:.4f}")

    print(f"Done with comparing")

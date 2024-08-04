import math
import sys
import os

import open3d as o3d
import pc2img
import numpy as np
import numba as nb
import imageAnalyser
import contourMatcher
import cv2
from threading import Thread
from matplotlib import pyplot as plt

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
    print('Starting contour search')
    image_blur = cv2.blur(image, (2, 2))
    contours = imageAnalyser.getContours(image_blur, 10, limit=10, segment_distance=1000000)
    #contourMatcher.display_contours_in_image(contours, image, name="with image", wait=1,
    #                                         colors=[255, 255, 0], save_name="with_con.jpg")
    #contourMatcher.display_contours(contours, name="without", wait=0,
    #                                colors=[255, 255, 255], save_name="only_con.jpg")

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


@nb.njit(parallel=False, fastmath=True)
def find_closest_point(i, j, arr_2, search_radius):
    max_distance = 500
    smallest_dist = max_distance
    best_match = 0, 0
    #print(f"Arr2 is: {arr_2}")
    for x in range(-search_radius, search_radius):
        for y in range(-search_radius, search_radius):
            a = i + x
            b = j + y
            #print(f"i, j: {i}, {j}: point to check: {a},{b} arr2: len: {len(arr_2)},{len(arr_2[a])}")
            if 0 <= a < len(arr_2) and 0 <= b < len(arr_2[a]):
                if arr_2[a, b]:
                    point_dist = contourMatcher.compare_point((a, b), (i, j))
                    #print(f"Smallest distance is: {int(smallest_dist*1000)}")
                    if point_dist == 0:
                        return 0, i - a, j - b
                    if point_dist < smallest_dist:
                        smallest_dist = point_dist
                        best_match = i - a, j - b
            else:
                pass
                #print(f"Skipped because of arr boundries")
    if smallest_dist >= max_distance:
        return -1, 0, 0
    return smallest_dist, best_match[0], best_match[1]


@nb.njit(parallel=False, fastmath=True)
def check_2_contours(arr_1, arr_2, progress, search_radius, result: []):
    #print(f"Comparing contour arrays...")
    distance = 0
    vectors = []
    len1 = len(arr_1)
    len2 = len(arr_1[-1])
    for i in range(len1):
        for j in range(len2):
            if arr_1[i, j]:
                smallest_dist, t1, t2 = find_closest_point(i, j, arr_2, search_radius)
                if smallest_dist != -1:
                    vectors.append((t1, t2))
                    distance += smallest_dist
        #print(f"Progress: {i/len1:.3f}, distance: {distance:.2f}")
    result[0] = distance
    return distance / len(arr_1), vectors


def convert_to_2d_array(contour: []):
    x, y, w, h = contourMatcher.get_boundaries([contour])
    array = np.zeros(shape=(w + 1, h + 1), dtype=bool)
    for point in contour:
        x = point[0]
        y = point[1]
        array[x, y] = True
    return array


def get_horizontal_distance(con1, con2):
    contour_a1 = convert_to_2d_array(con1)
    contour_a2 = convert_to_2d_array(con2)
    center_x1, center_y1 = get_center_of_mass(con1)
    center_x2, center_y2 = get_center_of_mass(con2)
    length1 = len(contour_a1)
    length2 = len(contour_a2)
    depth1 = len(contour_a1[int(length1 / 2)])
    depth2 = len(contour_a2[int(length2 / 2)])
    # Horizontal (middle)
    search_index_horizontal = math.floor((center_y1 + center_y1) / 2)
    matches_a1_hor = []
    matches_a2_hor = []
    line_con = []
    for i in range(max(length1, length2)):
        line_con.append((i, search_index_horizontal))
        if i < len(contour_a1) and contour_a1[i, search_index_horizontal]:
            matches_a1_hor.append((i, search_index_horizontal))
        if i < len(contour_a2) and contour_a2[i, search_index_horizontal]:
            matches_a2_hor.append((i, search_index_horizontal))

    if not (len(matches_a1_hor) >= 2 and len(matches_a2_hor) >= 2):
        print(f"Didnt find 2 or more matches per line hor, abort")
        return
    top_distance_hor = contourMatcher.compare_point(matches_a1_hor[0], matches_a2_hor[0])
    bot_distance_hor = contourMatcher.compare_point(matches_a1_hor[1], matches_a2_hor[1])
    print(f"Distance top hor: {top_distance_hor}")
    print(f"Distance bot hor: {bot_distance_hor}")


def get_vertical_distance(con1, con2, image_paths: [], index_1, index_2):
    print(f"Getting vertical distances")
    contour_a1 = convert_to_2d_array(con1)
    contour_a2 = convert_to_2d_array(con2)

    center_x1, center_y1 = get_center_of_mass(con1)
    center_x2, center_y2 = get_center_of_mass(con2)
    scanline_index = int((center_x1 + center_x2) / 2)
    smallest_distance = []
    all_vectors = []
    search_length = min(len(contour_a1), len(contour_a2))
    for scanline in range(0, search_length):
        line_con, vectors = find_distance_in_line(contour_a1, contour_a2, scanline)
        if len(vectors) == 2:
            all_vectors.append((vectors[0][1], vectors[1][1]))
            smallest_distance.append((vectors[0][1], vectors[1][1], line_con))
        else:
            all_vectors.append((0, 0))
    smallest_distance.sort(key=lambda length: abs(length[0]) + abs(length[1]))
    print(f"matches found: {len(smallest_distance)}")

    average_top_sum = 0
    average_bot_sum = 0
    for v in all_vectors:
        average_top_sum += v[0]
        average_bot_sum += v[1]
    average_top = average_top_sum/ len(all_vectors)
    average_bot = average_bot_sum / len(all_vectors)
    print(f"Average distance in middle vertical: {average_top},{average_bot}")
    print(f"Smallest distance: {smallest_distance[0][0]}, {smallest_distance[0][1]}")
    y1_val = [x[0] for x in all_vectors]
    y2_val = [x[1] for x in all_vectors]
    both_val = [x[0]-x[1] for x in all_vectors]
    y_line = [0, len(all_vectors)]
    x_val = [x for x in range(len(all_vectors))]
    # saving data to file:
    image_name_1 = image_paths[0].split('/')[-1].split('.')[0]
    image_name_2 = image_paths[1].split('/')[-1].split('.')[0]
    try:
        os.mkdir(f"deformation_data")
    except FileExistsError:
        pass
    file_name = f'deformation_data/{image_name_1}_{image_name_2}_{index_1}_{index_2}'
    file1 = open(f'{file_name}.txt', 'w')
    index = 0
    for val in both_val:
        file1.write(f"{index}, {val}\n")
        index += 1
    file1.close()

    contourMatcher.display_contours([con1, con2], name="with_line", wait=1,
                                    colors=[(255, 255, 0), (255, 0, 255), (0, 255, 255)],
                                    text=f"Difference {average_top:.2f}, {average_bot:.2f}",
                                    save_name=f"{file_name}.png")

    return all_vectors


@nb.njit(parallel=False, fastmath=True)
def find_distance_in_line(contour_a1, contour_a2, scanline_index):
    matches_a1 = []
    matches_a2 = []
    line_con = []
    length1 = len(contour_a1)
    length2 = len(contour_a2)
    depth1 = len(contour_a1[int(length1 / 2)])
    depth2 = len(contour_a2[int(length2 / 2)])
    #vertical
    for i in range(max(depth1, depth2)):
        line_con.append((scanline_index, i))
        if i < len(contour_a1[scanline_index]) and contour_a1[scanline_index, i]:
            matches_a1.append((scanline_index, i))
        if i < len(contour_a2[scanline_index]) and contour_a2[scanline_index, i]:
            matches_a2.append((scanline_index, i))

    if not (len(matches_a1) >= 2 and len(matches_a2) >= 2):
        #print(f"Didnt find 2 or more matches per line ver, abort")
        return line_con, [(0, 0)]

    # find smallest dist
    smallest_dists = []
    vectors = []
    vector = (0, 0)
    for a1 in matches_a1:
        smallest_dist = sys.maxsize
        for a2 in matches_a2:
            dist = contourMatcher.compare_point(a1, a2)
            if dist < smallest_dist:
                vector = contourMatcher.get_translation(a1, a2)
                smallest_dist = dist
            smallest_dist = min(smallest_dist, contourMatcher.compare_point(a1, a2))
        smallest_dists.append(smallest_dist)
        if vector != (0, 0):
            vectors.append(vector)

    return line_con, vectors


def move_and_check(con1, con2, progress, init_translation, result_distances: [], image_paths: [],
                   index_1, index_2, search_area=50, ):
    diff = abs(len(con1) - len(con2))
    min_len = 2500
    if diff > 20000 or len(con1) < min_len or len(con2) < min_len:
        print(f"Diff {diff} to high, abort")
        return (0, 0), 0
    #print(f"Checking contours with size {len(con1)} and {len(con2)}, diff: {diff}")
    moved_c = con2
    contour_a1 = convert_to_2d_array(con1)
    translation = init_translation
    length_init_t = max(100, contourMatcher.compare_point(translation, (0, 0)) * 1.5)
    print(f"Search radius is: {length_init_t}")
    if length_init_t > 500:
        length_init_t = 200
        print(f"Search radius too big, abort")
        #return (0, 0), 0
    final_translation = translation
    last_translations = []
    while True:
        if len(last_translations) > 4:
            last_translations.pop(0)
        moved_c = contourMatcher.move_contour(moved_c, translation)
        contour_a2 = convert_to_2d_array(moved_c)

        contourMatcher.display_contours([con1, moved_c], name="matching", wait=1,
                                        colors=[(255, 255, 0), (255, 0, 255)])

        result = [0]
        distance, vectors = check_2_contours(contour_a1, contour_a2, progress, search_radius=length_init_t,
                                             result=result)
        if len(vectors) == 0:
            print(f"Found no matches: distances {distance}")
            return (0, 0), 0

        sum1 = 0
        sum2 = 0
        for vector in vectors:
            sum1 += vector[0]
            sum2 += vector[1]
        sum1 = sum1 / len(vectors)
        sum2 = sum2 / len(vectors)
        last_translations.append((sum1, sum2))
        print(f"Distance: {distance}, transformation average: {sum1},{sum2}")
        t1 = math.ceil(sum1)
        t2 = math.ceil(sum2)
        if sum1 < 0:
            t1 = math.floor(sum1)
        if sum2 < 0:
            t2 = math.floor(sum2)
        translation = (t1, t2)
        final_translation = (final_translation[0] + translation[0],
                             final_translation[1] + translation[1])
        #print(f"Next translation is: {translation}")
        looping = (len(last_translations) > 3 and (sum1 == last_translations[-3][0] and
                                                   sum2 == last_translations[-3][1]))
        if looping or (abs(sum1) < 0.1 and abs(sum2) < 0.1):
            print(f"Checking done. complete translation: {final_translation}")
            #contourMatcher.display_contours([con1, moved_c], name="final", wait=1,
            #                                colors=[(255, 255, 0), (255, 0, 255)],
            #                                save_name=f"matched_{len(con1)}_{len(con2)}_{distance:.2f}.png",
            #                                text=f"Avg. Pixel distance: {distance:.4f}")
            break
    final_moved = contourMatcher.move_contour(con2, final_translation)
    all_distances_vertical = get_vertical_distance(con1, final_moved, image_paths, index_1, index_2)
    result_distances[0] = all_distances_vertical
    return final_translation, distance


def compare_images(image_paths: [], progress: [], result_distances: []):
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

    cons_1 = contours_1[0]
    cons_2 = contours_2[0]

    contourMatcher.display_contours(cons_1, name="without1", wait=0,
                                    colors=[255, 255, 255],
                                    save_name=f"contours_{image_paths[0].split('/')[-1]}")

    all_results = []
    for index_1 in range(len(cons_1)):
        for index_2 in range(len(cons_2)):
            x1, y1 = get_center_of_mass(cons_1[index_1])
            #print(f"Average is: {x1:.2f};{y1:.2f}")
            x2, y2 = get_center_of_mass(cons_2[index_2])
            #print(f"Average is: {x2:.2f};{y2:.2f}")

            all_c = [cons_1[index_1], cons_2[index_2]]
            for i in range(-5, 5):
                for j in range(-5, 5):
                    all_c += [[(int(x1 + i), int(y1 + j))]]
                    all_c += [[(int(x2 + i), int(y2 + j))]]

            #colors_1 = [(255, 255, 0)] * len(cons_1)
            #colors_2 = [(0, 255, 255)] * len(cons_2)
            #contourMatcher.display_contours(all_c, name="without1", wait=1,
            #                                colors=colors_1 + colors_2,
            #                                save_name=f"contours_{image_paths[0].split('/')[-1]}")

            translation = (int(x1 - x2), int(y1 - y2))
            #print(f"Translation is: {translation}")

            final_translation, distance = move_and_check(cons_1[index_1],
                                                         cons_2[index_2],
                                                         progress, translation, result_distances,
                                                         image_paths, index_1, index_2,
                                                         search_area=25)
            all_results.append((index_1, index_2, final_translation, distance))

    #print("Index1, Index2, Translation, Distance")
    distances = [sys.maxsize]
    for i in range(len(all_results)):
        if all_results[i][3] != 0:
            if len(distances) <= all_results[i][0]:
                distances.append(sys.maxsize)
            #print(f"{all_results[i]}")
            distances[all_results[i][0]] = min(distances[all_results[i][0]], (all_results[i][3]))

    print(f"Final: {distances} average: {np.mean(distances)}")

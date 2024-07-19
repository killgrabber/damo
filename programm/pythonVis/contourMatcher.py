import copy
import math
import sys

import cv2
import random as rng
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import statistics
import collections

# Returns the translation needed from a contour to a given point
@nb.njit(parallel=False, fastmath=True)
def get_translation(source_point: (float, float), target_point: (float, float)) -> (float, float):
    return target_point[0] - source_point[0], target_point[1] - source_point[1]


# Moves a contour by a given translation
@nb.njit(parallel=False, fastmath=True)
def move_contour(contour: [(float, float)], translation: (float, float)) -> []:
    new_contour = []
    for i in nb.prange(len(contour)):
        point = contour[i][0] + translation[0], contour[i][1] + translation[1]
        new_contour.append(point)
    return new_contour


def match_contour(top: [(float, float)],
                  bot: [(float, float)],
                  progress: []) -> (float, float):
    blue = (255, 0, 0)
    green = (0, 255, 0)
    if len(top) == 0 or len(bot) == 0:
        return 0, 0
    print(f"Matching contours with len: {len(top)} and len: {len(bot)}")
    distances = collect_distance(top, bot)
    if np.mean(np.array(distances)) > 2000:
        return 0, 0
    min_length = 100
    if abs(len(top) - len(bot)) > 200 or len(top) < min_length or len(bot) < min_length:
        return 0, 0

    # Perfomace reasons lol
    top = move_contour(top, (0, 0))
    bot = move_contour(bot, (0, 0))
    #display_contours([top, bot], [blue, green], wait=1,
    #                 save_name="before_matching.png")
    matching_pair, best_match = check_2_contours(top, bot, progress)
    matching_pair_reverse, best_match_reverse = check_2_contours(bot, top, progress)
    final_transition = get_translation(bot[matching_pair[0]], top[-matching_pair[1]])
    if best_match_reverse > best_match:
        matching_pair, best_match = matching_pair_reverse, best_match_reverse
        final_transition = get_translation(bot[-matching_pair[1]], top[matching_pair[0]])
    bot = move_contour(bot, final_transition)
    print(f"Matching pair: {matching_pair}, t: {final_transition}, confidence: {best_match:.2f}, "
          f"len top:{len(top)} len bot: {len(bot)}")
    display_contours([top, bot], colors=[blue, green], wait=0, name=str(best_match).format(5),
                     save_name=str(best_match).format(5)+"contours.png")

    if best_match < 0.01:
        final_transition = (0, 0)
    return final_transition

@nb.njit(parallel=False, fastmath=True)
def check_2_contours(top, bot, progress):
    best_match = 0
    matching_pair = (0, 0)
    for i in range(0, len(bot)):
        for j in range(0, 50, 1):
            # Move target to index of source and check distances
            translation = get_translation(bot[i], top[-j])
            bot = move_contour(bot, translation)
            distances = collect_distance(top, bot)
            percentage_of_zero = distances.count(0) / len(distances)
            #print(f"Percentage zeros: {percentage_of_zero * 100:.2f}%, i: {i}, j: {j}")
            blue = (255, 255, 0)
            green = (0, 255, 0)
            #display_contours([top, bot], [blue, green], wait=1,
            #                         text=f"{percentage_of_zero:.4f}")

            if percentage_of_zero > best_match:
                best_match = percentage_of_zero
                matching_pair = i, j
            #display_contours([top, bot], wait=1)
            progress[0] += 2
    return matching_pair, best_match


def display_plots(datas: [[]]):
    x_axis = 1020
    for data in datas:
        plt.plot(data, color='r')
    plt.show()


@nb.njit(parallel=False, fastmath=True)
def collect_distance(source: [(float, float)], target: [(float, float)]):
    distances = []
    for i in nb.prange(len(source)):
        index, distance = find_nearest_point(source[i], target)
        distances.append(distance)
        #get the vectors as well
    return distances


@nb.njit(parallel=False, fastmath=True)
def find_nearest_point(source, target: []) -> (int, float):
    #print(f"Finding nearest point")
    min_dist = sys.maxsize
    last_index = -1
    for i in nb.prange(len(target)):
        distance = compare_point(source, target[i])
        if distance < min_dist:
            last_index = i
            min_dist = distance
        if distance == 0:
            break
    return last_index, min_dist


@nb.njit(fastmath=True)
def compare_point(a, b):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def get_boundaries(contours: [[(float, float)]]):
    x, y, w, h = (sys.maxsize, sys.maxsize, 0, 0)
    for c in contours:
        for point in c:
            x = min(x, math.floor(point[0]))
            y = min(y, math.floor(point[1]))
            w = max(w, math.ceil(point[0]))
            h = max(h, math.ceil(point[1]))
    return x, y, w, h


def display_contours(contours: [[(float, float)]], colors: [],
                     wait=0, name="Contour", save_name="", text=""):
    x, y, w, h = get_boundaries(contours)
    # move the contour to (0,0)
    temp_contours = copy.deepcopy(contours)
    for i in range(len(temp_contours)):
        t = get_translation((x, y), (0, 0))
        temp_contours[i] = move_contour(temp_contours[i], t)

    new_blank_image = np.zeros((h + 1 - y, max(w + 1 - x, 500), 3), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(new_blank_image,
                text,
                (250-len(text)-10, 100),
                font, 1,
                (0, 255, 0), 2)
    index = 0
    for contour in contours:
        for p in contour:
            new_blank_image[p[1] - y, p[0] - x] = colors[index % len(colors)]
        index += 1
    show_image(new_blank_image, wait, name, save_name)
    return new_blank_image

def display_contours_in_image(contours: [[(float, float)]], image, colors: [], offset=(0, 0),
                              wait=0, name="Contour",
                              save_name=""):
    tmp_image = image.copy()
    index = 0
    for contour in contours:
        for p in contour:
            tmp_image[p[1] + offset[0], p[0] + offset[1]] = colors[index % len(colors)]
        index += 1
    show_image(tmp_image, wait, name, save_name)


def show_image(image, wait=0, name="Damo", save_name=""):
    scale = 0.5
    if image.shape[0] > 1000 or image.shape[1] > 1000:
        scale = 0.2
    copy = image.copy()
    small = cv2.resize(copy, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imshow(name, small)
    cv2.waitKey(wait)
    if save_name != "":
        print(f"Trying to save to: {save_name}...")
        cv2.imwrite(f"output/{save_name}", copy)

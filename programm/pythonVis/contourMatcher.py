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
def move_contour(contour: [(float, float)], translation: (float, float)) -> []:
    new_contour = []
    for point in contour:
        point = point + translation
        new_contour.append(point)
    return new_contour


def match_contour(source: [(float, float)],
                  target: [(float, float)],
                  progress: []) -> (float, float):
    print(f"Matching contours with len: {len(source)} and len: {len(target)}")
    distances = collect_distance(source, target)
    if statistics.fmean(distances) > 2000:
        print(f"To far away, abort. Average distance {statistics.fmean(distances)}")
        return
    if abs(len(source) - len(target)) > 200 or len(source) < 50 or len(target) < 50:
        print("Difference too high, abort")
        return

    translation = get_translation(source[0], target[-1])
    final_transition = (0,0)
    best_match = 0
    for i in range(0, len(source)):
        for j in range(1, 10):
            # Move target to index of source and check distances
            source = move_contour(source, translation)
            final_transition = (final_transition[0] + translation[0],
                                final_transition[1] + translation[1],)
            distances = collect_distance(source, target)
            percentage_of_zero = distances.count(0) / len(distances)
            #print(f"Percentage zeros: {percentage_of_zero * 100:.2f}%")
            if percentage_of_zero > best_match:
                best_match = percentage_of_zero
            display_contours([source, target], wait=1)
            progress[0] = i / (len(source) * 10)
            translation = get_translation(source[i], target[-j])
    print(f"Best match found: {best_match}")
    if best_match < 0.1:
        final_transition = (0,0)
    print(f"Final translation: {final_transition}")
    return final_transition


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
    min_dist = sys.maxsize
    last_index = -1
    for i in range(len(target)):
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


def display_contours(contours: [[(float, float)]], color=(255, 255, 255), wait=0):
    x, y, w, h = get_boundaries(contours)
    # move the contour to (0,0)
    temp_contours = copy.deepcopy(contours)
    for i in range(len(temp_contours)):
        t = get_translation((x, y), (0, 0))
        temp_contours[i] = move_contour(temp_contours[i], t)

    new_blank_image = np.zeros((h + 1 - y, w + 1 - x, 3), np.uint8)
    for contour in temp_contours:
        for p in contour:
            new_blank_image[round(p[1]), round(p[0])] = color
    show_image(new_blank_image, wait)


def show_image(image, wait=0):
    scale = 0.7
    if image.shape[0] > 2000 or image.shape[1] > 2000:
        scale = 0.3
    copy = image.copy()
    small = cv2.resize(copy, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imshow('Contours', small)
    cv2.waitKey(wait)

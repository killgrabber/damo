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
def get_translation(source_point: (float, float), target_point: (float, float)) -> (float, float):
    # find translation
    x = target_point[0] - source_point[0]
    y = target_point[1] - source_point[1]
    return x, y


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
    if len(source) != len(target):
        print("Warning: source and target not equal length, results may vary")
    if abs(len(source) - len(target)) > 100 or len(source) < 100 or len(source) < 100:
        print("Difference too high, abort")
        return 0, 0
    i = 0
    translation = (0, 0)
    best_translation = translation
    best_match = 0
    while i < len(source):
        # Move target to index of source and check distances
        translation = get_translation(source[i], target[0])
        moved_contour = move_contour(source, translation)
        distances = collect_distance(target, moved_contour)
        percentage_of_zero = distances.count(0) / len(distances)
        print(f"Percentage zeros: {percentage_of_zero*100:.2f}%")
        if percentage_of_zero > best_match:
            best_match = percentage_of_zero
            best_translation = translation
        #display_plots([distances])
        #plt.hist(distances, bins = 1000)
        #plt.show()
        display_contours([moved_contour, target])
        i += 1
        progress[0] = i / len(target)

    return best_translation


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


def display_contours(contours: [[(float, float)]], color=(255, 255, 255)):
    x, y, w, h = (sys.maxsize, sys.maxsize, 0, 0)
    for c in contours:
        for point in c:
            x = min(x, point[0])
            y = min(y, point[1])
            w = max(w, point[0])
            h = max(h, point[1])
    #print(f"Bounds are [{x},{y}:{w},{h}]")
    new_blank_image = np.zeros((h + 1, w + 1, 3), np.uint8)
    for contour in contours:
        for p in contour:
            new_blank_image[round(p[1]), round(p[0])] = color
    show_image(new_blank_image, 1)


def show_image(image, wait=0):
    scale = 0.7
    if image.shape[0] > 2000 or image.shape[1] > 2000:
        scale = 0.3
    copy = image.copy()
    small = cv2.resize(copy, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imshow('Contours', small)
    cv2.waitKey(wait)

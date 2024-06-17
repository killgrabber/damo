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


def match_contour(source: [(float, float)], target: [(float, float)], inverted=False) -> (float, float):
    print("Matching contours...")
    if len(source) != len(target):
        print("Warning: source and target not equal length, aborting")
        return
    i = 100
    translation = (0,0)
    while i < len(target):
        # Move target to index of source and check distances
        translation = get_translation(source[i], target[0])
        moved_contour = move_contour(source, translation)
        distances = collect_distance(target, moved_contour)
        distance_min_25 = np.percentile(distances, [5, 15, 10, 15, 18, 25], method="normal_unbiased")
        #print(f"distance percentage: {distance_min_25}")
        if distance_min_25[4] == 0:
            #found best match
            print(f"Found best match at {i}, transformation: {translation}, distance 18%: {distance_min_25}")
            break

        #display_plots([distances])
        #plt.hist(distances, bins = 1000)
        #plt.show()
        display_contours([moved_contour, target])
        i += 1
        if i%100 == 0:
            print(f"Done: {i/len(target)}")
    if inverted:
        translation = (translation[0] * -1,
                       translation[1] * -1)

    return translation

def display_plots(datas: [[]]):
    x_axis = 1020
    for data in datas:
        plt.plot( data, color='r')
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
    max_values = np.max(np.max(contours, axis=0), axis=0)
    min_values = np.min(np.min(contours, axis=0), axis=0)
    shape = max_values - min_values
    new_blank_image = np.zeros((shape[1] + 10, max(shape[0] + 10, 400), 3), np.uint8)
    for contour in contours:
        for p in contour:
            new_blank_image[round(p[1])-min_values[1], round(p[0])-min_values[0]] = color
    show_image(new_blank_image, 1)

def show_image(image, wait=0):
    scale = 0.7
    if image.shape[0] > 2000 or image.shape[1] > 2000:
        scale = 0.3
    copy = image.copy()
    small = cv2.resize(copy, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imshow('Contours', small)
    cv2.waitKey(wait)
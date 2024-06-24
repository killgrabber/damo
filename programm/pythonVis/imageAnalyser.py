import math
import sys

import cv2
import random
import numpy as np
import numba as nb
import string
import contourMatcher
import difflib

IMAGES = [
    'pcd_2916383.png',
    'pcd_3061898.png'
]


def importImages():
    images = []
    for i in range(len(IMAGES)):
        img = cv2.imread(IMAGES[i], cv2.IMREAD_GRAYSCALE)
        img = cv2.blur(img, (10, 10))
        images.append(img)
    return images


def getContours(img):
    # draw contours on the original image
    height, width = img.shape[:2]
    image_contour_blue = img.copy()
    image_gray = cv2.cvtColor(image_contour_blue, cv2.COLOR_BGRA2GRAY)
    contours1, hierarchy1 = cv2.findContours(image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print("Contours found: ", len(contours1))
    #cv2.drawContours(image_contour_blue, contours1, -1, (0, 255, 0), 3)
    #cv2.imshow("blue contours", image_contour_blue)
    #cv2.waitKey(1)
    # remove contour points on the edge
    all_contours = []
    max_distance = 5
    for contour in contours1:
        all_points = []
        for points in contour:
            for point in points:
                if not (point[0] < max_distance or point[1] < max_distance or
                        abs(point[0] - height) < max_distance or
                        abs(point[0] - width) < max_distance or
                        abs(point[1] - height) < max_distance or
                        abs(point[1] - width) < max_distance):
                    all_points.append(point)
        all_contours.append(all_points)
    all_contours.sort(key=len, reverse=True)
    return all_contours


def drawContours(image, counturs, hierarchy1, x, y):
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255)
    ]
    for i in range(len(counturs)):
        print("{:.4f}%".format(round((i / len(counturs)) * 100, 4)), end="\r")

        cv2.drawContours(image, counturs, i, colors[i % 3], 2,
                         cv2.LINE_8, hierarchy1, 0, offset=(x, y))
    return image


def save_image(image, name: str):
    cv2.imwrite(name, image)

def showAndSaveImage(image, wait=0, name=""):
    scale = 0.5
    if image.shape[0] > 2000 or image.shape[1] > 2000:
        scale = 0.3
    if image.shape[0] > 5000 or image.shape[1] > 5000:
        scale = 0.2
    copy = image.copy()
    small = cv2.resize(copy, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imshow('Contours', small)
    cv2.waitKey(wait)
    if name != "":
        cv2.imwrite(name + '.jpg', image)
    #cv2.destroyAllWindows()


def move_contour(contour: [(float, float)], transform: (float, float)) -> []:
    new_contour = []
    for point in contour:
        point = point + transform
        new_contour.append(point)
    return new_contour


# returns the transformation coordinates to match them
# retunrs x, y transformation and confidence
def match2contours(con1: [], con2: []):
    if len(con1) != len(con2):
        print("Warning: Not same amount of points...")
        return
    # remove the first and last elements
    # con1 = con1[math.floor(len(con1) * 0.1):]
    # con2 = con2[:math.floor(len(con2) * 0.9)]

    # Move the center points on top of each other
    transformation = (0, 0)
    basic_transform = basic_align(con1, con2)
    con2 = move_contour(con2, basic_transform)
    transformation = tuple(np.add(transformation, basic_transform))

    max_iterations = 10000
    last_icp_sum = sys.maxsize
    inital_step_size = -100
    last_transformation = (0, inital_step_size)
    switched = False
    old_distance = 0
    # first sum up the distance between every point to the closest point
    for j in range(max_iterations):
        new_sum, amount_points = icp_step(con1, con2)
        if new_sum != 0 and amount_points > 100:
            distance = new_sum - last_icp_sum
            # small enough, go right and left now
            print("distance, Diff:", distance, old_distance - distance)
            if abs(old_distance - distance) < 0.001:
                if switched:
                    #print("Finished!")
                    break
                #print("Changing to left and right")
                last_transformation = (inital_step_size, 0)
                switched = True
            old_distance = distance
            # if distance has increased, choose another direction
            if new_sum > last_icp_sum:
                #print("Distance has increased, go back and decrease step size")
                last_transformation = (last_transformation[0] * -0.5,
                                       last_transformation[1] * -0.5)
            last_icp_sum = new_sum
        # move con2 and repeat
        con2 = move_contour(con2, last_transformation)
        # accumulate the transformations we did
        transformation = tuple(np.add(transformation, last_transformation))
        #print("transformation: ", transformation)
        new_blank_image = np.zeros((5000, 5000, 3), np.uint8)
        yellow = (0, 255, 255)
        light_blue = (255, 255, 0)
        pink = (255, 0, 255)
        draw_points(new_blank_image, con1, 500, 0, yellow)
        draw_points(new_blank_image, con2, 500, 0, light_blue)
        showAndSaveImage(new_blank_image, 2)

    print("last sum: ", last_icp_sum)
    return transformation, last_icp_sum


def basic_align(con1: [], con2: []) -> (float, float):
    # get the center points and align those
    center_1 = con1[math.floor(len(con1) / 2)]
    center_2 = con2[math.floor(len(con1) / 2)]
    t_vector = (center_1[0] - center_2[0],
                center_1[1] - center_2[1])
    print("Distance vector: ", t_vector)
    return t_vector


def icp_step(con1: [], con2: []) -> (float, int):
    icp_sum = 0
    point_amount = 0
    for i in nb.prange(len(con1)):
        index, distance = find_nearest_point(con1[i], con2)
        if len(con1) * 0.15 < index < len(con1) * 0.85:
            #print("Distace, index: ", distance, index)
            icp_sum += distance
            point_amount += 1
    print("matching points: ", point_amount)
    return icp_sum, point_amount


# returns the index and distance to the closest point in target
def find_nearest_point(source, target: []) -> (int, float):
    min_dist = sys.maxsize
    last_index = -1
    for i in range(len(target)):
        distance = compare_point(source, target[i])
        if distance < min_dist:
            last_index = i
            min_dist = distance
    return last_index, min_dist


@nb.njit(fastmath=True)
def compare_point(a, b):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def compareContour(con1, con2):
    sum = 0
    maxRange = min(len(con1), len(con2))
    minDiff = 1000000
    diff = 0
    for i in range(len(con1)):
        for j in range(len(con2)):
            diff = compare_point(con1[i][0], con2[j][0])
            if (diff < minDiff):
                minDiff = diff
        sum += diff
    return sum


def compareContours(contours1: np.array, contours2: np.array):
    diff = compareContour(contours1[0], contours2[0])
    print(diff)


def cropImage(image, top, size=1000, offset=50):
    height, width = image.shape[:2]
    if top:
        crop_img = image[height - size - offset:height-offset, 0:width]
    else:
        crop_img = image[offset:size, 0:width]
    return crop_img


def draw_points(image, points, offset_x, offset_y, color=(255, 255, 255)):
    for p in points:
        image[round(p[1]) + offset_x, round(p[0]) + offset_y] = color
    return image


def cropContour(contour, x1, y1, x2, y2):
    points = []
    for c in contour:
        for c1 in c:
            for c2 in c1:
                x, y = c2
                if x1 < x < x2 and y1 < y < y2:
                    points.append((x, y))
    return points


def translate_image(img, x, y):
    matrix = np.float32([
        [1, 0, round(y)],
        [0, 1, round(x)]
    ])
    warp_dst = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
    return warp_dst


def combine_2_images(img1, img2, overlap=200):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((h1 + h2, max(w1, w2), 3), np.uint8)
    vis[:h1 - overlap, :w1] = img1[:h1 - overlap, :w1]
    vis[h1 - overlap:h1 + h2 - overlap, :w2] = img2
    return vis

def get_matching_name(name1: str, name2: str) -> str:
    top_name = name1.split('/')[-1].replace('.png', '')
    bot_name = name2.split('/')[-1].replace('.png', '')
    matcher = difflib.SequenceMatcher(a=top_name, b=bot_name)
    match = matcher.find_longest_match(0, len(top_name), 0, len(bot_name))
    matching_name = matcher.a[match.a:match.a + match.size][:-1]
    return matching_name

def cut_borders(image: cv2.Mat):
    # Crop borders
    print("Cropping ")
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(f"Found {len(contours)} contours")
    max_c = max(contours, key=len)
    x, y, w, h = cv2.boundingRect(max_c)
    crop = image[y:y + h, x:x + w]
    return crop



def stitch_images_path(top_image_path: str, bot_image_path: str, progress: [], result: []):
    top_image = cv2.imread(top_image_path, cv2.IMREAD_GRAYSCALE)
    bot_image = cv2.imread(bot_image_path, cv2.IMREAD_GRAYSCALE)
    stitch_images(top_image, bot_image, progress, result)


def stitch_images(top_image: cv2.Mat, bot_image: cv2.Mat, progress: [], result: []):
    top_image = cv2.blur(top_image, (5, 5))
    bot_image = cv2.blur(bot_image, (5, 5))

    cut_size = 500
    crop_top = cropImage(top_image, True, cut_size, 100)
    crop_bot = cropImage(bot_image, False, cut_size, 100)
    height, width = crop_bot.shape[:2]
    #showAndSaveImage(crop_top)
    #showAndSaveImage(crop_bot)

    contours_top = getContours(crop_top)
    contours_bot = getContours(crop_bot)

    # Move bottom contours to the original image can be translated with the same matrix
    bottom_contours = []
    for con in contours_bot:
        bottom_contours.append(move_contour(con, height))

    transitions = []
    # Match all contours
    for c_top in contours_top:
        for c_bot in bottom_contours:
            transitions.append(contourMatcher.match_contour(c_top, c_bot, progress))

    final_transition = (0,0)
    for t in transitions:
        final_transition = t

    # move second image
    overlap = 200
    moved_image = translate_image(bot_image, final_transition[1] + overlap, final_transition[0] + cut_size)
    combined_image = combine_2_images(top_image, moved_image, overlap)
    result.append(combined_image)
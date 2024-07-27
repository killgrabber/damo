import math
import sys

import cv2
import random
import numpy as np
import numba as nb
import string
import contourMatcher
import difflib
import random as rng
from collections import Counter
from threading import Thread


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


def getContours(img, min_distance=5, limit=100, segment_distance=10):
    # draw contours on the original image
    height, width = img.shape[:2]
    image_contour_blue = img.copy()
    image_gray = cv2.cvtColor(image_contour_blue, cv2.COLOR_BGRA2GRAY)
    contours1, hierarchy1 = cv2.findContours(image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print("Contours found: ", len(contours1))
    #for i in range(len(contours1)):
    #    color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    #    cv2.drawContours(image_contour_blue, contours1, i, color, 2, cv2.LINE_8, hierarchy1, 0)

    #cv2.imshow("con", cv2.resize(image_contour_blue, (0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA))
    #cv2.imwrite("test.png", image_gray)
    #cv2.waitKey(0)
    # remove contour points on the edge
    all_contours = []
    for contour in contours1:
        all_points = []
        for points in contour:
            for point in points:
                if not (point[0] < min_distance or point[1] < min_distance or
                        abs(point[0] - height) < min_distance or
                        abs(point[0] - width) < min_distance or
                        abs(point[1] - height) < min_distance or
                        abs(point[1] - width) < min_distance):
                    all_points.append(point)
        all_contours.append(all_points)
    all_contours.sort(key=len, reverse=True)

    # return all_contours
    # seperate contours
    new_contours = []
    for i in range(len(all_contours)):
        new_points = []
        for j in range(1, len(all_contours[i])):
            distance = compare_point(all_contours[i][j], all_contours[i][j - 1])
            if distance < segment_distance:
                new_points.append(all_contours[i][j].copy())
            else:
                print("New segment")
                new_contours.append(new_points.copy())
                new_points.clear()
                new_points.append(all_contours[i][j].copy())
        new_contours.append(new_points.copy())
    print(f"Returning {len(new_contours)} contours")
    if limit < len(new_contours):
        return new_contours[:limit]
    return new_contours


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


def showAndSaveImage(image, wait=0, name="", window_name="test"):
    scale = 1
    if image.shape[0] > 2000 or image.shape[1] > 2000:
        scale = 0.5
    if image.shape[0] > 5000 and image.shape[1] > 5000:
        scale = 0.2
    copy = image.copy()
    small = cv2.resize(copy, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imshow(window_name, small)
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
        crop_img = image[height - size - offset:height - offset, offset:width - offset]
    else:
        crop_img = image[offset:size + offset, offset:width - offset]
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
        [1, 0, round(x)],
        [0, 1, round(y)]
    ])
    warp_dst = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
    return warp_dst


def combine_2_images(img1, img2, overlap=200):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    print(f"Shape1: {h1},{w1}, shape2: {h2},{w2}")
    vis = np.zeros((h1 + h2 - overlap, max(w1, w2), 3), np.uint8)
    vis[:h1 - overlap, :w1] = img1[:h1 - overlap, :w1]
    vis[h1 - overlap:h1 + h2 - overlap, :w2] = img2
    for i in range(0, overlap):
        for j in range(0, min(w1, w2)):
            r1, g1, b1 = img1[h1 - i - 1, j]
            r2, g2, b2 = img2[overlap - i, j]
            vis[h1-i-1, j] = (max(r1, r2), max(g1, g2), max(b1, b2))

    return vis


def get_matching_name(name1: str, name2: str, ending = '.png') -> str:
    top_name = name1.split('/')[-1].replace(ending, '')
    bot_name = name2.split('/')[-1].replace(ending, '')
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


def match_all_contours(contours_top, bottom_contours, progress: []):
    transitions = [(0,0)]
    blue = (255, 255, 100)
    green = (200, 255, 200)
    #contourMatcher.display_contours(contours_top, colors=[blue, green], wait=0,
     #                               name="top_contours",
     #                               save_name="top_contours.png")
    # Match all contours
    matcher_threads = []

    for c_top in contours_top:
        for c_bot in bottom_contours:
            print(f"Starting contour matcher thread...")
            #transitions.append(contourMatcher.match_contour(c_top, c_bot, transitions, progress))
            matcher_thread = Thread(target=contourMatcher.match_contour,
                                    args=(c_top, c_bot, transitions, progress))
            matcher_thread.start()
            matcher_threads.append(matcher_thread)

    for t in matcher_threads:
        t.join()

    return transitions


def show_image_and_contours(image, contours, wait=0, window_name="test"):
    temp_image = image.copy()
    for contour in contours:
        temp_image = draw_points(temp_image, contour, 0, 0, (0, 0, 255))

    showAndSaveImage(temp_image, wait, window_name=window_name)


def stitch_images(top_image: cv2.Mat, bot_image: cv2.Mat, progress: [], result: []):
    if isinstance(top_image, str):
        top_image = cv2.imread(top_image)
    if isinstance(bot_image, str):
        bot_image = cv2.imread(bot_image)
    progress[0] += 10
    blur_size = 2
    top_image_blur = cv2.blur(top_image, (blur_size, blur_size))
    bot_image_blur = cv2.blur(bot_image, (blur_size, blur_size))
    cut_size = 500
    offset = 200
    crop_top = cropImage(top_image_blur, True, cut_size, offset=offset)
    crop_bot = cropImage(bot_image_blur, False, cut_size, offset=offset)
    progress[0] += 10

    height, width = crop_top.shape[:2]
    #print(f"Size topcrop: {crop_top.shape}, ")
    #print(f"Size botcrop: {crop_bot.shape}, ")
    #showAndSaveImage(crop_top)
    #showAndSaveImage(crop_bot)
    min_distance_from_border = 5
    contours_top = getContours(crop_top, min_distance_from_border, limit=20)
    print(f"Len of con: {len(contours_top)}")
    progress[0] += 5
    contours_bot = getContours(crop_bot, min_distance_from_border, limit=20)
    print(f"Len of con: {len(contours_bot)}")
    progress[0] += 5

    # move bot contour
    bot_contours = []
    for bot_c in contours_bot:
        bot_contours.append(move_contour(bot_c, (0, 0)))

    blue = (255, 255, 100)
    green = (200, 255, 200)
    contourMatcher.display_contours(contours_top + bot_contours, wait=0, colors=[blue, green])

    transitions = match_all_contours(contours_top, bot_contours, progress)
    transitions_dict = Counter(transitions)

    print(f"transitions: {transitions_dict}")

    # move second image
    stitch_offset = -831
    overlap = 100
    for transition, amount in sorted(transitions_dict.items(), key=lambda item: item[1], reverse=True):
        print(f"Moving t {transition}")
        moved_image_tmp = translate_image(bot_image, transition[0]+1,
                                      stitch_offset + transition[1])
        combined_image_tmp = combine_2_images(top_image, moved_image_tmp, 100)
        showAndSaveImage(combined_image_tmp, name="stitched")

    moved_image = translate_image(bot_image, round(transitions[0][0]) + 1,
                                  stitch_offset + round(transitions[0][1]))
    combined_image = combine_2_images(top_image, moved_image, overlap)
    progress[0] = 100
    #showAndSaveImage(combined_image)
    result.append(combined_image)


def show_image_tresh(top_image_path, bot_image_path, treshold_min: [], treshold_max: []):
    #cv2.namedWindow("loop")
    cv2.namedWindow("loop2")
    top_image = cv2.imread(top_image_path)
    bot_image = cv2.imread(bot_image_path)
    cut_size = 500
    offset = 200
    crop_top = cropImage(top_image, True, cut_size, offset)
    crop_bot = cropImage(bot_image, False, cut_size, offset)

    #top_image_blur = cv2.blur(crop_top, (5, 5))
    #bot_image_blur = cv2.blur(crop_bot, (5, 5))
    small_top = cv2.resize(top_image, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
    small_bot = cv2.resize(bot_image, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
    overlap = 50
    while True:
        do_some_threshing(small_bot, treshold_min, treshold_max)
        continue
        moved_image = translate_image(small_bot, treshold_min[0], treshold_max[0])
        #contours_top = getContours(cv2.blur(small_top, (5, 5)))
        #contours_bot = getContours(cv2.blur(moved_image, (5, 5)))
        combined_image = combine_2_images(small_top, moved_image, overlap)
        #index = 0
        #for i in range(0):
        #    for j in range(len(contours_bot)):
        #        if contours_top[i] and contours_bot[j]:
        #            distances = contourMatcher.collect_distance(contours_top[i], contours_bot[j])
        #            if sum(distances) > 100000:
        #                continue
        #            contourMatcher.display_contours([contours_top[i], contours_bot[j]],
        #                                            name=f"{i},{j}", wait=1,
        #                                            save_name="start_cons.png")
        #            percentage_of_zero = distances.count(0) / len(distances)
        #            match_text = (f"i,j: {i:2.0f},{j:2.0f} zeros: {percentage_of_zero:2.3f}, "
        #                          f"sum: {sum(distances):.3f}, "
        #                          f"lens:{len(contours_top[i])}, {len(contours_bot[j])}")
        #            cv2.putText(combined_image, match_text, (500, (index) * 30 + 80), 1, 2, 255)
        #            index += 1

        text = f"Min: {treshold_min[0]} max {treshold_max[0]}"
        print(text)
        cv2.putText(combined_image, text, (500, 50), 1, 2, 255)
        cv2.imshow("loop2", combined_image)

        cv2.waitKey(1)


def do_some_threshing(image_gray, treshold_min, treshold_max):
    print(f"Min: {treshold_min[0]} max {treshold_max[0]}")
    ret, thresh = cv2.threshold(image_gray, treshold_min[0], treshold_max[0], 0)
    #contours1, hierarchy1 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #blank = np.zeros(image_gray.shape)
    #print("Contours found: ", len(contours1))
    #for i in range(len(contours1)):
    #    color = (255, 255, 255)
    #    cv2.drawContours(blank, contours1, i, color, 2, cv2.LINE_8, hierarchy1, 0)
    # cv2.imshow("loop", thresh)
    cv2.imshow("loop2", thresh)
    cv2.waitKey(1)

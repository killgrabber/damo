import math
import sys

import cv2
import random as rng
import numpy as np
import numba as nb

IMAGES = [
    'pcd_2944702.png',
    'pcd_3072377.png'
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
    height, width = img.shape
    image_contour_blue = img.copy()
    image_contour_blue = cv2.cvtColor(image_contour_blue, cv2.COLOR_GRAY2RGB)
    contours1, hierarchy1 = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print("Contours found: ", len(contours1))

    # remove contour points on the edge
    all_contours = []
    max_distance = 10
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
        print("Adding points to contours: ", len(all_points))
        all_contours.append(all_points)
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


def showAndSaveImage(image, wait=0):
    scale = 0.5
    if image.shape[0] > 2000 or image.shape[1] > 2000:
        scale = 0.3
    copy = image.copy()
    small = cv2.resize(copy, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imshow('Contours', small)
    cv2.waitKey(wait)
    #cv2.imwrite('image.jpg', image)
    #cv2.destroyAllWindows()


def move_contour(contour: [(float, float)], transform: (float, float)) -> []:
    new_contour = []
    for point in contour:
        point = point + transform
        new_contour.append(point)
    return new_contour


@nb.njit(fastmath=True)
def compare_point(a, b):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


# returns the transformation coordinates to match them
# retunrs x, y transformation and confidence
def match2contours(con1: [], con2: []):
    if len(con1) != len(con2):
        print("Warning: Not same amount of points...")
        return
    # remove the first and last elements
    # con1 = con1[math.floor(len(con1) * 0.1):]
    # con2 = con2[:math.floor(len(con2) * 0.9)]

    max_iterations = 10000
    last_icp_sum = sys.maxsize
    inital_step_size = -100
    last_transformation = (0, inital_step_size)
    transformation = (0, 0)
    switched = False
    old_distance = 0
    # first sum up the distance between every point to the closest point
    for j in range(max_iterations):
        new_sum, amount_points = icp_step(con1, con2)
        if new_sum != 0 and amount_points > 100:
            distance = new_sum - last_icp_sum
            # small enough, go right and left now
            print("Diff:", old_distance - distance)
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


def icp_step(con1: [], con2: []) -> (float, int):
    icp_sum = 0
    point_amount = 0
    for i in nb.prange(len(con1)):
        index, distance = find_nearest_point(con1[i], con2)
        if len(con1) * 0.1 < index < len(con1) * 0.9:
            #print("Distace, index: ", distance, index)
            icp_sum += distance
            point_amount += 1
    print("matching points: ")
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


def cropImage(image, top, length=1000):
    height, width = image.shape
    if top:
        crop_img = image[height - length:height, 0:width]
    else:
        crop_img = image[0:length, 0:width]
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
        [1, 0, y],
        [0, 1, x]
    ])
    warp_dst = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
    return warp_dst


def main():
    images = importImages()
    # images[0] = cropImage(images[0], True)
    # images[1] = cropImage(images[1], False)
    crop_top = cropImage(images[0], True, 1000)
    crop_bot = cropImage(images[1], False, 1000)
    height, width = crop_bot.shape
    tr = (0, height)
    # showAndSaveImage(crop_top)
    # showAndSaveImage(crop_bot)

    contours1 = getContours(crop_top)
    contours2 = getContours(crop_bot)
    for i in range(len(contours2)):
        contours2[i] = move_contour(contours2[i], tr)

    height, width = images[0].shape
    blank_image1 = np.zeros((height + 200, width + 200, 3), np.uint8)
    new_image = cv2.cvtColor(images[0], cv2.COLOR_GRAY2RGB)

    print("con1[0]: ", len(contours1[0]))
    print("con1[1]: ", len(contours1[1]))
    print("con2[0]: ", len(contours2[0]))
    print("con2[1]: ", len(contours2[1]))

    transformation_1, distance_1 = match2contours(contours1[1], contours2[0])
    distance_1 = 0
    distance_2 = 1
    transformation_1 = (-7.958984375, 258.935546875)
    transformation_2 = (-7.958984375, 258.935546875)
    x = 1000 - math.floor(transformation_1[0])
    y = math.floor(transformation_1[1])
    print("Final t: ", transformation_1)
    #transformation_2, distance_2 = match2contours(contours1[0], contours2[1])
    print("Final t2: ", transformation_2)
    transformation = transformation_1
    if distance_2 > distance_1:
        transformation = transformation_2
    t_image = translate_image(cv2.cvtColor(images[1], cv2.COLOR_GRAY2RGB), -x, 0)
    vis = np.concatenate((cv2.cvtColor(images[0], cv2.COLOR_GRAY2RGB), t_image),
                         axis=0)

    showAndSaveImage(vis)
    exit()

    new_con = move_contour(contours2[0], transformation)
    new_con2 = move_contour(contours2[1], transformation)

    yellow = (0, 255, 255)
    light_blue = (255, 255, 0)
    pink = (255, 0, 255)

    border_image = cv2.copyMakeBorder(new_image, 0, 500, 0, 0, 0)
    border_image = draw_points(border_image, contours1[0], height - 1000, 0, yellow)
    border_image = draw_points(border_image, new_con2, height - 1000, 0, pink)

    border_image = draw_points(border_image, contours1[1], height - 1000, 0, yellow)
    #border_image = draw_points(border_image, contours2[0], height - 1000, 0, light_blue)
    border_image = draw_points(border_image, new_con, height - 1000, 0, pink)

    showAndSaveImage(border_image)

    exit()
    for img in images:
        height, width = img.shape
        contours, hierarchy = getContours(img)
        cropped_con = cropContour(contours, 500, 0, height, 500)
        all_c.append(cropped_con)
        all_h.append(hierarchy)
    all_c[1] = moveContours(all_c[1], height)
    while True:
        print("--------------------")
        all_c[1] = moveContours(all_c[1], -100)
        print("Moved contour")
        # compareContours(all_c[0], all_c[1])
        new_blank_image = np.zeros((height * 2 + 100, width + 10, 3), np.uint8)
        print("blank image size: ", new_blank_image.shape)
        image = drawPoints(new_blank_image, all_c[0])
        # image = drawPoints(new_blank_image, all_c[1])
        showAndSaveImage(image)


if __name__ == "__main__":
    main()

import cv2
import random as rng
import numpy as np

IMAGES = [
    '..\..\output\images\pcd_2944702.png',
    '..\..\output\images\pcd_3072377.png'
]


def importImages():
    images = []
    for i in range(len(IMAGES)):
        img = cv2.imread(IMAGES[i], cv2.IMREAD_GRAYSCALE)
        img = cv2.blur(img, (10, 10))
        images.append(img)
    return images


def getContours(img) -> []:
    # draw contours on the original image
    height, width = img.shape
    image_contour_blue = img.copy()
    image_contour_blue = cv2.cvtColor(image_contour_blue, cv2.COLOR_GRAY2RGB)
    contours1, hierarchy1 = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print("Contours found: ", len(contours1))

    # remove contour points on the edge
    all_points = []
    max_distance = 10
    for contour in contours1:
        for points in contour:
            for point in points:
                if not (point[0] < max_distance or point[1] < max_distance or
                        abs(point[0] - height) < max_distance or
                        abs(point[0] - width) < max_distance or
                        abs(point[1] - height) < max_distance or
                        abs(point[1] - width) < max_distance):
                    all_points.append(point)
    return all_points


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


def showAndSaveImage(image):
    copy = image.copy()
    small = cv2.resize(copy, (0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    cv2.imshow('Contours', small)
    cv2.waitKey(0)
    #cv2.imwrite('image.jpg', image)
    cv2.destroyAllWindows()


def moveContours(contours: np.array, offset):
    x_offset, y_offset = 0, offset
    newContours = []
    for contour in contours:
        # contour with new offset is created
        new_contour = contour + (x_offset, y_offset)
        # draw the new contour on the image
        newContours.append(new_contour)
    return newContours


def comparePoint(a, b):
    return np.linalg.norm(a - b)


def compareContour(con1, con2):
    sum = 0
    maxRange = min(len(con1), len(con2))
    minDiff = 1000000
    diff = 0
    for i in range(len(con1)):
        for j in range(len(con2)):
            diff = comparePoint(con1[i][0], con2[j][0])
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
        image[p[1] + offset_x, p[0] + offset_y] = color
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


def main():
    images = importImages()
    # images[0] = cropImage(images[0], True)
    # images[1] = cropImage(images[1], False)
    # showAndSaveImage(images[0])
    # showAndSaveImage(images[1])
    crop_top = cropImage(images[0], True)
    crop_bot = cropImage(images[1], False)
    contours1 = getContours(crop_top)
    contours2 = getContours(crop_bot)
    height, width = images[0].shape
    blank_image1 = np.zeros((height + 200, width + 200, 3), np.uint8)
    new_image = cv2.cvtColor(images[0], cv2.COLOR_GRAY2RGB)
    border_image = cv2.copyMakeBorder(new_image, 0, 500, 0, 0, 0)
    border_image = draw_points(border_image, contours1, height - 1000, 0, (0, 0, 255))
    border_image = draw_points(border_image, contours2, height - 730, 0, (0, 0, 255))
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

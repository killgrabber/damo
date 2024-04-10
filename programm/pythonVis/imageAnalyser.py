import cv2
import random as rng

# read the image
img = cv2.imread('test4.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.blur(img, (10,10))
#cv2.imshow('Contour detection using blue channels only', src_gray)
#cv2.waitKey(0)
# detect contours using blue channel and without thresholding
 
# draw contours on the original image
image_contour_blue = img.copy()
image_contour_blue = cv2.cvtColor(image_contour_blue,cv2.COLOR_GRAY2RGB)
#cv2.imshow('Contour detection using blue channels only', image_contour_blue)
#cv2.waitKey(0)

contours1, hierarchy1 = cv2.findContours(img, cv2.RETR_LIST , cv2.CHAIN_APPROX_SIMPLE)
print(len(contours1))
for i in range(len(contours1)):
    print("{:.4f}%".format(round((i/len(contours1))*100, 4)), end="\r")
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    cv2.drawContours(image_contour_blue, contours1, i, color, 2, cv2.LINE_8, hierarchy1, 0)
# see the results
small = cv2.resize(image_contour_blue, (0,0), fx=0.5, fy=0.5) 
cv2.imshow('Contour detection using blue channels only', small)
cv2.waitKey(0)
cv2.imwrite('blue_channel.jpg', small)
cv2.destroyAllWindows()

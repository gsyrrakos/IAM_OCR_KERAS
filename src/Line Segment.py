import imutils

import numpy as np

import cv2

from src.Predict import Predict, PredictLine
from src.SpellChecker import correct_sentence

image = cv2.imread('C:/Users/giorgos/Desktop/prof.png')
# cv2.imshow('orig',image)
# cv2.waitKey(0)

# grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', gray)
cv2.waitKey(0)

# binary
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow('second', thresh)
cv2.waitKey(0)

# dilation
kernel = np.ones((2, 100), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
# cv2.imshow('dilated', img_dilation)
cv2.waitKey(0)

# find contours
ctrs, hier, = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


def sort_contours(cnts, method="top-to-bottom"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def draw_contour(image, c, i):
    # compute the center of the contour area and draw a circle
    # representing the center
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # draw the countour number on the image
    cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2)
    # return the image with the contour number drawn on it
    return image


# load the image and initialize the accumulated edge image
image = image
accumEdged = np.zeros(image.shape[:2], dtype="uint8")
# loop over the blue, green, and red channels, respectively
for chan in cv2.split(image):
    # blur the channel, extract edges from it, and accumulate the set
    # of edges for the image
    chan = cv2.medianBlur(chan, 11)
    edged = cv2.Canny(chan, 50, 200)
    accumEdged = cv2.bitwise_or(accumEdged, edged)
# show the accumulated edge map
cv2.imshow("Edge Map", accumEdged)

# find contours in the accumulated image, keeping only the largest
# ones
cnts = cv2.cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

orig = image.copy()

imgs = []

for i, ctr in enumerate(cnts):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROItransfr
    roi = image[y:y + h, x:x + w]

    # show ROI
    # cv2.imshow('segment no:'+str(i),roi)
    # grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)
    cv2.waitKey(0)
    imgs.append(gray)

    # cv2.imread(roi[i], cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('segment no:' + str(i), roi)

    # img=cv2.imwrite("C:/Users/giorgos/Desktop/Handwritten-Line-Text-Recognition-using-Deep-Learning-with-Tensorflow-master\src/photosegment/segment_no_" + str(i) + ".png", roi)

    # cv2.rectangle(image, (x, y), (x + w, y + h), (90, 0, 255), 2)
    cv2.waitKey(0)


def Line_seg():
    string = []
    for i in range(len(imgs)):
        # cv2.imshow('segment no:' + str(i), imgs[i])
        string.append(PredictLine(imgs[i]))

    string.reverse()
    string = ''.join([str(elem) + '\n' for elem in string])
    print("Without Correction", string)
    print("With Correction", correct_sentence(string))


Line_seg()

import numpy as np
import cv2

import skimage as sk
from numpy.core.multiarray import ndarray
from skimage import transform
from skimage import util
from skimage import io
import random


def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array, mode='gaussian', clip=True)


def blur_filter(img_array: ndarray):
    # blur the image
    return cv2.blur(img_array, (8, 8))

# for IAM dataset

def reduce_line_thickness(image: ndarray):
    kernel = np.ones((4,4), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


def random_stretch(img: ndarray):
    stretch = (random.random() - 0.5)  # -0.5 .. +0.5
    wStretched = max(int(img.shape[1] * (1 + stretch)), 1)  # random width, but at least 1
    img = cv2.resize(img, (wStretched, img.shape[0]))  # stretch horizontally by factor 0.5 .. 1.5
    return img





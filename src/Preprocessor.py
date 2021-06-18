import random
import numpy as np
import cv2

from src.Augment import *


def preprocessor(img, imgSize, enhance=False, dataAugmentation=False):
    "put img into target img of size imgSize, transpose for TF and normalize gray-values"

    # there are damaged files in IAM dataset - just use black image instead
    if img is None:
        img = np.zeros([imgSize[1], imgSize[0]])  # (64,800)
        print("Image None!")

    # increase dataset size by applying random stretches to the images
    img = img.astype(np.float)
    if dataAugmentation:
        # print("Image Aug!")

        # photometric data augmentation
        #reduce_line_thickness(img)

        # if random.random() < 0.25:
        # img = random_noise(img)

        if random.random() < 0.25:
              img = reduce_line_thickness(img)

        # if random.random() < 0.25:
        # print()
        # img = blur_filter(img)

        # if random.random() < 0.25:
        # img = random_stretch(img)

    # create target image and copy sample image into it
    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1),
               max(min(ht, int(h / f)), 1))  # scale according to f (result at least 1 and at most wt or ht)
    img = cv2.resize(img, newSize,
                     interpolation=cv2.INTER_CUBIC)  # INTER_CUBIC interpolation best approximate the pixels image
    # see this https://stackoverflow.com/a/57503843/7338066
    target = np.ones([ht, wt]) * 255  # shape=(64,800)
    target[0:newSize[1], 0:newSize[0]] = img

    # transpose for TF
    img = cv2.transpose(target)
    return img

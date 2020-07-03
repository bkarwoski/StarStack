#!/usr/bin/env python3
from matplotlib import pyplot as plt
import cv2
import numpy as np
import scipy
import os
import glob
import copy
from sklearn.neighbors import KDTree
import sys
from multiprocessing import Pool
from util import *

# imgs = load_images()
imgs_path = os.path.join(pathlib.Path().absolute(), "raw/*")
imgs_list = sorted(glob.glob(imgs_path))
#how many stars to look for in each frame
num_stars = 200
# showStarCoords(imgs[0], getStarCoords(imgs[0], num_stars=num_stars))

transforms = []
trans_prior = np.identity(3)
transforms.append(trans_prior)
init_img = load_raw(imgs_list[0])
points_init = getStarCoords(init_img, num_stars=num_stars)
weight = 1.0 / len(imgs_list)
img_stack = np.zeros((init_img.shape), dtype=np.float32)

for idx, fname in enumerate(imgs_list, start=1):
    img = load_raw(fname)
    # img = load_image(fname)
    points = getStarCoords(img, num_stars=num_stars)
    transform = icpTransform(points_init, points, prior=trans_prior)
    transforms.append(transform)
    trans_prior = transform
    img_stack_dim = img.shape
    print(str(idx), "transform found")
    frame_dim = (init_img.shape[1], init_img.shape[0])
    img_warp = cv2.warpAffine(img, transform[:2], frame_dim,
    borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 0, 0),
    flags=cv2.WARP_INVERSE_MAP)
    img_stack += img_warp * weight
    print(str(idx), "stacked")
print("images stacked")
# print("image stack size: ", str(sys.getsizeof(img_stack) / 1000000), " MB")

cv2.cvtColor(img_stack, cv2.COLOR_BGR2RGB)
cv2.imwrite("stack_mean.jpg", img_stack)
plotTransforms(transforms)

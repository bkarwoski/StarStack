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
from util import *

imgs_path = os.path.join(pathlib.Path().absolute(), "jpg/*")
imgs_list = sorted(glob.glob(imgs_path))
#how many stars to look for in each frame
num_stars = 100

transforms = []
trans_prior = np.eye(3, 3, dtype=np.float32)
transforms.append(trans_prior)
init_img = load_image(imgs_list[0])
points_init = get_star_coords(init_img, num_stars=num_stars)
img_stack = np.zeros((init_img.shape), dtype=np.float32)

for fname in imgs_list[1:]:
    img = load_image(fname)
    # points = getStarCoords(img, num_stars=num_stars)
    transform = ecc_transform(init_img, img, prior=trans_prior)
    # print(transform)
    transforms.append(transform)
    trans_prior = transform
    img_stack_dim = img.shape
    frame_dim = (init_img.shape[1], init_img.shape[0])
    img_warp = cv2.warpPerspective(img, transform, frame_dim,
    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 255, 0),
    flags=cv2.WARP_INVERSE_MAP)
    img_stack += img_warp
    # img_stack = np.maximum.reduce([img_stack, img_warp])
    print(fname, "stacked")
img_stack /= len(imgs_list)
out_name = "stack_mean.jpg"
cv2.imwrite(out_name, img_stack)
print(out_name, "saved")

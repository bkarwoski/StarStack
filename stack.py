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

imgs_path = os.path.join(pathlib.Path().absolute(), "images/*")
raw_path = os.path.join(pathlib.Path().absolute(), "raw/*")
imgs_list = sorted(glob.glob(imgs_path))
raw_list = sorted(glob.glob(raw_path))
#how many stars to look for in each frame
num_stars = 100

transforms = []
trans_prior = np.eye(3, 3, dtype=np.float32)
transforms.append(trans_prior)
init_img = load_image(imgs_list[0])
points_init = get_star_coords(init_img, num_stars=num_stars)
img_stack = np.zeros((init_img.shape), dtype=np.float32)

denoise_idxs = [x ** 2 - 1 for x in range(1, 12)]

for idx, fname in enumerate(imgs_list[1:]):
    img = load_image(fname)
    raw = load_image(raw_list[idx])
    # points = getStarCoords(img, num_stars=num_stars)
    transform = ecc_transform(init_img, img, prior=trans_prior)
    # print(transform)
    transforms.append(transform)
    trans_prior = transform
    img_stack_dim = img.shape
    frame_dim = (init_img.shape[1], init_img.shape[0])
    img_warp = cv2.warpPerspective(raw, transform, frame_dim,
    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 255, 0),
    flags=cv2.WARP_INVERSE_MAP)
    img_stack += img_warp
    if idx in denoise_idxs:
        center = (img_stack.shape[0] // 2, img_stack.shape[1] // 2)
        h = 200
        w = 300
        img_out = img_stack[center[0] - h:center[0] + h,
                            center[1] - w:center[1] + w] / (idx + 1)
        out = str(idx + 1) + '_frames_mean.jpg'
        print(out) 
        cv2.imwrite(out, img_out)
    print(str(idx), fname, "stacked")
img_stack /= len(imgs_list)
out_name = "stack_mean.jpg"
cv2.imwrite(out_name, img_stack)
print(out_name, "saved")

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

jpgs_path = os.path.join(pathlib.Path().absolute(), "jpg/*")
raw_path = os.path.join(pathlib.Path().absolute(), "raw/*")
jpgs_list = sorted(glob.glob(jpgs_path))
raw_list = sorted(glob.glob(raw_path))

trans_prior = np.eye(3, 3, dtype=np.float32)
init_img = load_image(jpgs_list[0])
img_stack = init_img.astype(np.float32)

denoise_idxs = [x ** 2 - 1 for x in range(1, 12)]

for idx, fname in enumerate(jpgs_list[1:]):
    img = load_image(fname)
    raw = load_image(raw_list[idx])
    transform = ecc_transform(init_img, img, prior=trans_prior)
    trans_prior = transform
    img_stack_dim = img.shape
    frame_dim = (init_img.shape[1], init_img.shape[0])
    img_warp = cv2.warpPerspective(raw, transform, frame_dim,
    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    flags=cv2.WARP_INVERSE_MAP)
    img_stack += img_warp
    if idx in denoise_idxs:
        center = (img_stack.shape[0] // 2, img_stack.shape[1] // 2)
        h = 200
        w = 300
        img_out = img_stack[center[0] - h:center[0] + h,
                            center[1] - w:center[1] + w] / (idx + 1)
        out = 'to_gif/frame_' + str(idx).zfill(3) + '.jpg'
        print(out, "saved") 
        cv2.imwrite(out, img_out)
    print(fname, "stacked, frame", str(idx + 2), "of", str(len(jpgs_list)))

img_stack /= len(jpgs_list)
stars, background = rm_background(img_stack)
stars = cv2.cvtColor(stars, cv2.COLOR_BGR2RGB)
out_name = "stars.jpg"
cv2.imwrite(out_name, stars)
cv2.imwrite("removed_background.jpg", background)
print(out_name, "saved")
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
imgs_path = os.path.join(pathlib.Path().absolute(), "jpg/*")
imgs_list = sorted(glob.glob(imgs_path))
#how many stars to look for in each frame
num_stars = 200
# showStarCoords(imgs[0], getStarCoords(imgs[0], num_stars=num_stars))

points = []
transforms = []
transforms.append(np.identity(3))
img_stack_dim = (0, 0, 0)
for idx, fname in enumerate(imgs_list):
    img = load_image(fname)
    points.append(getStarCoords(img, num_stars=num_stars))
    if idx != 0:
        transforms.append(icpTransform(points[0], points[idx],
                          prior=transforms[idx - 1]))
    img_stack_dim = img.shape
    print(str(idx), "transform found")
print("transformations calculated")
frame_dim = (img_stack_dim[1], img_stack_dim[0])
img_stack = np.zeros((img_stack_dim), dtype=np.float32)

weight = 1.0 / len(imgs_list)
for idx, fname in enumerate(imgs_list):
    img = load_image(fname)
    # warp_identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    # img_warp = cv2.warpAffine(img, warp_identity, tuple(frame_dim), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 0, 0))
    img_warp = cv2.warpAffine(img, transforms[idx][:2], frame_dim, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 0, 0), flags=cv2.WARP_INVERSE_MAP)
    img_stack += img_warp * weight
    print(str(idx), "stacked")
print("images stacked")
# print("image stack size: ", str(sys.getsizeof(img_stack) / 1000000), " MB")

cv2.cvtColor(img_stack, cv2.COLOR_BGR2RGB)
cv2.imwrite("stack_mean.jpg", img_stack)
plotTransforms(transforms)

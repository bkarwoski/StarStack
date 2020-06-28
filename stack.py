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

imgs = load_images()
imgs.reverse()
#how many stars to look for in each frame
num_stars = 200

showStarCoords(imgs[0], getStarCoords(imgs[0], num_stars=num_stars))
points = []
transforms = []
for idx, img in enumerate(imgs):
    points.append(getStarCoords(img, num_stars=num_stars))
    transforms.append(icpTransform(points[0], points[idx], threshold=20))
# print(transforms)
# affineTransform = icpTransform(points[0], points[-1])
# affineTransform = getRigidTransform(points[0], points[1], numIter=8, img=imgs[0])
frame_dim = [imgs[0].shape[1], imgs[0].shape[0]]
img_stack_dim = (len(imgs), imgs[0].shape[0], imgs[0].shape[1],
                 imgs[0].shape[2])
img_stack = np.zeros((img_stack_dim), dtype=int)

for idx, img in enumerate(imgs):
    # warp_identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    # img_warp = cv2.warpAffine(img, warp_identity, tuple(frame_dim), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 0, 0))
    img_warp = cv2.warpAffine(img, transforms[idx][:2], tuple(frame_dim), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 0, 0), flags=cv2.WARP_INVERSE_MAP)
    img_stack[idx] = img_warp
print("images stacked")
print(img_stack.shape)
print("image stack size: ", str(sys.getsizeof(img_stack) / 1000000), " MB")

max_img = np.ndarray.max(img_stack, axis=0)
print("max img shape: ", str(max_img.shape))

# cv2.imwrite("last_img_in_stack_warped.jpg", img_stack[-1])
# cv2.cvtColor(max_img, cv2.COLOR_BGR2RGB)
cv2.imwrite("stack_max.jpg", max_img)
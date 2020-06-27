#!/usr/bin/env python3
from matplotlib import pyplot as plt
import cv2
import numpy as np
import scipy
import os
import glob
import copy
from sklearn.neighbors import KDTree
from multiprocessing import Pool
from util import *

imgs = load_images()
#how many stars to look for in each frame
num_stars = 40

showStarCoords(imgs[0], getStarCoords(imgs[0], num_stars=num_stars))
points = []
for img in imgs:
    points.append(getStarCoords(img, num_stars=num_stars))
affineTransform = getRigidTransform(points[0], points[1], numIter=8, img=imgs[0])
out_size = (2 * imgs[-1].shape[1], 2 * imgs[-1].shape[0])
im2_warp = cv2.warpAffine(imgs[-1], affineTransform[:2], out_size)
im2_warp[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
im2_warp = cv2.cvtColor(im2_warp, cv2.COLOR_BGR2RGB)
cv2.imwrite("affine_transformed_im2.jpg", im2_warp)
#!/usr/bin/env python3

import cv2
from functools import partial
import glob
from itertools import repeat
from multiprocessing import Pool
import numpy as np
import os
CHECKERBOARD = (7,9)

def get_points(fname):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # print(str(fname), " - processing")
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_shape = gray.shape[::-1]
    # print("gray shape", str(gray.shape[::-1]))
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret == True:
        # objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (20,20),(-1,-1), criteria)
        # imgpoints.append(corners2)
    else:
        print("chessboard not found for img ", str(fname))
    # print(str(fname), " - done")
    # print("corners2 shape", str(corners2.shape))
    return corners2

if __name__ == "__main__":
    images = glob.glob('./chessboard/*.JPG')
    # objpoints = []
    # imgpoints = []
    pool = Pool()
    # get_points_p = partial(get_points, objpoints=objpoints, imgpoints=imgpoints)
    points = pool.map(get_points, images)
    print(np.array(points).shape)

    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    print("objp shape", objp.shape)
    objpoints = np.tile(objp, (len(imgpoints), 1, 1))

    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    print("obj points shape", str(np.array(objpoints).shape))
    print("img points array shape", str(np.array(imgpoints).shape))
    print("gray shape", str(tuple(gray_shape)))
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, tuple(gray_shape), None, None)

    np.save("camera_matrix", mtx)
    np.save("dist_coeffs", dist)
    print("Camera matrix : \n")
    print(mtx)
    print("dist : \n")
    print(dist)
    print("rvecs : \n")
    print(rvecs)
    print("tvecs : \n")
    print(tvecs)
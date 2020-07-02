import copy
import cv2
import glob
from matplotlib import pyplot as plt
from multiprocessing import Pool
import numpy as np
import open3d as o3d
import os
import pathlib
import scipy
from sklearn.neighbors import KDTree

def load_image(fname):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def getStarCoords(img, num_stars=100):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #remove median light pollution
    #TODO: gradient estimation?
    img = cv2.subtract(img, np.median(img))

    #reduce effect of single hot pixels and non-prominent stars
    img = cv2.GaussianBlur(img, (0,0), 3)

    #create binary mask of star blobs
    starMask = np.zeros_like(img)
    imgStd = np.std(img)
    thresh = imgStd * 8 
    starBlobs = img > thresh
    starMask[starBlobs] = 1

    #get contours
    _, contours, _ = cv2.findContours(starMask, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    contours.sort(key=len, reverse=True)
    contours = contours[:num_stars]
    if len(contours) != num_stars:
        raise ValueError("error - did not find " + str(num_stars) + " stars")

    #get each star location from average position of blob perimeter
    starCoords = np.zeros((len(contours), 2), dtype=int)
    for c in range(len(contours)):
        starCoords[c, :] = np.mean(contours[c], axis=0)[0]
    return starCoords

def showCorrespondances(pts1, pts2, img):
    pts1 = np.array(pts1, dtype=np.int)
    pts2 = np.array(pts2, dtype=np.int)
    correspondancePlot = copy.deepcopy(img)
    correspondancePlot = cv2.cvtColor(correspondancePlot, cv2.COLOR_BGR2RGB)
    for i in range(pts1.shape[0]):
        cv2.line(correspondancePlot, tuple(pts1[i]), tuple(pts2[i]), (0, 255, 0), thickness=2)
    cv2.imwrite("correspondances.png", correspondancePlot)
    print("correspondances.png saved")

def showStarCoords(img, coords, filename="major_stars.png"):
    starPlot = copy.deepcopy(img)
    for point in coords:
        cv2.circle(starPlot, tuple(point), 20, (0, 255, 0), thickness=3)
    starPlot = cv2.cvtColor(starPlot, cv2.COLOR_BGR2RGB)
    cv2.imwrite(filename, starPlot)
    print(filename + " saved")

def plotCorrespondances(dyn, nn):
    plt.scatter(dyn[:,0], dyn[:,1], marker="o", color="red", label="shifted points")
    plt.scatter(nn[:,0], nn[:,1], marker="o", color="blue", label="nearest neighbor static points")
    for i in range(len(dyn)):
        plt.plot([dyn[i,0], nn[i,0]],
                 [dyn[i,1], nn[i,1]])
    plt.legend()
    plt.show()
    return

def h_2d_to_3d(H):
    '''converts planar 2D transform to a 4x4 3D equivalent, 
    assuming normal to z axis'''
    H_3d = np.array([[H[0,0], H[0,1], 0, H[0,2]],
                    [H[1,0], H[1,1], 0, H[1,2]],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    return H_3d

def h_3d_to_2d(H):
    '''converts full 4x4 3D transform into a 3x3 planar transform.'''
    H = np.array(H)
    H_2d = np.identity(3)
    H_2d[:2, :2] = H[:2, :2]
    H_2d[:2, 2] = H[:2, 3]
    return H_2d

def icpTransform(pts_static, pts_dynamic, num_iter=30, threshold=100,
                 prior=np.identity(3)):
    '''returns a 3x3 matrix transform to shift pts_dynamic to pts_static'''

    if pts_static.shape[1] != 2:
        raise ValueError("Error - pts_static should have 2 columns")
    if pts_dynamic.shape[1] != 2:
        raise ValueError("Error - pts_dynamic should have 2 columns")

    pts_static = np.hstack((pts_static, np.zeros((pts_static.shape[0], 1))))
    pts_dynamic = np.hstack((pts_dynamic, np.zeros((pts_dynamic.shape[0], 1))))
    pcd_static = o3d.geometry.PointCloud()
    pcd_dynamic = o3d.geometry.PointCloud()
    pcd_static.points = o3d.utility.Vector3dVector(pts_static)
    pcd_dynamic.points = o3d.utility.Vector3dVector(pts_dynamic)

    reg_p2p = o3d.registration.registration_icp(pcd_static, pcd_dynamic,
        threshold, h_2d_to_3d(prior),
        o3d.registration.TransformationEstimationPointToPoint(),
        o3d.registration.ICPConvergenceCriteria(max_iteration=num_iter))
    Rt_3d = reg_p2p.transformation
    Rt_2d = h_3d_to_2d(Rt_3d)
    return Rt_2d

def plotTransforms(transforms):
    '''debug tool, to show transform between each image.'''
    theta = np.zeros(len(transforms))
    x = np.zeros(len(transforms))
    y = np.zeros(len(transforms))
    for idx, t in enumerate(transforms):
        theta[idx] = np.arccos(t[0,0]) * 180 / np.pi
        x[idx] = t[0,2]
        y[idx] = t[1,2]
    plt.plot(theta)
    plt.plot(x)
    plt.plot(y)
    plt.legend(['theta [deg]', 'x [pix]', 'y [pix]'])
    plt.xlabel('image #')
    plt.show()
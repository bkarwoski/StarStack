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

def load_images():
    imsPath = os.path.join(pathlib.Path().absolute(), "jpg/*")
    imgs = []
    for img in glob.glob(imsPath):
        nextImg = cv2.imread(img)
        nextImg = cv2.cvtColor(nextImg, cv2.COLOR_BGR2RGB)
        imgs.append(nextImg)
    if len(imgs) == 0:
        raise ValueError("error - no images loaded")
    return imgs

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

def getRigidTransform(static, dynamic, numIter=8, img=None):
    '''
    Estimates a rigid body transform between two sets of points using ICP.
    returns 3x3 homogenous transformation matrix
    static: list of points in base image
    dynamic: list of point in image to be shifted
    '''
    if static.shape[1] != 2:
        raise ValueError("error - expected static to have 2 columns")
    if dynamic.shape[1] != 2:
        raise ValueError("error - expected static to have 2 columns")

    Rt = np.identity(3)
    #homogeneous padding
    ones = np.ones((dynamic.shape[0], 1))
    dynamic = np.hstack((dynamic, ones))
    tree = KDTree(static)
    for i in range(numIter):
        #apply transform to dynamic points
        dynamic = np.matmul(dynamic, Rt)
        #find correspondances for each point
        #for each dynamic point, find the 1 closest static point in the tree
        _, idxs = tree.query(dynamic[:,:-1], k=1)
        nnList = static[idxs]
        nnList = nnList[:,0,:]
        #debug
        # plotCorrespondances(dynamic, nnList)
        if img is not None:
            showCorrespondances(nnList, dynamic[:, :-1], img)

        nnCentroid = np.mean(nnList, axis=0)
        dynamicCentroid = np.mean(dynamic[:, :-1], axis=0)
        # dynamicCentroid = dynamicCentroid[:2]
        nnm = nnList - nnCentroid
        dynamicm = dynamic[:,:-1] - dynamicCentroid
        # dynamicm = dynamicm[:,:-1]
        H = np.matmul(nnm.T, dynamicm)
        # print(H.shape)
        U, _, Vt = np.linalg.svd(H)
        # print(U)
        # print(Vt)
        R = np.matmul(U, Vt)
        t = np.matmul(-R, nnCentroid) + dynamicCentroid
        Rt[:2, :2] = R
        Rt[:2, 2] = t
    print("rigid transform: ")
    print(Rt)
    return Rt

def icpTransform(pts_static, pts_dynamic, num_iter=30, threshold=1000):
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
        threshold, np.identity(4),
        o3d.registration.TransformationEstimationPointToPoint(),
        o3d.registration.ICPConvergenceCriteria(max_iteration=num_iter))
    Rt_3d = reg_p2p.transformation
    Rt_2d = h_3d_to_2d(Rt_3d)
    return Rt_2d


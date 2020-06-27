from util import getRigidTransform
import numpy as np
import cv2
from matplotlib import pyplot as plt
import open3d as o3d
import copy
from util import getRigidTransform, showCorrespondances

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def h_2d_to_3d(H):
    '''converts planar 2D transform to a 4x4 3D equivalent, 
    assuming normal to z axis'''
    H_3d = np.array([[H[0,0], H[0,1], 0, H[0,2]],
                    [H[1,0], H[1,1], 0, H[1,2]],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    return H_3d

pts_static = 100 * np.random.rand(20,2)
ones = np.ones((pts_static.shape[0], 1))
pts_dynamic = np.hstack((pts_static, ones))
#sample homography matrix
x = 20
y = -30
theta = np.pi / 6
H = np.array([[np.cos(theta), -np.sin(theta), x],
              [np.sin(theta),  np.cos(theta), y],
              [0, 0, 1]])

pts_dynamic = np.matmul(pts_dynamic, H)
pts_dynamic = pts_dynamic[:,:2]
Rt = getRigidTransform(pts_static, pts_dynamic, numIter=10)

pcd_static = o3d.geometry.PointCloud()
pts_static = np.hstack((pts_static, np.zeros((pts_static.shape[0], 1))))
pcd_static.points = o3d.utility.Vector3dVector(pts_static)
pcd_dynamic = o3d.geometry.PointCloud()
pts_dynamic = np.hstack((pts_dynamic, np.zeros((pts_dynamic.shape[0], 1))))
pcd_dynamic.points = o3d.utility.Vector3dVector(pts_dynamic)
threshold = 1000
reg_p2p = o3d.registration.registration_icp(pcd_static, pcd_dynamic,
        threshold, np.identity(4),
        o3d.registration.TransformationEstimationPointToPoint(),
        o3d.registration.ICPConvergenceCriteria(max_iteration=30))
print("reg_p2p transformation:")
print(reg_p2p.transformation)
evaluation_Rt = o3d.registration.evaluate_registration(pcd_static, pcd_dynamic, threshold, h_2d_to_3d(Rt))
print(evaluation_Rt)
draw_registration_result(pcd_static, pcd_dynamic, reg_p2p.transformation)
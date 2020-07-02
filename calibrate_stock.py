import numpy as np
import cv2
import glob
CHESSBOARD = (9,7)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((CHESSBOARD[0]*CHESSBOARD[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CHESSBOARD[0],0:CHESSBOARD[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('./chessboard/*.JPG')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD,None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        print(fname, "refined subpix")
        imgpoints.append(corners2)

        # Draw and display the corners
        # img = cv2.drawChessboardCorners(img, CHESSBOARD, corners2,ret)
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('image', 3000, 2000)
        # cv2.imshow('image',img)
        # cv2.waitKey(500)
    else:
        print(fname, "chessboard not found")

# cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

#test undistort feature

img = cv2.imread('./chessboard/DSC07511.JPG')
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
cv2.imwrite('./chessboard/DSC07511_cal.JPG', dst)
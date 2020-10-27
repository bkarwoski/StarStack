import copy
import cv2
from matplotlib import pyplot as plt
import numpy as np
import rawpy

def blur(img):
    return cv2.GaussianBlur(img, (0,0), 5)

def load_image(fname):
    if fname[-3:] in ('jpg', 'JPG'):
        img = cv2.imread(fname)
        if img is None:
            raise Exception("Error - could not load jpg image")
        return crop(img)
    if fname[-3:] in ('ARW', 'arw'):
        with rawpy.imread(fname) as raw:
            img = raw.postprocess(no_auto_bright=False)
            if img is None:
                raise Exception("Error - could not load raw image")
            return crop(img)
    else:
        raise Exception("Did not recognize file extension")

def get_star_mask(img, std=8):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype('float64')
    #remove median light pollution
    median = np.median(img, keepdims=True)
    median = median.astype('float64')
    img = cv2.subtract(img, median)

    #reduce effect of single hot pixels and non-prominent stars
    # img = cv2.GaussianBlur(img, (0,0), 3)

    #create binary mask of star blobs
    star_mask = np.zeros_like(img, dtype=np.uint8)
    img_std = np.std(img)
    thresh = img_std * std
    star_blobs = img > thresh
    star_mask[star_blobs] = 1
    return star_mask

def ecc_transform(source, shifted, prior=np.eye(3, 3, dtype=np.float32)):
    source = cv2.cvtColor(source,cv2.COLOR_BGR2GRAY)
    shifted = cv2.cvtColor(shifted,cv2.COLOR_BGR2GRAY)
    warp_mode = cv2.MOTION_HOMOGRAPHY
    number_of_iterations = 20
    termination_eps = 1e-10
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    cc, warp_matrix = cv2.findTransformECC (source, shifted, prior, warp_mode, criteria)
    print("correlation coefficient: ", f"{cc:.4}")
    return warp_matrix

def rm_background(img, ratio=1):
    img_nostar = copy.deepcopy(img)
    star_mask = get_star_mask(img, std=4)
    img_nostar[star_mask == 1] = np.median(img, axis=(0,1))
    img_blur = cv2.GaussianBlur(img_nostar, (0, 0), 100, cv2.BORDER_REFLECT_101)
    return cv2.subtract(img, img_blur * ratio), img_blur

### functions for debugging and visualization ###

def get_star_coords(img, num_stars=100):
    star_mask = get_star_mask(img)
    _, contours, _ = cv2.findContours(star_mask, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    contours.sort(key=len, reverse=True)
    contours = contours[:num_stars]
    if len(contours) != num_stars:
        raise ValueError("error - did not find " + str(num_stars) + " stars")

    #get each star location from average position of blob perimeter
    starCoords = np.zeros((len(contours), 2), dtype=int)
    for c in range(len(contours)):
        starCoords[c, :] = np.mean(contours[c], axis=0)[0]
    return starCoords

def plot_transforms(transforms):
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

def plot_correspondances(pts1, pts2):
    plt.scatter(pts1[:,0], pts1[:,1], marker="o", color="red", label="pts1")
    plt.scatter(pts2[:,0], pts2[:,1], marker="o", color="blue", label="pts2")
    for i in range(len(pts1)):
        plt.plot([pts1[i,0], pts2[i,0]],
                 [pts1[i,1], pts2[i,1]])
    plt.legend()
    plt.show()
    return

def plot_hist(img):
    histogram, bin_edges = np.histogram(img, bins=256, range=(0, 1))
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixels")
    plt.xlim([0.0, 1.0])  # <- named arguments do not work here
    plt.plot(bin_edges[0:-1], histogram)  # <- or here
    # plt.yscale('log')
    cv2.imwrite("raw image stock.jpg", img)
    plt.show()

def show_star_coords(img, coords, filename="major_stars.png"):
    starPlot = copy.deepcopy(img)
    for point in coords:
        cv2.circle(starPlot, tuple(point), 20, (0, 255, 0), thickness=3)
    starPlot = cv2.cvtColor(starPlot, cv2.COLOR_BGR2RGB)
    cv2.imwrite(filename, starPlot)
    print(filename + " saved")

def show_correspondances(pts1, pts2, img):
    pts1 = np.array(pts1, dtype=np.int)
    pts2 = np.array(pts2, dtype=np.int)
    correspondancePlot = copy.deepcopy(img)
    correspondancePlot = cv2.cvtColor(correspondancePlot, cv2.COLOR_BGR2RGB)
    for i in range(pts1.shape[0]):
        cv2.line(correspondancePlot, tuple(pts1[i]), tuple(pts2[i]), (0, 255, 0), thickness=2)
    cv2.imwrite("correspondances.png", correspondancePlot)
    print("correspondances.png saved")

def crop(img):
    img = img[:2900, :]
    return img
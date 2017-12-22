import cv2
import numpy as np
from numpy import uint8
from skimage import io
from skimage.viewer import ImageViewer

def read_image_as_rgb(path):
    # img = io.imread(path)
    bgr_img = cv2.imread(path)
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    # img = cv2.imread(path)
    return img

def read_image_as_bgr(path):
    img = cv2.imread(path)
    return img

def rgb2bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def bgr2rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def show_image(im_name, image):
    image = image/255
    cv2.imshow(im_name, image)
    cv2.waitKey(0)
    cv2.destroyWindow(im_name)

    # viewer = ImageViewer(image)
    # viewer.show()

def write_image(im_name, img):
    cv2.imwrite(im_name, img)

def laplacian(image):
    # kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    kernel = [[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]]
    kernel = np.array(kernel)
    laplacian = cv2.filter2D(image,-1,kernel)
    # laplacian = cv2.Laplacian(image,cv2.CV_64F)
    return laplacian

def gaussian(image):
    # lap = laplacian(image)
    gaussian = cv2.GaussianBlur(image,(5,5),0)
    return gaussian

import numpy as np
import cv2
from functions import *
from genHoughTransform import *
from scipy.ndimage import convolve

def match_template_fourier(img,template):
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    template_gray = cv2.cvtColor(template,cv2.COLOR_RGB2GRAY)
    template_gray_pad = cv2.copyMakeBorder(template_gray,0,img_gray.shape[0]-template_gray.shape[0],0,img_gray.shape[1]-template_gray.shape[1],cv2.BORDER_CONSTANT,value=0)
    img_f = np.fft.fftshift(np.fft.fft2(img_gray))
    template_f = np.fft.fftshift(np.fft.fft2(template_gray_pad))
    conv = template_f.conj()*img_f
    conv = conv/np.abs(conv)
    corr = np.fft.ifft2(np.fft.fftshift(conv))
    corr = np.real(corr)
    corner = np.unravel_index(corr.argmax(),corr.shape)
    mask = np.zeros(img_gray.shape,dtype=bool)
    mask[corner[0]:corner[0]+template_gray.shape[0],corner[1]:corner[1]+template_gray.shape[1]] = True
    return mask

def match_template_hough(img,template,angle):
    mask = hough(img,template,angle)
    return mask

def match_template(img,template,match_method=0,angle=2):
    mask = match_template_fourier(img,template) if match_method == 0 else match_template_hough(img,template,angle)
    return mask

if __name__ == "__main__":
    img = load_img("13.jpg",1)
    template = load_img("t13.jpg",1)
    template_mask = match_template(img,template,0)
    show_multiple_img(["Image","Template","Mask"],[img,template,((template_mask*255).astype(np.uint8))],0)
import numpy as np
import cv2
from scipy.ndimage import convolve
from functions import *

HSOBEL_WEIGHTS = np.array([[ 1, 2, 1],
                           [ 0, 0, 0],
                           [-1,-2,-1]]) / 4.0
VSOBEL_WEIGHTS = HSOBEL_WEIGHTS.T

def e1(img_gray):
    dx = np.diff(img_gray.astype(np.int64),axis=1)
    dx = np.concatenate((dx,dx[:,-1].reshape(-1,1)),axis=1)
    dy = np.diff(img_gray.astype(np.int64),axis=0)
    dy = np.concatenate((dy,dy[-1,:].reshape(1,-1)),axis=0)
    e1 = np.absolute(dx) + np.absolute(dy)
    return e1.astype(np.uint8)

def sobel_e1(img_gray):
    img_gray_f = img_gray.astype(np.float32)
    dx = convolve(img_gray_f, HSOBEL_WEIGHTS)
    dy = convolve(img_gray_f, VSOBEL_WEIGHTS)
    sobel_e1 = np.sqrt(dx**2 + dy**2)/np.sqrt(2)
    return sobel_e1.astype(np.uint8)

def entropy(img_gray):
    img_gray_pad = cv2.copyMakeBorder(img_gray,8,8,8,8,cv2.BORDER_REPLICATE)
    entropy = np.empty(img_gray_pad.shape)
    for x in range(0,img_gray.shape[0]+9):
        for y in range(0,img_gray.shape[1]+9):
            entropy[x,y] = get_entropy(img_gray_pad[x:x+9,y:y+9])
    e1_energy = e1(img_gray)
    entropy_roi = entropy[8:img_gray.shape[0]+8,8:img_gray.shape[1]+8]
    entropy_energy = entropy_roi.astype(np.uint8) + e1_energy
    return entropy_energy

def hog_e1(img_gray):
    hist = get_hog(img_gray)
    hog_e1 = e1(img_gray)/np.max(hist)
    return hog_e1.astype(np.uint8)

def get_energy(img_gray,energy_function=1,mask=[]):
    energy = e1(img_gray) if energy_function == 0 else sobel_e1(img_gray) if energy_function == 1 else entropy(img_gray) if energy_function == 2 else hog_e1(img_gray)
    if np.sum(mask) != 0:
        modified_energy = energy.copy().astype(np.float32)
        modified_energy[mask] = -1000
        return modified_energy.astype(np.float32)
    return energy.astype(np.float32)

if __name__ == "__main__":
    img = load_img("5.jpg",1)
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    e1_energy = e1(img_gray).astype(np.uint8)
    sobel_e1_energy = sobel_e1(img_gray).astype(np.uint8)
    entropy_energy = entropy(img_gray).astype(np.uint8)
    hog_e1_energy = hog_e1(img_gray).astype(np.uint8)
    e1_energy_C = cv2.applyColorMap(e1_energy,cv2.COLORMAP_JET)
    sobel_e1_energy_C = cv2.applyColorMap(sobel_e1_energy,cv2.COLORMAP_OCEAN)
    entropy_energy_C = cv2.applyColorMap(entropy_energy,cv2.COLORMAP_AUTUMN)
    # hog_e1_energy_C = cv2.applyColorMap(hog_e1_energy,cv2.COLORMAP_RAINBOW)
    show_multiple_img(["Image","E1","Sobel E1","Entropy"],[img,e1_energy_C,sobel_e1_energy_C,entropy_energy_C],0)
    # show_multiple_img(["Image","E1","Sobel E1","Entropy","HOG E1"],[img,e1_energy_C,sobel_e1_energy_C,entropy_energy_C,hog_e1_energy_C],0)
    # save_img(["e1_6","sob_6","ent_6","hog_6"],[e1_energy_C,sobel_e1_energy_C,entropy_energy_C,hog_e1_energy_C])
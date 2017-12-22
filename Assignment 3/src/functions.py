import numpy as np
import cv2
from scipy.stats import entropy as scipy_entropy

def load_img(name: str,mode: int) -> np.ndarray:
    return cv2.imread("Test Images/"+name, mode)

def show_img(title: str,img: np.ndarray,wait: int) -> int:
    cv2.imshow(title, img)
    k = cv2.waitKey(wait)
    cv2.destroyWindow(title)
    return k

def show_multiple_img(title: list,img: list,wait: int) -> int:
    if type(title) != list or len(title) == 0:
        title = [str(x) for x in range(len(img))]
    for x, y in zip(title, img):
        cv2.imshow(x,y)
    k = cv2.waitKey(wait)
    for x in title:
        cv2.destroyWindow(x)
    return k

def save_img(name,img):
    if type(name) == str:
        cv2.imwrite("out/"+name+".jpg",img)
    else:
        [cv2.imwrite("out/"+n+".jpg",i) for n,i in zip(name,img)]

def get_fourier(img_gray):
    img_f = np.fft.fft2(img_gray)
    img_fshift = np.fft.fftshift(img_f)
    img_fmag = np.log(np.abs(img_fshift))
    img_fmag_scaled = ((img_fmag - img_fmag.min())/(img_fmag.max() - img_fmag.min())*255).astype(np.uint8)
    return img_fmag_scaled

def get_entropy(img_gray):
    _, counts = np.unique(img_gray,return_counts=True)
    entropy = scipy_entropy(counts,base=2)
    return entropy

# Implementation taken from stackoverflow answer - https://stackoverflow.com/questions/6090399/get-hog-image-features-from-opencv-python
def get_hog(img_gray):
    img_gray = img_gray.astype(np.float32)/255
    gx = cv2.Sobel(img_gray,cv2.CV_32F,1,0)
    gy = cv2.Sobel(img_gray,cv2.CV_32F,0,1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bin_n = 8
    bin = np.int32(bin_n*ang/(2*np.pi))
    bin_cells = []
    mag_cells = []
    cellx = celly = 11
    for i in range(0,int(img_gray.shape[0]/celly)):
        for j in range(0,int(img_gray.shape[1]/cellx)):
            bin_cells.append(bin[i*celly:i*celly+celly,j*cellx:j*cellx+cellx])
            mag_cells.append(mag[i*celly:i*celly+celly,j*cellx:j*cellx+cellx])   
    hists = [np.bincount(b.ravel(),m.ravel(),bin_n) for b,m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    eps = 1e-7
    hist /= hist.sum() + eps
    hist = np.sqrt(hist)
    hist /= np.linalg.norm(hist) + eps
    return hist
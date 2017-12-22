import numpy as np
import cv2
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

def hist(img: np.ndarray) -> np.ndarray:
    return np.bincount(img.flatten())

def load_img(name: str,mode: int) -> np.ndarray:
    # return cv2.imread("./CAPE/Test Images/"+name, mode)
    return cv2.imread("./Test Images/"+name, mode)

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

def get_coordinates(xywh: tuple) -> tuple:
    p_x = xywh[0]
    p_y = xywh[1]
    q_x = xywh[2] + xywh[0]
    q_y = xywh[3] + xywh[1]
    (p_x,q_x) = (q_x,p_x) if p_x > q_x else (p_x,q_x)
    (p_y,q_y) = (q_y,p_y) if p_y > q_y else (p_y,q_y)
    return (p_x,p_y,q_x,q_y)

def hist_eq(img_intensity: np.ndarray):
    h = hist(img_intensity)
    pdf = h/np.prod(img_intensity.shape)
    cdf = pdf.cumsum()
    bins = h.shape[0]
    eq = np.vectorize(lambda x: cdf[x]*bins)
    img_intensity_eq = eq(img_intensity)
    return img_intensity_eq

def hist_smoothing(img_intensity: np.ndarray,ksize=30,sigma=10) -> np.ndarray:
    h = cv2.calcHist([img_intensity],[0],None,[255],[1,256]).T.ravel()
    plt.plot(h)
    plt.show()
    h = np.correlate(h,cv2.getGaussianKernel(ksize,sigma).ravel(),'same')
    plt.plot(h)
    plt.show()
    return h

def bimodal_detect(hist: np.ndarray):
    peaks = argrelextrema(hist,np.greater,order=5)[0]
    if len(peaks) < 2:
        return False, -1, -1, -1
    max_peak = np.argmax(hist[peaks])
    left = peaks[:max_peak]
    right = peaks[max_peak+1:]
    refined_peaks_left = argrelextrema(hist[left],np.greater_equal,order=1,mode='clip')[0]
    refined_peaks_right = argrelextrema(hist[right],np.greater_equal,order=1,mode='clip')[0]
    other_peak = -1
    if len(refined_peaks_left) > 0 and len(refined_peaks_right) > 0:
        other_peak = left[refined_peaks_left[0]] if refined_peaks_left[0] < refined_peaks_right[-1] else right[refined_peaks_right[-1]]
    elif len(refined_peaks_left) > 0:
        other_peak = left[refined_peaks_left[0]]
    else:
        other_peak = right[refined_peaks_right[-1]]      
    if other_peak == -1:
        dist_from_max = np.absolute(peaks-peaks[max_peak])
        max_distant_peak = np.argmax(dist_from_max)
        other_peak = peaks[max_distant_peak]       
    d = peaks[max_peak] if peaks[max_peak] < other_peak else other_peak
    b = peaks[max_peak] if peaks[max_peak] > other_peak else other_peak
    m = np.argmin(hist[d:b])+d
    return True, d, m, b

    # def refine_peaks(peaks_list,pivot,direction):
    #     if len(peaks_list) == 1:
    #         return peaks_list[0]
    #     elif len(peaks_list) == 2:
    #         return peaks_list[0] if peaks_list[0] < peaks_list[1] else peaks_list[1]
    #     else:
    #         new_pivot = np.argmax(hist[peaks_list])
    #         left_peak = refine_peaks(peaks_list[pivot:new_pivot],new_pivot)
    #         right_peak = refine_peaks(peaks_list[new_pivot:],new_pivot)
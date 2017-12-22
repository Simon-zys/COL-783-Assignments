import numpy as np
import cv2
from scipy.signal import convolve2d
from math import sqrt
from functions import *

def aindane(img_color: np.ndarray):
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img_n = img/255
    eps = 1e-6
    def ale():
        """ Adaptive luminance enhancement """
        cdf = hist(img).cumsum()
        L = np.array([np.searchsorted(cdf, 0.1*img.shape[0]*img.shape[1], side='right')])
        z = np.piecewise(L,[L<=50,L>50 and L<=150,L>150],[0,(L-50)/100,1])
        img_np = (img_n**(0.75*z+0.25) + (1-img_n)*0.4*(1-z) + img_n**(2-z))/2
        return img_np
    
    def ace(c=5):
        """ Adaptive contrast enhancement """
        sigma = int(round(sqrt(c**2 /2)))
        gaussian_kernel = cv2.getGaussianKernel(sigma*3,sigma)
        gaussian_kernel_normalized = (gaussian_kernel * gaussian_kernel.T)/ np.sum(gaussian_kernel * gaussian_kernel.T)
        img_conv = convolve2d(img,gaussian_kernel_normalized,mode='same',boundary='fill',fillvalue=0)
        img_sigma = np.array([np.std(img)])
        p = np.piecewise(img_sigma,[img_sigma<=3,img_sigma>3 and img_sigma<10,img_sigma>=10],[3,(27-2*img_sigma)/7,1])
        E = ((img_conv+eps)/(img+eps))**p
        S = 255*np.power(img_np,E)
        return S

    def color_restoration(l=[1,1,1]):
        S_restore = np.zeros(img_color.shape)
        for j in range(3):
            S_restore[...,j] = S*(img_color[...,j]/(img+eps))*l[j]
        return S_restore

    img_np = ale()
    S = ace()
    S_restore = color_restoration()
    S_final = np.clip(S_restore, 0, 255).astype('uint8')
    return S_final
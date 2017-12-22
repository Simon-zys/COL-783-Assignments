import numpy as np
import cv2
from functions import *
import matplotlib.pyplot as plt
from scipy import ndimage
from skin_detection import *
from wls_filter import *
from eacp import *
from pySaliencyMap import pySaliencyMap
from face_enhancement import face_detector

def get_energy_map(img_lab: np.ndarray,mask: list,t=0.4,s=15):
    eps = 1e-5
    img_gray = cv2.cvtColor(cv2.cvtColor(img_lab, cv2.COLOR_Lab2BGR),cv2.COLOR_BGR2GRAY)
    grad = np.gradient(img_gray)
    abs_grad = np.absolute(grad[0]) + np.absolute(grad[1])
    ker = cv2.getGaussianKernel(s*3,s)
    k = ker*ker.T
    lab_fine_grad_l = s*(ndimage.convolve(img_gray,k)) 
    lab_fine_abs_grad_l = np.absolute(lab_fine_grad_l[0]) + np.absolute(lab_fine_grad_l[1])
    lab_fine_grad_a = s*(ndimage.convolve(img_lab[:,:,1],k))
    lab_fine_abs_grad_a = np.absolute(lab_fine_grad_a[0]) + np.absolute(lab_fine_grad_a[1])
    lab_fine_grad_b = s*(ndimage.convolve(img_lab[:,:,2],k))
    lab_fine_abs_grad_b = np.absolute(lab_fine_grad_b[0]) + np.absolute(lab_fine_grad_b[1])
    final_grad = abs_grad + lab_fine_abs_grad_l + lab_fine_abs_grad_a + lab_fine_abs_grad_b
    final_grad = final_grad/final_grad.max()
    Hab,xedges,yedges = np.histogram2d(img_lab[...,1].ravel(),img_lab[...,2].ravel(),bins=10,normed=True)
    e_H = np.empty_like(img_lab[:,:,1],dtype='float')
    for (x,y),a in np.ndenumerate(img_lab[...,1]):
        b = img_lab[x,y,2]
        H = Hab[int((a-xedges[0])/(xedges[1] - xedges[0])-eps), int((b-yedges[0])/(yedges[1] - yedges[0])-eps)]
        if H >0.015:
            e_H[x,y] = 1.0/(H*100)
        else:
            e_H[x,y] = 1.0/(0.015*100)
    e_face = 1.0*t*mask
    x, y = img_gray.shape
    maxd_E = sqrt((x/2)**2+(y/2)**2)
    xy = np.empty([x,y,2])
    for i in range(x):
        xy[i,:,0] = i
    for j in range(y):
        xy[:,j,1] = j
    w_spac = 1-(np.sqrt((xy[:,:,0]-(x/2))**2+(xy[:,:,1]-(y/2))**2)/maxd_E)**2
    base = (final_grad + e_H + e_face)*w_spac
    base = eacp(base,img_lab[:,:,0])
    return base/base.max()

def shadow_enhancement(img: np.ndarray,xywh: list,base: np.ndarray,detail: np.ndarray):
    img_lab = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
    # show_img("Base",base,0)
    base = base*255
    detail = detail*255
    saliency_map = pySaliencyMap(img.shape[1],img.shape[0])
    energy_map = saliency_map.SMGetSM(img)
    # skin_mask = np.zeros((img.shape[0],img.shape[1]),dtype=bool)
    # for x,y,w,h in xywh:
    #     skin_mask[y:y+h,x:x+w] = True
    # energy_map = get_energy_map(img_lab,skin_mask)
    plt.imshow(energy_map,cmap="rainbow")
    plt.show()
    L = img_lab[:,:,0]
    dark = (L<50) & (np.maximum.reduce([img[:,:,0],img[:,:,1],img[:,:,2]]) - np.minimum.reduce([img[:,:,0],img[:,:,1],img[:,:,2]])>5)
    dark_base, dark_detail = wlsFilter(L[dark].reshape((-1,1)))
    f_sal = min(2.0,1.0*np.percentile(L[~dark],35)/np.percentile(dark_base,95))
    img_lab[:,:,0] = ((f_sal*energy_map*base + (1-energy_map)*base)+detail).clip(0,255).astype('uint8')
    i_res = cv2.cvtColor(img_lab,cv2.COLOR_Lab2BGR)
    show_img("Image",i_res,0)
    return i_res

# Best Results - image_24
if __name__ == '__main__':
    img = load_img("image_24.jpg",1)
    img_lab = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
    base, detail = wlsFilter(img_lab[:,:,0])
    show_img("Base",base,0)
    # base = cv2.bilateralFilter(img_lab[:,:,1],5,10,12)
    xywh = face_detector(img)
    shadow_enhancement(img,xywh,base,detail)
    
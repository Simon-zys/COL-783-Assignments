import numpy as np
import cv2
from functions import *
from wls_filter import *
from eacp import *
from aindane import *
from face_detection import *
from skin_detection import *
import matplotlib.pyplot as plt

def face_detector(img: np.ndarray) -> list:
    vj_xywh = viola_jones(img)
    a_vj_xywh = viola_jones(aindane(img))
    man_xywh = []
    k = 0
    if len(vj_xywh) != 0 or len(a_vj_xywh) != 0:
        k = show_detected_face(img,[vj_xywh,a_vj_xywh],["VJ","A-VJ"]) - 48
    if k != 1 and k != 2:
        man_xywh = manual_face_detector(img)
    if k == 3:
        man_xywh.append(vj_xywh)
    if k == 4:
        man_xywh.append(a_vj_xywh)
    return vj_xywh if k == 1 else a_vj_xywh if k == 2 else man_xywh

def sidelight_correction(img_l,i_out,H,S,xywh,eacp_l=0.2):
    i_out = (i_out*255).clip(0,255).astype(np.uint8)
    A = np.ones(i_out.shape)
    W = np.zeros(i_out.shape)
    sig = 255 * 3
    for i in range(len(H)):
        bimodal, d, m, b = bimodal_detect(H[i])
        s = (S[i]*255)
        if bimodal == True:
            f = (b-d)/(m-d)
            print("d,m,b,f = ",d,m,b,f)
            x,y,w,h = xywh[i]
            A[y:y+h,x:x+w][(s < m) & (s > 0)] = f
            miu = (i_out[A == f]).mean()
            W = np.exp(-(i_out - miu) ** 2 / sig ** 2)
            W[...] = 1 - W[...]
            W[...] = 1
        else:
            print("Bimodal not detected - ",i)
    if not (A == 1).all():
        A = eacp(A,img_l,W,l=eacp_l)
    return A*i_out

def exposure_correction(img_l,i_out,i_side,masks,xywh,eacp_l=0.2):
    i_out = (i_out*255).clip(0,255).astype(np.uint8)
    A = np.ones(i_out.shape)
    i_out_exp = i_side.copy()
    face_skin = get_skin(i_side,xywh,masks,1)
    S = [x[1] for x in face_skin]
    faces = [x[0] for x in face_skin] 
    for i in range(len(S)):
        x,y,w,h = xywh[i]
        cum_sum = cv2.calcHist([(S[i]).astype('uint8')],[0],None,[255],[1,256]).T.ravel().cumsum()
        plt.plot(cum_sum)
        plt.show()
        p = np.searchsorted(cum_sum, cum_sum[-1] * 0.75)
        if p < 120:
            print('Underexposed')
            f = (120+p)/((2*p) + 1e-6)
            print("f = ",f)
            if f > 2 or f < 1:
                continue
            A[y:y+h,x:x+w][faces[i] > 0] = f
            eacp_A = eacp(A,i_out,l=0.4,alpha=0.5)
            i_out_exp = eacp_A*i_side
    return i_out_exp

def face_enhancement(img: np.ndarray,base: np.ndarray,detail: np.ndarray) -> np.ndarray:
    img_lab = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
    # show_img("Base",base,0)
    i_out = base
    xywh = face_detector(img)
    if xywh == []:
        return [], img
    faces_xywh = get_face_roi(img,xywh)
    faces = [x[1] for x in faces_xywh]
    skin_masks = skin_detection_cape(img,xywh)
    masks = [x[1] for x in skin_masks]
    if len(faces) == 0:
        print("No faces detected")
        return [], img
    if sum([mask.any() for mask in masks]) == 0:
        print("No skin detected")
        return [], img
    i_out_faces = get_face_roi(i_out,xywh)
    face_skin = get_skin(i_out,xywh,masks,1)
    S = [x[1] for x in face_skin]
    H = [hist_smoothing((s*255).clip(0,255).astype(np.uint8)) for s in S]
    i_side = sidelight_correction(img_lab[:,:,0],i_out,H,S,xywh)
    i_exp = exposure_correction(img_lab[:,:,0],i_out,i_side,masks,xywh)
    img_lab[:,:,0] = (i_side + 255*detail).astype('uint8')
    i_res = cv2.cvtColor(img_lab,cv2.COLOR_Lab2BGR)
    show_img("Image",i_res,0)
    return xywh,i_res

# Best Results - 1,2,3,image_24,image_20
if __name__ == '__main__':
    img = load_img("image_24.jpg",1)
    img_lab = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
    base, detail = wlsFilter(img_lab[:,:,0])
    show_img("Base",base,0)
    # base = cv2.bilateralFilter(img_lab[:,:,1],5,10,12)
    face_enhancement(img,base,detail)
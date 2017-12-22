import sys
import numpy as np
import cv2
from functions import *
from face_enhancement import *
from sky_detection import create_sky_prob_map
from shadow_enhancement import *

def cape(img: np.ndarray) -> np.ndarray:
    img_lab = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
    base, detail = wlsFilter(img_lab[:,:,0])
    show_img("Base",base,0)
    # base = cv2.bilateralFilter(img_lab[:,:,1],5,10,12)
    xywh, i_res_face = face_enhancement(img,base,detail)
    _ = create_sky_prob_map(img)
    i_res_face_shadow = shadow_enhancement(i_res_face,xywh,base,detail)
    i_res = i_res_face_shadow.copy()
    show_img("CAPE",i_res,0)
    return i_res

# Best Results - image_24
if __name__ == '__main__':
    name = "image_24.jpg"
    if len(sys.argv) > 1:
        name = sys.argv[1]
    img = load_img(name,1)
    show_img("Image",img,0)
    out = cape(img)
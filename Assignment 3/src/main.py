import numpy as np
import cv2
from functions import *
from energy import *
from seam import *
from seam_resize import *
from template_matching import *

if __name__ == "__main__":
    img = load_img("15.jpg",1)
    template = load_img("t15.jpg",1)
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    action = 2
    if action == 0:                 # Display Energy Maps
        energy_function = 1         # 0 - E1, 1 - Sobel E1, 2 - Entropy, 3 - HoG
        energy = get_energy(img_gray,energy_function).astype(np.uint8)
        energy_map = cv2.applyColorMap(energy,cv2.COLORMAP_JET)
        show_multiple_img(["Image","Energy Map"],[img,energy_map],0)
    elif action == 1:               # Display Seams
        v_seam_count = 20
        h_seam_count = 20
        v_seams = transform_vert_seams(repeated_vert_seams(img,v_seam_count))
        h_seams = transform_hor_seams(repeated_hor_seams(img,h_seam_count))
        img_copy = img.copy()
        for v_seam in v_seams:
            img_copy[v_seam[:,0],v_seam[:,1]] = [0,0,255]
        for h_seam in h_seams:
            img_copy[h_seam[:,0],h_seam[:,1]] = [0,0,255]
        show_img("Image Seams",img_copy,0)
    elif action == 2:               # Resize Image
        final_width = img.shape[1]-20
        final_height = img.shape[0]-20
        resized_img = resize(img,final_width,final_height)
        show_multiple_img(["Image","Resized Image"],[img,resized_img],0)
        # save_img("reduce_20_50",resized_img)
    elif action == 3:               # Remove Object
        match_method = 0            # 0 - Fourier Correlation, 1 - Hough
        order = 0                   # 0 - Vertical Seam Removal, 1 - Horizontal Seam Removal 
        angle = 2
        energy_function = 1
        mask = match_template(img,template,match_method,angle)
        img_without_template = remove_obj(img,mask,energy_function,order)
        show_multiple_img(["Image","Template","Mask","Img-Template"],[img,template,mask.astype(np.uint8)*255,img_without_template],0)
        save_img("15_fourier",img_without_template)
        save_img("15_fourier_mask",mask.astype(np.uint8)*255)
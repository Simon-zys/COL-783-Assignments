import numpy as np
import cv2
from functions import *
from seam import *

def hor_decrease(img,decreased_width,energy_function=1):
    img_copy = img.copy()
    width_decrease = img.shape[1] - decreased_width
    v_seams = repeated_vert_seams(img,width_decrease,energy_function)
    for v_seam in v_seams:
        img_copy[v_seam[:,0],v_seam[:,1]] = [0,0,255]
        show_img("Image",img_copy,0)
        mask = np.ones(img_copy.shape,dtype=bool)
        mask[v_seam[:,0],v_seam[:,1]] = False
        img_copy = img_copy[mask].reshape(img_copy.shape[0],img_copy.shape[1]-1,3)
    return img_copy

def hor_increase(img,increased_width,energy_function=1):
    img_copy = img.copy()
    width_increase = increased_width - img.shape[1]
    v_seams = transform_vert_seams(transform_vert_seams(repeated_vert_seams(img,width_increase,energy_function)))
    for v_seam in v_seams:
        mask = np.ones((img_copy.shape[0],img_copy.shape[1]+1,3),dtype=bool)
        mask[v_seam[:,0],v_seam[:,1]] = False
        out = np.empty(mask.shape,dtype=np.float32)
        out[mask] = img_copy.ravel()
        idx = v_seam[:,1]-1
        idx[idx < 0] = 0
        out[~mask] = ((img_copy[v_seam[:,0],idx].astype(np.float32) + img_copy[v_seam[:,0],v_seam[:,1]].astype(np.float32))/2).ravel()
        img_copy[v_seam[:,0],v_seam[:,1]] = [0,0,255]
        show_img("Image",img_copy,0)
        img_copy = out.astype(np.uint8)
    return img_copy

def vert_decrease(img,decreased_height,energy_function=1):
    img_copy = img.copy()
    height_decrease = img.shape[0] - decreased_height
    h_seams = repeated_hor_seams(img,height_decrease,energy_function)
    for h_seam in h_seams:
        img_copy[h_seam[:,0],h_seam[:,1]] = [0,0,255]
        show_img("Image",img_copy,0)
        img_copy = np.rollaxis(img_copy,1)
        mask = np.ones(img_copy.shape,dtype=bool)
        mask[h_seam[:,1],h_seam[:,0]] = False
        img_copy = img_copy[mask].reshape(img_copy.shape[0],img_copy.shape[1]-1,3)
        img_copy = np.rollaxis(img_copy,1)
    return img_copy

def vert_increase(img,increased_heigth,energy_function=1):
    img_copy = img.copy()
    height_increase = increased_heigth - img.shape[0]
    h_seams = transform_hor_seams(transform_hor_seams(repeated_hor_seams(img,height_increase,energy_function)))
    for h_seam in h_seams:
        img_copy_rolled = np.rollaxis(img_copy,1)
        mask = np.ones((img_copy_rolled.shape[0],img_copy_rolled.shape[1]+1,3),dtype=bool)
        mask[h_seam[:,1],h_seam[:,0]] = False
        out = np.empty(mask.shape,dtype=np.float32)
        out[mask] = img_copy_rolled.ravel()
        idx = h_seam[:,0]-1
        idx[idx < 0] = 0
        out[~mask] = ((img_copy_rolled[h_seam[:,1],idx].astype(np.float32) + img_copy_rolled[h_seam[:,1],h_seam[:,0]].astype(np.float32))/2).ravel()
        out = np.rollaxis(out,1)
        img_copy[h_seam[:,0],h_seam[:,1]] = [0,0,255]
        show_img("Image",img_copy,0)
        img_copy = out.astype(np.uint8)
    return img_copy

def resize(img,final_width,final_height,energy_function=1):
    width_resized_img = hor_decrease(img,final_width,energy_function) if final_width < img.shape[1] else hor_increase(img,final_width,energy_function) if final_width > img.shape[1] else img
    resized_img = vert_decrease(width_resized_img,final_height,energy_function) if final_height < img.shape[0] else vert_increase(width_resized_img,final_height,energy_function) if final_height > img.shape[0] else width_resized_img
    return resized_img

def remove_obj(img,mask,energy_function=1,order=0):
    img_copy = img.copy()
    v_size = np.max(np.sum(mask,axis=0))
    h_size = np.max(np.sum(mask,axis=1))
    print(h_size,v_size)
    show_multiple_img(["Image","Mask"],[img_copy,mask.astype(np.uint8)*255],0)
    while(np.sum(mask) != 0):
        print(np.sum(mask))
        img_copy2 = img_copy.copy()
        img_gray = cv2.cvtColor(img_copy,cv2.COLOR_RGB2GRAY)
        img_energy = get_energy(img_gray,energy_function,mask)
        if order == 1:
            h_seam = hor_seam(img_energy)
            img_copy = np.rollaxis(img_copy,1)
            mask = np.rollaxis(mask,1)
            seam_mask = np.ones(img_copy.shape,dtype=bool)
            seam_mask[h_seam[:,1],h_seam[:,0]] = False
            img_copy2[h_seam[:,0],h_seam[:,1]] = [0,0,255]
            img_copy = img_copy[seam_mask].reshape(img_copy.shape[0],img_copy.shape[1]-1,3)
            mask = mask[seam_mask[:,:,0]].reshape(img_copy.shape[0],img_copy.shape[1])
            img_copy = np.rollaxis(img_copy,1)
            mask = np.rollaxis(mask,1)
        else:
            v_seam = vert_seam(img_energy)
            seam_mask = np.ones(img_copy.shape,dtype=bool)
            seam_mask[v_seam[:,0],v_seam[:,1]] = False
            img_copy2[v_seam[:,0],v_seam[:,1]] = [0,0,255]
            img_copy = img_copy[seam_mask].reshape(img_copy.shape[0],img_copy.shape[1]-1,3)
            mask = mask[seam_mask[:,:,0]].reshape(img_copy.shape[0],img_copy.shape[1])
        # show_multiple_img(["Image","Seam","Mask"],[img_copy,img_copy2,mask.astype(np.uint8)*255],0)
    return img_copy

if __name__ == "__main__":
    img = load_img("11.jpg",1)
    resized_img = resize(img,img.shape[1]-50,img.shape[0])
    show_multiple_img(["Original Image","Resized Image"],[img,resized_img],0)
    # save_img("reduce_11_50",resized_img)
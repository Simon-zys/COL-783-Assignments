import numpy as np
import cv2
from functions import *
from energy import *
import time

def vert_seam(img_e1):
    dp = np.empty((img_e1.shape[0],img_e1.shape[1],2),dtype=np.int64)
    dp[0,:,0] = img_e1[0,:]
    dp[0,:,1] = 0
    for i in range(1,img_e1.shape[0]):
        for j in range(0,img_e1.shape[1]):
            dp[i,j,1] = np.argmin([dp[i-1,j,0],dp[i-1,j+1,0]]) if j == 0 else np.argmin([dp[i-1,j-1,0],dp[i-1,j,0]]) - 1 if j == img_e1.shape[1]-1 else np.argmin([dp[i-1,j-1,0],dp[i-1,j,0],dp[i-1,j+1,0]]) - 1
            dp[i,j,0] = img_e1[i,j] + dp[i-1,j+dp[i,j,1],0]
    l = np.argmin(dp[-1,:,0])
    seam_path = [(img_e1.shape[0]-1,l)]
    for i in range(img_e1.shape[0]-2,-1,-1):
        l = l+dp[i+1,l,1]
        seam_path = seam_path + [(i,l)]
    seam_path.reverse()
    return np.array(seam_path,dtype=np.int64)

def repeated_vert_seams(img,n,energy_function=1):
    v_seams = []
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    print(img_gray.shape)
    img_copy = img_gray.copy()
    for i in range(n):
        print(i)
        img_energy = get_energy(img_copy,energy_function)
        v_seams.append(vert_seam(img_energy))
        mask = np.ones(img_copy.shape,dtype=bool)
        mask[v_seams[-1][:,0],v_seams[-1][:,1]] = False
        img_copy = img_copy[mask].reshape(img_copy.shape[0],img_copy.shape[1]-1)
    return v_seams

def compose_vert_seams(v_seam1,v_seam2):
    compose = np.vectorize(lambda x,y : y+1 if x <= y else y)
    composed_seam = np.empty(v_seam2.shape,dtype=np.int64)
    composed_seam[:,0] = v_seam2[:,0]
    composed_seam[:,1] = compose(v_seam1[:,1],v_seam2[:,1])
    return composed_seam

def transform_vert_seams(v_seams):
    transformed_seams = [v_seams[0]]
    for v_seam in v_seams[1:]:
        temp_seam = v_seam
        for transformed_seam in transformed_seams[::-1]:
            temp_seam = compose_vert_seams(transformed_seam,temp_seam)
        transformed_seams.append(temp_seam)
    return transformed_seams

def hor_seam(img_e1):
    dp = np.empty((img_e1.shape[0],img_e1.shape[1],2),dtype=np.int64)
    dp[:,0,0] = img_e1[:,0]
    dp[:,0,1] = 0
    for j in range(1,img_e1.shape[1]):
        for i in range(0,img_e1.shape[0]):
            dp[i,j,1] = np.argmin([dp[i,j-1,0],dp[i+1,j-1,0]]) if i == 0 else np.argmin([dp[i-1,j-1,0],dp[i,j-1,0]]) - 1 if i == img_e1.shape[0]-1 else np.argmin([dp[i-1,j-1,0],dp[i,j-1,0],dp[i+1,j-1,0]]) - 1
            dp[i,j,0] = img_e1[i,j] + dp[i+dp[i,j,1],j-1,0]
    l = np.argmin(dp[:,-1,0])
    seam_path = [(l,img_e1.shape[1]-1)]
    for j in range(img_e1.shape[1]-2,-1,-1):
        l = l+dp[l,j+1,1]
        seam_path = seam_path + [(l,j)]
    seam_path.reverse()
    return np.array(seam_path,dtype=np.int64)

def repeated_hor_seams(img,n,energy_function=1):
    h_seams = []
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    print(img_gray.shape)
    img_copy = img_gray.copy()
    for i in range(n):
        print(i)
        img_energy = get_energy(img_copy,energy_function)
        h_seams.append(hor_seam(img_energy))
        img_copy = np.rollaxis(img_copy,1)
        mask = np.ones(img_copy.shape,dtype=bool)
        mask[h_seams[-1][:,1],h_seams[-1][:,0]] = False
        img_copy = img_copy[mask].reshape(img_copy.shape[0],img_copy.shape[1]-1)
        img_copy = np.rollaxis(img_copy,1)
    return h_seams

def compose_hor_seams(h_seam1,h_seam2):
    compose = np.vectorize(lambda x,y : y+1 if x <= y else y)
    composed_seam = np.empty(h_seam2.shape,dtype=np.int64)
    composed_seam[:,1] = h_seam2[:,1]
    composed_seam[:,0] = compose(h_seam1[:,0],h_seam2[:,0])
    return composed_seam

def transform_hor_seams(h_seams):
    transformed_seams = [h_seams[0]]
    for h_seam in h_seams[1:]:
        temp_seam = h_seam
        for transformed_seam in transformed_seams[::-1]:
            temp_seam = compose_hor_seams(transformed_seam,temp_seam)
        transformed_seams.append(temp_seam)
    return transformed_seams

if __name__ == "__main__":
    img = load_img("6.jpg",1)
    t0 = time.time()
    v_seams = transform_vert_seams(repeated_vert_seams(img,50))
    t1 = time.time()
    print(t1-t0)
    for v_seam in v_seams:
        img[v_seam[:,0],v_seam[:,1]] = [0,0,255]
    show_img("img",img,0)
    # save_img("seams_1_100_hog",img)
    # t0 = time.time()
    # h_seams = transform_hor_seams(repeated_hor_seams(img,10))
    # t1 = time.time()
    # print(t1-t0)
    # for h_seam in h_seams:
    #     img[h_seam[:,0],h_seam[:,1]] = [0,0,255]
    # show_img("img",img,0)
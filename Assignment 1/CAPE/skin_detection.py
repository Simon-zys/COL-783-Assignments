import numpy as np
import cv2
from functions import *
from face_detection import *
from scipy.signal import convolve2d
from math import sqrt

def skin_detection_multiple_thresholding(img: np.ndarray,xywh: list) -> np.ndarray:
    def get_hsv_mask(img: np.ndarray) -> np.ndarray:
        lower_thresh = np.array([0, 50, 0], dtype=np.uint8)
        upper_thresh = np.array([120, 150, 255], dtype=np.uint8)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        msk_hsv = cv2.inRange(img_hsv, lower_thresh, upper_thresh)
        msk_hsv[msk_hsv < 128] = 0
        msk_hsv[msk_hsv >= 128] = 1
        return msk_hsv.astype(float)
    def get_rgb_mask(img: np.ndarray) -> np.ndarray:
        lower_thresh = np.array([45, 52, 108], dtype=np.uint8)
        upper_thresh = np.array([255, 255, 255], dtype=np.uint8)
        mask_a = cv2.inRange(img, lower_thresh, upper_thresh)
        mask_b = 255 * ((img[:, :, 2] - img[:, :, 1]) / 20)
        mask_c = 255 * ((np.max(img, axis=2) - np.min(img, axis=2)) / 20)
        msk_rgb = np.bitwise_and(mask_c.astype(np.uint8), np.bitwise_and(mask_a.astype(np.uint8), mask_b.astype(np.uint8)))
        msk_rgb[msk_rgb < 128] = 0
        msk_rgb[msk_rgb >= 128] = 1
        return msk_rgb.astype(float)
    def get_ycrcb_mask(img: np.ndarray) -> np.ndarray:
        lower_thresh = np.array([90, 100, 130], dtype=np.uint8)
        upper_thresh = np.array([230, 120, 180], dtype=np.uint8)
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
        msk_ycrcb = cv2.inRange(img_ycrcb, lower_thresh, upper_thresh)
        msk_ycrcb[msk_ycrcb < 128] = 0
        msk_ycrcb[msk_ycrcb >= 128] = 1
        return msk_ycrcb.astype(float)
    def grab_cut_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        kernel = np.ones((50, 50), np.float32) / (50 * 50)
        dst = cv2.filter2D(mask, -1, kernel)
        dst[dst != 0] = 255
        free = np.array(cv2.bitwise_not(dst), dtype=np.uint8)
        grab_mask = np.zeros(mask.shape, dtype=np.uint8)
        grab_mask[:, :] = 2
        grab_mask[mask == 255] = 1
        grab_mask[free == 255] = 0
        if np.unique(grab_mask).tolist() == [0, 1]:
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            if img_col.size != 0:
                mask, bgdModel, fgdModel = cv2.grabCut(img_col, grab_mask, None, bgdModel, fgdModel, 5,
                                                    cv2.GC_INIT_WITH_MASK)
                mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
        return mask
    def closing(mask: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        return mask
    faces_roi = get_face_roi(img,xywh)
    skin_masks = []
    thresh = 0.5
    for face in faces_roi:
        mask_hsv = get_hsv_mask(face[1])
        mask_rgb = get_rgb_mask(face[1])
        mask_ycrcb = get_ycrcb_mask(face[1])
        n_masks = 3.0
        # mask = cv2.bitwise_or(mask_hsv, cv2.bitwise_or(mask_rgb, mask_ycrcb))
        mask = (mask_rgb + mask_hsv + mask_ycrcb) / n_masks
        mask[mask < thresh] = 0.0
        mask[mask >= thresh] = 255.0
        mask = mask.astype(np.uint8)
        mask = closing(mask)
        mask = grab_cut_mask(face[1], mask)
        skin_masks.append((face[0],mask.astype(np.uint8)))
    return skin_masks

def hsv_skin_detection(img: np.ndarray,xywh: list) -> list:
    faces_roi = get_face_roi(img,xywh)
    skin_masks = []
    for face in faces_roi:
        face_hsv = cv2.cvtColor(face[1], cv2.COLOR_BGR2HSV)
        lower = np.array([0, 48, 40], dtype = "uint8")
        upper = np.array([20, 255, 255], dtype = "uint8")
        skin_mask = cv2.inRange(face_hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skin_mask = cv2.erode(skin_mask, kernel, iterations = 2)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations = 2)
        skin_mask = cv2.GaussianBlur(skin_mask,(3,3),0)
        skin = cv2.bitwise_and(face[1],face[1] ,mask = skin_mask)
        skin_masks.append((face[0],skin_mask))
    return skin_masks

def skin_detection_cape(img: np.ndarray,xywh: list) -> list:
    faces_roi = get_face_roi(img,xywh)
    skin_masks = []
    for coordinates,face in faces_roi:
        face_lab = cv2.cvtColor(face,cv2.COLOR_BGR2Lab)
        face_hsv = cv2.cvtColor(face,cv2.COLOR_BGR2HSV)
        skin_mask = np.zeros((face.shape[0],face.shape[1]))
        average_mask = np.ones((5,5))/25
        a_mean = convolve2d(face_lab[:,:,1],average_mask,mode='same',boundary='symm')
        b_mean = convolve2d(face_lab[:,:,2],average_mask,mode='same',boundary='symm')
        ellipse_test = np.vectorize(lambda a,b: 1 if ((((a-143)/6.5)**2)+(((b-148)/12)**2)) < 1 else (sqrt((a-143)**2+(b-148)**2)-90)/-78 if sqrt((a-143)**2+(b-148)**2)-90 < 0 else 0)
        skin_mask = ellipse_test(a_mean,b_mean)
        hsv_test = np.vectorize(lambda s,h,m: 1 if m > 0.95 and s >= 0.25 and s <= 0.75 and h <= 0.095 else 0)
        skin_mask = hsv_test(face_hsv[:,:,1]/256,face_hsv[:,:,0]/180,skin_mask)
        # large_ellipse_test = np.vectorize(lambda m,a,b: 1 if ((((a-143)/6.5)**2)+(((b-148)/12)**2)) < 1.25 and m == 1 else 0)
        # skin_mask = large_ellipse_test(skin_mask,a_mean,b_mean)
        skin_masks.append((coordinates,skin_mask.astype(np.uint8)))
    return skin_masks

def get_skin(img: np.ndarray,xywh: list,skin_masks: list,show=0) -> None:
    faces_roi = get_face_roi(img,xywh)
    face_skin = []
    for face,mask in zip(faces_roi,skin_masks):
        skin = cv2.bitwise_and(face[1],face[1] ,mask = mask)
        face_skin.append((face[1],skin))
        if show != 0:
            cv2.imshow("Face - Skin", np.hstack([face[1],skin]))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return face_skin
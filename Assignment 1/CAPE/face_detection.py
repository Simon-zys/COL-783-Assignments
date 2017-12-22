import numpy as np
import cv2
from functions import *

def manual_face_detector(img: np.ndarray) -> list:
    img_copy = img.copy()
    vertex_list = []
    def select_faces(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            vertex_list.append((x,y))
            if(len(vertex_list)%2 == 0):
                a = vertex_list[-2]
                b = vertex_list[-1]
                _ = cv2.rectangle(img_copy,(a[0],a[1]),(b[0],b[1]),(255,0,0),2)
            else:
                _ = cv2.circle(img_copy,(vertex_list[-1][0],vertex_list[-1][1]),1,(255,0,0))
    cv2.namedWindow("Face Selection")
    cv2.setMouseCallback("Face Selection",select_faces)
    while(1):
        cv2.imshow("Face Selection",img_copy)
        k = cv2.waitKey(1)
        if k == 122 and len(vertex_list) != 0:
            vertex_list.pop()
            img_copy = img.copy()
            for x,y in zip(vertex_list[0::2],vertex_list[1::2]):
                _ = cv2.rectangle(img_copy,x,y,(255,0,0),2)
            if len(vertex_list)%2 == 1:
                _ = cv2.circle(img_copy,(vertex_list[-1][0],vertex_list[-1][1]),1,(255,0,0))
            continue
        if k != -1:
            break
    cv2.destroyWindow("Face Selection")
    rectangle_list = [(x[0],x[1],y[0]-x[0],y[1]-x[1]) for x,y in zip(vertex_list[0::2],vertex_list[1::2])]
    return rectangle_list

def viola_jones(img: np.ndarray) -> list:
    face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_xywh = face_cascade.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=6,minSize=(30,30))
    return faces_xywh.tolist() if len(faces_xywh) > 0 else []

def show_detected_face(img: np.ndarray,xywh: list,title=[]) -> None:
    img_copy = []
    for coordinates in xywh:
        img_copy.append(img.copy())
        for (x,y,w,h) in coordinates:
            _ = cv2.rectangle(img_copy[-1],(x,y),(x+w,y+h),(255,0,0),2)
    return show_multiple_img(title,img_copy,0)

def get_face_roi(img: np.ndarray,xywh: list) -> list:
    face_roi = []
    for i in xywh:
        p_x,p_y,q_x,q_y = get_coordinates(i)
        face_roi.append((i,img[p_y:q_y,p_x:q_x].copy()))
    return face_roi
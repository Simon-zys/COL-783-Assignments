import numpy as np
import cv2
from numpy import float32
from basic_fns import*
import dlib
import os

PREDICTOR_PATH = os.getcwd() + "/shape_predictor_68_face_landmarks.dat"

def get_rect_and_68control_points(rgb_image):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    rectangles = detector(rgb_image, 1)
    shape = None
    rectangle = None
    for k, rect in enumerate(rectangles):
        # Get the landmarks/parts for the face in box d.
        shape = predictor(rgb_image, rect)
        rectangle = rect

    return rectangle, shape

def show_rect_and_68control_points(rgb_image, rectangle, shape):

    win = dlib.image_window()   #creating a window where the image and its control points will be shown
    win.clear_overlay()
    win.set_image(rgb_image)    #setting the original image
    win.add_overlay(shape)      #adding the control points
    win.add_overlay(rectangle) #adding the facial box
    dlib.hit_enter_to_continue()

def draw_control_pts(image):

    img = np.copy(image)
    rect, shape = get_rect_and_68control_points(img)
    points_and_indices = get_points_and_indices(shape)
    for point_and_index in points_and_indices:
        pt1 = (point_and_index[0][0], point_and_index[0][1])
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(img, str(point_and_index[1]), pt1, font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    return img

def get_points_and_indices(shape):

    points_and_indices = []
    for i in range(shape.num_parts):
        x = shape.part(i).x
        y = shape.part(i).y
        point = np.array([x, y])
        index = i
        point_and_index = (point, index)
        points_and_indices.append(point_and_index)

    return points_and_indices

def all_regions(image):
    #36 to 41  is left eye
    #42 to 47 is right eye
    #60 to 67 is mouth cavity
    #48 to 59 is lips c2
    #rest is considered skin area c1

    rectangle, shape = get_rect_and_68control_points(image)
    points_and_indices = get_points_and_indices(shape)
    left_eye=[]
    right_eye=[]
    cavity = []
    lips = []
    skin = []
    left_brow=[]
    right_brow=[]
    for point_and_index in points_and_indices:
        if(17<=point_and_index[1]<=21):
            left_brow.append(point_and_index[0])
        if(22<=point_and_index[1]<=26):
            right_brow.append(point_and_index[0])
        if(48<=point_and_index[1]<=59):
            lips.append(point_and_index[0])
        if(36<=point_and_index[1]<=41):
            left_eye.append(point_and_index[0])
        if(42<=point_and_index[1]<=47):
            right_eye.append(point_and_index[0])
        if(60<=point_and_index[1]<=67):
            cavity.append(point_and_index[0])
        else:
            skin.append(point_and_index[0])

    return [left_eye, right_eye, cavity, lips, skin, left_brow, right_brow]

def createMask(image):

    regions = all_regions(image)
    rows = image.shape[0]
    cols = image.shape[1]
    masks = []

    for i, region in enumerate(regions):
        hull_contours = cv2.convexHull(np.vstack(np.array(region)))
        hull = np.vstack(hull_contours)
        # black image
        mask = np.zeros((rows, cols), dtype=np.uint8)
        # blit our contours onto it in white color
        cv2.drawContours(mask, [hull], 0, 1, -1)
        masks.append(mask)

    masks[3] = masks[3]-masks[2]
    masks[4] = masks[4] - masks[0] - masks[1] - masks[2] - masks[3] - masks[5] - masks[6]
    return masks

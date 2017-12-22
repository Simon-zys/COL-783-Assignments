import numpy as np
import cv2
from basic_fns import *
import control_pts as ctrl
from numpy import float32

def make_delaunay_class(image, rectangle, shape):

    rectangle = (0, 0, image.shape[1], image.shape[0])
    delaunay  = cv2.Subdiv2D(rectangle)
    for k in range(shape.num_parts):
        delaunay.insert((shape.part(k).x, shape.part(k).y))
    return delaunay

def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

def draw_delaunay_tri(img):

    rectangle, shape = ctrl.get_rect_and_68control_points(img)
    md = make_delaunay_class(img, rectangle, shape)
    triangleList = md.getTriangleList()
    triangleList = triangleList
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList :
        if(885 in t or -885 in t):
            color = (255, 0, 0)
            continue
        else:
            color = (0, 0, 255)
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
            cv2.line(img, pt1, pt2, color, 1)
            cv2.line(img, pt2, pt3, color, 1)
            cv2.line(img, pt3, pt1, color, 1)

    return img

def get_rect(rect):
    return (rect.left(), rect.top(), rect.right(), rect.bottom())

def warp_example_to_subject(example, subject):

    dsize = example.shape; rows = example.shape[0]; cols = example.shape[1]
    ex_rect, ex_shape = ctrl.get_rect_and_68control_points(example)
    ex_points_and_indices = ctrl.get_points_and_indices(ex_shape)
    ex_del = make_delaunay_class(example, ex_rect, ex_shape)
    sub_rect, sub_shape = ctrl.get_rect_and_68control_points(subject)
    sub_points_and_indices = ctrl.get_points_and_indices(sub_shape)
    sub_del = make_delaunay_class(subject, sub_rect, sub_shape)
    #we have calculated control points and delaunay triangles of both example and subject

    warped_tri = sub_del.getTriangleList()
    input_tri = ex_del.getTriangleList()
    output = np.ones(example.shape, dtype=float32)

    for i in range(warped_tri.shape[0]):
        # Affine Transformation
        dst = np.reshape(warped_tri[i, :], newshape=(3, 2))
        if(885 in dst or -885 in dst):
            continue
        tri2 = dst.astype(float32)
        pt_num=[]
        for pt in tri2:
            for k in sub_points_and_indices:
                if(pt[0] == k[0][0] and pt[1] == k[0][1]):
                    pt_num.append(k[1])

        src = np.ndarray(shape=(3, 2))
        for i, num in enumerate(pt_num):
            src[i, :] = ex_points_and_indices[num][0]
        tri1 = src.astype(dtype=float32)

        r1 = cv2.boundingRect(tri1) #xywh
        r2 = cv2.boundingRect(tri2)
        tri1Cropped = []
        tri2Cropped = []
        for i in range(3):
          tri1Cropped.append(((tri1[i][0] - r1[0]),(tri1[i][1] - r1[1])))
          tri2Cropped.append(((tri2[i][0] - r2[0]),(tri2[i][1] - r2[1])))
        img1Cropped = example[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        warpMat = cv2.getAffineTransform(np.float32(tri1Cropped), np.float32(tri2Cropped))
        img2Cropped = cv2.warpAffine( img1Cropped, warpMat, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
        # Get mask by filling triangle
        mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)

        cv2.fillConvexPoly(mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0);
        # Apply mask to cropped region
        img2Cropped = img2Cropped * mask

        # Copy triangular region of the rectangular patch to the output image
        output[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = output[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
        output[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = output[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Cropped

    return output

def merge_example_and_image(example, subject):

    subject_face = warp_example_to_subject(subject, subject)
    output = subject - subject_face
    warped_example = warp_example_to_subject(example, subject)
    output = output + warped_example
    return output


#
# example = read_image_as_rgb('example.png')
# ex = read_image_as_rgb('example.png')
# subject = read_image_as_rgb('subject.png')
# sub = np.copy(subject)
#
# del_ex = draw_delaunay_tri(ex)
# write_image('del_ex.png', rgb2bgr(del_ex))
# del_sub = draw_delaunay_tri(sub)
# write_image('del_sub.png', rgb2bgr(del_sub))
#
# output = warp_example_to_subject(example,  subject)
# show_image('please', rgb2bgr(output))
# write_image('example2subject.png', rgb2bgr(output))

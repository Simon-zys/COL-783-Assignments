import cv2
import numpy as np
from numpy import float32
from basic_fns import *
import layer_decomposition as layers
import transfers
import control_pts as ctrl
import rgb2lab
import delaunay as dela

#taking input
image1 = 'subject.png'
image2 = 'example.png'
subject = read_image_as_bgr(image1)
example = read_image_as_bgr(image2)
shape = subject.shape

#presentation
x = np.copy(example)
s=np.copy(subject)
pts = ctrl.draw_control_pts(s)
show_image('control points', pts)
del_tris = dela.draw_delaunay_tri(s)
show_image('delaunay triangles', del_tris)

pts = ctrl.draw_control_pts(x)
show_image('control points', pts)
del_tris = dela.draw_delaunay_tri(x)
show_image('delaunay triangles', del_tris)

#1. face alignment
example_warped = dela.merge_example_and_image(example, subject)
show_image('warped_image', example_warped)

#2. layer decomposition into face structure and skin detail and color using bilateral filter
layersSubject = layers.get_layers_bilateral(subject)
layersExample = layers.get_layers_bilateral(example_warped.astype('uint8'))

#3. skin detail transfer
detailResult = transfers.skin_detail_transfer(layersSubject[1], layersExample[1])

#4. color transfer
masks = ctrl.createMask(subject)
colorResult = transfers.color_transfer(masks, layersSubject[2], layersExample[2], r = 0.8)

#5. highlight and shading transfer
structureResult = transfers.highlight_shading_transfer(masks, layersSubject[0], layersExample[0])

#6. lip makeup transfer
lipsResult = transfers.lip_makeup_transfer(layersSubject, layersExample, masks)

#compiling lab of output image
temp = np.ones(shape, dtype='uint8')
temp = temp[:, :, 0]-masks[3]
structureResult = np.multiply(structureResult, temp)
Lout = (structureResult + lipsResult + detailResult).astype('uint8')

shape = (Lout.shape[0], Lout.shape[1], 3)
finalOut = np.ndarray(shape=shape, dtype='uint8')
finalOut[:, :, 0] = Lout
finalOut[:, :, 1] = colorResult[:, :, 0]
finalOut[:, :, 2] = colorResult[:, :, 1]

finalOut = rgb2lab.lab_to_rgb(finalOut)
finalOut = rgb2bgr(finalOut)
show_image('final please', finalOut)
write_image('final_image.png', finalOut)

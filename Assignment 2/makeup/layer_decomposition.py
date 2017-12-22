import cv2
import numpy as np
from numpy import float32
from numpy import uint8
import rgb2lab
from basic_fns import *
from wls_filter import wlsFilter

def get_layers_bilateral(inputImage):

    cielab_img = cv2.cvtColor(inputImage, cv2.COLOR_BGR2Lab)
    structure_layer = cv2.bilateralFilter(cielab_img[:,:,0],9,75,75)
    detail_layer = cielab_img[:,:,0] - structure_layer
    color_layer = cielab_img[:, :, 1:3]
    layersImage = [structure_layer, detail_layer, color_layer]

    return layersImage

def beta(inputImage):

    beta = np.ones(shape=shape)
    return beta

def get_layers_wlsFilter(inputImage):


    return layersImage

def get_layers_wls(inputImage):
    cielab_img = cv2.cvtColor(inputImage, cv2.COLOR_BGR2Lab)
    structure_layer, detail_layer = wlsFilter(cielab_img[:,:,0])
    color_layer = cielab_img[:, :, 1:3]
    detail_layer = (detail_layer*255).astype('uint8')
    return [structure_layer, detail_layer, color_layer]

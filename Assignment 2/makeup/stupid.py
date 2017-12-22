import cv2
from skimage import io, color
from basic_fns import *
#import layer_decomposition

filename = 'subject.png'

rgb = io.imread(filename)
lab = color.rgb2lab(rgb)
print(type(lab))
print(lab.shape)
show_img('lab', lab[:, :, 0])
rgb = color.lab2rgb(lab)
show_img('rgb', rgb*255)

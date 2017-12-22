import cv2
from skimage import io, color
from basic_fns import *

def lab_to_rgb(inputImage):

    outputImage = cv2.cvtColor(inputImage, cv2.COLOR_LAB2RGB)
    return outputImage

def rgb_to_lab(inputImage):

    outputImage = cv2.cvtColor(inputImage, cv2.COLOR_RGB2LAB)
    return outputImage

def rgb_pixel_2_lab_pixel (inputColor):
   RGB = [0, 0, 0]

   for i, value in enumerate(inputColor) :
       value = float(value) / 255

       if value > 0.04045 :
           value = ( ( value + 0.055 ) / 1.055 ) ** 2.4
       else :
           value = value / 12.92

       RGB[i] = value * 100

   XYZ = [0, 0, 0,]

   X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
   Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
   Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505
   XYZ[ 0 ] = round( X, 4 )
   XYZ[ 1 ] = round( Y, 4 )
   XYZ[ 2 ] = round( Z, 4 )

   XYZ[ 0 ] = float( XYZ[ 0 ] ) / 95.047         # ref_X =  95.047   Observer= 2°, Illuminant= D65
   XYZ[ 1 ] = float( XYZ[ 1 ] ) / 100.0          # ref_Y = 100.000
   XYZ[ 2 ] = float( XYZ[ 2 ] ) / 108.883        # ref_Z = 108.883

   for i, value in enumerate(XYZ) :
       if value > 0.008856 :
           value = value**(0.3333333333333333)
       else :
           value = (7.787*value) + (16/116)

       XYZ[i] = value

   Lab = [0, 0, 0]

   L = ( 116 * XYZ[ 1 ] ) - 16
   a = 500 * ( XYZ[ 0 ] - XYZ[ 1 ] )
   b = 200 * ( XYZ[ 1 ] - XYZ[ 2 ] )

   Lab [ 0 ] = round( L, 4 )
   Lab [ 1 ] = round( a, 4 )
   Lab [ 2 ] = round( b, 4 )

   return Lab

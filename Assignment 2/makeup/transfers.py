import cv2
import numpy as np
from basic_fns import *
from scipy import signal

def skin_detail_transfer(detailSubject, detailExample, subject_weight = 0, example_weight = 1):
    detailResult = subject_weight*detailSubject + example_weight*detailExample
    return detailResult

def color_transfer(masks, colorSubject, colorExample, r = 0.8):

    shape = colorExample.shape
    colorResult = np.zeros(shape=shape, dtype='uint8')
    mask = masks[0]+masks[1]+masks[2]
    for i in range(shape[0]):
        for j in range(shape[1]):
            if(mask[i, j] == 0):
                colorResult[i, j] = (1-r)*colorSubject[i, j] + r*colorExample[i, j]
            else:
                colorResult[i, j] = colorSubject[i, j]

    return colorResult

def highlight_shading_transfer(masks, structureSubject, structureExample):
    shape = structureSubject.shape
    structureResult = np.ndarray(shape=shape, dtype='uint8')
    der_structureResult = np.ndarray(shape=shape, dtype='uint8')

    der_structureExample = laplacian(structureExample)
    der_structureSubject = laplacian(structureSubject)
    # gauss_structureExample = gaussian(structureExample)
    gauss_structureSubject = gaussian(structureSubject)

    beta = masks[4]

    for i in range(shape[0]):
        for j in range(shape[1]):
            if(abs(der_structureExample[i, j])*beta[i, j] > abs(der_structureSubject[i, j])):
                der_structureResult[i, j] = der_structureExample[i, j]
            else:
                der_structureResult[i, j] = der_structureSubject[i, j]

    structureResult = der_structureResult + gauss_structureSubject

    return structureResult

def lip_point(mask, Lsubject, Lexample, p):

    maxi = 0
    argmax = (0,0)
    shape = Lexample.shape

    for qi in range(shape[0]):
        for qj in range(shape[1]):
            if(mask[qi, qj] == 1):
                Ip = Lsubject[p[0], p[1]]
                Eq = Lexample[qi, qj]
                argument = np.exp((-(p[0]-qi)**2) + -(p[1]-qj)**2)*np.exp((-(Ip-Eq)**2))
                if(argument>maxi):
                    maxi = argument
                    argmax = (qi, qj)

    return (argmax[0], argmax[1])

def lip_makeup_transfer_1(layersSubject, layersExample, masks):

    Lsubject = layersSubject[0]+layersSubject[1]
    Lexample = layersExample[0]+layersExample[1]
    mask = masks[3]
    shape = Lexample.shape
    #do histogram equalization of Lsubject and Lexample


    lips_output = np.ndarray(shape=shape, dtype='uint8')

    for i in range(shape[0]):
        for j in range(shape[1]):
            if(mask[i, j]==1):
                p = (i, j)
                q = lip_point(mask, Lsubject, Lexample, p)
                lips_output[i, j] = Lexample[q[0], q[1]]

    return lips_output

def lip_makeup_transfer(layersSubject, layersExample, masks):

    shape = layersSubject[0].shape
    lip_output = np.ndarray(shape=shape, dtype='uint8')
    lip_output = np.multiply((layersSubject[1] + layersExample[0]), masks[3])
    return lip_output

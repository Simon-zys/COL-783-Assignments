import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
from functions import *

def eacp(g,img_lum,w=None,l=0.2,alpha=0.3):
    eps = 1e-4
    g = g.flatten(1)
    w = np.zeros(g.shape) if w is None else w.flatten(1)
    shape = img_lum.shape
    cells = shape[0]*shape[1]
    dy = -l / (np.absolute(np.diff(img_lum,1,0)) ** alpha + eps)
    # dy = (np.concatenate((dy,[np.zeros((shape[1],shape[2]))]),axis=0)).flatten(1)
    dy = (np.vstack((dy, np.zeros(shape[1], )))).flatten(1)
    dx = -l / (np.absolute(np.diff(img_lum,1,1)) ** alpha + eps)
    # dx = (np.concatenate((dx,np.zeros((shape[0],1,shape[2]))),axis=1)).flatten(1)
    dx = (np.hstack((dx, np.zeros(shape[0], )[:, np.newaxis]))).flatten(1)
    a = spdiags(np.vstack((dx,dy)),[-shape[0], -1],cells,cells)
    d = w - (dx + np.roll(dx,shape[0]) + dy + np.roll(dy,1))
    a = a + a.T + spdiags(d, 0,cells,cells)
    f = spsolve(a, w*g).reshape(shape[::-1])
    A = np.rollaxis(f,1)
    return A
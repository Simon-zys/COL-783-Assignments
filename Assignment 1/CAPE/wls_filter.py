import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve

def wlsFilter(img_lum: np.ndarray,l=0.4,alpha=1.2) -> tuple:
    eps=1e-4
    img_lum = img_lum.astype(np.float)/255.0
    shape = img_lum.shape
    cells = shape[0]*shape[1]
    dy = -l / (np.absolute(np.diff(img_lum,1,0)) ** alpha + eps)
    # dy = (np.concatenate((dy,[np.zeros((shape[1],shape[2]))]),axis=0)).flatten(1)
    dy = (np.vstack((dy, np.zeros(shape[1], )))).flatten(1)
    dx = -l / (np.absolute(np.diff(img_lum,1,1)) ** alpha + eps)
    # dx = (np.concatenate((dx,np.zeros((shape[0],1,shape[2]))),axis=1)).flatten(1)
    dx = (np.hstack((dx, np.zeros(shape[0], )[:, np.newaxis]))).flatten(1)
    a = spdiags(np.vstack((dx,dy)),[-shape[0], -1],cells,cells)
    d = 1 - (dx + np.roll(dx,shape[0]) + dy + np.roll(dy,1))
    a = a + a.T + spdiags(d, 0,cells,cells)
    u = spsolve(a, img_lum.flatten(1)).reshape(shape[::-1])
    base = np.rollaxis(u,1)
    detail = img_lum - base
    return (base, detail)
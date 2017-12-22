import cv2, os
import numpy as np
path = os.getcwd() + "/Test Images/"

def load_img(img: str) -> np.ndarray:
    return cv2.imread(path+img,0)

def show_img(title: list,img: list,wait: int) -> None:
    for x, y in zip(title, img):
        cv2.imshow(x,y)
    cv2.waitKey(wait)
    for x in title:
        if cv2.getWindowProperty(x,0) != -1:
            cv2.destroyWindow(x)

def show_img_cont(title: str,img: np.ndarray) -> None:
    while(1):
        cv2.imshow(title,img)
        k = cv2.waitKey(1)
        if k != -1:
            break
    cv2.destroyWindow(title)

def zoom_img_builtin(img: np.ndarray,x: int,y: int) -> np.ndarray:
    return cv2.resize(img,None,fx=x,fy=y)

def zoom_img_builtin_rec(img: np.ndarray,x: int,y: int) -> np.ndarray:
    return cv2.resize(img,None,fx=1/x,fy=1/y)

def zoom_img_replication(img: np.ndarray) -> np.ndarray:
    zoom_img = np.zeros((img.shape[0]*3,img.shape[1]*3),dtype=np.uint8)
    for x in range(0,zoom_img.shape[0]):
        for y in range(0,zoom_img.shape[1]):
            zoom_img[x,y] = img[int(x/3),int(y/3)]
    return zoom_img

def zoom_img_replication_rec(img: np.ndarray) -> np.ndarray:
    zoom_img_rec = np.zeros((int(img.shape[0]/3),int(img.shape[1]/3)),dtype=np.uint8)
    for x in range(0,img.shape[0],3):
        for y in range(0,img.shape[1],3):
            zoom_img_rec[int(x/3),int(y/3)] = img[x,y]
    return zoom_img_rec

def zoom_img_interpolate_linear(img: np.ndarray) -> np.ndarray:
    img = (img.copy()).astype('float')
    z = 4
    interpolate = lambda x,y,m: [np.round(x+(i*(y-x)/(m-1))).astype(int) for i in range(1,m)]
    x_interpolate = np.zeros((img.shape[0],(z-1)*img.shape[1]-(z-2)))
    for row in range(len(img)):
        l = [img[row,0]]
        for i in range(len(img[row,:-1])):
            l = l + interpolate(img[row,i],img[row,i+1],z)
        x_interpolate[row] = (np.array(l)).astype(int)
    x_interpolate = x_interpolate.copy().astype('float')
    img_interpolate = np.zeros(((z-1)*img.shape[0]-(z-2),(z-1)*img.shape[1]-(z-2)))
    for column in range(len(x_interpolate[0])):
        l = [x_interpolate[0,column]]
        for i in range(len(x_interpolate[:-1,column])):
            l = l + interpolate(x_interpolate[i,column],x_interpolate[i+1,column],z)
        img_interpolate[:,column] = (np.array(l)).astype(int)
    img_interpolate = img_interpolate.copy().astype('uint8')
    return img_interpolate

def zoom_img_interpolate_linear_rec(img: np.ndarray) -> np.ndarray:
    zoom_img_rec = np.zeros((int(np.ceil(img.shape[0]/3)),int(np.ceil(img.shape[0]/3))),dtype=np.uint8)
    for x in range(0,img.shape[0],3):
        for y in range(0,img.shape[1],3):
            zoom_img_rec[int(x/3),int(y/3)] = img[x,y]
    return zoom_img_rec

def zoom_click(event,x,y,flags,param):
    global point_x, point_y
    width = img.shape[1]
    height = img.shape[0]
    size = int(2**(width.bit_length()-2)) if width < height else int(2**(height.bit_length()-2))
    if event == cv2.EVENT_MOUSEMOVE or event == cv2.EVENT_LBUTTONDOWN:
        point_x = int(x-size/2) if x > size/2 else 0
        point_y = int(y-size/2) if y > size/2 else 0
        point_x = int(width-size) if x+size/2 > width else point_x
        point_y = int(height-size) if y+size/2 > height else point_y
    if event == cv2.EVENT_MOUSEWHEEL:
        # delta = cv2.getMouseWheelDelta(flags)
        # print(delta)
        # show_img(["Test"],[zoom_img(img[point_y:point_y+size,point_x:point_x+size],3,3)],10000)
        print("MOUSE WHEEL DETECTED")
    if event == cv2.EVENT_LBUTTONDOWN:
        img_titles = ["ROI","Replication_REC","Built-in_REC","Linear_Interpolation_REC","Replication","Built-in","Linear_Interpolation"]
        roi = img[point_y:point_y+size,point_x:point_x+size]
        zoom_images = []
        zoom_images.append(zoom_img_replication(roi))
        zoom_images.append(zoom_img_builtin(roi,3,3))
        zoom_images.append(zoom_img_interpolate_linear(roi))
        reconstructed_images = []
        reconstructed_images.append(zoom_img_replication_rec(zoom_images[0]))
        reconstructed_images.append(zoom_img_builtin_rec(zoom_images[1],3,3))
        reconstructed_images.append(zoom_img_interpolate_linear_rec(zoom_images[2]))
        show_img(img_titles,[roi]+reconstructed_images+zoom_images,10000)

point_x = 0
point_y = 0
img = load_img("cat.png")
cv2.namedWindow("Image")
cv2.setMouseCallback("Image",zoom_click)
show_img_cont("Image",img)
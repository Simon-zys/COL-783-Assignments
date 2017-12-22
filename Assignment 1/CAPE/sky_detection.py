import cv2
import numpy as np
from functions import *
from colorthief import ColorThief
import time

def get_swatch(img: np.ndarray) -> list:
    img_copy = img.copy()
    vertex_list = []
    def select_swatch(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            vertex_list.append((x,y))
            if(len(vertex_list)%2 == 0):
                a = vertex_list[-2]
                b = vertex_list[-1]
                _ = cv2.rectangle(img_copy,(a[0],a[1]),(b[0],b[1]),(255,0,0),2)
            else:
                _ = cv2.circle(img_copy,(vertex_list[-1][0],vertex_list[-1][1]),1,(255,0,0))
    cv2.namedWindow("Swatch Selection")
    cv2.setMouseCallback("Swatch Selection",select_swatch)
    while(1):
        cv2.imshow("Swatch Selection",img_copy)
        k = cv2.waitKey(1)
        if k == 122 and len(vertex_list) != 0:
            vertex_list.pop()
            img_copy = img.copy()
            for x,y in zip(vertex_list[0::2],vertex_list[1::2]):
                _ = cv2.rectangle(img_copy,x,y,(255,0,0),2)
            if len(vertex_list)%2 == 1:
                _ = cv2.circle(img_copy,(vertex_list[-1][0],vertex_list[-1][1]),1,(255,0,0))
            continue
        if k != -1:
            break
    cv2.destroyWindow("Swatch Selection")
    rectangle_list = [(x[0],x[1],y[0]-x[0],y[1]-x[1]) for x,y in zip(vertex_list[0::2],vertex_list[1::2])]
    rectangle_list = None if len(rectangle_list) == 0 else rectangle_list
    return rectangle_list

def create_bilateral_blur_img(img):
    return cv2.bilateralFilter(img,9,75,75)

def get_swatched_image(img):
    swatch = get_swatch(img)
    if swatch is None:
        return None
    a = swatch[0][0]
    b = swatch[0][1]
    da = swatch[0][2]+a
    db = swatch[0][3]+b

    if a > da:
        a, da = da ,a
    if b > db:
        b, db = db ,b
    return img[b:db, a:da, :]

def find_dom_color_img(img):
    if img is None:
        return None
    #preprocessing
    cv2.imwrite('./Data/swatched_part.png',img)
    time.sleep(5)
    color_thief = ColorThief('./Data/swatched_part.png')
    dominant_color = color_thief.get_color(quality=1)
    temp0 = dominant_color[0]
    temp1 = dominant_color[1]
    temp2 = dominant_color[2]
    dominant_color = []
    dominant_color.append(temp2)
    dominant_color.append(temp1)
    dominant_color.append(temp0)
    return dominant_color

def find_binary_sky_mask(ideal_blue_ref, bilateral_blur_img):
    if ideal_blue_ref is None:
        return None
    #ideal blue characteristics
    blue_mean = ideal_blue_ref[0]
    green_mean = ideal_blue_ref[1]
    red_mean = ideal_blue_ref[2]
    stddev = 30
    #setting ranges for acceptable values in the 3 channels
    blue_min = (blue_mean - stddev)
    blue_max = (blue_mean + stddev)
    green_min = (green_mean - stddev)
    green_max = (green_mean + stddev)
    red_min = (red_mean - stddev)
    red_max = (red_mean + stddev)
    #finding channel wise filters
    x = bilateral_blur_img.shape[0]
    y = bilateral_blur_img.shape[1]
    bilateral_blur_img_blue = bilateral_blur_img[:, :, 0]
    bilateral_blur_img_blue = np.resize(bilateral_blur_img_blue, (x, y))
    blue_mask = (bilateral_blur_img_blue>=blue_min) & (bilateral_blur_img_blue<=blue_max)
    binary_blue_channel = blue_mask.astype(np.int)
    bilateral_blur_img_green = bilateral_blur_img[:, :, 1]
    bilateral_blur_img_green = np.resize(bilateral_blur_img_green, (x, y))
    green_mask = (bilateral_blur_img_green>=green_min) & (bilateral_blur_img_green<=green_max)
    binary_green_channel = green_mask.astype(np.int)
    bilateral_blur_img_red = bilateral_blur_img[:, :, 2]
    bilateral_blur_img_red = np.resize(bilateral_blur_img_red, (x, y))
    red_mask = (bilateral_blur_img_red>=red_min) & (bilateral_blur_img_red<=red_max)
    binary_red_channel = red_mask.astype(np.int)
    #finding net binary sky mask.
    sum_of_3_channels = binary_blue_channel + binary_red_channel + binary_green_channel
    total_mask = (sum_of_3_channels == 3)
    binary_sky_mask = total_mask.astype(np.int)
    return binary_sky_mask

def refine_binary_mask(binary_sky_mask):
    binary_sky_mask = binary_sky_mask.astype(float)
    laplacian = cv2.Laplacian(binary_sky_mask,cv2.CV_64F)
    laplacian = laplacian/np.max(laplacian)
    exponentially_decreasing_laplacian = np.multiply(np.exp(-laplacian), (binary_sky_mask == 1).astype(int))
    return exponentially_decreasing_laplacian

def get_image_from_mask(binary_sky_mask, bilateral_blur_img):
    #applying mask to the color image to get only blue sky picture
    x = bilateral_blur_img.shape[0]
    y = bilateral_blur_img.shape[1]
    z = bilateral_blur_img.shape[2]
    colored_sky_image = np.empty([x, y, z])
    colored_sky_image[:, :, 0] = np.multiply(binary_sky_mask, bilateral_blur_img[:, :, 0])
    colored_sky_image[:, :, 1] = np.multiply(binary_sky_mask, bilateral_blur_img[:, :, 1])
    colored_sky_image[:, :, 2] = np.multiply(binary_sky_mask, bilateral_blur_img[:, :, 2])
    colored_sky_image = colored_sky_image.copy().astype('uint8')
    return colored_sky_image

def create_sky_prob_map(img):
    #reading the image for which sky has to be detected
    # img = cv2.imread('./Test Images/' + a + '.jpg')

    #creating the bilaterally filtered image
    bilateral_blur_img = create_bilateral_blur_img(img)

    #taking a swatch of that image to get a sample of sky color
    swatch_image_blue = get_swatched_image(img)
    if swatch_image_blue is None:
        print("No Sky Detected")
        return []
    #using swatch to find the ideal blue color
    ideal_blue_ref = find_dom_color_img(swatch_image_blue)
    #binary sky mask is found by searching for ideal blue color on blurred image
    binary_sky_mask_blue = find_binary_sky_mask(ideal_blue_ref, bilateral_blur_img)

    #taking a swatch of bilateral blur image to get a sample of the cloud color
    swatch_image_cloud = get_swatched_image(img)
    if swatch_image_cloud is None:
        print("No Clouds Detected")
    #binary sky mask is found by searching ideal cloud color on blurred image
    ideal_cloud_ref = find_dom_color_img(swatch_image_cloud)
    #binary sky mask is found by searching for ideal cloud color on blurred image
    binary_sky_mask_cloud = find_binary_sky_mask(ideal_cloud_ref, bilateral_blur_img)

    #adding blue mask and sky mask to get cloud mask
    binary_sky_mask = binary_sky_mask_blue if binary_sky_mask_cloud is None else binary_sky_mask_blue + binary_sky_mask_cloud

    #refining the binary mask by setting values proportional to exponentially decreasing gradient
    refined_sky_mask = refine_binary_mask(binary_sky_mask)

    #applying mask to the color image to get only blue sky picture
    colored_sky_image = get_image_from_mask(refined_sky_mask, bilateral_blur_img)
    show_img("Sky",colored_sky_image,0)
    return colored_sky_image

if __name__ == '__main__':
    img = load_img("4.jpg",1)
    show_img("Image",img,0)
    # base = cv2.bilateralFilter(img_lab[:,:,1],5,10,12)
    create_sky_prob_map(img)
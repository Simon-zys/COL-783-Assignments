# Assignment 2 - Digital Face Makeup by Example
> By Anoop, 2015CS10265 and Akash, 2015EE30522

- **Face Alignment** :
  - <u>Finding Control Points for Faces</u> -  
    The popular machine learning library dlib was used to find the control points consistently for the faces. The repo can be found here, https://github.com/davisking/dlib  
  - <u>Delauney Triangulation</u> -  
    Using the popular image processing library, opencv's (https://github.com/opencv/opencv) inbuilt function we obtained delauney triangles for the control points.
  - <u>Alignment</u> -  
    Faces were aligned by performing an affine transform (using opencv's method) over the obtained triangles. 

- **Makeup Transfer** :
  - <u>Colorspace</u> -  
    All the computations have been done in the CIELAB colorspace. The conversion from rgb to CIELAB was performed using opencv's function.
  - <u>WLS Filter</u> -  
    "https://github.com/drakeguan/cp11fall_project1/blob/master/wlsFilter/wlsFilter.m"  
    Took ideas from above matlab implementation to implementt WLS Filter in python3.  
    Alternatively a Bilateral Filter (using opencv function) approach was also used.  
  - <u>Decompositions and Transfers</u> -  
    The transfer of makeup layer by layer has been implemented by following the paper mentioned religiously. 
  -<u>Lip Transfer</u> -  
    The method of finding the max argument of a difference of gaussian funtions as described in the paper was computationally heavy.
    Although that was implemented, an attempt at a simple more intuitive method is also made. It simply invloves using the detail 
    layer of the subject image (to retain texture) and the structure layer of the example image (to retain color). 
    The results seemed to vary depending on images.   
  
- **Areas of Improvement** :
  - <u>Control Point Selection</u> -  
    Only points obtained using dlib were used. But these points ignored the forehead, making the results awkward in several cases.
    This however can be overcome by selecting extra control points manually. 
  - <u>Post Processing</u> -  
    No post processing has been done but adding simple image enhancement steps may help. 
    
    Code & ideas from this excellent blog post have been used. Follow below link to see more, 
    https://www.learnopencv.com/delaunay-triangulation-and-voronoi-diagram-using-opencv-c-python/

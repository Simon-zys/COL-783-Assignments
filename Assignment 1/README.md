# Assignment 1 - Content-Aware Automatic Photo Enhancement
> By Anoop, 2015CS10265 and Akash, 2015EE30522

- **Face Detection** :
  - <u>AINDANE</u> -  
    Implemented "Adaptive and integrated neighborhood-dependent approach for nonlinear enhancement of color images" paper
  - <u>Skin Segmentation</u> -  
    Tried multiple techniques from different papers/tutorials. Found best implementation to be the one given in CAPE paper Appendix B
  - <u>WLS Filter</u> -  
    "https://github.com/drakeguan/cp11fall_project1/blob/master/wlsFilter/wlsFilter.m"  
    Took ideas from above matlab implementation to implementt WLS Filter in python3

- **Sky Detection** :
  - <u>Swatch Selection</u>:
    Swatch selection followed by sky segmentation has been implemented. This is an alternate method proposed in the class.

- **Shadow-Saliency Enhancement** :
  - <u>Saliency Map</u> -  
    Implementation taken from - https://github.com/akisato-/pySaliencyMap
  - <u>CAPE Enhancement</u> -  
    Using the saliency map obtained, further enhancements have been made according to CAPE.
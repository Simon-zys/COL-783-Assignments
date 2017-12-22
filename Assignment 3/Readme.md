# Assignment 3 - Seam Carving for Content-Aware Image Resizing
> By Anoop, 2015CS10265 and Akash, 2015EE30522

- **Seam Generation** :
  - <u>Energy Maps</u> -  
    Used multiple energy functions - Successive Difference, Sobel (Best results), Entropy, Histogram of Oriented Gradients.
  - Identified the minimum energy seam using Dynamic Programming approach.
  
- **Image Aspect Ratio Changing** :  
  Seams are added/removed according to output aspect ratio requirements.

- **Object Removal through Template Matching** :
  - <u>Fourier Correlation</u>
  - <u>Generalized Hough Transform</u> -  
    Implemented Generalized Hough Transform for custom quantization in angle.

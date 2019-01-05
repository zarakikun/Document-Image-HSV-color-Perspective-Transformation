# Document-Image---HSV-color---Perspective-Transformation
This script uses OpenCV to generate the upper and lower HSV ranges of any pixel in an RGB document image, tolerances could be modified.

The selected color range is used for color subtraction, followed by canny edge detection,hough transform, line instersections and lastly perspective transformation

# Extreme Points
# 
# This program was written to find the extreme points of palm of hands 
# and drawing the convex hull for the set of points in the contour
# This can form the basis of gesture recognition as well

import cv2
import argparse
from dominant_color import DominantColor

# Driver function
def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Detect the shapes in image")
    parser.add_argument('-i', help="Process image at the specified path", required=True)
    path = parser.parse_args().i


    # Open the image file
    image = cv2.imread(path)


    # Extract dominant color
    dcObj = DominantColor(image, 1)
    dColor = dcObj.extractColor()


    # Convert image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # Applying Gaussian blurring
    image_blur = cv2.GaussianBlur(image_gray, (7, 7), 0)


    # Thresholding
    image_thresh = None
    
    # Assuming background occupies most of the space and is a shade of either black or white
    if max(dColor[0]) > 225:
        # White background
        image_thresh = cv2.threshold(image_blur, 225, 255, cv2.THRESH_BINARY_INV)[1]
    elif max(dColor[0]) < 100:
        # Black background
        image_thresh = cv2.threshold(image_blur, 100, 255, cv2.THRESH_BINARY)[1]


    # Set of erosions and dilations to remove noise
    image_thresh = cv2.erode(image_thresh, None, iterations=4)
    image_thresh = cv2.dilate(image_thresh, None, iterations=4)

    
    # Find contours in the thresholded image
    contours, heirarchy = cv2.findContours(image_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

   
    # Make a copy of image for displaying output
    image_output = image.copy()

    
    # Pick the max contour by area
    contour_max = max(contours, key=cv2.contourArea)

    
    # Drawing the convex hull
    convex_hull = cv2.convexHull(contour_max)
    cv2.drawContours(image_output, [convex_hull], -1, (255, 0, 0), 2)
    
    
    # Find extreme points
    # Topmost
    top = contour_max[contour_max[:, :, 1].argmin()]
    cv2.circle(image_output, (top[0, 0], top[0, 1]), 4, (0, 0, 255), -1)
    
    # Bottommost
    bottom = contour_max[contour_max[:, :, 1].argmax()]
    cv2.circle(image_output, (bottom[0, 0], bottom[0, 1]), 4, (0, 0, 255), -1)
    
    # Leftmost
    left = contour_max[contour_max[:, :, 0].argmin()]
    cv2.circle(image_output, (left[0, 0], left[0, 1]), 4, (0, 0, 255), -1)
    
    # Rightmost
    right = contour_max[contour_max[:, :, 0].argmax()]
    cv2.circle(image_output, (right[0, 0], right[0, 1]), 4, (0, 0, 255), -1)

    
    # Display output image
    cv2.imshow("Extreme Points", image_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()
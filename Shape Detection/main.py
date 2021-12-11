# Shape Detection
# 
# This is the driver program where the image is preprocessed before
# passing it to an object of class Shape for detecting the shapes
#
# The images are assumed to have only black and white backgrounds

import cv2
import argparse
from shape import Shape

# Driver function
def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Detect the shapes in image")
    parser.add_argument('-i', help="Process image at the specified path", required=True)
    path = parser.parse_args().i


    # Open the image file
    image = cv2.imread(path)


    # Convert image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # Applying Gaussian blurring
    image_blur = cv2.GaussianBlur(image_gray, (7, 7), 0)


    # For white background
    # Thresholding : threshold value of 225 was obtained by trial and error
    image_thresh = cv2.threshold(image_blur, 225, 255, cv2.THRESH_BINARY_INV)[1]

    # Set of erosions and dilations to remove noise
    image_thresh = cv2.erode(image_thresh, None, iterations=4)
    image_thresh = cv2.dilate(image_thresh, None, iterations=4)

    # Find contours in the thresholded image
    contours, heirarchy = cv2.findContours(image_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Text color
    text_color = (0, 0, 0)

    # Contour color
    c_color = (210, 210, 210)


    # If there is only 1 contour, then background is black
    if len(contours) == 1:
        # Thresholding : threshold value of 80 was obtained by trial and error
        image_thresh = cv2.threshold(image_blur, 80, 255, cv2.THRESH_BINARY)[1]

        # Set of erosions and dilations to remove noise
        image_thresh = cv2.erode(image_thresh, None, iterations=4)
        image_thresh = cv2.dilate(image_thresh, None, iterations=4)

        # Find contours in the thresholded image
        contours, heirarchy = cv2.findContours(image_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Text color
        text_color = (210, 210, 210)
        
        # Contour color
        c_color = (255, 255, 255)


    # Make a copy of image for displaying output
    image_output = image.copy()


    # Create shape object
    shapeObj = Shape()


    # Draw contours and write the shape to the image
    for c in contours:
        # Draw contours on the image
        cv2.drawContours(image_output, [c], -1, c_color, 2)


        # Draw centers
        m = cv2.moments(c)
        pX = 0
        pY = 0
        if m["m00"] != 0:
            pX = int(m["m10"] / m["m00"])
            pY = int(m["m01"] / m["m00"])
            cv2.circle(image_output, (pX, pY), 3, (0, 0, 0), -1)


        # Get the shape of the contour
        contour_shape = shapeObj.getShape(contour=c)
    

        # Add the shape above center of contour
        cv2.putText(image_output, contour_shape, (pX - 20, pY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)


    # Display output image
    cv2.imshow("Shape Detection", image_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()
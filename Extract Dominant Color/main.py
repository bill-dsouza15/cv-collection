# Extract Dominant Color
# 
# This is the driver program where the image is read and passed to
# an object of class DominantColor for getting the list of dominant colors

import cv2
import argparse
import numpy as np
from dominant_color import DominantColor

# Driver function
def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Detect the shapes in image")
    parser.add_argument('-i', help="Process image at the specified path", required=True)
    parser.add_argument('-c', choices=['1','2','3','4','5','6','7','8'], help="Number of colors to be extracted", required=True)
    path = parser.parse_args().i
    cluster = int(parser.parse_args().c)


    # Open the image file
    image = cv2.imread(path)


    # Create DominantColor object
    dcObj = DominantColor(image, cluster)


    # Extract list of colors
    colors = dcObj.extractColor()


    # Setting 480 as the default height for a better output view
    if cluster > 6:
        height = 70*cluster
    elif cluster > 0 and cluster <= 6:
        height = 480

    
    # Output image
    image_output = np.zeros((height, 350, 3), dtype="uint8")


    # Display the list of extracted colors
    f = 0
    for c in colors:
        c = ( int (c[0]), int (c[1]), int (c[2]) )
        cv2.rectangle(image_output, (5, 5+f), (50, 50+f), color = (255, 255, 255), thickness=1)
        cv2.rectangle(image_output, (7, 7+f), (48, 48+f), color = c, thickness=-1)
        cv2.putText(image_output, str(c), (150, 31+f),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        f += 70

    
    # Concatenate original image and output image
    image = cv2.resize(image, (int(height * image.shape[1] / image.shape[0]), height))
    image_output = np.concatenate((image, image_output), axis=1)


    # Display output image
    cv2.imshow("Extract Dominant Color", image_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()
# Counting coins
# 
# In this program, I have counted the number of coins in the coin images 
# in the image folder.
# The process involves converting image to grayscale, apply thresholding,
# finding contours, drawing contours and displaying the count on the image.
# I have even found the center for each coin using the moments function
# and drawn them on the coins

import cv2
import numpy as np

# Driver function
def main():
    # Open the image file
    image = cv2.imread(r"Counting Coins/image/coins1.jpg")


    # Convert image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # Thresholding : threshold value of 80 was obtained by trial and error
    image_thresh = cv2.threshold(image_gray, 80, 255, cv2.THRESH_BINARY)[1]


    # Find contours in the thresholded image
    contours, heirarchy = cv2.findContours(image_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_contours = image.copy()

    for c in contours:
        # Draw contours on the image
        cv2.drawContours(image_contours, [c], -1, (90, 200, 190), 2)


        # Draw centers
        m = cv2.moments(c)
        if m["m00"] != 0:
            pX = int(m["m10"] / m["m00"])
            pY = int(m["m01"] / m["m00"])
            cv2.circle(image_contours, (pX, pY), 3, (0, 0, 255), -1)
  

    # Add count to topleft corner of image
    text = "{} coins".format(len(contours))
    cv2.putText(image_contours, text, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 200, 190), 2)


    # Create a copy of image
    image_output = image.copy()

    # Append image_gray to image_output
    image_output = np.concatenate((image_output, cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)), axis=1)

    # Append image_thresh to image_output
    image_output = np.concatenate((image_output, cv2.cvtColor(image_thresh, cv2.COLOR_GRAY2BGR)), axis=1)
    
    # Append image_contours to image_output
    image_output = np.concatenate((image_output, image_contours), axis=1)


    # Display image
    cv2.imshow("Counting Objects", image_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()

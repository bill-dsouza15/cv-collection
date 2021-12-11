# Understanding convolutions and kernels
#
# Kernels are small matrices that are used for different
# image processing operations like blurring, sharpening,
# edge detection, etc.
#
# In this program, I have used kernels corresponding
# to different operations such as sharpening, blurring, edge detection
# and have performed convolutions between the kernels and the image.

import cv2
import numpy as np
from skimage.exposure import rescale_intensity

# Perform convolution on image using kernel matrix (kernel is assumed to be a square matrix of odd dimensions)
def convolution(image, kernel):
    # Obtain dimensions of image and kernel
    (iheight, iwidth) = image.shape
    (kheight, kwidth) = kernel.shape


    # To retain original size of image after convolution
    # a padding has to be added to the image
    padding = (kheight - 1) // 2
    
    # Add padding to image
    image_padded = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)


    # Create an output matrix
    image_output = np.zeros((iheight, iwidth), dtype="float")


    # Perform the sliding of kernel over the image (convolution)
    for i in range(padding, iheight, 1):
        for j in range(padding, iwidth, 1):
            # Fetch the ROI for convolution
            image_roi = image_padded[i-padding:i+padding+1, j-padding:j+padding+1]


            # Perform convolution
            image_con = (image_roi * kernel).sum()


            # Add value to final array
            image_output[i, j] = image_con


    # Scale values to between 0 and 255
    image_output = rescale_intensity(image_output, in_range=(0, 255))
    image_output = (image_output * 255).astype("uint8")

    return image_output


# Driver function
def main():
    # Open the image file
    image = cv2.imread(r"Understanding Convolution & Kernels/image/pugs.jpg")


    # Convert image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # Create kernels matrix for different operations
    # Sharpen
    sharpen = np.array((
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]), dtype="int")


    # Smoothing
    boxblur = np.array((
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]), dtype="float")
    gaussianblur = np.array((
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16]), dtype="float")


    # Edge detection
    sobelx = np.array((
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]), dtype="int")
    sobely = np.array((
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]), dtype="int")
    laplacian = np.array((
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]), dtype="int")


    # Create a dictionary
    kerneldict = {"Sharpen": sharpen, "BoxBlur": boxblur, "GaussianBlur": gaussianblur,
                  "SobelX": sobelx, "SobelY": sobely, "Laplacian": laplacian}


    # Create copies of original image and grayscale image (converted)
    image1 = image.copy()
    image2 = image_gray.copy()
    image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)


    # Call convolute function for all kernel matrices
    for key, values in kerneldict.items():
        # Concatenate images
        image_output = np.concatenate((image1, image2), axis=1)


        # Call convolute function
        image3 = convolution(image_gray, values)
        image3 = cv2.cvtColor(image3, cv2.COLOR_GRAY2BGR)


        # Concatenate image3 to image_output
        image_output = np.concatenate((image_output, image3), axis=1)


        # Display image
        cv2.imshow(key, image_output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


main()

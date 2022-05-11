# Augmenting my data in order to use a more diverse dataset:
# This program will be run on the computer-generated mask dataset, because 
# these masks are all the exact same blue color. I will change some of them to 
# black, red, or a darker blue, in order to improve the quality of the CNN. 

import cv2 # OpenCV - image editing
import argparse # Argparse - testing with different images
import numpy as np # NumPy - arrays, etc. 

# Basic introductory code: ask for an image (to test the data editing on)
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # convert to HSV to edit color more easily
cv2.imshow("Original Image", image)
cv2.waitKey(0)

# Here, we can try to establish limits for which colors are edited by this program.
# This was done using the "mask swatch" image (should be included) and varying the
# average hue, which was determined to be hue 199.
# Note: in OpenCV, hue is done from 0 to 180, not 0 to 360. I used 188 and 206.
maskBlueStart = np.array([94, 0, 0]) # darkest mask color swatched from dataset
maskBlueEnd = np.array([107, 255, 255]) # lightest mask color swatched from dataset

maskOfMask = cv2.inRange(hsvImage, maskBlueStart, maskBlueEnd)

image[maskOfMask > 0] = (0, 0, 255) # highlight the selected region as red

cv2.imshow("Masked Image", image)
cv2.waitKey(0)
# Because I've decided to run the code focusing on two classes (correct mask or
# not) rather than three classes (correct, incorrect, nonexistent), I've tried
# to make up for the data imbalance by duplicating and augmenting the correct
# picutres.

# Imports: reading images in
import os
from os import listdir
# Imports: saving and modifying images
import cv2
# Imports: other methods
import random # used for random rotation

folder_dir = "mask-present-dataset/with_passing_mask"
slay = 0
for imageName in os.listdir(folder_dir):
	# Load the image in
	image = cv2.imread("mask-present-dataset/with_passing_mask/" + imageName)
	# TODO FIX THIS!!
	# Rotate the image slightly by a random amount
	rotationMatrix = cv2.getRotationMatrix2D((64, 64), random.randint(-45, 45), 1.0)
	image = cv2.warpAffine(image, rotationMatrix, (128, 128))
	# Zoom in on the image to cut out black space
	image = image[20:110, 20:110]
	# Resize image to 128 x 128 for the dataset
	image = cv2.resize(image, (128, 128), interpolation = cv2.INTER_AREA)
	cv2.imwrite('mask-present-dataset/with_passing_mask/augmented_' + imageName, image)
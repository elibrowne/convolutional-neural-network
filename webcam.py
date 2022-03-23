import cv2 # OpenCV is used for the webcam and face finding
import sys # information from the system
# Tensorflow/Keras imports
import tensorflow.keras.models as models 
from keras.preprocessing import image
import numpy as np

# Set up the webcam through OpenCV's built-in video capture method
camera = cv2.VideoCapture(0)

# Prepare our classifier to find faces
faceCascade = cv2.CascadeClassifier("faceFinder.xml")

# You can use this code to load any model saved on the system.
savedModel = models.load_model("weights-from-runs/mar22-1")
# Some datasets classify three (mask good, mask bad, no mask), some only test for mask good
TESTING_THREE_CLASSES = False

# Declare method for guessing if someone is wearing a mask (see test.py)
def testImage(imageData):
	try: 
		# The image be 128 x 128 x 3, which this method resolves.
		imageData = cv2.resize(imageData, (128, 128))
		# This is to resolve an issue with the tensor. The prediction requires
		# a batch size, so this adds a batch size of 1 to the start of the array.
		cv2.imshow("testing image", imageData)
		imageData = np.expand_dims(imageData, axis = 0)
		prediction = savedModel.predict(imageData)
		verdict = np.argmax(prediction, axis = 1) 
		# 3 cases: 0 for mask wrong, 1 for mask right, 2 for no mask
		# 2 cases: 0 for mask right, 1 for mask wrong
		return verdict
	except:
		print ("broke")
		return 3 # this is the "image capture broke" return

# This loop determines the camera's function.
while True:
	check, frame = camera.read() # capture the camera view for processing

	# We can look for faces using OpenCV's CascadeClassifier (initialized above)
	# The XML file for this classifier has been taken from the internet. 
	grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale( # parameters for detection
		grayscale,
		minNeighbors = 2, # trying to get faces with masks too (difficult)
		minSize = (200, 200), # trying to not detect small objects like eyes
		flags = cv2.CASCADE_SCALE_IMAGE,
	)

	""" Using this built-in face finder with a CascadeClassifier has been rough.
	The code does its best, but there is much to be desired, especially when the
	subject of the frame is wearing a mask (it is quite strong without a mask!). 
	I'd like to work on improving this, but for now, the somewhat poor face 
	detection could be an advantage: it 'cleans' the data a bit! Luckily, it is
	best at detecting bare faces, which is when one would want an alert to show 
	up anyways. """
	
	# For each face found, extract the image, test, and label accordingly
	for (x, y, w, h) in faces:
		# Crop out the image of the face (any dimension)
		croppedFace = frame[y:y+h, x:x+w]
		# Plug the data from the cropped face into the code
		wearingMask = testImage(croppedFace)
		print(wearingMask)
	
		color = (255, 0, 0) # blue (error) will be the default settings
		# THREE CLASSES (CORRECT, INCORRECT, NONE)
		if TESTING_THREE_CLASSES:
			if wearingMask == 1: # 1 corresponds to a poorly worn mask
				color = (0, 255, 255)
			elif wearingMask == 2: # 2 corresponds to a good mask
				color = (0, 255, 0)	
			elif wearingMask == 0: # 0 corresponds to no mask
				color = (0, 0, 255)
		# TWO CLASSES (CORRECT, INCORRECT/NONE)
		else:
			if wearingMask == 1: # flipping... are the classes wrong?
				color = (0, 255, 0)
			elif wearingMask == 0:
				color = (0, 0, 255)
		
		# Using the result from testing the image, color and draw the rectangle
		cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

	# Once the rectangles have been drawn, we can dispaly the output on a frame.
	cv2.imshow("Camera Output", frame) 

	# This block waits for the escape key to close the window
	key = cv2.waitKey(1)
	if key == 27:
		break

# End the program and destroy the windows once the loop is broke
camera.release()
cv2.destroyAllWindows()

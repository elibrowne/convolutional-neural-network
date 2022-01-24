import cv2 # OpenCV is used for the webcam and face finding
import sys # information from the system

# Set up the webcam through OpenCV's built-in video capture method
camera = cv2.VideoCapture(0)

# Prepare our classifier to find faces
faceCascade = cv2.CascadeClassifier("faceFinder.xml")

while True:
	check, frame = camera.read() # capture the camera view

	# We can look for faces using OpenCV's CascadeClassifier (initialized above)
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
	detection could be an advantage: it 'cleans' the data a bit! """

	# Draw rectangles around the found faces (TODO use color to determine mask accuracy)
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

	# Once the rectangles have been drawn, we can dispaly the output on a frame.
	cv2.imshow("Camera Output", frame) 

	# This block waits for the escape key to close the window
	key = cv2.waitKey(1)
	if key == 27:
		break


camera.release()
cv2.destroyAllWindows()

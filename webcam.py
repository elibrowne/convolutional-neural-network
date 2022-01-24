import cv2

camera = cv2.VideoCapture(0)

while True:
	check, frame = camera.read()
	cv2.imshow("Webcam", frame)

	key = cv2.waitKey(1)
	if key == 27:
		break


camera.release()
cv2.destroyAllWindows()

import tensorflow.keras.models as models 
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.callbacks as cb
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.utils as utils
from keras.preprocessing import image
import numpy as np

savedModel = models.load_model("weights-from-runs")

def testImage(pathToImage):
	# Load the image from the path using Keras
	imageToTest = image.load_img(pathToImage, target_size = (128, 128))
	imageToTest = image.img_to_array(imageToTest) # convert to array
	# This is to resolve an issue with the tensor. The prediction requires
	# a batch size, so this adds a batch size of 1 to the start of the array.
	imageToTest = np.expand_dims(imageToTest, axis = 0)
	prediction = savedModel.predict(imageToTest)
	print("Numeric prediction values for " + pathToImage + ":")
	print(prediction)
	print("Verdict: ")
	print(np.argmax(prediction, axis = 1))

# Load in test images (outside dataset) using Keras preprocessing
testImage('test-images/eli_w_mask.jpg')
testImage('test-images/eli_w_bad_mask.jpg')
testImage('test-images/kasie_w_mask.jpg')
testImage('test-images/jasper_w_mask.jpg')
testImage('test-images/sasha_w_bad_mask.jpg')
testImage('test-images/sasha_w_mask.jpg')
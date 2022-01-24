'''
CONVOLUTIONAL NEURAL NETWORK NOTES
(I took these notes during a lecture in this file and have left them here for
reference.)
Input: 250 x 250 x 3 (example)

Layer 1: Convolution 2D
 - Frame: 13 x 13
 - Step size: 3
 - No padding
 - Depth: 8
The first convolution in a CNN is looking for features.

Layer 2: Max Pool 2D
 - 2x2 frame and a step size of 2
 - Result: 40 x 40 x 8
The goal of a MaxPool is twofold: contracting the data and highlighting the most
prominent features of it.

Layer 3: Convolution 2D
 - Frame: 3 x 3
 - Step size: 1
 - Depth: 16
 - Result: 38 x 38 x 16
Convolutions two and up look for combinations of different features.

Layer 4: Max Pool 2D
 - 2x2 frame and 2 step size
 - Result: 19 x 19 x 16
Similar to the first MaxPool, this one focuses on highlighting the most prominent
combinations of features while further condensing the data.

Layer 5: Flatten 1D
 - 5716 size
This facilitates the transition from two dimensions to one.

Layers 6, 7, 8, 9...: Densify (1024 -> 256 -> 64 -> 5)
Densification makes connections among information from above layers that are 
physically (in terms of location) separated from each other. 
'''

# Always use Tensorflow's Keras (and not Keras by itself)
import tensorflow.keras.models as models 
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.callbacks as cb
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.utils as utils
from keras.preprocessing import image
import numpy as np

class Net():
	def __init__(self, image_size):
		self.model = models.Sequential() 
		# Conv2D (output depth, frame size, kwargs)
		self.model.add(layers.Conv2D(16, 11, strides = 3, 
			input_shape = image_size, activation = 'relu'))
		# Output of first layer - 80 x 80 x 8
		
		# MaxPool2D (frame size) - default stride is frame size
		self.model.add(layers.MaxPool2D(pool_size = 2))
		# Output of second layer - 40 x 40 x 8

		# Conv2D (output depth, frame size) - default stride is 1
		self.model.add(layers.Conv2D(24, 3, activation = 'relu'))
		# Output of third layer - 38 x 38 x 16

		# MaxPool2D (frame size)
		# self.model.add(layers.MaxPool2D(pool_size = 2))
		# Output of fourth layer - 19 x 19 x 16

		# layers.Flatten() - no arguments needed to flatten the layers!
		self.model.add(layers.Flatten())
		# Output of fifth layer - 5776 length of items in a 1D shape

		# Dense (layer size) is pretty obvious from here for the next layers
		# They require an activation function (the same as Conv2D)
		self.model.add(layers.Dense(1024, activation = 'relu'))
		self.model.add(layers.Dense(256, activation = 'relu'))
		self.model.add(layers.Dense(64, activation = 'relu'))

		# 'softmax' rescales the values to make a list of probabilities
		# The values are all positive and add to one!
		self.model.add(layers.Dense(3, activation = 'softmax'))

		# Calculating loss
		self.loss = losses.MeanSquaredError()
		# Another interesting model is CategoricalCrossentropy() per Dr. J

		# Optimizing with stochastic gradient descent (needs a learning rate)
		self.optimizer = optimizers.SGD(lr = 0.0001) # 0.001 for CCE

		# Compile model!
		self.model.compile(
			loss = self.loss, # loss function used to determine algorithm's success
			optimizer = self.optimizer,
			metrics = ['accuracy'] # not using accuracy: outputting accuracy @ each epoch
		)

	def __str__(self):
		# Prints when the object itself is printed
		self.model.summary()
		return "" # has to return an empty string even if it doesn't print

# Create our training data (notes included on omitted parameters)
train = utils.image_dataset_from_directory(
	'masks', # directory
	# labels are inferred as the names of the folders (omit)
	label_mode = 'categorical', # one-hot encoding 
	# class names are skipped because labeling is inferred
	# colormode is skipped because photos are already in RGB
	# batch_size is skipped because 32 is acceptable
	image_size = (128, 128), # the size of images isn't 256 x 256
	shuffle = True, # shuffling the images is based on the seed
	seed = 8, 
	validation_split = 0.3, # 30% of data is validation, 70% is training
	subset = 'training', # for git purposes :( 
	# interpolation isn't used because we aren't resizing any images
	# no symbolic links so, we don't need to follow links
)

# The test data is formed using the same parameters, but it's a 'validation' subset
test = utils.image_dataset_from_directory(
	'masks', # directory
	label_mode = 'categorical', # one-hot encoding 
	image_size = (128, 128), # the size of images isn't 256 x 256
	shuffle = True, # shuffling the images is based on the seed
	seed = 8, 
	validation_split = 0.3, # 30% of data is validation, 70% is training
	subset = 'validation', # for git purposes :( 
)

# Create a model
net = Net((128, 128, 3)) # size is 128 x 128 x 3: starting size for our net
print(net)

# Train the model
net.model.fit(
	train,
	batch_size = 32,
	epochs = 100,
	verbose = 2, # 2 = one line per epoch, 1 = progress bar, 0 = silent
	validation_data = test,
	validation_batch_size = 32,
	callbacks = cb.EarlyStopping(monitor = 'val_loss', patience = 2, restore_best_weights = True) # end the model when it's good enough
)

def testImage(pathToImage):
	# Load the image from the path using Keras
	imageToTest = image.load_img(pathToImage, target_size = (128, 128))
	imageToTest = image.img_to_array(imageToTest) # convert to array
	# This is to resolve an issue with the tensor. The prediction requires
	# a batch size, so this adds a batch size of 1 to the start of the array.
	imageToTest = np.expand_dims(imageToTest, axis = 0)
	prediction = net.model.predict(imageToTest)
	print("Numeric prediction values for " + pathToImage + ":")
	print(prediction)
	print("Verdict: ")
	print(np.argmax(prediction, axis = 1))

net.model.save("weights-from-runs/4") # make a new folder for each new saved run

# Load in test images (outside dataset) using Keras preprocessing
testImage('test-images/eli_w_mask.jpg')
testImage('test-images/eli_w_bad_mask.jpg')
testImage('test-images/kasie_w_mask.jpg')
testImage('test-images/jasper_w_mask.jpg')
testImage('test-images/sasha_w_bad_mask.jpg')
testImage('test-images/sasha_w_mask.jpg')
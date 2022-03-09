# Always use Tensorflow's Keras (and not Keras by itself)
import tensorflow.keras.models as models 
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.callbacks as cb
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.utils as utils
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# This code was added to prevent any issues with the remote desktop.
phys = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(
    phys[0],
    [
            tf.config.LogicalDeviceConfiguration(memory_limit=2048),
    ]
)

class Net():
	def __init__(self, image_size):
		self.model = models.Sequential() 
		# Conv2D (output depth, frame size, kwargs)
		self.model.add(layers.Conv2D(16, 11, strides = 3, 
			input_shape = image_size))
		# Add LeakyReLU activation to the layer afterwise
		self.model.add(layers.LeakyReLU(alpha=0.1))
		# Output of first layer - 80 x 80 x 8
		self.model.add(layers.Dropout(0.2))
		
		# MaxPool2D (frame size) - default stride is frame size
		self.model.add(layers.MaxPool2D(pool_size = 2))
		# Output of second layer - 40 x 40 x 8
		self.model.add(layers.BatchNormalization()) # normalize batches after MaxPool

		# Conv2D (output depth, frame size) - default stride is 1
		self.model.add(layers.Conv2D(24, 3, strides = 1))
		self.model.add(layers.LeakyReLU(alpha=0.1))
		# Output of third layer - 38 x 38 x 16
		self.model.add(layers.Dropout(0.2))

		# MaxPool2D (frame size)
		self.model.add(layers.MaxPool2D(pool_size = 2))
		# Output of fourth layer - 19 x 19 x 16
		self.model.add(layers.BatchNormalization()) # normalize again

		# layers.Flatten() - no arguments needed to flatten the layers!
		self.model.add(layers.Flatten())
		# Output of fifth layer - 5776 length of items in a 1D shape

		# Dense (layer size) is pretty obvious from here for the next layers
		# They require an activation function (the same as Conv2D)
		self.model.add(layers.Dense(1024))
		self.model.add(layers.LeakyReLU(alpha=0.1))
		self.model.add(layers.Dense(256))
		self.model.add(layers.LeakyReLU(alpha=0.1))
		self.model.add(layers.Dense(64))
		self.model.add(layers.LeakyReLU(alpha=0.1))

		# batch normalization and dropout (small early/large dense layers)

		# 'softmax' rescales the values to make a list of probabilities
		# The values are all positive and add to one!
		self.model.add(layers.Dense(2, activation = 'softmax'))

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
	'mask-present-dataset', # directory
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

# This code works on data augmentation: I've noticed that my dataset would result
# in different things based on lighting, which I aim to change here. 

# Because I've decided to run the code focusing on two classes (correct mask or
# not) rather than three classes (correct, incorrect, nonexistent), I've tried
# to make up for the data imbalance. 

duplicatedPassingImagesTrain = utils.image_dataset_from_directory(
	'mask-present-imgs',
	label_mode = 'categorical',
	image_size = (128, 128),
	shuffle = True,
	seed = 18, # different seed? I don't think it matters
	validation_split = 0.3, # same ratio as above
	subset = 'training'
)

# Do some changes on the duplicated images so they look different.
augmentDuplicatedData = models.Sequential([
	layers.RandomRotation(0.17, input_shape = (128, 128, 3))
])

# Change to class one (correctly worn mask) b/c it's at 0 right now
addToTrainDataset = duplicatedPassingImagesTrain.map(lambda x, y: (augmentDuplicatedData(x), y))
train = train.concatenate(addToTrainDataset)

# Here, we augment all of the data, including the duplicated images.
changeContrast = models.Sequential([
	# You can use layers to change contrast but not brightness.
    layers.RandomContrast(0.2, input_shape = (128, 128, 3))
])

# FROM DR. J'S DATA AUGMENTATION
# train.map() applies the transformation in parentheses to each pair x,y 
# in the dataset.  We only need to transform the x-values, we just pass
# the y-values along passively.  Notice that the output of the lambda
# function is a 2-tuple, which is the transformed image followed
modifiedImages = train.map(lambda x, y: (changeContrast(x), y))
# Here, I change brightness (I think this is how one does that?)
modifiedImages = train.map(lambda x, y: (tf.image.random_brightness(x, 0.3), y))
train = train.concatenate(modifiedImages)

# The test data is formed using the same parameters, but it's a 'validation' subset
test = utils.image_dataset_from_directory(
	'mask-present-dataset', # directory
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
history = net.model.fit(
	train,
	batch_size = 32,
	epochs = 4000, # do a lot! no worries overnight
	verbose = 1, # 2 = one line per epoch, 1 = progress bar, 0 = silent
	validation_data = test,
	validation_batch_size = 32,
	callbacks = cb.ModelCheckpoint(filepath = "weights-from-runs/mar9-1", verbose = 1, save_only_best_model = True)
)
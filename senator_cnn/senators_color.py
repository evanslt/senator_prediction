import numpy as np
import sys
import os
import sklearn
from sklearn import metrics

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from PIL import Image

def compile_training_set(files, input_shape):
	num_files = len(files)
	X = np.zeros((num_files,) + input_shape)
	Y = np.zeros(num_files)

	for idx, f in enumerate(files):
		image = np.array(Image.open(f))
		if image.shape[0] != input_shape[0] and image.shape[1] != input_shape[1]:
			new_image = np.zeros(input_shape)
			new_image[:image.shape[0], :image.shape[1], :] = image
			image = new_image
		if image.shape[0] != input_shape[0]:
			new_image = np.zeros(input_shape)
			new_image[:image.shape[0], :, :] = image
			image = new_image
		elif image.shape[1] != input_shape[1]:
			new_image = np.zeros(input_shape)
			new_image[:, :image.shape[1], :] = image
			image = new_image
		# normalize image
		image = (image - np.mean(image, axis = (0, 1), keepdims = True))/(np.var(image, axis = (0, 1), keepdims = True))
		if image.shape == (250, 250, 3):
			X[idx, ...] = image
			Y[idx, ...] = float('Winner' in f)
		else:
			print(image.shape)
	return X, Y

def cnn_model_v2(image_shape, metrics=['accuracy']):
	model = Sequential()
	model.add(Conv2D(filters = 32, kernel_size = (3, 3),
		activation = 'relu', input_shape = image_shape,
		name = 'conv-1'))
	model.add(MaxPooling2D(pool_size = (2,2), name = 'maxpool-1'))
	model.add(Dropout(0.5, name = 'dropout-1'))
	model.add(Conv2D(filters = 8, kernel_size = (2, 2),
		activation = 'relu', input_shape = image_shape,
		name = 'conv-2'))
	model.add(MaxPooling2D(pool_size = (2,2), name = 'maxpool-2'))
	model.add(Dropout(0.5, name = 'dropout-2'))
	model.add(Flatten())
	model.add(Dense(64, activation = 'relu', name = 'dense-1'))
	model.add(Dropout(0.5, name = 'dropout-3'))
	model.add(Dense(1, activation = 'sigmoid', name = 'dense-2'))
	model.compile(loss='binary_crossentropy',
				optimizer='adam',
				metrics = metrics)
	return model

def load_filenames(option = 'original_train', rootdir = 'color'):
	'''
	options:
		original_train:
		all_train:
		original_test:
		all_test 
	'''
	filenames = []

	# iterate through the images in the directory
	for root, subFolders, files in os.walk(rootdir):
		for f in files:
			if option == 'original_train':
				if ('2016' not in f) and ('block' not in f) and ('shift' not in f) and ('blur' not in f) and ('rotate' not in f) and ('noise' not in f):
					filenames.append(rootdir + '/' + f)
			elif option == 'all_train':
				if '2016' not in f and ('rotate' not in f):
					filenames.append(rootdir + '/' + f)
			elif option == 'original_test':
				if ('2016' in f) and ('block' not in f) and ('shift' not in f) and ('blur' not in f) and ('rotate' not in f) and ('noise' not in f):
					filenames.append(rootdir + '/' + f)
			elif option == 'all_test':
				if '2016' in f:
					filenames.append(rootdir + '/' + f)
	return filenames

rootdir = 'color'
# load training files
filenames = load_filenames('all_train', rootdir)
X, Y = compile_training_set(filenames, (250, 250, 3))
# set up model
model = cnn_model_v2((250, 250, 3))
# fit model
model.fit(X, Y, batch_size = 64, epochs = 10, verbose = 1, validation_split = 0.2)
#print("Training set loss and accuracy: ", model.test_on_batch(X, Y))
# load test set
filenames = load_filenames('original_test', 'color_test')
X_test, Y_test = compile_training_set(filenames, (250, 250, 3))
# accuracy and predictions
print("Test set loss and accuracy: ", model.test_on_batch(X_test, Y_test))
print("Predictions on test: ", model.predict(X_test))
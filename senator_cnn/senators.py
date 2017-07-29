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

def fit_generator(files, batch_size, input_shape, epoch_sizes = []):
	X = np.zeros((batch_size,) + input_shape)
	Y = np.zeros(batch_size)

	while True:
		# randomize the images
		np.random.shuffle(files)
		total_cnt = 0
		batch_cnt = 0
		for f in files:
			# add an image to our current batch
			image = np.array(Image.open(f))
			# normalize image
			image = (image - np.mean(image))/np.var(image)
			X[batch_cnt, ...] = np.expand_dims(image, 2) # expand the dimension since keras expects a color channel
			Y[batch_cnt, ...] = float('Winner' in f)
			batch_cnt += 1

			# if we've completed a batch, send it to keras
			if batch_cnt >= batch_size:
				# normalize batch
				yield X, Y
				total_cnt += batch_cnt
				batch_cnt = 0

		# return an incomplete batch
		if batch_cnt > 0:
			yield X[:batch_cnt, ...], Y[:batch_cnt, ...]
			total_cnt += batch_cnt
		epoch_sizes.append(total_cnt)

def compile_training_set(files, input_shape):
	num_train = len(files)
	X = np.zeros((num_train,) + input_shape)
	Y = np.zeros(num_train)

	for idx, f in enumerate(files):
		image = np.array(Image.open(f))
		# normalize image
		image = (image - np.mean(image))/np.var(image)
		X[idx, ...] = np.expand_dims(image, 2) # expand the dimension since keras expects a color channel
		Y[idx, ...] = float('Winner' in f)

	return X, Y


def cnn_model_v1(image_shape, metrics=['accuracy']):
	# we'll first try a simple CNN with only one convolutional layer, one max pooling, and one dense layer
	# the worry is that we'll overfit with too many parameters, since our dataset is so small
	model = Sequential()
	model.add(Conv2D(filters = 16, kernel_size = (2, 2),
		activation = 'relu', input_shape = image_shape,
		name = 'conv-1'))
	model.add(MaxPooling2D(pool_size = (2,2), name = 'maxpool-1'))
	model.add(Dropout(0.5, name = 'dropout-1'))
	model.add(Flatten())
	model.add(Dense(128, activation = 'relu', name = 'dense-1'))
	model.add(Dropout(0.5, name = 'dropout-2'))
	model.add(Dense(1, activation = 'sigmoid', name = 'dense-2'))
	model.compile(loss='binary_crossentropy',
			optimizer='adam',
			metrics = metrics)
	return model

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

def cnn_model_v3(image_shape, metrics=['accuracy']):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=image_shape, name = "conv1"))
    model.add(Conv2D(64, (3, 3), activation='relu', name = "conv2"))
    model.add(MaxPooling2D(pool_size=(2, 2), name = "maxpool1"))
    model.add(Dropout(0.25, name = "dropout1"))
    model.add(Flatten(name = "flatten1"))
    model.add(Dense(128, activation='relu', name = "dense1"))
    model.add(Dropout(0.5, name = "dropout2"))
    model.add(Dense(1, activation='sigmoid', name = "softmax1"))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics = metrics)
    return model

def feedforward_model(image_shape, metrics=['accuracy']):
    model = Sequential()
    model.add(Dense(256, activation = 'relu', name = "dense1", input_shape = image_shape))
    model.add(Dropout(0.5, name = "dropout1"))
    model.add(Dense(128, activation = 'relu', name = "dense2"))
    model.add(Dropout(0.5, name = "dropout2"))
    model.add(Dense(64, activation = 'relu', name = "dense3"))
    model.add(Dropout(0.5, name = "dropout3"))
    model.add(Flatten(name = "flatten1"))
    model.add(Dense(1, activation='sigmoid', name = "softmax1"))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics = metrics)
    return model

def load_filenames(option = 'black_white_all', rootdir = 'Images/Senators'):
	'''
	options:
		black_white_all: all black and white photos, including the distorted ones; but not the blacked out images
		black_white_block: black and white photos with the face block
		black_white_clean: black and white photos with no distortions
		color: color images, TODO: make this more specific
	'''
	filenames = []

	# iterate through the images in the directory
	for root, subFolders, files in os.walk(rootdir):
		for f in files:
			if option == 'black_white_all':
				if f[-3:] == 'bmp' and ('block' not in f):
					filenames.append(rootdir + '/' + f)
			elif option == 'black_white_block':
				if f[-3:] == 'bmp' and ('block' in f):
					filenames.append(rootdir + '/' + f)
			elif option == 'black_white_clean':
				if f[-3:] == 'bmp' and ('block' not in f) and ('shift' not in f) and ('blur' not in f) and ('rotate' not in f) and ('noise' not in f):
					filenames.append(rootdir + '/' + f)
			elif option == 'color':
				if f[-3:] == 'jpg':
					filenames.append(rootdir + '/' + f)
	return filenames

### training script ###
# load filenames
rootdir = 'Images/Senators'
filenames = load_filenames('black_white_all', rootdir)
# generate training set
X, Y = compile_training_set(filenames, (130, 100, 1))
# create model
model = cnn_model_v2((130, 100, 1))
# fit the model
model.fit(X, Y, batch_size = 32, epochs = 20, verbose = 1, validation_split = 0.2)
# accuracy on full training set
print("Training set loss and accuracy: ", model.test_on_batch(X, Y))
# test model on block images
filenames = load_filenames('black_white_block', rootdir)
X_block, Y_block = compile_training_set(filenames, (130, 100, 1))
print("Accuracy on block data: ", model.test_on_batch(X_block, Y_block))

### fit generator code, probably don't need for senators stuff

#fg = fit_generator(filenames, 16, (130, 100, 1))

#history = model.fit_generator(fg
#                              , steps_per_epoch=18
#                              , epochs=20)

# calculate accuracy on training set
#X, Y = compile_training_set(filenames, (130, 100, 1))
#model.fit(X, Y, batch_size = 32, epochs = 30, verbose = 1, validation_split = 0.4)
#prediction_probabilities = model.predict(X)[:,0]
#print(prediction_probabilities)
#print(Y)
#model.save("senator_model.h5")

#generate confusion matrix for predictions on training data
#print(sklearn.metrics.confusion_matrix(Y, prediction_probabilities > .5))

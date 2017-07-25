import numpy as np
import sys
import os

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
			X[batch_cnt, ...] = np.expand_dims(np.array(Image.open(f)), 2) # expand the dimension since keras expects a color channel
			Y[batch_cnt, ...] = float('Winner' in f)
			batch_cnt += 1

			# if we've completed a batch, send it to keras
			if batch_cnt >= batch_size:
				yield X, Y
				total_cnt += batch_cnt
				batch_cnt = 0
		
		# return an incomplete batch
		if batch_cnt > 0:
			yield X[:batch_cnt, ...], Y[:batch_cnt, ...]
			total_cnt += batch_cnt
		epoch_sizes.append(total_cnt)

def cnn_model_v1(image_shape):
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
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

filenames = []
rootdir = 'Images/Senators'

# iterate through the images in the directory
for root, subFolders, files in os.walk(rootdir):
	for file in files:
		# only use the normalized images
		if file[-3:] == 'bmp':
			filenames.append(rootdir + '/' + file)

# generate the model
model = cnn_model_v1((130, 100, 1))

fg = fit_generator(filenames, 16, (130, 100, 1))

history = model.fit_generator(fg
                              , steps_per_epoch=17
                              , epochs=10)

model.save("senator_model.h5")




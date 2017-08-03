from keras_vggface.vggface import VGGFace
from keras.layers import Dense
from keras.layers import Flatten
from keras.engine import Model
import numpy as np
import sys
import os
from PIL import Image

def compile_images(files, input_shape):
	num_files = len(files)
	X = np.zeros((num_files,) + input_shape)
	Y = np.zeros(num_files)

	for idx, f in enumerate(files):
		image = np.array(Image.open(f))
		# handles images that aren't 250 x 250; might not be necessary anymore
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
		X[idx, ...] = image
		Y[idx, ...] = float('Winner' in f)
	return X, Y

def load_filenames(option = 'original_train', rootdir = 'color'):
	'''
	options:
		original_train: the original dataset without augmentation
		all_train: augmented dataset
		original_test: test set without augmentation
		all_test: augmented test set 
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

# grab and modify the VGG model
vgg_model = VGGFace(include_top=False, input_shape=(250,250,3))
last_layer = vgg_model.get_layer('pool5').output
flattened =  Flatten(name = 'flatten')(last_layer)
output = Dense(1, activation = 'sigmoid', name='sigmoid-1')(flattened)
mod_vgg = Model(vgg_model.input, output)
num_layers = len(mod_vgg.layers)
for layer in range(num_layers - 1):
	mod_vgg.layers[layer].trainable = False
mod_vgg.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# get training data
train_rootdir = 'color'
test_rootdir = 'color_test'
filenames = load_filenames('all_train', train_rootdir)
X, Y = compile_images(filenames, (250, 250, 3))
# fit updated model on senators
mod_vgg.fit(X, Y, batch_size = 8, epochs = 5, verbose = 1, validation_split = 0.2)
# test
filenames_test = load_filenames('original_test', test_rootdir)
X_test, Y_test = compile_images(filenames_test, (250, 250, 3))
print("Test results: ", mod_vgg.test_on_batch(X_test, Y_test))


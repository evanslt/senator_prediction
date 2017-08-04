
#trying to transfer learn from mdoel 120
#comand to run on FH: floyd run --env keras --message "Transfer Learning w model 120" --data psachdeva/datasets/senators/2:test --data psachdeva/datasets/senators/1:training  --data arielle/projects/cdips-test/120/output:out --gpu "python transfer_120.py"

from keras.models import load_model
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
			print(f)
			if f == '.floyddata':
				continue
			if f == '.DS_Store':
				continue
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
faces_model = load_model('/out/face_model_5120_b64_e50_s80-reorder-v0.h5')

last_layer=faces_model.get_layer('maxpool1').output
flattened =  Flatten(name = 'flatten')(last_layer)
output = Dense(1, activation = 'sigmoid', name='sigmoid-1')(flattened)

final_model = Model(faces_model.input, output)
num_layers = len(final_model.layers)

for layer in range(num_layers - 1):
	final_model.layers[layer].trainable = False

final_model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# get training data

rootdir_train='/training'
filenames = load_filenames('all_train', rootdir_train)
#print(filenames)

X, Y = compile_images(filenames, (250, 250, 3))
# fit updated model on senators
final_model.fit(X, Y, batch_size = 8, epochs = 10, verbose = 1, validation_split = 0.2)
# test


rootdir_test='/test'
filenames_test = load_filenames('all_test', rootdir_test)
X_test, Y_test = compile_images(filenames_test, (250, 250, 3))


print("Test results: ", final_model.test_on_batch(X_test, Y_test))

predictions=final_model.predict(X_test)

for i,f in enumerate(filnames_test):
	print(f, " with truth value: ", Y_test[i], predictions[i])

final_model.save("/output/transfer_modle120.h5")


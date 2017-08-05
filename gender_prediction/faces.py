# For reference: https://github.com/karthikv2k/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb

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
from keras.callbacks import EarlyStopping
import tensorflow as tf
import pickle
import random
import cv2

def fit_generator(files, batch_size, input_shape, trainflag, epoch_sizes=[]):

    X = np.zeros((batch_size,) + input_shape)
    Y = np.zeros(batch_size)

    while True:
        np.random.shuffle(files)
        total_cnt = 0
        batch_cnt = 0
        for f in files:
            image = cv2.imread(f)
            # image = np.array(Image.open(f))
            image = (image - np.mean(image, axis=(0, 1), keepdims=True)) / (np.var(image, axis=(0, 1), keepdims=True))
            # image = (image - np.mean(image)) / np.var(image) # normalizing/standardizing
            #
            flip = random.randint(0, 1)
            if flip == 0 & trainflag:
                image = add_noise(image)  # randomly augment half-ish of the data

            X[batch_cnt, ...] = image
            Y[batch_cnt, ...] = float('female' in f)
            batch_cnt += 1

            if batch_cnt >= batch_size:
                yield X, Y
                total_cnt += batch_cnt
                batch_cnt = 0

        if batch_cnt > 0:
            yield X[:batch_cnt, ...], Y[:batch_cnt, ...]
            total_cnt += batch_cnt
        epoch_sizes.append(total_cnt)

        #if print_stats:
        #    print("Total samples cnt: {}, execution time: {}".format(total_cnt, time.time() - start_ts))

def add_noise(X, noise_ratio = 0.01):
    X_rnd = X + np.random.random(X.shape)*noise_ratio
    X_rnd[X_rnd>1] = 1
    return X_rnd

def rotate(X, rnd_rotation=True, degree=None, range=[-30,30]):
    X_rotated = np.zeros(X.shape)
    n = X_rotated.shape[0]
    rows,cols = X.shape[1:3]
    if rnd_rotation:
        deg = range[0] + (np.random.rand(n)*(range[1]-range[0]))
    else:
        deg = np.repeat(degree, n)
    if X_rotated.shape[-1]==1: #grayscale needs special treatment
        for i,d in enumerate(deg):
            M = cv2.getRotationMatrix2D((cols/2,rows/2),d,1)
            X_rotated[i,...] = cv2.warpAffine(X[i,...],M,(cols,rows)).reshape(rows,cols,1)
    else:
        for i,d in enumerate(deg):
            M = cv2.getRotationMatrix2D((cols/2,rows/2),d,1)
            X_rotated[i,...] = cv2.warpAffine(X[i,...],M,(cols,rows))
    return X_rotated

def basic_cnn_model_v0(image_shape, metrics=['accuracy']):
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
                  metrics=metrics)
    return model

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

filenames_train = [];
filenames_test = [];

cnt_male = 0;
cnt_female = 0;

rootdir = '/Images/male/'
for root, subFolders, files in os.walk(rootdir):
    for file in files:
        cnt_male += 1;
        if cnt_male <= 3072:
            filenames_train.append(rootdir + '/' + file)
        else:
            filenames_test.append(rootdir + '/' + file)

rootdir = '/Images/female/'
for root, subFolders, files in os.walk(rootdir):
    for file in files:
        cnt_female += 1;
        if cnt_female <= 1024:
            filenames_train.append(rootdir + '/' + file)
        else:
            filenames_test.append(rootdir + '/' + file)

# cnt_male 5280, cnt_female 1409, 6689 total

# early stopping, cover up parts of images, see how that affects score
# two pictures of same person, make sure scores are similar -- e.g. bernie sanders

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
callbacks=[early_stopping]

pickle.dump(filenames_test, open("/output/filenames_test.p", "wb"))
pickle.dump(filenames_train, open("/output/filenames_train.p", "wb"))

# model 0
model = basic_cnn_model_v0((250,250,3))
fg = fit_generator(filenames_train, 64, (250, 250, 3), trainflag=1)
fg_val = fit_generator(filenames_test, 64, (250, 250, 3), trainflag=0)
history = model.fit_generator(fg
                              , steps_per_epoch=64
                              , epochs=100, callbacks=callbacks, validation_data=fg_val, validation_steps=40)
model.save("/output/face_modelv0_somenoise.h5")

model1 = cnn_model_v1((250,250,3))
fg = fit_generator(filenames_train, 64, (250, 250, 3), trainflag=1)
fg_val = fit_generator(filenames_test, 64, (250, 250, 3), trainflag=0)
history1 = model1.fit_generator(fg
                              , steps_per_epoch=64
                              , epochs=100, callbacks=callbacks, validation_data=fg_val, validation_steps=40)
model1.save("/output/face_modelv1_somenoise.h5")

### notes
# model 0 with 100 epochs  1793/2593 69% vs. 1424/2593 55%, 14 diff (old data)
# model 1 with 100 epochs, 2060/2593 79% vs. 1671/2593 64%, 15 diff (old data)
# model 1 with 20 epochs, 1606/2593 (61.9%) vs. 1537/2593 (59.3%), 2.6 diff
# model 1 with 20 epochs, 1671/2593 (64%) vs. 1019/2593 (39%) adding some noise, 25 diff
# model 1 with 35 epochs, 1828/2593 (70%) vs. 1626/2593 (62.7%), 7.3 diff, adding some noise - run 104
# model 1 with 20 epochs, 1757/2593 (67.8%) vs. 1313/2593 (50%), 17.8 diff, no noise - run 105
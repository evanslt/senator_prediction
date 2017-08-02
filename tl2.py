## NEVERMIND TL WORKS, just slow

import numpy as np
import sys
import os

import numpy
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from PIL import Image
import tensorflow as tf
import pickle
import random
import cv2
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras import applications

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

pickle.dump(filenames_test, open("/output/filenames_test.p", "wb"))
pickle.dump(filenames_train, open("/output/filenames_train.p", "wb"))

# early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
# callbacks=[early_stopping]

## ABOVE IS IDENTICAL TO FACES

checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

# Building model
img_width, img_height = 250, 250
nb_train_samples = len(filenames_train)
nb_validation_samples = len(filenames_test)
batch_size = 64
epochs = 20

model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
for layer in model.layers[:5]:
    layer.trainable = False

#Adding custom Layers
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(1, activation="softmax")(x)

# creating the final model
model_final = Model(input = model.input, output = predictions)

# compile the model
model_final.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
##

callbacks=[early, checkpoint]

fg = fit_generator(filenames_train, 64, (250, 250, 3), trainflag=1)
fg_val = fit_generator(filenames_test, 64, (250, 250, 3), trainflag=0)
history1 = model_final.fit_generator(fg
                              , steps_per_epoch=batch_size
                              , epochs=epochs, callbacks=callbacks, validation_data=fg_val, validation_steps=40)
model_final.save("/output/test_tf.h5")
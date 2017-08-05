from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import pickle
import numpy as np
import cv2
import random
import os
from keras_vggface.vggface import VGGFace
from keras.models import load_model
import numpy as np
import sys
import os
from PIL import Image

# Get Data
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

def compile_images(files, input_shape):
    num_train = len(files)
    X = np.zeros((num_train,) + input_shape)
    Y = np.zeros(num_train)
    X_block = np.zeros((num_train,) + input_shape)

    for idx, f in enumerate(files):
        image = cv2.imread(f)
        image = (image - np.mean(image, axis=(0, 1), keepdims=True)) / (np.var(image, axis=(0, 1), keepdims=True))
        X[idx, ...] = image # expand the dimension since keras expects a color channel
        Y[idx, ...] = float('female' in f)

        img_len, img_wid = image.shape[:2]
        image[int(0.15 * img_len):int(0.85 * img_len), int(0.15 * img_wid):int(0.85 * img_wid), :] = 0
        X_block[idx, ...] = image;

    return X, Y, X_block

# Fit Generator
def fit_generator(files, batch_size, input_shape, epoch_sizes=[]):
    X = np.zeros((batch_size,) + input_shape)
    Y = np.zeros(batch_size)

    while True:
        np.random.shuffle(files)
        total_cnt = 0
        batch_cnt = 0
        for f in files:
            image = cv2.imread(f)
            image = (image - np.mean(image, axis=(0, 1), keepdims=True)) / (np.var(image, axis=(0, 1), keepdims=True))

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

# Grab and modify the VGG model
vgg_model = VGGFace(include_top=False, input_shape=(250, 250, 3))
last_layer = vgg_model.get_layer('pool5').output
flattened = Flatten(name='flatten')(last_layer)
output = Dense(1, activation='sigmoid', name='sigmoid-1')(flattened)
mod_vgg = Model(vgg_model.input, output)
num_layers = len(mod_vgg.layers)

# Freeze layers
for layer in range(num_layers - 1):
    mod_vgg.layers[layer].trainable = False

# Compile Model
mod_vgg.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Get training data
X, Y, X_block = compile_images(filenames_train, (250,250,3))

# Fit updated model on senators
early_stopping = EarlyStopping(monitor='val_loss', patience=15)
fg = fit_generator(filenames_train, 64, (250, 250, 3))
fg_val = fit_generator(filenames_test, 64, (250, 250, 3))
history = mod_vgg.fit_generator(fg
                              , steps_per_epoch=64
                              , epochs=10, callbacks=[early_stopping], validation_data=fg_val, validation_steps=40)
mod_vgg.save("/output/vgg-faces-gender.h5")

# test
X_test, Y_test = compile_images(filenames_test, (250, 250, 3))
print("Test results: ", mod_vgg.test_on_batch(X_test, Y_test))

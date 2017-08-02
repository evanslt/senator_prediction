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

def add_noise(X, noise_ratio = 0.07):
    X_rnd = X + np.random.random(X.shape)*noise_ratio
    X_rnd[X_rnd>1] = 1
    return X_rnd

def compile_training_set(files, input_shape):
    num_train = len(files)
    X = np.zeros((num_train,) + input_shape)
    Y = np.zeros(num_train)

    for idx, f in enumerate(files):
        image = np.array(Image.open(f))
        # normalize image
        #image = (image - np.mean(image))/np.var(image)
        image = (image - np.mean(image, axis = (0, 1), keepdims = True))/(np.var(image, axis = (0, 1), keepdims = True))
        X[idx, ...] = image # expand the dimension since keras expects a color channel
        Y[idx, ...] = float('female' in f)

    return X, Y



def fit_generator(files, batch_size, input_shape, noise, 
    print_stats=True, 
                  one_epoch=False, epoch_sizes=[]):

    X = np.zeros((batch_size,) + input_shape)
    Y = np.zeros(batch_size)

    while True:
        np.random.shuffle(files)
        total_cnt = 0
        batch_cnt = 0
        for f in files:
            image = np.array(Image.open(f))
            #image = (image - np.mean(image)) / np.var(image) # normalizing/standardizing
            image = (image - np.mean(image, axis = (0, 1), keepdims = True))/(np.var(image, axis = (0, 1), keepdims = True))
            
            if noise == 1:
                image=add_noise(image)

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

        if one_epoch:
            break


def basic_cnn_model_v0(image_shape, metrics=['accuracy']):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=image_shape, name = "conv1"))
    model.add(Conv2D(64, (3, 3), activation='relu', name = "conv2"))

    model.add(MaxPooling2D(pool_size=(2, 2), name = "maxpool1"))
    #model.add(Conv2D(64, (3, 3), activation='relu', name = "conv3"))
    #model.add(MaxPooling2D(pool_size=(2, 2), name = "maxpool2"))
    model.add(Dropout(0.25, name = "dropout1"))
    model.add(Flatten(name = "flatten1"))
    model.add(Dense(128, activation='relu', name = "dense1"))
    model.add(Dropout(0.5, name = "dropout2"))
    model.add(Dense(1, activation='sigmoid', name = "softmax1"))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=metrics)
    return model

def basic_cnn_model_v1(image_shape, metrics=['accuracy']):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=image_shape, name = "conv1"))
    model.add(Conv2D(64, (3, 3), activation='relu', name = "conv2"))
    model.add(Dropout(0.25, name = "dropout1"))
    model.add(MaxPooling2D(pool_size=(2, 2), name = "maxpool1"))
    model.add(Dropout(0.25, name = "dropout2"))
    model.add(Flatten(name = "flatten1"))
    model.add(Dense(128, activation='relu', name = "dense1"))
    model.add(Dropout(0.5, name = "dropout3"))
    model.add(Dense(128, activation='relu', name = "dense2"))
    model.add(Dropout(0.5, name = "dropout4"))
    model.add(Dense(1, activation='sigmoid', name = "softmax1"))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=metrics)
    return model


def basic_cnn_model_v2(image_shape, metrics=['accuracy']):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=image_shape, name = "conv1"))
    model.add(Conv2D(64, (3, 3), activation='relu', name = "conv2"))
    model.add(MaxPooling2D(pool_size=(2, 2), name = "maxpool1"))
    model.add(Dropout(0.25, name = "dropout1"))
    model.add(Conv2D(64, (3, 3), activation='relu', name = "conv3"))
    model.add(MaxPooling2D(pool_size=(2, 2), name = "maxpool2"))
    model.add(Dropout(0.25, name = "dropout2"))
    model.add(Flatten(name = "flatten1"))
    model.add(Dense(128, activation='relu', name = "dense1"))
    model.add(Dropout(0.5, name = "dropout3"))
    model.add(Dense(128, activation='relu', name = "dense2"))
    model.add(Dropout(0.5, name = "dropout4"))
    model.add(Dense(n_classes, activation='softmax', name = "softmax1"))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=metrics)
    return model

filenames_train = [];
filenames_test = [];

cnt_male = 0;
cnt_female = 0;

rootdir = '/Images/male'
for root, subFolders, files in os.walk(rootdir):
    for file in files:
        cnt_male += 1;
        if cnt_male < 5150:
            filenames_train.append(rootdir + '/' + file)
        elif cnt_male<5155:
            filenames_test.append(rootdir + '/' + file)

rootdir = '/Images/female'
for root, subFolders, files in os.walk(rootdir):
    for file in files:
        cnt_female += 1;
        if cnt_female < 1250:
            filenames_train.append(rootdir + '/' + file)
        elif cnt_female<1360:
            filenames_test.append(rootdir + '/' + file)


# currently doing 208 total for training
# remember to transfer the test set to the predict script

# atch 64 steps_per_epoch 20, one will be missing 

X, Y = compile_training_set(filenames_test, (250, 250, 3))

model = basic_cnn_model_v0((250,250,3))


fg = fit_generator(filenames_train, 64, (250, 250, 3), noise=1)

history = model.fit_generator(fg
                              , steps_per_epoch=100
                              , epochs=50)

model.save("/output/face_model_6400_b64_e50-reorder-v0-noise.h5")

prediction_probabilities = model.predict(X)[:,0]
print(filenames_test)
print(prediction_probabilities)
print(Y)
#

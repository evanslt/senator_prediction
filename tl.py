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

## starting by getting vgg to transfer to male/female
## can use senator images as "legit" test set for the male/female

# get data
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

# Building model
img_width, img_height = 250, 250
nb_train_samples = len(filenames_train)
nb_validation_samples = len(filenames_test)

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

# # Initiate the train and test generators with data augumentation
# train_datagen = ImageDataGenerator(
# rescale = 1./255,
# horizontal_flip = True,
# fill_mode = "nearest",
# zoom_range = 0.3,
# width_shift_range = 0.3,
# height_shift_range=0.3,
# rotation_range=30)
#
# # Save the model according to the conditions
# checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
# early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
#
# def compile_set(files, input_shape):
#     num_train = len(files)
#     X = np.zeros((num_train,) + input_shape)
#     Y = np.zeros(num_train)
#     X_block = np.zeros((num_train,) + input_shape)
#
#     for idx, f in enumerate(files):
#         image = cv2.imread(f)
#         # image = np.array(Image.open(f))
#         image = (image - np.mean(image, axis=(0, 1), keepdims=True)) / (np.var(image, axis=(0, 1), keepdims=True))
#         X[idx, ...] = image # expand the dimension since keras expects a color channel
#         Y[idx, ...] = float('female' in f)
#
#     return X, Y
#
# x_train, y_train = compile_set(filenames_train, (250,250,3))
# x_test, y_test = compile_set(filenames_test, (250,250,3))
#
# train_generator = train_datagen.flow(x_train, y_train, batch_size=64)
# validation_generator = train_datagen.flow(x_test, y_test)
#
# # Train the model
# model_final.fit_generator(
# train_generator,
# samples_per_epoch = nb_train_samples,
# epochs = epochs,
# validation_data = validation_generator,
# nb_val_samples = nb_validation_samples,
# callbacks = [checkpoint, early])

##### try fit generator instead, having issues with flow function
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

def add_noise(X, noise_ratio=0.01):
    X_rnd = X + np.random.random(X.shape) * noise_ratio
    X_rnd[X_rnd > 1] = 1
    return X_rnd

# Save the model according to the conditions
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

# def compile_set(files, input_shape):
#     num_train = len(files)
#     X = np.zeros((num_train,) + input_shape)
#     Y = np.zeros(num_train)
#     X_block = np.zeros((num_train,) + input_shape)
#
#     for idx, f in enumerate(files):
#         image = cv2.imread(f)
#         image = (image - np.mean(image, axis=(0, 1), keepdims=True)) / (np.var(image, axis=(0, 1), keepdims=True))
#         X[idx, ...] = image # expand the dimension since keras expects a color channel
#         Y[idx, ...] = float('female' in f)
#
#     return X, Y
#
# train_generator = fit_generator(filenames_train, 64, (250, 250, 3), trainflag=1)
# validation_generator = fit_generator(filenames_test, 64, (250, 250, 3), trainflag=0)
#
# # Train the model
# model_final.fit_generator(
# train_generator,
# samples_per_epoch = nb_train_samples,
# epochs = epochs,
# validation_data = validation_generator,
# nb_val_samples = nb_validation_samples,
# callbacks = [checkpoint, early])

###
callbacks=[early, checkpoint]

fg = fit_generator(filenames_train, 64, (250, 250, 3), trainflag=1)
fg_val = fit_generator(filenames_test, 64, (250, 250, 3), trainflag=0)
history1 = model_final.fit_generator(fg
                              , steps_per_epoch=64
                              , epochs=20, callbacks=callbacks, validation_data=fg_val, validation_steps=40)
model_final.save("/output/test_tf.h5")
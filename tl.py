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

# get data
filenames_train = pickle.load(open("/out/filenames_train.p", "rb"))
filenames_test = pickle.load(open("/out/filenames_test.p", "rb"))

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
predictions = Dense(16, activation="softmax")(x)

# creating the final model
model_final = Model(input = model.input, output = predictions)

# compile the model
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

# Initiate the train and test generators with data Augumentation
train_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

# Save the model according to the conditions
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

def compile_set(files, input_shape):
    num_train = len(files)
    X = np.zeros((num_train,) + input_shape)
    Y = np.zeros(num_train)
    X_block = np.zeros((num_train,) + input_shape)

    for idx, f in enumerate(files):
        image = cv2.imread(f)
        # image = np.array(Image.open(f))
        image = (image - np.mean(image, axis=(0, 1), keepdims=True)) / (np.var(image, axis=(0, 1), keepdims=True))
        X[idx, ...] = image # expand the dimension since keras expects a color channel
        Y[idx, ...] = float('female' in f)

    return X, Y

x_train, y_train = compile_set(filenames_train, (250,250,3))
x_test, y_test = compile_set(filenames_test, (250,250,3))

train_generator = train_datagen.flow(x_train, y_train, batch_size=64)
validation_generator = train_datagen.flow(x_test, y_test)

# Train the model
model_final.fit_generator(
train_generator,
samples_per_epoch = nb_train_samples,
epochs = epochs,
validation_data = validation_generator,
nb_val_samples = nb_validation_samples,
callbacks = [checkpoint, early])


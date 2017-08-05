from keras_vggface.vggface import VGGFace
from keras.layers import Dense, Flatten, Dropout
from keras.engine import Model
from keras.callbacks import EarlyStopping
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
        image = (image - np.mean(image, axis=(0, 1), keepdims=True)) / (np.var(image, axis=(0, 1), keepdims=True))
        X[idx, ...] = image
        Y[idx, ...] = float('Winner' in f)
    return X, Y


def load_filenames(option='original_train', rootdir='color'):
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
            if f[0] != '.':
                if option == 'original_train':
                    if ('2016' not in f) and ('block' not in f) and ('shift' not in f) and ('blur' not in f) and (
                        'rotate' not in f) and ('noise' not in f):
                        filenames.append(rootdir + '/' + f)
                elif option == 'all_train':
                    if '2016' not in f and ('rotate' not in f):
                        filenames.append(rootdir + '/' + f)
                elif option == 'original_test':
                    if ('2016' in f) and ('block' not in f) and ('shift' not in f) and ('blur' not in f) and (
                        'rotate' not in f) and ('noise' not in f):
                        filenames.append(rootdir + '/' + f)
                elif option == 'all_test':
                    if '2016' in f:
                        filenames.append(rootdir + '/' + f)
    return filenames

# Adding custom layers to vgg-face
vggfaces = VGGFace(include_top=False, input_shape=(250, 250, 3))
x = vggfaces.get_layer('pool5').output
x = Flatten(name='flatten')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid', name='sigmoid-1')(x)

mod_vgg = Model(vggfaces.input, output)
num_layers = len(mod_vgg.layers)
vggfaces.summary()
mod_vgg.summary()
print("vgg_model lay", len(vggfaces.layers))
print("mod_vgg layers", len(mod_vgg.layers))

# Freeze layers
for layer in range(num_layers - 2):
    mod_vgg.layers[layer].trainable = False

mod_vgg.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# get training data
train_rootdir = '/color'
test_rootdir = '/color_test'
filenames = load_filenames('all_train', train_rootdir)
X, Y = compile_images(filenames, (250, 250, 3))

# fit updated model on senators
early_stopping = EarlyStopping(monitor='val_loss', patience=15)
mod_vgg.fit(X, Y, batch_size=64, epochs=75, verbose=1, validation_split=0.2, callbacks=[early_stopping])

# test
filenames_test = load_filenames('original_test', test_rootdir)
X_test, Y_test = compile_images(filenames_test, (250, 250, 3))
print("Test results: ", mod_vgg.test_on_batch(X_test, Y_test))
mod_vgg.save("/output/model1.h5")

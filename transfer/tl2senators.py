from keras.models import load_model
import numpy as np
import sys
import os
from PIL import Image
from keras.engine import Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
import pickle

def compile_training_set(files, input_shape):
    num_files = len(files)
    X = np.zeros((num_files,) + input_shape)
    Y = np.zeros(num_files)

    for idx, f in enumerate(files):
        image = np.array(Image.open(f))
        # handles images that aren't 250 x 250; might not be necessary anymore
        try:
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
        except:
            pass
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

# load model trained on faces

#custom parameters
nb_class = 1
hidden_dim = 512

# model = VGGFace(include_top=False, input_shape=(250, 250, 3))


filename = "face_modelv0_somenoise.h5"
import h5py
model = h5py.File(filename, 'r+')
del model['optimizer_weights']
model.close()

model.summary()
num_layers = len(model.layers)

num_layers = len(model.layers)
# freeze all layers except the classifier
for layer in range(num_layers - 1):
	model.layers[layer].trainable = False

# # Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
# for layer in model.layers[:5]:
#     layer.trainable = False
#
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

model_final.summary()

# gather senator training data
train_rootdir = 'color'
test_rootdir = 'color_test'
filenames_train = load_filenames('all_train', train_rootdir)
filenames_test = load_filenames('all_train', test_rootdir)
X, Y = compile_training_set(filenames_train, (250, 250, 3))

pickle.dump(filenames_test, open("filenames_test.p", "wb"))
pickle.dump(filenames_train, open("filenames_train.p", "wb"))

# pickle.dump(filenames_test, open("/output/filenames_test.p", "wb"))
# pickle.dump(filenames_train, open("/output/filenames_train.p", "wb"))

print(filenames_train)

# fit updated model on senators
model_final.fit(X, Y, batch_size=64, epochs=10, verbose=1, validation_split=0.2)

# model_final.save("/output/tfmodel.h5")
model_final.save("tfmodel.h5")
import numpy as np
from PIL import Image
from keras.models import load_model
import cv2

def compile_set(files, input_shape):
    num_train = len(files)
    X = np.zeros((num_train,) + input_shape)
    Y = np.zeros(num_train)
    X_block = np.zeros((num_train,) + input_shape)

    for idx, f in enumerate(files):
        image = cv2.imread(f)
        # image = np.array(Image.open(f))

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

        image = (image - np.mean(image, axis=(0, 1), keepdims=True)) / (np.var(image, axis=(0, 1), keepdims=True))
        X[idx, ...] = image # expand the dimension since keras expects a color channel
        Y[idx, ...] = float('female' in f)

    return X, Y

filename = "model1_ps.h5"
model = load_model(filename)

filenames_test = ["2014_SouthDakota_Winner_pad.jpg", "2014_WestVirginia_Winner_pad.jpg", "2014_WestVirginia_Loser_pad.jpg"]

X_test, Y_test =compile_set(filenames_test, (250, 250, 3))
print(model.predict(X_test)[:, 0])
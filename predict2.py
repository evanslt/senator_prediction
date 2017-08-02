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
from keras.models import load_model
import pickle
import cv2

model = load_model("/out/face_modelv1_somenoise.h5")
filenames_test = pickle.load(open("/out/filenames_test.p", "rb"))

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

        # print(image.shape) # 250 250 3
        img_len, img_wid = image.shape[:2]
        image[int(0.15 * img_len):int(0.85 * img_len), int(0.15 * img_wid):int(0.85 * img_wid), :] = 0
        X_block[idx, ...] = image;

    return X, Y, X_block

X, Y, X_block = compile_set(filenames_test, (250,250,3))
print('should be 1 if female, 0 if male')

## calculate accuracy on test set
predictions = model.predict(X)[:, 0]
print("predictions", predictions)
print("actual", Y)
print("total", predictions.size)

scores=[]
filenames_short=[]
for file in filenames_test:
    filenames_short.append(file)

for i, val in enumerate(predictions):
    diff=val-Y[i]
    if np.around(diff, decimals=1) == 0:
        scores.append(1)
    else:
        scores.append(0)
print("number correct:", np.sum(scores))

file=open('/output/prediction.dat','wb')
data = np.zeros(np.size(filenames_test),dtype=[('1','S20'),('2',float), ('3', float)])
data['1']=filenames_short
data['2']=predictions
data['3']=Y
np.savetxt(file, data, fmt="%s %f %f")
file.close()

## calculate accuracy for blocked faces
print('with blocked faces')
predictions_block = model.predict(X_block)[:, 0]
print("predictions", predictions_block)
print("actual", Y)

scores_block=[]
filenames_short_block=[]
for file in filenames_test:
    filenames_short_block.append(file)

for i, val in enumerate(predictions_block):
    diff=val-Y[i]
    if np.around(diff, decimals=1) == 0:
        scores_block.append(1)
    else:
        scores_block.append(0)
print("number correct with blocked faces:", np.sum(scores_block))

file=open('/output/prediction_block.dat','wb')
data = np.zeros(np.size(filenames_test),dtype=[('1','S20'),('2',float), ('3', float)])
data['1']=filenames_short_block
data['2']=predictions_block
data['3']=Y
np.savetxt(file, data, fmt="%s %f %f")
file.close()




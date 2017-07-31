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

model = load_model("/out/face_model_2520_b64_e30.h5")

filenames = [];

cnt_male   = 0;
cnt_female = 0;

# Currently predicting for first 10 images in male and female

rootdir = '/Images/Images/male/'
for root, subFolders, files in os.walk(rootdir):
    for file in files:
        cnt_male += 1;
        if cnt_male > 1285:
            filenames.append(rootdir+'/'+file)
            print(file)
        if cnt_male > 1300:
            break;

rootdir = '/Images/Images/female/'
for root, subFolders, files in os.walk(rootdir):
    for file in files:
        cnt_female += 1;
        if cnt_female >1285:
            filenames.append(rootdir+'/'+file)
            print(file)
        if cnt_female > 1300:
            break;

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
# calculate accuracy on training set
X, Y = compile_training_set(filenames, (250,250,3))

print('should be 1 if female, 0 if male')
print("predictions", model.predict(X)[:, 0])
print("actual", Y)

pred=model.predict(X)[:, 0]
scores=[]
filenames_short=[]
for file in filenames:
    file = file[20::]
    filenames_short.append(file)

for i, val in enumerate(pred):
    diff=val-Y[i]
    if np.around(diff, decimals=1) == 0:
        scores.append(1)
    else:
        scores.append(0)
print("number correct:", np.sum(scores))

#file=open('/output/prediction_0731_test_230img-2.dat','wb')
#data = np.zeros(np.size(filenames),dtype=[('1','S20'),('2',float), ('3', float)])
#data['1']=filenames_short
#data['2']=model.predict(X)[:, 0]
#data['3']=Y
#np.savetxt(file, data, fmt="%s %f %f")
#file.close()
# name = "Brad_Wilk_0001"

# this currently predicts for one image
# img = np.array(Image.open("Images/male/" + name + ".jpg"))
# to_predict = np.zeros((1,) + img.shape)
# to_predict[0, ...] = img
# print(model.predict(to_predict))

# uncomment this section to write to file
# file=open('/output/predict_'+name+'.dat','wb')
# prediction=model.predict(to_predict)
# np.savetxt(file, prediction, fmt=['%f', '%f'])
# file.close()


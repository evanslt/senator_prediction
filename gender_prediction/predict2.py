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

model = load_model("/out/vgg-faces-gender.h5")
filenames_train = pickle.load(open("/out/filenames_train.p", "rb"))
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

# X, Y, X_block = compile_set(filenames_test, (250,250,3))
# print('should be 1 if female, 0 if male')

## calculate accuracy on test set
# predictions = model.predict(X)[:, 0]
# print("predictions", predictions)
# print("actual", Y)
# print("total", predictions.size)
#
# scores=[]
# filenames_short=[]
# for file in filenames_test:
#     filenames_short.append(file)
#
# for i, val in enumerate(predictions):
#     diff=val-Y[i]
#     if np.around(diff, decimals=1) == 0:
#         scores.append(1)
#     else:
#         scores.append(0)
# print("number correct:", np.sum(scores))
#
# file=open('/output/prediction.dat','wb')
# data = np.zeros(np.size(filenames_test),dtype=[('1','S20'),('2',float), ('3', float)])
# data['1']=filenames_short
# data['2']=predictions
# data['3']=Y
# np.savetxt(file, data, fmt="%s %f %f")
# file.close()
#
# ## calculate accuracy for blocked faces
# print('with blocked faces')
# predictions_block = model.predict(X_block)[:, 0]
# print("predictions", predictions_block)
# print("actual", Y)
#
# scores_block=[]
# filenames_short_block=[]
# for file in filenames_test:
#     filenames_short_block.append(file)
#
# for i, val in enumerate(predictions_block):
#     diff=val-Y[i]
#     if np.around(diff, decimals=1) == 0:
#         scores_block.append(1)
#     else:
#         scores_block.append(0)
# print("number correct with blocked faces:", np.sum(scores_block))
#
# file=open('/output/prediction_block.dat','wb')
# data = np.zeros(np.size(filenames_test),dtype=[('1','S20'),('2',float), ('3', float)])
# data['1']=filenames_short_block
# data['2']=predictions_block
# data['3']=Y
# np.savetxt(file, data, fmt="%s %f %f")
# file.close()

###### to make sure I'm matching arielle's validation process
# make the validation set out of training data.
X, Y, X_block = compile_set(filenames_train, (250,250,3))

#make the test set out of test data.
X_test, Y_test, X_test_block =compile_set(filenames_test, (250, 250, 3))

pred_val=model.predict(X)[:, 0]

pred_test=model.predict(X_test)[:, 0]
pred_test_blocked=model.predict(X_test_block)[:,0]

scores_val=[]
scores_test=[]
scores_blocked=[]
filenames_v_short=[]
filenames_t_short=[]

for file in filenames_train:
    file = file[13::]
    filenames_v_short.append(file)

for file in filenames_test:
    file = file[13::]
    filenames_t_short.append(file)

for i, val in enumerate(pred_val):
    diff=Y[i]-val
    if np.around(diff) == 0:
        scores_val.append(1)
    else:
        scores_val.append(0)

for i, val in enumerate(pred_test):
    diff=Y_test[i]-val
    if np.around(diff) == 0:
        scores_test.append(1)
        print(diff)
    else:
        scores_test.append(0)

        #print("Incorrect Test Prediction for:", filenames_test[i], "prediction:", val)
for i, val in enumerate(pred_test_blocked):
    diff=Y_test[i]-val
    if np.around(diff) == 0:
        scores_blocked.append(1)
        print(diff)
    else:
        scores_blocked.append(0)

for i, val in enumerate(pred_test):
    print ("Name: ", filenames_t_short[i], "Pred: ", val)

#find out how many male it gets wrong

wrong_fm=0
wrong_m=0

for i, val in enumerate(pred_test):
    diff=val-Y_test[i]
    if Y_test[i]==1:
        if np.around(diff)!= 0:
            #print("INCORRECT female", filenames_test[i])
            wrong_fm+=1;
        else:
            continue

    if Y_test[i]==0:
        if np.around(diff)!= 0:
            #print("INCORRECT male", filenames_test[i])
            wrong_m+=1
        else:
            continue


print("number correct in validation:", np.sum(scores_val))
print("out of:", np.size(scores_val), "or ", np.sum(scores_val)/np.size(scores_val))

print("number correct in test:", np.sum(scores_test))
print("out of:", np.size(scores_test), np.sum(scores_test)/np.size(scores_test))

print("number correct in test, blocked:", np.sum(scores_blocked))
print("out of:", np.size(scores_blocked), np.sum(scores_blocked)/np.size(scores_blocked))

# print("FOR THE TEST ONLY:")
# print("# incorrect female", wrong_fm, "out of", np.size(filenames_test)-n_male_test)
# print("# incorrect male", wrong_m, "out of", n_male_test)

file=open('/output/validation-125.dat','wb')
data = np.zeros(np.size(filenames_v_short),dtype=[('1','S20'),('2',float), ('3', float)])
data['1']=filenames_v_short
data['2']=pred_val
data['3']=Y
np.savetxt(file, data, fmt="%s %f %f")
file.close()

file=open('/output/test-125.dat','wb')
data = np.zeros(np.size(filenames_t_short),dtype=[('1','S20'),('2',float), ('3', float)])
data['1']=filenames_t_short
data['2']=pred_test
data['3']=Y_test
np.savetxt(file, data, fmt="%s %f %f")
file.close()





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

model = load_model("/out/face_modelv1_somenoise.h5")

filenames_validation = [];
filenames_test = [];


cnt_male   = 0;
cnt_female = 0;

# Currently predicting for first 10 images in male and female

rootdir = '/Images/male/'
for root, subFolders, files in os.walk(rootdir):
    for file in files:
        cnt_male += 1;
        if cnt_male > 0 and cnt_male<50:
            filenames_validation.append(rootdir+file)
            print(file)
        if cnt_male > 3900 and cnt_male<4141:
            filenames_test.append(rootdir+file)
        #if cnt_male>4000:
            #break


print("number in val male", np.size(filenames_validation))
print("number in test male", np.size(filenames_test))

n_male_test=np.size(filenames_test)

rootdir = '/Images/female/'
for root, subFolders, files in os.walk(rootdir):
    for file in files:
        cnt_female += 1;
        #if cnt_female >1250:
        if cnt_female >0 and cnt_female<50:
            filenames_validation.append(rootdir+file)
            #print(file)
        if cnt_female > 1250:
            filenames_test.append(rootdir+file)
            #break;


print("validation samples: ", np.size(filenames_validation))
print("test samples: ", np.size(filenames_test))

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

# make the validation set out of training data.
X, Y = compile_training_set(filenames_validation, (250,250,3))

#make the test set out of test data. 
X_test, Y_test =compile_training_set(filenames_test, (250, 250, 3))

pred_val=model.predict(X)[:, 0]

pred_test=model.predict(X_test)[:, 0]

scores_val=[]
scores_test=[]
filenames_v_short=[]
filenames_t_short=[]

for file in filenames_validation:
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



#find out how many male it gets wrong

wrong_fm=0
wrong_m=0

for i, val in enumerate(pred_test):
    diff=val-Y_test[i]
    if Y_test[i]==1:
        if np.around(diff)!= 0:
            print("INCORRECT female", filenames_test[i])
            wrong_fm+=1;
        else:
            continue

    if Y_test[i]==0:
        if np.around(diff)!= 0:
            print("INCORRECT male", filenames_test[i])
            wrong_m+=1
        else:
            continue


for i, val in enumerate(pred_test):
    print ("Name: ", filenames_t_short[i], "Pred: ", val)

print("number correct in validation:", np.sum(scores_val))
print("out of:", np.size(scores_val), "or ", np.sum(scores_val)/np.size(scores_val))

print("number correct in test:", np.sum(scores_test))
print("out of:", np.size(scores_test), np.sum(scores_test)/np.size(scores_test))
print("FOR THE TEST ONLY:")
print("# incorrect female", wrong_fm, "out of", np.size(filenames_test)-n_male_test)
print("# incorrect male", wrong_m, "out of", n_male_test)

file=open('/output/validation-120.dat','wb')
data = np.zeros(np.size(filenames_v_short),dtype=[('1','S20'),('2',float), ('3', float)])
data['1']=filenames_v_short
data['2']=pred_val
data['3']=Y
np.savetxt(file, data, fmt="%s %f %f")
file.close()


file=open('/output/test-120.dat','wb')
data = np.zeros(np.size(filenames_t_short),dtype=[('1','S20'),('2',float), ('3', float)])
data['1']=filenames_t_short
data['2']=pred_test
data['3']=Y_test
np.savetxt(file, data, fmt="%s %f %f")
file.close()
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


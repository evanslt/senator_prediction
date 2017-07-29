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


model = load_model("/out/face_model.h5")

#Step through many images

num_pred=10
predictions=[]
gender=[]

#first just try picking the first n images. Need to figure out how to pick random test images.

rootdir = '/Images/Images/male/'
cnt = 0;

men=[]
for root, subFolders, files in os.walk(rootdir):
    for file in files:
        men.append(file)
        cnt += 1;
        if(cnt>num_pred):
            break;     

for name in men:
	img = np.array(Image.open("/Images/Images/male/"+name))
	img = (img - np.mean(img)) / np.var(img)
	to_predict = np.zeros((1,) + img.shape)
	to_predict[0,...] = img
	predictions.append(model.predict(to_predict))
	gender.append(0)
#move to female directory
            
rootdir = '/Images/Images/female/'
cnt = 0;
women=[]
for root, subFolders, files in os.walk(rootdir):
    for file in files:
        women.append(file)
        cnt += 1;
        if(cnt>num_pred):
            break;
            
for name in women:
	img = np.array(Image.open("/Images/Images/female/"+name))
	img = (img - np.mean(img)) / np.var(img)
	to_predict = np.zeros((1,) + img.shape)
	to_predict[0,...] = img
	predictions.append(model.predict(to_predict))
	gender.append(1)
	
#concatenate names
names=np.concatenate((men, women), axis=0)
#write to file
print(np.size(predictions))
print(np.size(names))
print(np.shape(names))


file=open('/output/prediction2.dat','wb')
data = np.zeros(np.size(names),dtype=[('1','S20'),('2',float), ('3', float)])
data['1']=names
data['2']=predictions
data['3']=gender

np.savetxt(file, data, fmt="%s %i %s")
file.close()




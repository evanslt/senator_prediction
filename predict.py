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
img = np.array(Image.open("/Images/Images/male/Brad_Wilk_0001.jpg"))
to_predict = np.zeros((1,) + img.shape)
to_predict[0,...] = img
print(model.predict(to_predict))



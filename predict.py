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


model = load_model("models/model_2017_06_21_19_14_18_08_0.53.hdf5")
img = np.array(Image.open("/Images/Images/male/xxx.jpg"))
print model.predict(img)



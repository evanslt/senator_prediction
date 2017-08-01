# NOTES for running:
# Must run: pip install quiver-engine
# This script runs well from a .ipynb jupyter notebook, just put into a .py file for git sake.
# Model must be trained on .jpg and only .jpg files will load in the quiver visualization.
#

import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.utils.vis_utils import plot_model
import keras.backend

from quiver_engine import server

model = load_model("senator_model.h5")
model.summary()

server.launch(model, temp_folder='./tmp', input_folder='./Images/Senators/')

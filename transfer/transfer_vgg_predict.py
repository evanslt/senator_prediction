from keras_vggface.vggface import VGGFace
from keras.layers import Dense, Flatten, Dropout
from keras.engine import Model
from keras.callbacks import EarlyStopping
import numpy as np
import sys
import os
from PIL import Image
from keras.models import load_model

def compile_images(files, input_shape):
    num_files = len(files)
    X = np.zeros((num_files,) + input_shape)
    Y = np.zeros(num_files)

    for idx, f in enumerate(files):
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
        # normalize image
        image = (image - np.mean(image, axis=(0, 1), keepdims=True)) / (np.var(image, axis=(0, 1), keepdims=True))
        X[idx, ...] = image
        Y[idx, ...] = float('Winner' in f)
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
            if f[0] != '.':
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

# get training data
train_rootdir = '/color'
test_rootdir = '/color_test'

# Load model
model = load_model("/out/model1.h5")

# test
filenames_test = load_filenames('original_test', test_rootdir)
X_test, Y_test = compile_images(filenames_test, (250, 250, 3))

pred_test=model.predict(X_test)[:, 0]

scores_test=[]
filenames_v_short=[]
filenames_t_short=[]

for file in filenames_test:
    file = file[13::]
    filenames_t_short.append(file)

for i, val in enumerate(pred_test):
    diff=Y_test[i]-val
    if np.around(diff) == 0:
        scores_test.append(1)
        print(diff)
    else:
        scores_test.append(0)

for i, val in enumerate(pred_test):
    print ("Name: ", filenames_t_short[i], "Pred: ", val)

print("number correct in test:", np.sum(scores_test))
print("out of:", np.size(scores_test), np.sum(scores_test)/np.size(scores_test))

file=open('/output/test.dat','wb')
data = np.zeros(np.size(filenames_t_short),dtype=[('1','S20'),('2',float), ('3', float)])
data['1']=filenames_t_short
data['2']=pred_test
data['3']=Y_test
np.savetxt(file, data, fmt="%s %f %f")
file.close()
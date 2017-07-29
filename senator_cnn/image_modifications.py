import numpy as np
import sys
import os
import numpy
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import cv2

#shift the image in up/down and left/right directions - remainder is filled with black
def shift(X, shit_pct=0.4):
    rows,cols = X.shape[0:2]
    shift_pts_x = np.random.randint(int(cols*shit_pct))*(-1)**np.random.randint(1,3)
    shift_pts_y = np.random.randint(int(cols*shit_pct))*(-1)**np.random.randint(1,3)
    M = np.float32([[1,0,shift_pts_x],[0,1,shift_pts_y]])
    X_shifted = cv2.warpAffine(X,M,(cols,rows))

    X_shifted = cv2.cvtColor(X_shifted,cv2.COLOR_BGR2GRAY)
    X_shifted = np.array(Image.fromarray(X_shifted))
    return Image.fromarray(X_shifted)

#blur the image
def blur(X):
    X_blur = np.zeros(X.shape)
    blur_size = (np.random.randint(2,4),np.random.randint(2,4))
    if X_blur.shape[-1]==1: #grayscale needs special treatment
        print("special")
        for i in range(X.shape[0]):
            X_blur[i,...] = cv2.blur(X[i,...], blur_size).reshape(X_blur.shape[1],X_blur.shape[2],1)
    else:
        for i in range(X.shape[0]):
            X_blur[i,...] = cv2.blur(X[i,...], blur_size)

    X_blur = X_blur.astype(numpy.uint8)
    X_blur = cv2.cvtColor(X_blur,cv2.COLOR_BGR2GRAY)
    X_blur = np.array(Image.fromarray(X_blur))
    return Image.fromarray(X_blur)

#rotate the image +/- 30 degrees
def rotate(X, rnd_rotation=True, degree=None, range=[-30,30]):
    rows,cols = X.shape[0:2]
    if rnd_rotation:
        deg = np.random.rand(1)*(range[1]-range[0])
    else:
        deg = 90
    M = cv2.getRotationMatrix2D((cols/2,rows/2),deg[0],1)
    X_rotated = cv2.warpAffine(X,M,(cols,rows))

    X_rotated = cv2.cvtColor(X_rotated,cv2.COLOR_BGR2GRAY)
    X_rotated = np.array(Image.fromarray(X_rotated))
    return Image.fromarray(X_rotated)

#add random noise to the image
def add_noise(X, noise_var = 100):
    row,col,ch= X.shape
    mean = 0
    var = noise_var
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    X_rnd = X + gauss
    X_rnd = X_rnd.astype(numpy.uint8)
    X_rnd = cv2.cvtColor(X_rnd,cv2.COLOR_BGR2GRAY)
    X_rnd = np.array(Image.fromarray(X_rnd))
    return Image.fromarray(X_rnd)

def block_face(X):
    X_len, X_wid, _ = X.shape
    X[int(0.15 * X_len):int(0.85 * X_len), int(0.15 * X_wid):int(0.85 * X_wid), :] = 0
    return Image.fromarray(X)

filenames = []
rootdir = 'Images/Senators'

# iterate through the images in the directory
for root, subFolders, files in os.walk(rootdir):
	for f in files:
		# only use the normalized images
		if f[-3:] == 'bmp':
			filenames.append(rootdir + '/' + f)

#loop through all files, read each and apply transformations, write to .bmp files
for f in filenames:

    img = cv2.imread(f)   #read in as cv2 type image

    #shift
    shift_image = shift(img)
    shift_image.save(f+'_shift.bmp')

    #blur
    blur_image = blur(img)
    blur_image.save(f+'_blur.bmp')

    #rotate
    rotated_image = rotate(img)
    rotated_image.save(f+'_rotate.bmp')

    #add random noise
    noised_image = add_noise(img)
    noised_image.save(f+'_noise.bmp')

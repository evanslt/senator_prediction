#
# Script for manipulation of color images
# python image_modifications_color [--pad | --man]
# --pad option pads images with black on edges to reach a full size of 250x250 (to match faces images)
# --man option manimputaes the images
#
import numpy as np
import sys
import os
import numpy
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import math
import sys

#shift the image in up/down and left/right directions - remainder is filled with black
def shift(X, shit_pct=0.4):
    rows,cols = X.shape[0:2]
    shift_pts_x = np.random.randint(int(cols*shit_pct))*(-1)**np.random.randint(1,3)
    shift_pts_y = np.random.randint(int(cols*shit_pct))*(-1)**np.random.randint(1,3)
    M = np.float32([[1,0,shift_pts_x],[0,1,shift_pts_y]])
    X_shifted = cv2.warpAffine(X,M,(cols,rows))

    X_shifted = cv2.cvtColor(X_shifted,cv2.COLOR_BGR2RGB)
    X_shifted = np.array(Image.fromarray(X_shifted))
    return Image.fromarray(X_shifted)


#pad the image with a black background and cut vertically if the image is > 250 pixels tall
def pad(X):
    rows,cols = X.shape[0:2]
    faces_dim = 250
    left_border = math.ceil((faces_dim-cols)/2)
    right_border = math.floor((faces_dim-cols)/2)
    top_border = math.ceil((faces_dim-rows)/2)
    bottom_border = math.floor((faces_dim-rows)/2)


    if(top_border < 0 or bottom_border < 0):

        if(right_border <0 or left_border < 0):
            constant = cv2.copyMakeBorder(X,0,0,0,0,cv2.BORDER_CONSTANT,value=[0,0,0])
            tmp= [i[-1*left_border:(faces_dim-right_border-1)] for i in constant]
            constant = np.array(tmp)
            constant = constant[-1*top_border:(faces_dim-bottom_border-1)]

        else:
            constant = cv2.copyMakeBorder(X,0,0,left_border,right_border,cv2.BORDER_CONSTANT,value=[0,0,0])
            constant = constant[-1*top_border:(faces_dim-bottom_border-1)]

    elif(right_border <0 or left_border < 0):
        constant = cv2.copyMakeBorder(X,top_border,bottom_border,0,0,cv2.BORDER_CONSTANT,value=[0,0,0])
        tmp= [i[-1*left_border:(faces_dim-right_border-1)] for i in constant]
        constant = np.array(tmp)
    else:
        constant = cv2.copyMakeBorder(X,top_border,bottom_border,left_border,right_border,cv2.BORDER_CONSTANT,value=[0,0,0])
    constant = cv2.cvtColor(constant, cv2.COLOR_RGB2BGR)
    rows_final,cols_final = constant.shape[0:2]


    if(rows_final > 250 or cols_final > 250):

        print('rows:'+str(rows))
        print('cols:'+str(cols))
        print(left_border)
        print(right_border)
        print(top_border)
        print(bottom_border)

        print(top_border - bottom_border)
        print(right_border - left_border)

        print('rows final:'+str(rows_final))
        print('cols final:'+str(cols_final))

        print('\n\n')

    return Image.fromarray(constant)


#blur the image
#BUG: this flips the colors to a yellow-ish tint
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
    X_blur = cv2.cvtColor(X_blur,cv2.COLOR_RGB2BGR)
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

    X_rotated = cv2.cvtColor(X_rotated,cv2.COLOR_BGR2RGB)
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
    X_rnd = cv2.cvtColor(X_rnd,cv2.COLOR_BGR2RGB)
    X_rnd = np.array(Image.fromarray(X_rnd))
    return Image.fromarray(X_rnd)


if(len(sys.argv) < 2):
    print("Requires one of the following input flag options:")
    print("--pad : pads the images for input into transfer learning")
    print("--man : manipulates the senator images (shift, blur, rotate, noised)")

filenames = []
rootdir = 'Images/Senators'

# iterate through the images in the directory
for root, subFolders, files in os.walk(rootdir):
	for f in files:
		# only use the normalized images
		if f[-3:] == 'jpg':
			filenames.append(rootdir + '/' + f)

if sys.argv[1] == '--pad':
    #loop through all files, read each and apply transformations, write to .bmp files
    for f in filenames:
        img = cv2.imread(f)   #read in as cv2 type image
        #pad to size
        padded = pad(img)
        padded.save(f+'_pad.jpg')


if sys.argv[1] == '--man':
    #loop through all files, read each and apply transformations, write to .bmp files
    for f in filenames:
        img = cv2.imread(f)   #read in as cv2 type image
        #shift
        shift_image = shift(img)
        shift_image.save(f+'_shift.jpg')
        #blur
        blur_image = blur(img)
        blur_image.save(f+'_blur.jpg')
        #rotate
        rotated_image = rotate(img)
        rotated_image.save(f+'_rotate.jpg')
        #add random noise
        noised_image = add_noise(img)
        noised_image.save(f+'_noise.jpg')

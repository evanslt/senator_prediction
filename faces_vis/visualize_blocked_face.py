import cv2
import numpy as np
from PIL import Image

f = "Alanis_Morissette_0001.jpg"

num_train = 1
input_shape = (250,250,3)
idx = 0

X = np.zeros((num_train,) + input_shape)
Y = np.zeros(num_train)
X_block = np.zeros((num_train,) + input_shape)

image = cv2.imread(f)
# image = np.array(Image.open(f))
# image = (image - np.mean(image, axis=(0, 1), keepdims=True)) / (np.var(image, axis=(0, 1), keepdims=True))
# X[idx, ...] = image  # expand the dimension since keras expects a color channel
# Y[idx, ...] = float('female' in f)

# print(image.shape) # 250 250 3
img_len, img_wid = image.shape[:2]
image[int(0.15 * img_len):int(0.85 * img_len), int(0.15 * img_wid):int(0.85 * img_wid), :] = 0

im = Image.fromarray(image)
im.save("your_file.jpeg")

#%%
from keras.models import load_model
import numpy as np
import pandas as pd
import cv2 as cv 
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import PIL.ImageOps

#%%
model = load_model('MNIST_Classifier.h5')
# %%
image = Image.open('greyscale_image.jpg')
inverse = PIL.ImageOps.invert(image)
#%%
image = cv.imread('no5.jpeg')
inverse_n_greyscale = PIL.ImageOps.invert(Image.fromarray(image))
inverse_n_cropped = inverse_n_greyscale.convert('L')
#%%
inverse_n_contrast = inverse_n_cropped.crop((500,800,750,1200))
inverse = ImageEnhance.Contrast(inverse_n_contrast).enhance(1)
#%%
new = inverse.enhance(6).resize((28,28))

new = np.array(new)

new = new.reshape(1,28,28,1)

model.predict(new)

# %%
im = cv.imread('no.5.jpeg')
transmask = im[:,:,3] < 125
im[transmask] = [255,255,255,255]
new_im = cv.cvtColor(im, cv.COLOR_BGRA2BGR)
inverse_n_greyscale = PIL.ImageOps.invert(Image.fromarray(new_im))
inverse = inverse_n_greyscale.convert('L')
# %%

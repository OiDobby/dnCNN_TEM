import numpy as np
import math
import time
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 as cv

from keras.models import load_model
from CNN_utils import dir_check, load_img_data, plot_image

#number of pixels in the images
pixels = 64

base_path = os.getcwd()
DS_dir_name = 'Gen_DS/' # dataset dir name
DS_path = os.path.join(base_path, DS_dir_name)

#predict_X_name = 'noise_image/'   # noisy images dir name for predicting
predict_X_name = 'noise_image-test'

#Loading the training data.
predict_X_path = os.path.join(DS_path, predict_X_name)
print('loading predict_X')
predict_X = load_img_data(predict_X_path, pixels)

print ("predict_X shape is:", predict_X.shape)

print('Loading the model.')
model = load_model('CNN_model.h5')
model.summary()

print('Starting prediction.')

file_names = os.listdir(predict_X_path)
#file_names.sort(key=lambda x: int(x.split('.')[0]))
num_img = len(file_names)
predict_Y = np.zeros((num_img, pixels, pixels,0))

predict_Y = model.predict(predict_X)

print('Saving predicted imges.')

PD_dir_name = 'predict_image/'
dir_check(base_path, PD_dir_name)
PD_dir_path = os.path.join(base_path,PD_dir_name)

for i in range(num_img):
    predict_file_name = PD_dir_path + file_names[i]
    np.save(predict_file_name, predict_Y[i,:,:,0])

print('Program done')

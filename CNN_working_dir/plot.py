import numpy as np
import math
import time
import datetime
import os
import shutil
import matplotlib.pyplot as plt
import cv2 as cv

from keras.models import load_model
from CNN_utils import dir_check, load_img_data, plot_image

base_path = os.getcwd()
DS_dir_name = 'Gen_DS/' # dataset dir name
DS_path = os.path.join(base_path, DS_dir_name)

noise_dir_name = 'noise_image-test/'
norm_dir_name = 'norm_image-test/'
PD_dir_name = 'predict_image/'

noise_dir_path = os.path.join(DS_path, noise_dir_name)
norm_dir_path = os.path.join(DS_path, norm_dir_name)
PD_dir_path = os.path.join(base_path, PD_dir_name)

dir_path = [norm_dir_path, noise_dir_path, PD_dir_path]
plot_dir_name = ['plot_image_norm/', 'plot_image_noise/', 'plot_image_predict/']

for i in range(len(plot_dir_name)):
    plot_image(dir_path[i], plot_dir_name[i])

print('===================================================================')
print('All images are plotted.')

print('===================================================================')
print('We make and check the saving directory.')

x = datetime.datetime.now()
save_dir_name = x.strftime('%m_%d_%y') +'/'

group_dir_name = 'plotted_img/'

plotted_dir_path = os.path.join(base_path, group_dir_name)
dir_check(plotted_dir_path, save_dir_name)

save_dir_path = os.path.join(plotted_dir_path, save_dir_name)

for i in range(len(plot_dir_name)):
    time.sleep(2)
    shutil.move('./'+plot_dir_name[i], save_dir_path)


print('===================================================================')
print('All images are maved in dir_path;', save_dir_path)
print('===================================================================')

print('Program done')

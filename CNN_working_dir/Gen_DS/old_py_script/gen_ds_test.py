import numpy as np
import cv2
import math
import time
import os
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from gen_ds_utils import dir_check, slice_image, make_noisy_image, make_valid_set

base_path = os.getcwd()
dir_name = 'test_image/'
ori_file_dir = 'test_png/'       # It indicates directory for origin images. (.png format)
dir_name1 = 'norm_image-test/'   # We save normalized images in this directory. (.npy format)
dir_name2 = 'noise_image-test/'  # We save noisy images in thies directory (.npy format)

num_make_images = 8  # We make num_make_images*num_make_images images

slice_image(base_path, dir_name, ori_file_dir, num_make_images)

make_noisy_image(base_path, dir_name, dir_name1, dir_name2)

print('Program done')

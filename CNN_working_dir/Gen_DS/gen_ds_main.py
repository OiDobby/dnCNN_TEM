import numpy as np
import cv2
import math
import time
import os
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from gen_ds_utils import dir_check, img_proc, make_valid_set 

# Update in 03/13/2024

base_path = os.getcwd()

num_make_images = 8         # We make num_make_images*num_make_images images.
num_pixel = 64              # The images have constructed by num_pixel*num_pixel

ori_file_dir = 'ori_png/'   # It indicates directory for origin images. (.png format)
dir_name = 'ori_image/'     # Sliced iages are saving in this directory. (.npy format)
dir_name1 = 'norm_image/'   # We save normalized images in this directory. (.npy format)
dir_name2 = 'noise_image/'  # We save noisy images in thies directory. (.npy format)

noise_factor = [0.1, 0.15, 0.2, 0.25]  # We use various noises for traning images.

WB_option = 1   # White balancing option 
                # 0: Not use (default)
                # 1: Use

WB_val = [1, 0.2, 0.05]     # WB normal distribution values [type, mean, std].
                            # [0, 0.2, 0.05] is default for gaussian
                            # type can be 0 or 1. 
                            # Type; 0 = preprocessing beform noise, 1 = post-processing after noise

exp_img_option = 1  # 0: Using experimental images (already noise, do not use to 0 for traning sets).
                    # 1: Gaussian noise.
                    # 2: Salt-Pepper noise
                    # 3: Gaussian + Salt-Pepper noise
                    # 4: Salt-Pepper + Gausian noise

#img_proc(base_path, ori_file_dir, dir_name, dir_name1, dir_name2, num_make_images, num_pixel, noise_factor, WB_option, WB_val, exp_img_option)

#make_valid_set(base_path, dir_name1, dir_name2)

dir_name = 'test_image/'
ori_file_dir = 'test_png/'       # It indicates directory for origin images. (.png format)
dir_name1 = 'norm_image-test/'   # We save normalized images in this directory. (.npy format)
dir_name2 = 'noise_image-test/'  # We save noisy images in thies directory. (.npy format)

noise_factor = [0.3]

WB_option = 0   # White balancing option 
                # 0: Not use (default)
                # 1: Use

WB_val = [1, 0.2, 0.05]     # WB normal distribution values [type, mean, std].
                            # [1, 0.2, 0.05] is default for gaussian.
                            # type can be 0 or 1. 
                            # Type; 0 = preprocessing beform noise, 1 = post-processing after noise

exp_img_option = 0  # 0: Using experimental images (already noise).
                    # 1: Gaussian noise.
                    # 2: Salt-Pepper noise
                    # 3: Gaussian + Salt-Pepper noise
                    # 4: Salt-Pepper + Gausian noise

img_proc(base_path, ori_file_dir, dir_name, dir_name1, dir_name2, num_make_images, num_pixel, noise_factor, WB_option, WB_val, exp_img_option)

print('Program done')

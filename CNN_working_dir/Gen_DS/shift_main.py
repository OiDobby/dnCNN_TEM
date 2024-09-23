import numpy as np
import cv2
import math
import time
import os
import shutil
import random
import matplotlib.pyplot as plt
import tensorflow as tf

def shift_img(base_path, ori_file_dir, moved_pixel):
    file_dir_path = os.path.join(base_path, ori_file_dir)
    file_list = os.listdir(file_dir_path)
    file_num = len(file_list)

    file_name = [i+1 for i in range(file_num)]
    file_name = list(map(str, file_name))
    for i in range(file_num):
        file_name[i] = file_name[i] + '_shift-'

    print('===================================================================')

    for i in range(file_num):
        file_path = os.path.join(file_dir_path, file_list[i])
        print('Working in', file_path)
        ori_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) # image read, flag=gray scale - original image
        ori_X = ori_img.shape[1]
        ori_Y = ori_img.shape[0]
        if ori_X == ori_Y :
            print('Shape of the image is square.')
            ori_img = ori_img
        elif ori_X > ori_Y :
            print('Shape of the image is not square. We make the square image.')
            ori_img = ori_img[0:ori_Y, 0:ori_Y]
        elif ori_X < ori_Y :
            print('Shape of the image is not square. We make the square image.')
            ori_img = ori_img[0:ori_X, 0:ori_X]

        print('Resizing the image in', 512,'*', 512,'pixels...')
        resized_img = cv2.resize(ori_img, dsize = (512, 512), interpolation=cv2.INTER_AREA) # scaled image

        print('Resized image done.')
        print('-------------------------------------------------------------------')
        
        ori_X = resized_img.shape[1]
        ori_Y = resized_img.shape[0]
        #size_X = 512 # int(ori_X/num_make_images) # num = num_pixel
        #size_Y = 512 # int(ori_Y/num_make_images) # num = num_pixel

        for j in range(len(moved_pixel)):
            print('Shifting image in', moved_pixel[j], 'pixels')
            s_img = resized_img.copy()
            s_img = np.vstack((s_img, np.zeros((moved_pixel[j], ori_Y))))
            s_img = np.hstack((s_img, np.zeros((ori_X+moved_pixel[j], moved_pixel[j]))))

            for l in range(moved_pixel[j]):
                s_img[l+ori_X] = s_img[l]
            for l in range(ori_X+moved_pixel[j]):
                for m in range(moved_pixel[j]):
                    s_img[l][m+ori_Y] = s_img[l][m]

            for k in range(moved_pixel[j]):
                s_img = np.delete(s_img, 0, axis = 0)
            for k in range(moved_pixel[j]):
                s_img = np.delete(s_img, 0, axis = 1)

            save_fig_name = file_dir_path + file_name[i] + str(moved_pixel[j]) + 'px'

            print('plot the image')
            fig = plt.imshow(s_img)
            plt.axis('off'), plt.xticks([]), plt.yticks([])
            plt.tight_layout()
            plt.gray()
            plt.savefig(save_fig_name, bbox_inches = 'tight', pad_inches = 0, dpi = 300)
        print('-------------------------------------------------------------------')

    return

base_path = os.getcwd()

ori_file_dir = 'ori_png/'   # It indicates directory for origin images. (.png format)
moved_pixel = [8, 10, 12]

shift_img(base_path, ori_file_dir, moved_pixel)

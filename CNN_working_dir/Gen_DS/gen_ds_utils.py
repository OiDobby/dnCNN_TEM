import numpy as np
import cv2
import math
import time
import os
import shutil
import random
import matplotlib.pyplot as plt
import tensorflow as tf

# Update in 03/19/2024

def dir_check(base_path, dir_name):
    print('-------------------------------------------------------------------')
    dir_path = os.path.join(base_path,dir_name)
    print('Directory checking...')
    if os.path.exists(dir_path):
        print('Directory already exist. We remove the directory.')
        shutil.rmtree(dir_path)
        time.sleep(2)
        print('Directory removed completely.')
        time.sleep(1)

        os.mkdir(dir_path)
        print('Make a diretory.')

    else:
        os.mkdir(dir_path)
        print('Make a diretory.')

    print('-------------------------------------------------------------------')

    return base_path

def White_Balance(img, WB_option, WB_val):
    # white balance from considering the expriment image
    height, width = img.shape
    if WB_option == 0:
        img = img
    elif WB_option == 1:
        rand_num = random.uniform(0.0, 1.0)
        rand_num1 = random.uniform(0.4*WB_val[1], 0.6*WB_val[1])
        if rand_num < 0.15:
            WB_val_new = rand_num1
            WB_DV = 0
        elif rand_num >= 0.85:
            WB_val_new = 2*rand_num1
            WB_DV = 1
        else :
            WB_val_new = 1.5*rand_num1
            WB_DV = 2

        for l in range(height):
            for m in range(width):
                rand_num = random.uniform(0.8*WB_val[1], 1.2*WB_val[1])
                rand_num2 = np.random.normal(WB_val_new, WB_val[2])
                if WB_DV == 0:
                    if img[l][m] < WB_val_new:
                        img[l][m] = rand_num2
                    elif img[l][m] > 3*WB_val_new:
                        img[l][m] = img[l][m]-2.3*rand_num2
                    elif img[l][m] > 2*WB_val_new and img[l][m] <= 3*WB_val_new:
                        img[l][m] = img[l][m]-2.1*rand_num2
                    else :
                        img[l][m] = img[l][m]-1.9*rand_num2
                if WB_DV == 1:
                    if img[l][m] < WB_val_new:
                        img[l][m] = rand_num2
                    elif img[l][m] > 3*WB_val_new:
                        img[l][m] = img[l][m]-1.3*rand_num2
                    elif img[l][m] > 2*WB_val_new and img[l][m] <= 3*WB_val_new:
                        img[l][m] = img[l][m]-0.8*rand_num2
                    else :
                        img[l][m] = img[l][m]-0.3*rand_num2
                elif WB_DV == 2:
                    if img[l][m] < WB_val_new:
                        img[l][m] = rand_num2
                    elif img[l][m] > 3*WB_val_new:
                        img[l][m] = img[l][m]-1.8*rand_num2
                    elif img[l][m] > 2*WB_val_new and img[l][m] <= 3*WB_val_new:
                        img[l][m] = img[l][m]-1.3*rand_num2
                    else :
                        img[l][m] = img[l][m]-0.8*rand_num2
    else :
        img = img

    return img

def Noise_effect(exp_img_option, img_norm, noise_factor, x):
    # Noise effects
    if exp_img_option == 0:     #None
        img_noisy = img_norm.copy()
    elif exp_img_option == 1:   #G
        img_noisy = img_norm + noise_factor[x] * tf.random.normal(shape=img_norm.shape)
        img_noisy = tf.clip_by_value(img_noisy, clip_value_min = 0., clip_value_max=1.)
        img_noisy = img_noisy.numpy()
    elif exp_img_option == 2:   #SP
        img_noisy = img_norm.copy()
        height, width = img_noisy.shape
        for l in range(height):
            for m in range(width):
                rand_num = random.uniform(0.0, 1.0)
                if rand_num <= noise_factor[x]:
                    if img_noisy[l][m] < 0.5:
                        img_noisy[l][m] = 1
                    else :
                        img_noisy[l][m] = 0
                else :
                    img_noisy[l][m] = img_noisy[l][m]
    elif exp_img_option == 3:   #G+SP
        img_noisy = img_norm + noise_factor[x] * tf.random.normal(shape=img_norm.shape)
        img_noisy = tf.clip_by_value(img_noisy, clip_value_min = 0., clip_value_max=1.)
        img_noisy = img_noisy.numpy()
        height, width = img_noisy.shape
        for l in range(height):
            for m in range(width):
                rand_num = random.uniform(0.0, 1.0)
                if rand_num <= noise_factor[x]:
                    if img_noisy[l][m] < 0.5:
                        img_noisy[l][m] = 1
                    else :
                        img_noisy[l][m] = 0
                else :
                    img_noisy[l][m] = img_noisy[l][m]
    elif exp_img_option == 4:   # SP+G
        img_noisy = img_norm.copy()
        height, width = img_noisy.shape
        for l in range(height):
            for m in range(width):
                rand_num = random.uniform(0.0, 1.0)
                if rand_num <= noise_factor[x]:
                    if img_noisy[l][m] < 0.5:
                        img_noisy[l][m] = 1
                    else :
                        img_noisy[l][m] = 0
                else :
                    img_noisy[l][m] = img_noisy[l][m]
        img_noisy = img_noisy + noise_factor[x] * tf.random.normal(shape=img_norm.shape)
        img_noisy = tf.clip_by_value(img_noisy, clip_value_min = 0., clip_value_max=1.)
        img_noisy = img_noisy.numpy()
    else :
        print('You set an unavailable value for "exp_img_option". Check this first!')
        print('We do not make the noise effect on images.')
        img_noisy = img_norm.copy()

    return img_noisy 

def img_proc(base_path, ori_file_dir, dir_name, dir_name1, dir_name2, num_make_images, num_pixel, noise_factor, WB_option, WB_val, exp_img_option):
    dir_check(base_path, dir_name)
    dir_check(base_path, dir_name1)
    dir_check(base_path, dir_name2)    

    dir_path = os.path.join(base_path, dir_name)
    dir_path1 = os.path.join(base_path,dir_name1)
    dir_path2 = os.path.join(base_path,dir_name2)

    file_dir_path = os.path.join(base_path, ori_file_dir)
    file_list = os.listdir(file_dir_path)
    file_num = len(file_list)

    file_name = [i+1 for i in range(file_num)]
    file_name = list(map(str, file_name))
    for i in range(file_num):
        file_name[i] = 'set' + file_name[i]

    noise_factor_name = list(map(str, noise_factor))

    print('===================================================================')
    print('Start the image processing')
    print('Read from original image')

    print('Checking "WB_option".')
    if WB_option == 0:
        print('"WB_option" OFF')
        print('No whith balancing is applied to the images.')
    elif WB_option == 1:
        print('"WB_option" ON')
    else :
        print('You do not use the value specified for "WB_option".')
        print('No whith balancing is applied to the images.')

    print('Checking "WB_val".')
    if WB_option == 0:
        print('"WB" in pre-proc. before noise')
    elif WB_option == 1:
        print('"WB" in post-proc. after noise')
    else :
        print('You do not use the value specified for "WB_val[0]".')
        print('We use default value')
        print('"WB" in post-proc. after noise')


    for x in range(len(noise_factor_name)):
        print('===================================================================')
        if exp_img_option == 0:
            print('Process of images is end of exp_img.')
        else :
            print('Working in noise_factor:', noise_factor_name[x])
        print('-------------------------------------------------------------------')
        for k in range(file_num):
            file_path = os.path.join(file_dir_path, file_list[k])
            print(file_path)
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

            print('Resizing the image in', num_make_images*num_pixel,'*',num_make_images*num_pixel ,'pixels...')
            resized_img = cv2.resize(ori_img, dsize = (num_make_images*num_pixel, num_make_images*num_pixel), interpolation=cv2.INTER_AREA) # scaled image

            print('Resized image done.')
                
            ori_X = resized_img.shape[1]
            ori_Y = resized_img.shape[0]
            size_X = num_pixel # int(ori_X/num_make_images) # num = num_pixel
            size_Y = num_pixel # int(ori_Y/num_make_images) # num = num_pixel
    
            print('Start the image processing; slicing, norm, noisy.')

            sliced_np_name = [i+1 for i in range(num_make_images*num_make_images)]
            sliced_np_name = list(map(str, sliced_np_name))
                
            if len(noise_factor_name) > 1:
                for i in range (0,num_make_images*num_make_images):
                    sliced_np_name[i] = noise_factor_name[x] + '_' + file_name[k] + '.' + sliced_np_name[i] + '.npy'
            elif len(noise_factor_name) == 1:
                for i in range (0,num_make_images*num_make_images):
                    sliced_np_name[i] = file_name[k] + '.' + sliced_np_name[i] + '.npy'


            for j in range (0,num_make_images):
                for i in range (0,num_make_images):
                    #image slicing
                    re_img = resized_img[j*size_Y:(j+1)*size_Y, i*size_X:(i+1)*size_X].copy()
                    np_file_name = sliced_np_name[num_make_images*j+i]
                    np_file_path0 = os.path.join(dir_path, np_file_name)
                    np.save(np_file_path0, re_img)

                    #image normalization
                    img_norm = re_img.copy() #img_norm_ori = re_img.copy()
                    height, width = img_norm.shape #img_norm = img_norm_ori.copy()
                    img_norm = img_norm.astype(np.float32)
                    for l in range(height):
                        for m in range(width):
                            img_norm[l][m] = img_norm[l][m]/255
                    np_file_path1 = os.path.join(dir_path1, np_file_name)
                    np.save(np_file_path1, img_norm)

                    if WB_val[0] == 0:
                        #white balance (pre)
                        White_Balance(img_norm, WB_option, WB_val)
                        #image noise incorporate
                        img_noisy = Noise_effect(exp_img_option, img_norm, noise_factor, x)
                    elif WB_val[0] == 1:
                        #image noise incorporate
                        img_noisy = Noise_effect(exp_img_option, img_norm, noise_factor, x)
                        #white valance (post)
                        White_Balance(img_noisy, WB_option, WB_val)
                    else :
                        #image noise incorporate
                        img_noisy = Noise_effect(exp_img_option, img_norm, noise_factor, x)
                        #white valance (post)
                        White_Balance(img_noisy, WB_option, WB_val)

                    np_file_path2 = os.path.join(dir_path2, np_file_name)
                    np.save(np_file_path2, img_noisy)
                        
            print('End of the image processing for file:', file_list[k])
        
        print('-------------------------------------------------------------------')
        if exp_img_option == 0:
            print('Process of images is end of exp_img.')
        else :
            print('Process of images is end in noise_factor:', noise_factor_name[x])

    print('Finished the image progress (Slicing, Normalization, Incorporating noise) for all images.')
    return base_path


def make_valid_set(base_path, dir_name1, dir_name2):
    dir_path1 = base_path + '/' + dir_name1
    dir_path2 = base_path + '/' + dir_name2
    
    file_list = os.listdir(dir_path1)
    file_num = len(file_list)

    rand_file_num = int(file_num*0.2)
    rand_file_list = random.sample(file_list, rand_file_num)

    print('===================================================================')
    print('Copy some images for valid set from training set')

    dir_VS_name1 = dir_name1.replace('/', '') + '-val/'
    dir_VS_name2 = dir_name2.replace('/', '') + '-val/'

    dir_check(base_path, dir_VS_name1)
    dir_check(base_path, dir_VS_name2)

    dir_VS_path1 = base_path + '/' + dir_VS_name1
    dir_VS_path2 = base_path + '/' + dir_VS_name2

    for i in range(rand_file_num):
        shutil.copyfile(dir_path1 + rand_file_list[i] , dir_VS_path1 + rand_file_list[i])
        shutil.copyfile(dir_path2 + rand_file_list[i] , dir_VS_path2 + rand_file_list[i])

    print('Complete to make the valid set')
    print('===================================================================')

    return base_path

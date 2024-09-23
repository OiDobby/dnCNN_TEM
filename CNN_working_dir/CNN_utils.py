import numpy as np
import math
import time
import os
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import BatchNormalization, Conv2D, PReLU, Dropout
from keras.initializers import Constant
import cv2 as cv
import re

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


def load_img_data(load_path, pixels):
    file_names = os.listdir(load_path)
#    file_names.sort(key=lambda x: int(x.split('.')[0]))
    print ("loading data...")
#    print (file_names)

    #number of images should be the same for X and Y
    num_img = len(file_names)
    #Then I load the other images in the array train_X
    images = np.zeros((num_img, pixels, pixels,1))
    r = list(range(num_img))
    #random.shuffle(r)
    for i in r:
        file_name = file_names[i]
        file_path = os.path.join(load_path, file_name)
        images[i,:,:,0] = np.load(file_path)
    return images


def get_uncompiled_model_filter(model_number, dropout_rate, pixels, num_filters):
    if model_number == 0: #from the ref. paper
        model = models.Sequential()
        #put some layers
        #block1 down
        model.add(Conv2D(num_filters, (3, 3), input_shape=(pixels, pixels, 1), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        #model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Conv2D(num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        #model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(layers.MaxPooling2D((2, 2)))
        #block 2 down
        model.add(Conv2D(2*num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        #model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Conv2D(2*num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        #model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(layers.MaxPooling2D((2, 2)))
        #lower block
        model.add(Conv2D(4*num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        #model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Conv2D(4*num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        #model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(layers.UpSampling2D())
        #block 1 up
        model.add(Conv2D(2*num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        #model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Conv2D(2*num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        #model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(layers.UpSampling2D())
        #block 2 up
        model.add(Conv2D(num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        #model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Conv2D(num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        #model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        #final
        model.add(Conv2D(1, (3, 3), padding='same', activation='tanh'))

    if model_number == 1: #my model 1
        model = models.Sequential()
        #put some layers
        #block1 down
        model.add(Conv2D(num_filters, (3, 3), input_shape=(pixels, pixels, 1), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Conv2D(num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(layers.MaxPooling2D((2, 2)))
        #block 2 down
        model.add(Conv2D(2*num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Conv2D(2*num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(layers.MaxPooling2D((2, 2)))
        #lower block
        model.add(Conv2D(4*num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Conv2D(4*num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(layers.UpSampling2D())
        #block 1 up
        model.add(Conv2D(2*num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Conv2D(2*num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(layers.UpSampling2D())
        #block 2 up
        model.add(Conv2D(num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Conv2D(num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        #final
        model.add(Conv2D(1, (3, 3), padding='same', activation='relu'))

    if model_number == 2: #my model 2, Not yet completed
        model = models.Sequential()
        #put some layers
        #block1 down
        model.add(Conv2D(num_filters, (3, 3), input_shape=(pixels, pixels, 1), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Conv2D(num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(layers.MaxPooling2D((2, 2)))
        #block 2 down
        model.add(Conv2D(2*num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Conv2D(2*num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(layers.MaxPooling2D((2, 2)))
        #lower block
        model.add(Conv2D(4*num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Conv2D(4*num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(layers.UpSampling2D())
        #block 1 up
        model.add(Conv2D(2*num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Conv2D(2*num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(layers.UpSampling2D())
        #block 2 up
        model.add(Conv2D(num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Conv2D(num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        #final
        model.add(Conv2D(1, (3, 3), padding='same', activation='relu'))

    return model


#returns the compiled model
def get_compiled_model(model_number, loss, alpha, dropout_rate, pixels, learning_rate, num_filters):
    model = get_uncompiled_model_filter(model_number, dropout_rate, pixels, num_filters)
    print("compiling the model...")
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if loss == "MAE":
        model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=opt, metrics=['mean_absolute_error'])
    if loss == "MSE":
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer= opt, metrics=['mean_squared_error'])
    if loss == "SSIM":
        model.compile(loss=my_loss_function_SSIM(alpha), optimizer=opt, metrics=['mean_absolute_error'])
    if loss == "SSIMm":
        model.compile(loss=my_loss_function_SSIM_multi(alpha), optimizer= opt, metrics=['mean_absolute_error'])
    return model

#plotting files
def plot_image(data_dir_path, plot_dir_name):
    print('===================================================================')
    print('Plotting the images.')
    base_path = os.getcwd()
    
    #dir_path1 = os.path.join(base_path,data_dir_name)
    #dir_path2 = os.path.join(base_path,plot_dir_name)

    dir_path1 = data_dir_path
    dir_path2 = os.path.join(base_path,plot_dir_name)

    dir_check(base_path, plot_dir_name)

    file_list = os.listdir(dir_path1)
#    file_list.sort(key=lambda x: int(x.split('.')[0]))
    file_num = len(os.listdir(dir_path1))

    plot_name = [0 for i in range(file_num)]
    for i in range(file_num):
        plot_name[i] = file_list[i].replace('.npy', '.png')

    for i in range(file_num):
        plot_data_name = dir_path1 + file_list[i]
        plot_fig_name = dir_path2 + plot_name[i]

        predict_image = np.load(plot_data_name)

        fig = plt.imshow(tf.squeeze(predict_image))
        plt.axis('off'), plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.gray()
        plt.savefig(plot_fig_name, bbox_inches = 'tight', pad_inches = 0, dpi = 300)
    print('Finish! we plotted the all files!')

import numpy as np
import cv2
import math
import os
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf

def plot_noisy_image(file_num, dir_path1, noise_file_name, plot_name, dir_path2):
    print('Plotting the noisy images')
    for i in range(file_num):
        plot_file_name = dir_path1 + noise_file_name[i]
        plot_fig_name = dir_path2 + plot_name[i]

        noise_image = np.load(plot_file_name)

        fig = plt.imshow(tf.squeeze(noise_image))
        plt.axis('off'), plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.gray()
        plt.savefig(plot_fig_name, bbox_inches = 'tight', pad_inches = 0, dpi = 35)
    print('Finish! we plotted the all files!')

base_path = os.getcwd()
dir_name1 = 'noise_image/'
dir_path1 = os.path.join(base_path,dir_name1)

file_list = os.listdir(dir_path1)
file_list.sort(key=lambda x: int(x.split('.')[0]))
file_num = len(os.listdir(dir_path1))
#file_name = dir_path1 + file_list[2]
#print(file_name)
#print (file_num)


dir_name2 = 'plot_noise_image/'
dir_path2 = os.path.join(base_path,dir_name2)

if os.path.exists(dir_path2):
    for file in os.scandir(dir_path2):
        os.remove(file.path)
    print('Directory for ploting noise images is existed, Files are all clear.')
    print('Make a diretory.')
else:
    os.mkdir('%s' %dir_name2)
    print('Directory for ploting noise imagse is not existed, Make a diretory.')

noise_file_name = []
plot_name = []
for i in range(file_num):
    noise_file_name.append(i+1)
    plot_name.append(i+1)

noise_file_name = list(map(str, noise_file_name))
plot_name = list(map(str, plot_name))

for i in range(file_num):
    noise_file_name[i] = noise_file_name[i] + '.npy'
    plot_name[i] = plot_name[i] + '_noisy.png'

plot_noisy_image(file_num, dir_path1, noise_file_name, plot_name, dir_path2)


#img2 = np.load('./1_noisy.npy')
#plt.imshow(tf.squeeze(img2))
#plt.gray()
#plt.show()

#img2_noisy = np.zeros((height, width), dtype=np.int64)
#for i in range(height):
#    for j in range(width):
#        img2_noisy[i][j] = math.floor(img1_noisy[i][j]*225)
#
#
#cv2.imwrite('./reimg.tif', img2_noisy)


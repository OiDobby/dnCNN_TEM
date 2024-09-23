import numpy as np
import math
import time
import os
import matplotlib.pyplot as plt
import cv2 as cv

#from keras.models import load_model
#from CNN_utils import dir_check, load_img_data, plot_image

base_path = os.getcwd()
DS_dir_name = 'Gen_DS/' # dataset dir name
DS_path = os.path.join(base_path, DS_dir_name)

noise_dir_name = 'noise_image-test/'
norm_dir_name = 'norm_image-test/'

noise_dir_path = os.path.join(DS_path, noise_dir_name)
norm_dir_path = os.path.join(DS_path, norm_dir_name)

PD_dir_name = 'predict_image/'
PD_dir_path = os.path.join(base_path, PD_dir_name)

pixels = 64
save_option =  1  # 0: norm_img, 1: noise_img, 2: PD_img

if save_option == 0:
    dir_path = norm_dir_path
    save_np_name = 'merged_image_norm.npy'
    save_fig_name = 'merged_image_norm'
elif save_option == 1:
    dir_path = noise_dir_path
    save_np_name = 'merged_image_noise.npy'
    save_fig_name = 'merged_image_noise'
elif save_option == 2:
    dir_path = PD_dir_path
    save_np_name = 'merged_image_PD.npy'
    save_fig_name = 'merged_image_PD'

file_list = os.listdir(dir_path)
num_img = len(file_list)
naxis_img = int(math.sqrt(num_img))

print('-------------------------------------------------------------------')
print('Searching files.')

file_name_list = [i for i in range(1,len(file_list)+1)]
file_name_list = list(map(str, file_name_list))
for i in range(len(file_name_list)):
    file_name_list[i] = 'set1.' + file_name_list[i] + '.npy'

print('-------------------------------------------------------------------')
print('Reading files.')
images = np.zeros((num_img, pixels, pixels))
for i in range(len(file_list)):
    file_name = file_name_list[i]
    file_path = os.path.join(dir_path, file_name)
    images[i,:,:] = np.load(file_path)

list_img = images.tolist()

print('-------------------------------------------------------------------')
print('Merging files in 1 image.')
tmp_img = []
merg_img = []
for k in range(naxis_img):
    for j in range(pixels):
        tmp_img = []
        for i in range(naxis_img):
            tmp_img.extend(list_img[naxis_img*k+i][j])
        merg_img.append(tmp_img)
tmp_img = []

np_merg_img = np.array(merg_img)

print('-------------------------------------------------------------------')
print('Saving merged file in numpy array.')
np.save(save_np_name, np_merg_img)

print('-------------------------------------------------------------------')
print('Plotting merged file in png format.')
fig = plt.imshow(np_merg_img)
plt.axis('off'), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.gray()
plt.savefig(save_fig_name, bbox_inches = 'tight', pad_inches = 0, dpi = 300)

print('===================================================================')
print('Program done')
print('===================================================================')

import numpy as np
import math
import time
import os
import matplotlib.pyplot as plt
import cv2 as cv

#######################################################################################
# This script is for evaluating the amout to loss of images (original and denoising). #
# We need to two images, which are the numpy format.                                  #
# You must be saved the images in numpy format before excuted the this script.        #
# Two images must be have exactly same size, which is the width and height.           #
#######################################################################################

# Edited in 03/26/2024

# Functions

def eval_PSNR(ori_img, noise_img, PD_img):
    width = ori_img.shape[0]
    height = ori_img.shape[1]

    corrupt = noise_img - ori_img
    corrupt_val0 = np.square(corrupt)
    corrupt_val1 = np.sum(corrupt_val0) / width*height
    corrupt_val = 10*np.log10(1/corrupt_val1)

    PD = PD_img - ori_img
    PD_val0 = np.square(PD)
    PD_val1 = np.sum(PD_val0) / width*height
    PD_val = 10*np.log10(1/PD_val1)

    return corrupt_val, PD_val

def eval_SSIM(ori_img, noise_img, PD_img):
    width = ori_img.shape[0]
    height = ori_img.shape[1]

    mean_ori = np.mean(ori_img)
    mean_noise = np.mean(noise_img)
    mean_PD = np.mean(PD_img)

    std_ori = np.std(ori_img, ddof=1)
    std_noise = np.std(noise_img, ddof=1)
    std_PD = np.std(PD_img, ddof=1)

    # covariance matix. It is pointless.
    #cov_ori = np.cov(ori_img)
    #cov_noise = np.cov(noise_img)
    #cov_PD = np.cov(PD_img)
    #cov_ori_noise = np.cov(ori_img, noise_img)
    #cov_ori_PD = np.cov(ori_img, PD_img)
    #print(cov_ori_noise.shape, cov_ori_PD.shape, corr_ori_PD.shape)

    cov_noise = np.sum((ori_img - mean_ori)*(noise_img - mean_noise))/(width*height-1)
    cov_PD = np.sum((ori_img - mean_ori)*(PD_img - mean_PD))/(width*height-1)
    #print(cov_noise, cov_PD)

    C1 = (0.01)**2
    C2 = (0.03)**2
    C3 = C2/2

    Lumi_corrupt = (2*mean_ori*mean_noise + C1)/(mean_ori**2 + mean_noise**2 +C1)
    Cont_corrupt = (2*std_ori*std_noise + C2)/(std_ori**2 + std_noise**2 + C2)
    Struct_corrupt = (cov_noise + C3)/(std_ori*std_noise + C3)

    Lumi_PD = (2*mean_ori*mean_PD + C1)/(mean_ori**2 + mean_PD**2 +C1)
    Cont_PD = (2*std_ori*std_PD + C2)/(std_ori**2 + std_PD**2 + C2)
    Struct_PD = (cov_PD + C3)/(std_ori*std_PD + C3)

    SSIM_corrupt = Lumi_corrupt*Cont_corrupt*Struct_corrupt
    SSIM_PD = Lumi_PD*Cont_PD*Struct_PD

    return SSIM_corrupt, SSIM_PD

def eval_M_SSIM(ori_img, noise_img, PD_img):
    width = ori_img.shape[0]
    height = ori_img.shape[1]

    num_image_X = width - 9
    num_image_Y = height - 9

    m_SSIM_corrupt = 0
    m_SSIM_PD = 0

    for i in range(0, num_image_X):
        for j in range(0, num_image_Y):
            ori = ori_img[i:i+10, j:j+10]
            noise = noise_img[i:i+10, j:j+10]
            PD = PD_img[i:i+10, j:j+10]
            
            SSIM_corrupt, SSIM_PD = eval_SSIM(ori, noise, PD)

            m_SSIM_corrupt = m_SSIM_corrupt + SSIM_corrupt
            m_SSIM_PD = m_SSIM_PD + SSIM_PD

    M_SSIM_corrupt = m_SSIM_corrupt / (num_image_X*num_image_Y)
    M_SSIM_PD = m_SSIM_PD / (num_image_X*num_image_Y)

    return M_SSIM_corrupt, M_SSIM_PD


# Inputs

base_dir_name = 'BB-0.36'
ori_img_name = 'merged_image_norm.npy'
noise_img_name = 'merged_image_noise.npy'
PD_img_name = 'merged_image_PD.npy'

eval_opt = 1    # 1: PSNR, 2:SSIM


# Set directory path

base_path1 = os.getcwd()
base_path = os.path.join(base_path1, base_dir_name)

ori_img_path = os.path.join(base_path1, ori_img_name)
noise_img_path = os.path.join(base_path, noise_img_name)
PD_img_path = os.path.join(base_path, PD_img_name)

ori_img = np.load(ori_img_path)
noise_img = np.load(noise_img_path)
PD_img = np.load(PD_img_path)

corrupt_val, PD_val = eval_PSNR(ori_img, noise_img, PD_img)
SSIM_corrupt, SSIM_PD = eval_SSIM(ori_img, noise_img, PD_img)
M_SSIM_corrupt, M_SSIM_PD = eval_M_SSIM(ori_img, noise_img, PD_img)

print('noise' ',' 'prediction')
print('PSNR =', corrupt_val, PD_val)
print('SSIM =', SSIM_corrupt, SSIM_PD)
print('M-SSIM =', M_SSIM_corrupt, M_SSIM_PD)

max_val1 = np.max(ori_img)
min_val1 = np.min(ori_img)
max_val2 = np.max(noise_img)
min_val2 = np.min(noise_img)
#max_val3 = np.max(PD_img)
#min_val3 = np.min(PD_img)
print('max =', max_val1, '/', 'min =', min_val1)
print('max =', max_val2, '/', 'min =', min_val2)
#
#print(max_val1, min_val1, max_val2, min_val2, max_val3, min_val3)
#print(ori_img.shape, noise_img.shape, PD_img.shape)
#
#width = PD_img.shape[0]
#height = PD_img.shape[1]
#result = PD_img[PD_img < 0]
#print('Number of negative value =', len(result))
#print('negative value ratio =', len(result)/(width*height))

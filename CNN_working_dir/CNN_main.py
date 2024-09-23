import numpy as np
import math
import time
import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 as cv

from CNN_utils import dir_check, load_img_data, get_compiled_model

#Getting the time for naming of the model
x = datetime.datetime.now()

#number of pixels in the images
pixels = 64
#number of filters in upper blocks
num_filters = 32
#The CNN model you chose (see CNN_utils.py)
#number 0 is model from the reference paper.
model_num = 1
#number of epochs
num_epochs = 150
#batch size
batch_size = 20
#Drop out rate
drop_out_rate = 0.0
#learning rate
learning_rate = 0.001
#loss (MAE works well)
loss = "MAE"   #choose MAE or SSIM or SSIMm
#Parameter to set if you use SSIM as a loss (see this paper for more info: https://aip.scitation.org/doi/10.1063/5.0054920)
alpha = 0 #for SSIM (1 is pure SSIM, 0 is MAE)
#I name the model for saving
#model_name = x.strftime("%m_%d_%y_%H_%M_%S_")+"Denoising_"+loss+"_alpha_"+str(alpha)+"_LR_"+str(learning_rate)+"_filter_"+str(num_filters)+"_DO_"+str(drop_out_rate)+"_epochs_"+str(num_epochs)+"_model_"+str(model_num)
model_name = x.strftime("%m_%d_%y_%H_%M_%S_")


base_path = os.getcwd()
DS_dir_name = 'Gen_DS/BB_train/' # dataset dir name
DS_path = os.path.join(base_path, DS_dir_name)

train_X_name = 'noise_image/'   # noisy images dir name for training
train_Y_name = 'norm_image/'    # ground truth dir name for training

test_X_name = 'noise_image-val/'   # ground truth dir name for test
test_Y_name = 'norm_image-val/'   # noisy images dir name for test

#Loading the training data.
train_X_path = os.path.join(DS_path, train_X_name)
print('loading train_X')
train_X = load_img_data(train_X_path, pixels)

train_Y_path = os.path.join(DS_path, train_Y_name)
print('loading train_Y')
train_Y = load_img_data(train_Y_path, pixels)

print ("train_X shape is:", train_X.shape)
print ("train_Y shape is:", train_Y.shape)

#Loading the test data. (Must will be finished after.)
test_X_path = os.path.join(DS_path, test_X_name)
print('loading test_X')
test_X = load_img_data(test_X_path, pixels)

test_Y_path = os.path.join(DS_path, test_Y_name)
print('loading test_Y')
test_Y = load_img_data(test_Y_path, pixels)

print ("test_X shape is:", test_X.shape)
print ("test_Y shape is:", test_Y.shape)

#Now I create the model
model = get_compiled_model(model_number = model_num,
                            loss = loss,
                            alpha = alpha,
                            dropout_rate = drop_out_rate,
                            pixels = pixels,
                            learning_rate = learning_rate,
                            num_filters = num_filters)

model.summary()

#Save the model
ckpt_dir_name = 'ckpt_files/'
dir_check(base_path, ckpt_dir_name)
ckpt_path = os.path.join(base_path, ckpt_dir_name)
checkpoint_path = ckpt_path+model_name+'.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)


# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

#Now I fit the model
print("We are fitting the model...")
history = model.fit(train_X,
                    train_Y,
                    epochs=num_epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(test_X, test_Y),
                    callbacks=[cp_callback])

print('Model trained! Save the model in current directory.')
model.save('CNN_model.h5')
print('Model Saved!')

#Then you plot the history
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label = 'Val. loss')
print("history:",history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
#plt.show()
plt.savefig('loss.png', dpi = 600)
#And you save the history
file_name = base_path+x.strftime("%m_%d_%y_%H_%M_%S_")+'loss.txt'
np.savetxt(file_name, history.history['loss'])
file_name = base_path+x.strftime("%m_%d_%y_%H_%M_%S_")+'val_loss.txt'
np.savetxt(file_name, history.history['val_loss'])

#I print the test loss and accuracy, for info.
test_loss, test_acc = model.evaluate(test_X,  test_Y, verbose=2)
print ("test_loss and test_accuracy are:")
print (test_loss, test_acc)

print('Program done')

#print('======================================================')
#print('loading test')
#print('------------------------------------------------------')
#ref_np_file_name = '2.npy'
#ref_np_file_path = os.path.join(train_X_path, ref_np_file_name)
#ref_np_file = np.load(ref_np_file_path)
#print(np.all(ref_np_file == train_X[1,:,:,0]))
#print('======================================================')


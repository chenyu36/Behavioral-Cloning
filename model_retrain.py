#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from resizeimage import resizeimage
from PIL import Image
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU, ELU
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import math

def shift_image_pixel(image, steer, trans_range):
    """ Shift the image a certain pixel distance lift or right 
    according to the steer parameter
    each pixel unit maps to 0.004 shift in steering angle
    """    
    # Shape of image
    rows, cols = image.shape[:2]
#     print('height is {}'.format(rows))
#     print('width is {}'.format(cols))
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_angle = steer + tr_x/trans_range*2*.2
#     print('pixel shift is {}'.format(tr_x))
#     print('angle shift is {}'.format(tr_x/trans_range*2*.2))
    tr_y = 40*np.random.uniform()-40/2
#    tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols, rows))
    
    return image_tr, steer_angle

def add_random_brightness(image_data):
    """ 
    Randomly adjust brightness of the image_data in the range of 0.25 to 1.0  
    """  
    # Convert to HSV from RGB
    hsv = cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV)
    # Generate random number to apply to brightness channel
    rand = random.uniform(0.25,1.0)
    hsv[:,:,2] = rand*hsv[:,:,2]
    # Convert back to RGB colorspace
    processed_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return processed_img 

# Crop the sky and car hood
def crop_image(image_data):
    """ 
    crop out the sky and the car hood to simplify the images for learning
    """  
    # Convert to HSV from RGB
    shape = image_data.shape
    cropped_img = image_data[CROP_PIXEL_FROM_TOP:shape[0]-CROP_PIXEL_FROM_BOTTOM,:]
    return cropped_img

def flip_image(img, angle):
    """ 
    flip (mirror) the image along the vertical axis
    """      
    flipped_img = cv2.flip(img,1)
    angle = -angle
    return flipped_img, angle

def select_image_and_process(index, LR_shift = 0.20):
    """ 
    process the indexed image with all the preprocessing methods
    1. randomly add brightness
    2. shift the pixel in x and y direction while adjusting steering angle
    3. crop out the sky and the car hood
    4. resize the image
    """     
    # pick a random image from one of the cameras: (left, center, right)
    # and add/subtract angles if necessary
    random_choice = np.random.randint(3)
    if (random_choice == 0):
        img_file = 'data/' + driving_log['left'].str.strip()
        shift_ang = LR_shift
    if (random_choice == 1):
        img_file = 'data/' + driving_log['center'].str.strip()
        shift_ang = 0.
    if (random_choice == 2):
        img_file = 'data/' + driving_log['right'].str.strip()
        shift_ang = -LR_shift 
#     print('random choice is {}'.format(random_choice))
    
#     print('file index is {}'.format(index))
#     print('file name is {}'.format(img_file[index]))    
    # create Numpy arrays of input data
    # and labels, using each row in the csv file
    x = cv2.imread(img_file[index])
    x = cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
    y = np.array(steering_angles[index])
#     print('steering angle is {}'.format(y))  
    x = add_random_brightness(x)
    x, y = shift_image_pixel(x, y, 100)
    x = crop_image(x)
        
    x = cv2.resize(x, (RESIZE_IMAGE_W, RESIZE_IMAGE_H), interpolation=cv2.INTER_AREA)
    y = np.add(y, shift_ang)
    
    # randomly fiip the image
    should_flip = np.random.randint(2)
    if should_flip==1:
        x, y = flip_image(x, y)

    return x, y # x: processed image, y: processed angle

def data_generator_with_batch(path, batch_size):
    """ 
    generate the data for training
    """         
    x_batch = np.zeros((batch_size, RESIZE_IMAGE_H, RESIZE_IMAGE_W, n_channels), dtype = np.float32)
    y_batch = np.zeros((batch_size,), dtype = np.float32)
    drop_threshold_for_low_angles = 0.33
    epoch_i = 1
    counter = 0
    while 1:
        for i in range(batch_size):
            if (counter < (n_split_train_data*3)):
                counter += 1
            else:
                # 1 epoch
                counter = 0
                epoch_i += 1
                print('\n drop_threshold_for_low_angles is {}'.format(1 / epoch_i))

            # threshold starts out as 1, i.e. chance of dropping small angles is high in early rounds
            # threshold decreases as iteration of epoch progresses
            drop_threshold_for_low_angles = 0.33 / epoch_i 
            keep_pr = 0
            while keep_pr == 0:
                random_pick = np.random.randint(n_train_data)
                x_batch[i], y_batch[i] = select_image_and_process(random_pick, LR_shift = 0.26)
                # if the angle is too small, we may need to drop this training data so that
                # it has less influence on the learning
                if abs(y_batch[i]) < 0.1:
                    pr_val = np.random.uniform()
                    if pr_val > drop_threshold_for_low_angles:
                        keep_pr = 1
                else:
                    keep_pr = 1
        yield x_batch, y_batch


# Getting the output (i.e. steering_angle) from the recorded data
user_data = False
if (user_data == True):
  csv_file_path = 'driving_log.csv'
else:
  csv_file_path = 'data//driving_log.csv'

driving_log = pd.read_csv(csv_file_path)
steering_angles = driving_log['steering']
steering_angles = np.array(steering_angles)
n_train_data = steering_angles.shape[0]

if user_data == True:
  image_file_names = driving_log['center']
else:
  image_file_names = 'data//' + driving_log['center']

print('number of train data is {}'.format(n_train_data))


# Define the model with Keras #
nb_filters = 32
nb_filters_2 = 64
nb_filters_3 = 128
kernel_size_w = 3
kernel_size_h = 3
dropout = 0.5
learning_rate = 0.0001
EPOCHS = 4 
img_width, img_height = 320, 160
n_channels = 3  # color image has 3 channels
RESIZE_IMAGE_W = 200
RESIZE_IMAGE_H = 66
batch_size = 128
CROP_PIXEL_FROM_TOP = 60
CROP_PIXEL_FROM_BOTTOM = 25

# Define the split if valiation set is used #
split = 0.1
n_val_data = int(n_train_data*split)
n_split_train_data = n_train_data - n_val_data

print('shape of steering_angles is {}'.format(steering_angles.shape))
input_shape = (RESIZE_IMAGE_H, RESIZE_IMAGE_W, n_channels)
model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5, input_shape = input_shape))
model.add(Convolution2D(nb_filters, kernel_size_w, kernel_size_h, border_mode='valid', subsample =(2,2)))
model.add(Dropout(dropout))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters_2, kernel_size_w, kernel_size_h, border_mode='valid', subsample =(2,2)))
model.add(Dropout(dropout))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters_3, kernel_size_w, kernel_size_h, border_mode='valid', subsample =(2,2)))
model.add(Dropout(dropout))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(dropout))
model.add(ELU())   # add an advanced activation
model.add(Dense(64))
model.add(ELU())   # add an advanced activation
model.add(Dense(16))
model.add(ELU())   # add an advanced activation
model.add(Dense(1))
adam = Adam(lr = learning_rate)
model.load_weights('model.h5')
model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])


# Train the Model #
# by feeding the fit_generator using the data generator
# since we are going to use all 3 camera's data, we multiple the number of train data by 3
# hence: samples_per_epoch=(3*n_split_train_data)       
train_data_generator = data_generator_with_batch(csv_file_path, batch_size)
model.fit_generator(train_data_generator, samples_per_epoch=(3*n_split_train_data), nb_epoch=EPOCHS, initial_epoch=0)


# Save the Model #
# serialize model to JSON and save the model's weights
model_name = 'model.json'
model_json = model.to_json()
with open(model_name, "w") as json_file:
    json_file.write(model_json)
    print("model saved to disk as {}".format(model_name))
model_weight_name = 'model.h5'
model.save_weights(model_weight_name)  # save the weights after training or during training
print("model weight saved to disk as {}".format(model_weight_name))






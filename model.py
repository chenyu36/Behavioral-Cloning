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
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import math

n_channels = 3 # color image has 3 channels



def driving_acc(y_true, y_pred, threshold = 0.5/25.):
    """
    Shows what percentage of the batch results are within 0.5 deg of true data
	
    Author: Thomas Antony
    """
    diff = K.abs(y_true - y_pred)
    good_rows = K.lesser(diff, threshold)
    good = K.sum(K.cast(good_rows, K.floatx()))
    total = K.sum(K.ones_like(y_true))
    return good*100/total

def add_random_brightness(image_data):
    # Convert to HSV from RGB
    hsv = cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV)
    # Generate random number to apply to brightness channel
    rand = random.uniform(0.3,1.0)
    hsv[:,:,2] = rand*hsv[:,:,2]
    # Convert back to RGB colorspace
    processed_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return processed_img 

    
def convert_image(image_data):
    # convert to yuv color space
    yuv = cv2.cvtColor(image_data, cv2.COLOR_BGR2YUV)
    # split into 3 channels
    y, u, v = cv2.split(yuv)
    # Contrast Limited Adaptive Histogram Equalization: CLAHE
    # http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
    # only apply CLAHE to the y channel
    contrastBoostedImg = clahe.apply(y)
    
    global n_channels
    n_channels = 1
    return contrastBoostedImg    

def normalize_image(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [a, b]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    a = -0.5
    b = 0.5
    channel_min = 0
    channel_max = 255
    return a + ( ( (image_data - channel_min)*(b - a) )/( channel_max - channel_min ) )
    
def resize_image(img):
    image = resizeimage.resize_width(img, RESIZE_IMAGE_W)
    return image

# parameters for the model
EPOCHS = 3 
img_width, img_height = 320, 160
RESIZE_IMAGE_W = 64
RESIZE_IMAGE_H = 64
batch_size = 32
user_data = False
n_split_train_data = 0 # will be determined at run time based on actual number of samples
n_val_data = 0         # will be determined at run time based on actual number of samples

if (user_data == True):
  csv_file_path = 'driving_log.csv'
else:
  csv_file_path = 'data//driving_log.csv'

# this is the output labels for each frame
driving_log = pd.read_csv(csv_file_path)
steering_angles = driving_log['steering']
steering_angles = np.array(steering_angles)
n_train_data = steering_angles.shape[0]
print('number of train data is {}'.format(n_train_data))

if user_data == True:
  image_file_names = driving_log['center']
else:
  image_file_names = 'data//' + driving_log['center']




y_train = []
x_train = []
# get data from disk
for index in range(n_train_data):
  #global y_train
  y = np.array(steering_angles[index])
  y_train.append(y)
  #global x_train
  x = cv2.imread(image_file_names[index],cv2.IMREAD_COLOR)
  x = add_random_brightness(x)
## test print the image
#   if (index == 1):
#     cv2.imshow('test train image',x)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
  x = cv2.resize(x, (RESIZE_IMAGE_W, RESIZE_IMAGE_H))
  x_train.append(x)


x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
print('shape of x_train is {}'.format(x_train.shape))
print('shape of y_train is {}'.format(y_train.shape))
print('x_train[0] is {}'.format(x_train[0]))

# split the data to training set and validation set
# x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size = 0.1, random_state = 0)

print('length of x_train is {}'.format(len(x_train)))
print('length of y_train is {}'.format(len(y_train)))
# print('length of x_validation is {}'.format(len(x_validation)))
# print('length of y_validation is {}'.format(len(y_validation)))
print('type of x_train is {}'.format(type(x_train)))
print('type of y_train is {}'.format(type(y_train)))

# shuffle the training data
x_train, y_train = shuffle(x_train, y_train)


   
# define the model with Keras
nb_filters = 16
nb_filters_2 = 24
nb_filters_3 = 48
kernel_size_w = 3
kernel_size_h = 3
dropout = 0.5
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
model.add(Dense(128))
model.add(Dropout(dropout))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer=Adam(lr = 0.0001), metrics=['accuracy'])
# history = model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=EPOCHS, validation_split=0.2)


def data_generator_with_batch(path, batch_size, split=0.2, data='train'):
    global n_val_data, n_split_train_data
    n_val_data = int(n_train_data*split)
    n_split_train_data = n_train_data - n_val_data

    # calculate number of batches
    if data =='train':
        number_of_batch = math.ceil(n_split_train_data/batch_size)
        offset = 0
        number_of_samples = n_split_train_data
    else:
        number_of_batch = math.ceil(n_val_data/batch_size)
        offset = n_split_train_data
        number_of_samples = n_val_data
    # calculate the last batch size
    if (number_of_samples % batch_size == 0):
        last_batch_size = batch_size
    else:
        last_batch_size = (number_of_samples % batch_size)

    while 1:         
        # shuffle the array read from csv
        global image_file_names, steering_angles
        image_file_names, steering_angles = shuffle(image_file_names, steering_angles)

        for batch_i in range(number_of_batch):
            x_batch = []
            y_batch = []

            # check if we are not in last batch
            if (batch_i < (number_of_batch - 1)):
                for i in range(batch_size):
                    # create Numpy arrays of input data
                    # and labels, using each row in the csv file
                    x = cv2.imread(image_file_names[offset + i + batch_i*batch_size],cv2.IMREAD_COLOR)
                    if data =='train':
                        x = add_random_brightness(x)
                    x = cv2.resize(x, (RESIZE_IMAGE_W, RESIZE_IMAGE_H))
                    y = np.array(steering_angles[offset + i + batch_i*batch_size])
                    x_batch = np.append(x_batch, x)
                    y_batch = np.append(y_batch, y)
                    x_batch = np.asarray(x_batch)
                    y_batch = np.asarray(y_batch)
                X = x_batch.reshape(batch_size,RESIZE_IMAGE_H, RESIZE_IMAGE_W,n_channels)
                Y = y_batch.reshape(batch_size,1)
                yield X, Y
            else:
                # the last batch
                for i in range(last_batch_size):
                    # create Numpy arrays of input data
                    # and labels, using each row in the csv file
                    x = cv2.imread(image_file_names[offset + i + batch_i*batch_size],cv2.IMREAD_COLOR)
                    if data =='train':
                        x = add_random_brightness(x)
                    x = cv2.resize(x, (RESIZE_IMAGE_W, RESIZE_IMAGE_H))
                    y = np.array(steering_angles[offset + i + batch_i*batch_size])
                    x_batch = np.append(x_batch, x)
                    y_batch = np.append(y_batch, y)
                    x_batch = np.asarray(x_batch)
                    y_batch = np.asarray(y_batch)
                X = x_batch.reshape(last_batch_size,RESIZE_IMAGE_H, RESIZE_IMAGE_W,n_channels)
                Y = y_batch.reshape(last_batch_size,1)
                yield X, Y
        


train_generator = data_generator_with_batch(csv_file_path, batch_size)
validation_generator = data_generator_with_batch(csv_file_path, batch_size, data='validation')
print(next(train_generator))
print(next(validation_generator))
model.fit_generator(train_generator, samples_per_epoch=n_split_train_data, 
    nb_epoch=EPOCHS, validation_data=validation_generator, nb_val_samples=n_val_data)

# serialize model to JSON
model_name = 'model.json'
model_json = model.to_json()
with open(model_name, "w") as json_file:
    json_file.write(model_json)
    print("model saved to disk as {}".format(model_name))
model_weight_name = 'model.h5'
model.save_weights(model_weight_name)  # save the weights after training or during training
print("model weight saved to disk as {}".format(model_weight_name))






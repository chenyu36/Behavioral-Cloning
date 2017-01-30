import numpy as np
import pandas as pd
from resizeimage import resizeimage
from PIL import Image
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
import random
from keras import backend as K
from sklearn.utils import shuffle

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


# dimensions of our images.
img_width, img_height = 320, 160
RESIZE_IMAGE_WIDTH = 32
RESIZE_IMAGE_HEIGHT = 16
csv_file_path = 'data//driving_log.csv'

# this is the output labels for each frame
driving_log = pd.read_csv(csv_file_path)
#steering_angles = pd.read_csv(csv_file_path, header=None, usecols=[3])
steering_angles = driving_log['steering']
steering_angles = np.array(steering_angles)
n_train_data = steering_angles.shape[0]
print('number of train data is {}'.format(n_train_data))

# file name of each image
# image_file_names = driving_log['center']
#image_file_names = pd.read_csv(csv_file_path, header=None, usecols=[0])
#image_file_names = np.array(image_file_names)
image_file_names = 'data//' + driving_log['center']
#print('image file name 0 is {}'.format(image_file_names[0][0])) #[row][column]
#print(type(image_file_names))
#print('shape of image_file_names is {}'.format(image_file_names.shape))

   
# define the model with Keras
nb_filters = 32
nb_filters_2 = 16
kernel_size_w = 3
kernel_size_h = 3
dropout = 0.25
print('shape of steering_angles is {}'.format(steering_angles.shape))
input_shape = (RESIZE_IMAGE_HEIGHT, RESIZE_IMAGE_WIDTH, 3)
model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size_w, kernel_size_h, border_mode='valid', input_shape = input_shape))
model.add(Dropout(dropout))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters_2, kernel_size_w, kernel_size_h, border_mode='valid'))
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
model.add(Dense(1))

# model.compile(loss='mse', optimizer='adam', metrics=[driving_acc])
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])


def resize_image(img):
    image = resizeimage.resize_width(img, RESIZE_IMAGE_WIDTH)
    return image

# Python data generator
def generate_arrays_from_file(path):
    while 1:
        # driving_log = pd.read_csv(csv_file_path)
        # image_file_names = driving_log['center']
        # steering_angles = driving_log['steering']
        # steering_angles = np.array(steering_angles)
        # c = list(zip(image_file_names, steering_angles))
        # random.shuffle(c)
        # image_file_names, steering_angles = zip(*c)
        # print('data shuffled')
#         for index, filename in enumerate(image_file_names):
#             # create Numpy arrays of input data
#             # and labels, using each row in the csv file
#             image = Image.open(filename)
#             image = resize_image(image)
#             x = img_to_array(image)
#             y = np.array(steering_angles[index])
#             y = steering_angles[index]
# 
#             yield x.reshape(1,RESIZE_IMAGE_HEIGHT, RESIZE_IMAGE_WIDTH,3), y.reshape(1)        
        for index in range(n_train_data):
            random_i = random.randint(0, n_train_data-1)
            # create Numpy arrays of input data
            # and labels, using each row in the csv file
            image = Image.open(image_file_names[random_i])
            image = resize_image(image)
            x = img_to_array(image)
            #y = np.array(steering_angles[index])
            y = steering_angles[random_i]
            x = x.reshape(1,RESIZE_IMAGE_HEIGHT, RESIZE_IMAGE_WIDTH,3)
            y = y.reshape(1)

            yield x, y
        x, y = shuffle(x, y)
        print('data shuffled')
        

generator = generate_arrays_from_file(csv_file_path)
print(next(generator))
model.fit_generator(generator, samples_per_epoch=n_train_data, nb_epoch=3)
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')  # save the weights after training or during training
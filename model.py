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
import cv2
from sklearn.model_selection import train_test_split

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
    
    global num_of_channels
    num_of_channels = 1
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
    image = resizeimage.resize_width(img, RESIZE_IMAGE_WIDTH)
    return image

# dimensions of our images.
img_width, img_height = 320, 160
RESIZE_IMAGE_WIDTH = 64
RESIZE_IMAGE_HEIGHT = 32
recovery_data = False

if (recovery_data == True):
  csv_file_path = 'driving_log.csv'
else:
  csv_file_path = 'data//driving_log.csv'

# this is the output labels for each frame
driving_log = pd.read_csv(csv_file_path)
#steering_angles = pd.read_csv(csv_file_path, header=None, usecols=[3])
steering_angles = driving_log['steering']
steering_angles = np.array(steering_angles)
n_train_data = steering_angles.shape[0]
print('number of train data is {}'.format(n_train_data))


if recovery_data == True:
  image_file_names = driving_log['center']
else:
  image_file_names = 'data//' + driving_log['center']

y_train = []
x_train = []


# get data from disk
for index in range(n_train_data):
  global y_train
  y = np.array(steering_angles[index])
  y_train.append(y)
  global x_train
  x = cv2.imread(image_file_names[index],cv2.IMREAD_COLOR)
  x = convert_image(x)
## test print the image
#   if (index == 1):
#     cv2.imshow('test train image',x)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
  x = cv2.resize(x, (RESIZE_IMAGE_WIDTH, RESIZE_IMAGE_HEIGHT))
  x = normalize_image(x.reshape(RESIZE_IMAGE_HEIGHT, RESIZE_IMAGE_WIDTH, 1))
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
input_shape = (RESIZE_IMAGE_HEIGHT, RESIZE_IMAGE_WIDTH, 1)
model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size_w, kernel_size_h, border_mode='valid', input_shape = input_shape))
model.add(Dropout(dropout))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters_2, kernel_size_w, kernel_size_h, border_mode='valid'))
model.add(Dropout(dropout))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters_3, kernel_size_w, kernel_size_h, border_mode='valid'))
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
model.add(Activation('linear'))
# model.compile(loss='mse', optimizer='adam', metrics=[driving_acc])
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=32, nb_epoch=10, validation_split=0.2)



# Python data generator
def generate_arrays_from_file(path):
    while 1:
        driving_log = pd.read_csv(csv_file_path)
        image_file_names = driving_log['center']
        steering_angles = driving_log['steering']
        for index, filename in enumerate(image_file_names):
            # create Numpy arrays of input data
            # and labels, using each row in the csv file
            image = Image.open(filename)
            image = resize_image(image)
            x = img_to_array(image)
            y = np.array(steering_angles[index])
            y = steering_angles[index]
            x = x.reshape(1,RESIZE_IMAGE_HEIGHT, RESIZE_IMAGE_WIDTH,3)
            y = y.reshape(1)
            yield x, y
        x, y = shuffle(x, y)
        print('data shuffled')

        

# generator = generate_arrays_from_file(csv_file_path)
# print(next(generator))
# model.fit_generator(generator, samples_per_epoch=n_train_data, nb_epoch=3)




# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')  # save the weights after training or during training
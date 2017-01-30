import numpy as np
import pandas as pd
from PIL import Image
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D

# dimensions of our images.
img_width, img_height = 320, 160
csv_file_path = 'data/driving_log.csv'
driving_log = pd.read_csv('data/driving_log.csv')
# this is the output labels for each frame
steering_angles = driving_log['steering']
n_train_data = len(steering_angles)

# file name of each image
image_file_names = 'data/' + driving_log['center']
print('image file name 0 is {}'.format(image_file_names[0]))
print('number of train data is {}'.format(n_train_data))
print('shape of label is {}'.format(driving_log.shape[0]))
   
# define the model with Keras
nb_filters = 32
kernel_size_w = 3
kernel_size_h = 3
print('shape of steering_angles is {}'.format(steering_angles.shape))
input_shape = (img_height, img_width, 3)
model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size_w, kernel_size_h, border_mode='valid', input_shape = input_shape))
#model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=input_shape))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=["accuracy"])

# Python data generator
def generate_arrays_from_file(path):
    while 1:
        for index in range(n_train_data):
            # create Numpy arrays of input data
            # and labels, using each row in the csv file
            image = Image.open(image_file_names[index])
            x = img_to_array(image)
            #x2 = imresize(x, (64, 64))
            #print('x shape is {}'.format(x.shape))
            y = np.array(steering_angles[index])
            #yield x2.reshape(1,64,64,3), y.reshape(1)
            yield x.reshape(1,160,320,3), y.reshape(1)

generator = generate_arrays_from_file(csv_file_path)
print(next(generator))
model.fit_generator(generator, samples_per_epoch=n_train_data, nb_epoch=10)
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')  # save the weights after training or during training
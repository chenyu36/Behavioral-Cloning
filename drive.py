import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model
import cv2
import time


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

RESIZE_IMAGE_WIDTH = 200
RESIZE_IMAGE_HEIGHT = 66
CROP_PIXEL_FROM_TOP = 60
CROP_PIXEL_FROM_BOTTOM = 25
start_time_small_angles = time.time()
small_angles = False
small_angle_time_threshold = 1.5

# Crop the sky and car hood


def crop_image(image_data):
    shape = image_data.shape
    cropped_img = image_data[
        CROP_PIXEL_FROM_TOP:shape[0] - CROP_PIXEL_FROM_BOTTOM, :]
    return cropped_img


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        # crop the image the same way as in the model
        image_array = crop_image(image_array)
        # resizing the image the same way as in the model
        image_array = cv2.resize(
            image_array, (RESIZE_IMAGE_WIDTH, RESIZE_IMAGE_HEIGHT))
        transformed_image_array = image_array[None, :, :, :]
        # This model currently assumes that the features of the model are just the
        # images. Feel free to change this.
        steering_angle = float(model.predict(
            transformed_image_array, batch_size=1))
        # The driving model currently just outputs a constant throttle. Feel free
        # to edit this.

        angle = abs(steering_angle)
        if angle > 0.5:
            throttle = 0.12
            global small_angles
            small_angles = False
        elif 0.18 <= angle <= 0.5:
            throttle = 0.18
            global small_angles
            small_angles = False
        elif 0.15 <= angle < 0.18:
            throttle = 0.20
            global small_angles
            small_angles = False
        else:
            if small_angles == False:
                print('straighter course: steering angle < 0.15')
                global small_angles
                small_angles = True
                global start_time_small_angles
                start_time_small_angles = time.time()
            now = time.time()
            # if we are in stright line for 1.5 seconds, allow changing to higher
            # throttle
            if (now - start_time_small_angles > small_angle_time_threshold):
                throttle = 0.35

        print(steering_angle, throttle)
        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
                timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
                image_filename = os.path.join(args.image_folder, timestamp)
                image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

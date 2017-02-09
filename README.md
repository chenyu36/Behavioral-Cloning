#P3 Behavior Cloning
The objective of this project is to clone the human's driving behavior using the car simulator and the convolutional neural network (CNN). To collect data to train the CNN, we use the simulator to record frame-by-frame images of the track and the corresponding steering angles input by the driver. A properly trained CNN is expected to use each frame as an input and predicts the steering angle as an output. As a result, the car in the simulator can continuously drive itself on the road without going off-track in the autonomous mode.

##Model Architecture
The color images from the dataset has the dimesion of 320x160 (WxH). After preprocessing, the image is resized to 220x60 (WxH). 220x60 is chosen based on Nvidia's paper [End to End Learning for Self-Driving Cars](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0ahUKEwjJ6sjqw__RAhXL64MKHa5NBd4QFggcMAA&url=https%3A%2F%2Fimages.nvidia.com%2Fcontent%2Ftegra%2Fautomotive%2Fimages%2F2016%2Fsolutions%2Fpdf%2Fend-to-end-dl-using-px.pdf&usg=AFQjCNGgCrFq0dg2NHSt-N0gi9ult70wig&sig2=C-E6D6aB57ozFW6uH0eYUw).  The architecture was chosen similar to the one used in Nvidia's paper. Instead of 5 convolutional layers, I use 3 for simplicity and thought it could speed up the training due to reduced amount of calculation.
The first layer, implemented with a Lambda function, performs normalization so that each color channel has the value between -0.5 and 0.5. (model.py lines 196) It is followed by 3 convolutional layers with 3x3 kernel, with the depths between 32 and 128 (model.py lines 197-203). The output from the last convolutional layer is flattened, and fed to 3 fully connected layers(512 -> 64  -> 16). After each convolutional layer, a dropout layer is used to reduced overfitting and a ReLU is used as the activation function to introduce nonlinearity. (model.py lines 198-204) In between the fully conneted layers, ELU is used. (model.py lines 209-213) [ELU](http://arxiv.org/pdf/1511.07289v1.pdf) has been shown to enable the network learn faster. The final output is the steering angle to control the car in the simulator. Since the steering angles vary continuously according to the track's curvature, the problem is a regression one. The Adam optimizer is used in the model, while a learning rate of 0.0001 is explictly specified. Here is a visualization of the architecture.

## Training Strategy
### Data selection
#### Udacity data
The selection of data is critical to the success of the training. Although training mode is available in the simulator to collect data, Udacity provides its data for the students as the starting point. Its data was recorded by 3 cameras: left, center and right. At first glance, it is not clear whether such data is sufficient to train the model. However, after converting the center image sequence to video and play it back, it appears the data consists of balanced amount of data of the car driving in the normal direction of the track and the opposite one.
Here is the Udacity data converted as video:
![Udacity data](/video/UdacityDataCenterImagesOnly.mov).
####User data
I recorded my own data by driving in the normal direction for 4 laps with a [PS3 controller](https://www.walmart.com/ip/Bluetooth-Wireless-Black-Game-Controller-for-PlayStation-3-PS3-USB-Wired-Available/50175929). I used the "Default Mac desktop Universal" simulator to record 3 camera data streams. In the initial comparison, under the same training condition, the model that trains with the User data does not perform as well as the one that trains with Udacity data. Therefore, I decided to use Udacity data to continue the training.

###How the dataset was generated
####Python generator
My first step is to learn how Python generator works since it was mentioned in the project specification. This proves to be a crucial method because of the amount of data required for training. The entire dataset is effectively divided into several batches, with only one batch residing in the memory at a time. For example, if there are 12800 images in the dataset, it will take 100 batches with a batch size of 128 (12800 = 128 * 100) to complete 1 epoch. With a batch size of 128, the Python generator yields 128 images to the CNN for training at a time.
####Early attemps
The early attemps were not successful.  I had no preprocessing other than greyscaling. I only utilized images from the center camera. The car went off the track without passing the first turn.
####Preprocessing and augmentation
Learning from the Nvidia's paper and other students in the Slack Carn-ND channel, I apply preprocesing and augmentation to the data. These steps involves:

* Brightness: randomly changing the brightness of the image in the range from 25% to 100%. (model.py line 44) This is meant to train the model so that it is invariant to scene's brightness.
   Example of brightness augmentation:
   
   <p align="center">Before</p>
   ![Before](images/add_random_brightness-before.jpg)
 <p align="center">After (brightness ratio = 0.635)</p>
   ![After](images/add_random_brightness-after.635.jpg)
* Cropping: cropping out the sky and the car hood in the image. This greatly reduces the complexity of the data and speeds up training. (model.py line 52)
   Example of cropping:
   
   <p align="center">Before</p>
   ![Before](images/crop_image-before.jpg)
 <p align="center">After (Image dimension = 320 x 75 (WxH))</p>
   ![After](images/crop_image-after.jpg)
* Augmenting data from left and right cameras: Images from the left and right cameras can simulate the view when car is not driving in the center of the track. By adding or subtracting an offset from the steering angle data of the center camera, we can generate a pair of input and output data using images from either the left or right camera. The offset is chosen emprically in the range of 0.15 - 0.26. For the purpose of simulating the recovery, we subtract the offset from the steering angle, when using the left camera's image. Similarly, we add the offset to the steering angle, when using the right camera's image. The selection of which camera's image to be augmented is random. (model.py lines 80-89)
* Pixel shifting: by shifting the pixel to the left or right, we can simulate the change in camera's view point. (model.py line 16) This augmented data is useful in further training the model for recovery. For every pixel shifting to the left, we subtract 0.004 steering angle unit from the given angle data. Similarly, for every pixel shifting to the right, we add 0.004 steering angle unit to the given angle data.









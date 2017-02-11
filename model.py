import pandas as pd
from random import uniform, randint
import os
import shutil
import cv2
import matplotlib.pyplot as plt
from numpy.random import random
import numpy as np
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers import Convolution2D
from keras.optimizers import Adam
#from keras.utils.visualize_util import plot
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.regularizers import l2
import json

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


# Read Udacity recorded data of track 1 which will be used for model training

df = pd.read_csv('./data/driving_log.csv') # Reading data csv file
steering_data_angles = df.steering # steering angles data for recorded images
center_images = df.center       # images from car center camera
left_images = df.left           # images from car left camera
right_images = df.right         # images from car right camera
    
    

# train features(images) generator for keras.fit_generator function

def gen_features_images(batch_size=128):
    
    while True:
        
        features = []   # train features (generated images)
        labels = []     # train labels (generaed steering angles)
        
        # Creating batch of generated features
        while len(features) < batch_size:
            
            
            
            # Randomly pick image and add it if it has left or right steering angle
            # Avoiding images with zero steering angle
            
            random_index  = randint(0,len(steering_data_angles)-1)
            
            if steering_data_angles[random_index] < 0 :
                
                image = cv2.imread('./data/' + center_images[random_index],1)
                image = cv2.resize(image[65:135,:],(64,64),interpolation=cv2.INTER_AREA)
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                features.append(image)
                labels.append(steering_data_angles[random_index])
            
  
            
            if steering_data_angles[random_index] > 0 :
                
                image = cv2.imread('./data/' + center_images[random_index],1)
                # Cropping and resizing image
                image = cv2.resize(image[65:135,:],(64,64),interpolation=cv2.INTER_AREA)
                
                # Convert image to HSV planes before adding it to the batch
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                features.append(image)
                labels.append(steering_data_angles[random_index])
                
                
            ##### Using left and right camera images 
            # Avoiding images with zero steering angle
            random_index  = randint(0,len(steering_data_angles)-1)
            
            if steering_data_angles[random_index] != 0 :
                
                #Add a small angle .3 to the left camera
                
                image = cv2.imread('./data/'+left_images[random_index][1:],1)
                image = cv2.resize(image[65:135,:],(64,64),interpolation=cv2.INTER_AREA)
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                features.append(image)
                labels.append(steering_data_angles[random_index] + 0.3)
                
                
                #Subtract angle of 0.3 from the right camera
                
                image = cv2.imread('./data/' + right_images[random_index][1:],1)
                image = cv2.resize(image[65:135,:],(64,64),interpolation=cv2.INTER_AREA)
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                features.append(image)
                labels.append(steering_data_angles[random_index] - 0.3)
            
            
            # Randomly choose image and add flipped version of it
            random_index  = randint(0,len(steering_data_angles)-1)
            
            if steering_data_angles[random_index] != 0 :
                
                image = cv2.imread('./data/' + center_images[random_index],1)
                flipped_image = cv2.flip(image,1)
                flipped_image = cv2.resize(image[65:135,:],(64,64),interpolation=cv2.INTER_AREA)
                
                flipped_image = cv2.cvtColor(flipped_image, cv2.COLOR_BGR2HSV)
                features.append(flipped_image)
                labels.append(-steering_data_angles[random_index])
                
            
            
            # Brightness augmentation
            random_index  = randint(0,len(steering_data_angles)-1)
            
            if steering_data_angles[random_index] != 0 :
                
                image = cv2.imread('./data/' + center_images[random_index],1)
                image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
                random_bright = .25+np.random.uniform()
                image[:,:,2] = image[:,:,2]*random_bright
                image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
                image = cv2.resize(image[65:135,:],(64,64),interpolation=cv2.INTER_AREA)
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                features.append(image)
                labels.append(steering_data_angles[random_index])
        
            
            # Do random shifting horizontal and vertical 
            random_index  = randint(0,len(steering_data_angles)-1)
    
            if steering_data_angles[random_index] != 0 :
                
                image = cv2.imread('./data/' + center_images[random_index],1)
                image = image[65:135,:]
                
                shift_x = 65*np.random.uniform()-65/2
                shift_y = 20*np.random.uniform()-20/2
                
                shift = np.float32([[1,0,shift_x],[0,1,shift_y]])
                
                rows = image.shape[0]
                cols = image.shape[1]
                
                image = cv2.warpAffine(image,shift,(cols,rows))
                
                image = cv2.resize(image,(64,64),interpolation=cv2.INTER_AREA)
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                features.append(image)
                labels.append(steering_data_angles[random_index] + shift_x/65*.4 )
            
            
        features = features[0:batch_size]
        labels = labels[0:batch_size]
        
        # shuffle data before sending to the fit generator function
        features, labels, = shuffle(features, labels, random_state=0)

        # yield the batch
        yield (np.array(features), np.array(labels))
            

# build model (Nvidia model) with input image shape 64x64x3     
def build_model(dropout=.5):
    
    # build sequential model
    model = Sequential()
    
    # input image shape
    img_shape = (64, 64, 3)
    
    # normalisation layer
    model.add(Lambda(lambda x: x * 1./127.5 - 1,
                     input_shape=(img_shape),
                     output_shape=(img_shape),
                     name='Normalization'))
    
    
    # convolution layers with dropout
    nb_filters = [24, 36, 48, 64, 64]
    kernel_size = [(5, 5), (5, 5), (5, 5), (3, 3), (3, 3)]
    strides = [(2, 2), (2, 2), (2, 2), (1, 1), (1, 1)]
    
    
    for filt in range(len(nb_filters)):
        
        model.add(Convolution2D(nb_filters[filt],
                                kernel_size[filt][0], kernel_size[filt][1],
                                border_mode='valid',
                                subsample=strides[filt],
                                activation='elu'))
        
        #Add Dropout
        model.add(Dropout(dropout))

    
    # flatten
    model.add(Flatten())
    
    
    # fully connected layers with dropout
    fc = [100, 50, 10]
    
    for f in range(len(fc)):
        model.add(Dense(fc[f], activation='elu'))
        model.add(Dropout(dropout))
    
    
    # steering angle output
    model.add(Dense(1, activation='elu', name='Out'))
          
          
    # Adam optimizer
    optimizer = Adam(lr=0.001)
    
    # model compile
    model.compile(optimizer=optimizer,
                  loss='mse')
            
            
    return model       
        



# Create Nvidia model     
model = build_model()
    
# print model summary
print(model.summary())
    
    
# train model using keras fit_generator
model.fit_generator(gen_features_images(batch_size= 128 ),
                    samples_per_epoch=100000,
                    nb_epoch=10)


# Save model
model.save('model.h5')

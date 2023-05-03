import tqdm
import time
import random
import pathlib
import itertools
import collections

import cv2
import einops
import numpy as np
import remotezip as rz
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os
#This is to stop the video file corrupting when stopping an
#inferance run
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
from sklearn.preprocessing import normalize

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential

class Conv2Plus1D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding, **kwargs):
        """
            A sequence of convolutional layers that first apply the convolution operation over the
            spatial dimensions, and then the temporal dimension. 
        """
        super(Conv2Plus1D, self).__init__(**kwargs)
        self.seq = keras.Sequential([  
            # Spatial decomposition
            layers.Conv3D(filters=filters,
                            kernel_size=(1, kernel_size[1], kernel_size[2]),
                            padding=padding),
            # Temporal decomposition
            layers.Conv3D(filters=filters, 
                            kernel_size=(kernel_size[0], 1, 1),
                            padding=padding)
            ])

    def call(self, x):
        return self.seq(x)
    
    def get_config(self):
        config = super(Conv2Plus1D, self).get_config()
        config.update({"seq":self.seq})
        return config

class ResidualMain(keras.layers.Layer):
    """
        Residual block of the model with convolution, layer normalization, and the
        activation function, ReLU.
    """
    def __init__(self, filters, kernel_size,**kwargs):
        super(ResidualMain, self).__init__(**kwargs)
        self.seq = keras.Sequential([
            Conv2Plus1D(filters=filters,
                        kernel_size=kernel_size,
                        padding='same'),
            layers.LayerNormalization(),
            layers.ReLU(),
            Conv2Plus1D(filters=filters, 
                        kernel_size=kernel_size,
                        padding='same'),
            layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)
    
    def get_config(self):
        config = super(ResidualMain, self).get_config()
        config.update({"seq":self.seq})
        return config

class Project(keras.layers.Layer):
    """
    Project certain dimensions of the tensor as the data is passed through different 
    sized filters and downsampled. 
    """
    def __init__(self, units, **kwargs):
        super(Project, self).__init__(**kwargs)
        self.seq = keras.Sequential([
            layers.Dense(units),
            layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)

    def get_config(self):
        config = super(Project, self).get_config()
        config.update({"seq":self.seq})
        return config

def add_residual_block(input, filters, kernel_size):
    """
    Add residual blocks to the model. If the last dimensions of the input data
    and filter size does not match, project it such that last dimension matches.
    """
    out = ResidualMain(filters, 
                        kernel_size)(input)

    res = input
    # Using the Keras functional APIs, project the last dimension of the tensor to
    # match the new filter size
    if out.shape[-1] != input.shape[-1]:
        res = Project(out.shape[-1])(res)

    return layers.add([res, out])

class ResizeVideo(keras.layers.Layer):
    def __init__(self, height, width, **kwargs):
        super(ResizeVideo, self).__init__(**kwargs)
        self.height = height
        self.width = width
        self.resizing_layer = layers.Resizing(self.height, self.width)

    def call(self, video):
        """
        Use the einops library to resize the tensor.  

        Args:
            video: Tensor representation of the video, in the form of a set of frames.

        Return:
            A downsampled size of the video according to the new height and width it should be resized to.
        """
        # b stands for batch size, t stands for time, h stands for height, 
        # w stands for width, and c stands for the number of channels.
        old_shape = einops.parse_shape(video, 'b t h w c')
        images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
        images = self.resizing_layer(images)
        videos = einops.rearrange(
            images, '(b t) h w c -> b t h w c',
            t = old_shape['t'])
        return videos
    
    def get_config(self):
        config = super(ResizeVideo, self).get_config()
        config.update({"height":self.height, "width":self.width, "resizing_layer":self.resizing_layer})
        return config
    
def format_frames(frame, output_size):
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame

def initialize_action_model():
    '''returns the loaded model so they it only has to inferred
    once'''
    CLASS_NAMES =  ["na", 
                "Person loading an Object to a Vehicle", 
                "Person Unloading an Object from a Car/Vehicle",
                "Person Opening a Vehicle/Car Trunk",
                "Person Closing a Vehicle/Car Trunk",
                "Person getting into a Vehicle",
                "Person getting out of a Vehicle",
                "Person gesturing",
                "Person digging",
                "Person carrying an object",
                "Person running",
                "Person entering a facility",
                "Person exiting a facility"]

    HEIGHT = 224
    WIDTH = 224

    input_shape = (None, 10, HEIGHT, WIDTH, 3)
    input = layers.Input(shape=(input_shape[1:]))
    x = input

    x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = ResizeVideo(HEIGHT // 2, WIDTH // 2)(x)

    # Block 1
    x = add_residual_block(x, 16, (3, 3, 3))
    x = ResizeVideo(HEIGHT // 4, WIDTH // 4)(x)

    # Block 2
    x = add_residual_block(x, 32, (3, 3, 3))
    x = ResizeVideo(HEIGHT // 8, WIDTH // 8)(x)

    # Block 3
    x = add_residual_block(x, 64, (3, 3, 3))
    x = ResizeVideo(HEIGHT // 16, WIDTH // 16)(x)

    # Block 4
    x = add_residual_block(x, 128, (3, 3, 3))
    #here
    #x = ResizeVideo(HEIGHT // 32, WIDTH // 32)(x)

    #Block 5, added to initial structre
    #x = add_residual_block(x, 256, (3, 3, 3))
    #to here

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Flatten()(x)

    x = layers.Dense(len(CLASS_NAMES))(x)

    model = keras.Model(input, x)

    model.load_weights('C:\\Users\\hugo\\Documents\\University\\YEAR_3\\ECM3401_Project\\Action Detection\\yolov7-main\\action_detection\\runs\\models\\new_action_model210423.h5')
    
    return model

def detect_action(modeli, framecoords, last10frames):
    '''function to return the actions detected in a short video'''
    start = time.time()
    output_size = (224,224)
    frames = []
    #[ids[i], roi, int(roi[1]), int(roi[1]+h) , int(roi[0]), int(roi[0]+w)]

    PATH = "C:\\Users\\hugo\\Documents\\University\\YEAR_3\\ECM3401_Project\\Action Detection\\yolov7-main\\action_detection\\data\\"
    if len(framecoords) != 10 and len(last10frames) != 10:
        return None, None
    
    videoframes = []
    for i, coord in enumerate(framecoords):
        videoframes.append([coord[0], coord[1], last10frames[i][coord[2]:coord[3], coord[4]:coord[5]]])

    for frame in videoframes:
        frames.append(format_frames(frame[2], output_size))
    
    frames = np.array(frames)[..., [2, 1, 0]]
    frames = np.expand_dims(frames, axis=0)
    
    predictions = modeli.predict(frames)

    maxpred = 0
    classid = 0
    normalisedpred = predictions[0]
    for i, num in enumerate(normalisedpred):
        if num < 0:
            normalisedpred[i] = 0
    
    x_norm = normalize([normalisedpred], norm="l1")

    for i, pred in enumerate(x_norm[0]):
        if pred > maxpred:
            maxpred = pred
            classid = i
    finish = time.time()
    #print(finish - start)
    if maxpred > 0.4:#maybe leave this at 0, due to many multiple detections
        return int(classid), maxpred
    else:
        return None, None
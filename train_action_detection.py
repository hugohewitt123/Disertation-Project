import tqdm
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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential

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

def format_frames(frame, output_size):
    """
    Pad and resize an image from a video.

    Args:
        frame: Image that needs to resized and padded. 
        output_size: Pixel size of the output frame image.

    Return:
        Formatted frame with padding of specified output size.
    """
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame

def frames_from_video_file(video_path, n_frames, output_size = (224,224), frame_step = 10):
  """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
  """
  # Read each video frame by frame
  result = []
  src = cv2.VideoCapture(str(video_path))

  video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

  need_length = 1 + (n_frames - 1) * frame_step

  if need_length > video_length:
    start = 0
  else:
    max_start = video_length - need_length
    start = random.randint(0, max_start + 1)

  src.set(cv2.CAP_PROP_POS_FRAMES, start)
  # ret is a boolean indicating whether read was successful, frame is the image itself
  ret, frame = src.read()
  result.append(format_frames(frame, output_size))

  for _ in range(n_frames - 1):
    for _ in range(frame_step):
      ret, frame = src.read()
    if ret:
      frame = format_frames(frame, output_size)
      result.append(frame)
    else:
      result.append(np.zeros_like(result[0]))
  src.release()
  result = np.array(result)[..., [2, 1, 0]]

  return result

class FrameGenerator:
    def __init__(self, path, n_frames, training = False):
        """ Returns a set of frames with their associated label.

        Args:
            path: Video file paths.
            n_frames: Number of frames. 
            training: Boolean to determine if training dataset is being created.
        """
        self.path = path
        self.n_frames = n_frames
        self.training = training
        #self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
        self.class_names = CLASS_NAMES

    def get_files_and_class_names(self):
        video_paths  = list(self.path.glob('videos\\*.mp4'))
        classes_path = list(self.path.glob('class_label\\*.txt'))
        return video_paths, classes_path

    def __call__(self):
        video_paths, classes = self.get_files_and_class_names()
        pairs = list(zip(video_paths, classes))

        if self.training:
            random.shuffle(pairs)

        for path, name in pairs:
            video_frames = frames_from_video_file(path, self.n_frames) 
            f = open(name, "r")
            class_id = f.read()
            yield video_frames, int(class_id)

n_frames = 10 # 10
batch_size = 8 # 8

output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))

PATH = "C:\\Users\\hugo\\Documents\\University\\YEAR_3\\ECM3401_Project\\Action Detection\\yolov7-main\\action_detection\\data\\"

train_ds = tf.data.Dataset.from_generator(FrameGenerator(Path(PATH+"train\\"), n_frames, training=True),
                                          output_signature = output_signature)


# Batch the data
train_ds = train_ds.batch(batch_size)

val_ds = tf.data.Dataset.from_generator(FrameGenerator(Path(PATH+"val\\"), n_frames),
                                        output_signature = output_signature)
val_ds = val_ds.batch(batch_size)

test_ds = tf.data.Dataset.from_generator(FrameGenerator(Path(PATH+"test\\"), n_frames),
                                         output_signature = output_signature)

test_ds = test_ds.batch(batch_size)

# Define the dimensions of one frame in the set of frames created
HEIGHT = 224
WIDTH = 224

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

input_shape = (None, n_frames, HEIGHT, WIDTH, 3)
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

#Block 5, added to initial structure
#x = add_residual_block(x, 256, (3, 3, 3))
#x = ResizeVideo(HEIGHT // 64, WIDTH // 64)(x)

#Block 6
#x = add_residual_block(x, 512, (3, 3, 3))
#to here

x = layers.GlobalAveragePooling3D()(x)
x = layers.Flatten()(x)
x = layers.Dense(len(CLASS_NAMES))(x)

model = keras.Model(input, x)


frames, label = next(iter(train_ds)) 

model.build(frames)

# Visualize the model
plot_model(model, expand_nested=True, dpi=60, show_shapes=True)

model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              optimizer = keras.optimizers.Adam(learning_rate = 0.001), 
              metrics = ['accuracy'])

history = model.fit(x = train_ds,
                    epochs = 150,
                    validation_data = val_ds)

def plot_history(history):
    """
        Plotting training and validation learning curves.

        Args:
        history: model history with all the metric measures
    """
    fig, (ax1, ax2) = plt.subplots(2)

    fig.set_size_inches(18.5, 10.5)

    # Plot loss
    ax1.set_title('Loss')
    ax1.plot(history.history['loss'], label = 'train')
    ax1.plot(history.history['val_loss'], label = 'test')
    ax1.set_ylabel('Loss')

    # Determine upper bound of y-axis
    max_loss = max(history.history['loss'] + history.history['val_loss'])

    ax1.set_ylim([0, np.ceil(max_loss)])
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation']) 

    # Plot accuracy
    ax2.set_title('Accuracy')
    ax2.plot(history.history['accuracy'],  label = 'train')
    ax2.plot(history.history['val_accuracy'], label = 'test')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1])
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'])

    #plt.show()
    plt.savefig('C:\\Users\\hugo\\Documents\\University\\YEAR_3\\ECM3401_Project\\Action Detection\\yolov7-main\\action_detection\\runs\\models\\history.png')

model.evaluate(test_ds, return_dict=True)

model.save('C:\\Users\\hugo\\Documents\\University\\YEAR_3\\ECM3401_Project\\Action Detection\\yolov7-main\\action_detection\\runs\\models\\new_action_model.h5')
print("model saved")

plot_history(history)


from google.colab import drive
import tensorflow as tf

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from helpers import *
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from pathlib import Path

import random
from torch.utils.data import Dataset, DataLoader, random_split


drive.mount('/content/drive')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


data_path = '/content/drive/MyDrive/Honey/data/water surface method'
partitionPath = '/content/drive/MyDrive/Honey/data/partition'
corner_file_path = '/content/drive/MyDrive/Colab Notebooks/cornerInfo.xlsx'
honey_drop_info_path = '/content/drive/MyDrive/Colab Notebooks/dropInfo.xlsx'
ground_truth_file_path = '/content/drive/MyDrive/Colab Notebooks/Bal_AI.xlsx'


video_files = os.listdir(partitionPath)  # Replace this with os.listdir(path) for actual files.
conditionsFolder = ['Boiled Water']#os.listdir(data_path)
corner_df = pd.read_excel(corner_file_path)
honey_drop_info_df = pd.read_excel(honey_drop_info_path)
ground_truth_file_df = pd.read_excel(ground_truth_file_path)
specialCorners = corner_df['File Name']



# Function to filter videos based on environmental condition
def filter_videos_by_condition(video_list, condition="Boiled Water"):
    return [video for video in video_list if condition in video]

# Shuffle and split function
def shuffle_and_split(video_list, train_ratio=0.8):
    random.shuffle(video_list)  # Shuffle videos
    train_size = int(len(video_list) * train_ratio)
    train_videos = video_list[:train_size]
    test_videos = video_list[train_size:]
    return train_videos, test_videos

class PathWrapper:
    def __init__(self, paths, root_path):
        """
        Wraps a list of paths and appends a root path to each.
  
        Args:
            paths (list): A list of paths as strings or Path objects.
            root_path (str): The root path to prepend to each path.
        """
        self.root_path = Path(root_path)
        self.paths = [self.root_path / Path(p) for p in paths]  # Combine root path with each path

    def iterdir(self):
        """Mimics Path.iterdir by yielding each path."""
        for path in self.paths:
            yield path

    def glob(self, pattern):
        """Mimics Path.glob by filtering paths matching the pattern."""
        for path in self.paths:
            if path.match(pattern):
                yield path

# Filter videos by environmental condition
boiled_water_videos = filter_videos_by_condition(video_files, condition="Boiled Water")

# Shuffle and split the data
train_videos, test_videos = shuffle_and_split(boiled_water_videos)

# Wrap train and test video lists
subset_paths = {
    'train': PathWrapper(train_videos, partitionPath),
    'test': PathWrapper(test_videos, partitionPath)
}



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


def frames_from_video_file(video_path, n_frames, output_size = (224,224), frame_step = 15):
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
      # If the video has ended before reaching n_frames, pad with the last frame
      result.append(result[-1]) #Changed from result.append(np.zeros_like(result[0]))
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
    self.class_names = sorted(set(p.name.split('-')[0] for p in self.path.iterdir())) #Change made here
    self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

  def get_files_and_class_names(self):
    video_paths = list(self.path.glob('*/*.mp4'))
    classes = [p.name.split('-')[0] for p in video_paths]
    return video_paths, classes

  def __call__(self):
    video_paths, classes = self.get_files_and_class_names()

    pairs = list(zip(video_paths, classes))

    if self.training:
      random.shuffle(pairs)

    for path, name in pairs:
      video_frames = frames_from_video_file(path, self.n_frames)
      label = self.class_ids_for_name[name] # Encode labels
      yield video_frames, label



# train_videos, test_videos
# partitionPath
example = os.path.join(partitionPath, train_videos[0])
fg = FrameGenerator(subset_paths['train'], 9, training=True)

frames, label = next(fg())

print(f"Shape: {frames.shape}")
print(f"Label: {label}")


# Create the training set
output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))
train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['train'], 9, training=True),
                                          output_signature = output_signature)


# Create the validation set
test_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['test'], 9),
                                        output_signature = output_signature)

# Print the shapes of the data
train_frames, train_labels = next(iter(train_ds))
print(f'Shape of training set of frames: {train_frames.shape}')
print(f'Shape of training labels: {train_labels.shape}')

val_frames, val_labels = next(iter(test_ds))
print(f'Shape of validation set of frames: {val_frames.shape}')
print(f'Shape of validation labels: {val_labels.shape}')


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)


total_samples = train_ds.cardinality().numpy()
# Set your batch size
batch_size = 64  # Modify with your batch size

# Calculate the number of steps per epoch
steps_per_epoch = total_samples // batch_size


train_ds = train_ds.batch(batch_size)
test_ds = test_ds.batch(batch_size)

tf.config.optimizer.set_jit(False)

# Define input shape
input_shape = (9, 224, 224, 3)

# Define 3D CNN model
model = Sequential([
    # First Conv3D layer
    Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same', input_shape=input_shape),
    MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'),
    BatchNormalization(),

    # Second Conv3D layer
    Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
    MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'),
    BatchNormalization(),

    # Flatten and Fully Connected Layers
    Flatten(),
    Dense(units=64, activation='relu'),
    Dropout(0.5),  # Add dropout for regularization
    Dense(units=32, activation='relu'),
    Dropout(0.5),
    Dense(units=1, activation='sigmoid')  # For binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Use sparse_categorical_crossentropy for multi-class
              metrics=['accuracy'])

# Print the model summary
model.summary()

# Example for fitting the model (replace with actual data)
history = model.fit(
    train_ds,
          epochs = 10,
          validation_data = test_ds, verbose = 1,
          callbacks = tf.keras.callbacks.EarlyStopping(patience = 2, monitor = 'val_loss')
)

# # Evaluate the model
loss, accuracy = model.evaluate(test_ds['X'], test_ds['y'])
print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")
#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#

#/***************************************************************************
# *   Copyright (C) 2024 -- 2025   by Marek Kowal                           *
# *                                  <M.Kowal@issi.uz.zgora.pl>             *
# *                                                                         *
# *                                                                         *
# *   Part of the QAutoEnc:                                                 *
# *         https://github.com/qMSUZ/QAutoEnc                               *
# *                                                                         *
# * Permission is hereby granted, free of charge, to any person obtaining   *
# * a copy of this software and associated documentation files              *
# * (the “Software”), to deal in the Software without restriction,          *
# * including without limitation the rights to use, copy, modify, merge,    *
# * publish, distribute, sublicense, and/or sell copies of the Software,    *
# * and to permit persons to whom the Software is furnished to do so,       *
# * subject to the following conditions:                                    *
# *                                                                         *
# * The above copyright notice and this permission notice shall be included *
# * in all copies or substantial portions of the Software.                  *
# *                                                                         *
# * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS *
# * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF              *
# * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  *
# * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY    *
# * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,    *
# * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH           *
# * THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.              *
# ***************************************************************************/



import PolLettDS as pld
import os, random, time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Cropping2D, MaxPooling2D, UpSampling2D, Activation, Dropout, Multiply, BatchNormalization, Dense, Flatten, concatenate
from tensorflow.keras.activations import softmax
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy, MeanSquaredError






def split(images, split_percentage=[70, 20]):
    """
    Splits a dataset of images into training, validation, and test sets.

    This function shuffles the input `images` and partitions them into
    training, validation, and test sets according to the percentages
    provided in `split_percentage`.

    Parameters
    ----------
    images : numpy.ndarray
        A 3D NumPy array of shape (N, H, W), where:
            - N is the number of images,
            - H is the height of each image,
            - W is the width of each image.
    split_percentage : list of int, optional
        A list specifying the percentage split for training and validation
        sets, respectively. The remainder is assigned to the test set.
        Defaults to [70, 20], meaning:
            - 70% training,
            - 20% validation,
            - 10% test.

    Returns
    -------
    tuple
        training : numpy.ndarray
            A 4D NumPy array of training images with shape 
            (N_train, H, W, 1).
        validation : numpy.ndarray
            A 4D NumPy array of validation images with shape 
            (N_val, H, W, 1).
        test : numpy.ndarray
            A 4D NumPy array of test images with shape 
            (N_test, H, W, 1).

    Raises
    ------
    ValueError
        If `images` is not a 3D array.
    ValueError
        If `split_percentage` does not contain exactly 2 integers.

    Example
    -------
    >>> training, validation, test = split(images, split_percentage=[80, 10])
    >>> training.shape
    (N_train, H, W, 1)
    """
    
    # Validate input dimensions
    if images.ndim != 3:
        raise ValueError(f"Expected a 3D numpy array of images, but got {images.ndim}D array")
    if len(split_percentage) != 2:
        raise ValueError("split_percentage must contain exactly two values: [train%, val%]")

    size = images.shape[0]  # total number of images

    # Generate shuffled indices for random splitting
    indices = np.arange(size)
    np.random.seed(42)  # ensure reproducibility
    np.random.shuffle(indices)

    # Shuffle the images using the indices
    images = images[indices]

    # Compute split sizes
    split_training = int(split_percentage[0] * size / 100)
    split_validation = split_training + int(split_percentage[1] * size / 100)

    # Add channel dimension to match CNN input format (N, H, W, 1)
    images = np.expand_dims(images, axis=3)

    # Perform the split
    training = images[0:split_training, :, :, :]
    validation = images[split_training:split_validation, :, :, :]
    test = images[split_validation:, :, :, :]

    return training, validation, test










def tune_training_ds(train_images, batch_size=32):
    """
    Prepares a TensorFlow training dataset for autoencoder-like models.

    This function converts `train_images` into a TensorFlow dataset,
    where each image is paired with itself as both input and target.
    The dataset is shuffled, batched, prefetched, and repeated indefinitely
    to optimize training performance.

    Parameters
    ----------
    train_images : numpy.ndarray or tf.Tensor
        A 4D array/tensor of training images with shape (N, H, W, C), where:
            - N is the number of images,
            - H is the height of each image,
            - W is the width of each image,
            - C is the number of channels.
    batch_size : int, optional
        The size of each training batch. Defaults to 32.

    Returns
    -------
    tf.data.Dataset
        A TensorFlow dataset object that yields batches of shape:
        (batch_size, H, W, C) for both inputs and targets.
        The dataset is shuffled, repeated indefinitely, and optimized
        with prefetching.

    Example
    -------
    >>> dataset = tune_training_ds(train_images, batch_size=64)
    >>> for batch_inputs, batch_targets in dataset.take(1):
    ...     print(batch_inputs.shape, batch_targets.shape)
    (64, H, W, C) (64, H, W, C)
    """

    # Convert training images to a TensorFlow tensor (float32 for neural networks)
    train_images = tf.convert_to_tensor(train_images, dtype=tf.float32)

    # Create a dataset where input == target (useful for autoencoders)
    dataset = tf.data.Dataset.from_tensor_slices((train_images, train_images))

    # Shuffle dataset for randomness (reshuffle at each epoch)
    dataset = dataset.shuffle(1024, reshuffle_each_iteration=True)

    # Batch the dataset, dropping incomplete batches
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Prefetch to overlap preprocessing and model execution
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # Repeat indefinitely (training loop defines stopping condition)
    dataset = dataset.repeat()

    return dataset





def tune_validation_ds(val_images, batch_size=32):
    """
    Prepares a TensorFlow validation dataset for autoencoder-like models.

    This function converts `val_images` into a TensorFlow dataset,
    where each image is paired with itself as both input and target.
    The dataset is batched and repeated indefinitely to be used during
    validation.

    Parameters
    ----------
    val_images : numpy.ndarray or tf.Tensor
        A 4D array/tensor of validation images with shape (N, H, W, C), where:
            - N is the number of images,
            - H is the height of each image,
            - W is the width of each image,
            - C is the number of channels.
    batch_size : int, optional
        The size of each validation batch. Defaults to 32.

    Returns
    -------
    tf.data.Dataset
        A TensorFlow dataset object that yields batches of shape:
        (batch_size, H, W, C) for both inputs and targets.
        The dataset is repeated indefinitely (no shuffling).

    Example
    -------
    >>> dataset = tune_validation_ds(val_images, batch_size=64)
    >>> for batch_inputs, batch_targets in dataset.take(1):
    ...     print(batch_inputs.shape, batch_targets.shape)
    (64, H, W, C) (64, H, W, C)
    """

    # Convert validation images to a TensorFlow tensor (float32 for model input)
    val_images = tf.convert_to_tensor(val_images, dtype=tf.float32)

    # Create a dataset where input == target (used in autoencoder validation)
    dataset = tf.data.Dataset.from_tensor_slices((val_images, val_images))

    # Batch the dataset, dropping incomplete batches
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Repeat indefinitely (validation loop defines stopping condition)
    dataset = dataset.repeat()

    return dataset








def tune_test_ds(test_images):
    """
    Prepares a TensorFlow test dataset for autoencoder-like models.

    This function converts `test_images` into a TensorFlow dataset,
    where each image is paired with itself as both input and target.
    The dataset is batched with a size of 1 (one sample per batch) and
    repeated indefinitely to support continuous evaluation.

    Parameters
    ----------
    test_images : numpy.ndarray or tf.Tensor
        A 4D array/tensor of test images with shape (N, H, W, C), where:
            - N is the number of images,
            - H is the height of each image,
            - W is the width of each image,
            - C is the number of channels.

    Returns
    -------
    tf.data.Dataset
        A TensorFlow dataset object that yields batches of shape:
        (1, H, W, C) for both inputs and targets.
        The dataset is repeated indefinitely (no shuffling).

    Example
    -------
    >>> dataset = tune_test_ds(test_images)
    >>> for batch_inputs, batch_targets in dataset.take(3):
    ...     print(batch_inputs.shape, batch_targets.shape)
    (1, H, W, C) (1, H, W, C)
    """

    # Convert test images to a TensorFlow tensor (float32 for model input)
    test_images = tf.convert_to_tensor(test_images, dtype=tf.float32)

    # Create a dataset where input == target (used in autoencoder testing)
    dataset = tf.data.Dataset.from_tensor_slices((test_images, test_images))

    # Use batch size = 1 for individual test evaluation
    dataset = dataset.batch(1)

    # Repeat indefinitely (test loop defines stopping condition)
    dataset = dataset.repeat()

    return dataset


def get_simple_autoencoder(input_shape=[64, 64, 1]):
    """
    Builds a simple convolutional autoencoder model.

    This function constructs an encoder-decoder convolutional neural
    network (CNN) for image reconstruction tasks. The encoder downsamples
    the input image into a compressed latent representation, while the
    decoder upsamples it back to the original size. The final output layer
    uses a sigmoid activation to normalize pixel values to [0, 1].

    Parameters
    ----------
    input_shape : list of int, optional
        Shape of the input image as [H, W, C], where:
            - H is the image height,
            - W is the image width,
            - C is the number of channels.
        Defaults to [64, 64, 1] (grayscale 64x64 images).

    Returns
    -------
    tuple
        model : tensorflow.keras.Model
            The compiled autoencoder model consisting of encoder and decoder.
        model_name : str
            A descriptive string identifier for the model architecture.

    Example
    -------
    >>> model, name = get_simple_autoencoder(input_shape=[32, 32, 1])
    >>> model.summary()
    Model: "autoencoder_c16x16x4_p153_EL2_DL3"
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
    ┃ Layer (type)                    ┃ Output Shape           ┃ Param #       ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
    ┃ input_1 (InputLayer)            ┃ (None, 32, 32, 1)      ┃ 0             ┃
    ┃ conv2d (Conv2D)                 ┃ (None, 16, 16, 1)      ┃ 10            ┃
    ┃ ...                             ┃ ...                    ┃ ...           ┃
    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━┛
    """

    model_name = "autoencoder"

    # Input layer
    inputs = Input(input_shape)

    # ----- Encoder -----
    # First convolutional block
    x = Conv2D(filters=1, kernel_size=(3, 3), padding='same', strides=2)(inputs)
    x = Activation('relu')(x)

    # Second convolutional block
    x = Conv2D(filters=4, kernel_size=(3, 3), padding='same', strides=2)(x)
    x = Activation('relu')(x)

    # ----- Decoder -----
    # First upsampling block
    x = Conv2DTranspose(filters=2, kernel_size=(3, 3), strides=2, padding='same')(x)
    x = Activation('relu')(x)

    # Second upsampling block
    x = Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=2, padding='same')(x)
    x = Activation('relu')(x)

    # Final reconstruction layer (sigmoid to normalize output between 0 and 1)
    outputs = Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)

    # Assemble autoencoder model
    model = Model(inputs=[inputs], outputs=[outputs])

    return model, model_name









#main part
print(f"Tenesorflow version: {tf.__version__}")


"""
Autoencoder training pipeline for Polish letter images.

This script loads letter images, preprocesses them (reshape + normalize),
excludes selected indices, splits the dataset into train/val/test, builds
tf.data datasets, constructs and compiles a simple convolutional autoencoder,
trains it with callbacks, saves the best model, and finally visualizes
reconstructions on test samples and training curves.

Steps
-----
1) Load raw data from binary files (`pld.load_pol_lett_ds_from_files`).
2) Preprocess:
   - Reshape to (N, 64, 64)
   - Normalize to [0, 1]
   - Exclude specified indices (test letters).
3) Split into train/val/test using `split`.
4) Build tf.data datasets with `tune_training_ds`, `tune_validation_ds`, `tune_test_ds`.
5) Build the model via `get_simple_autoencoder`, compile with Adam + MSE.
6) Train with EarlyStopping + ModelCheckpoint; time the run.
7) Plot training history (loss, val_loss).
8) Load the best model and visualize input→output pairs.

Notes
-----
- Assumes the following are imported elsewhere:
  numpy as np, matplotlib.pyplot as plt, time
  tensorflow as tf, from tensorflow import keras
  from tensorflow.keras.losses import MeanSquaredError
  and your own helpers: split, tune_training_ds, tune_validation_ds, tune_test_ds,
  get_simple_autoencoder, and pld (data loader module).

- Image shape is 64x64 (grayscale). Adjust `input_shape` in the model if needed.
"""

# -------------------------
# Configuration
# -------------------------
EPOCHS = 100
BATCH_SIZE = 32
LR = 0.001

# Indices of samples (letters) to exclude from the dataset.
# These excluded letters are reserved for a comparative test with the
# quantum autoencoder, therefore they must not appear in the training
# or validation sets.
EXCLUDED_INDICES = [4, 9, 23, 46, 79]  # indices of samples to exclude from the dataset

# -------------------------
# Load data (letters)
# -------------------------
loaded_data, loaded_labels, labels_count = pld.load_pol_lett_ds_from_files(
    'pol_lett_ds.bin',
    'pol_lett_ds_labels.bin'
)

# Reshape raw bytes/flat data to (N, 64, 64) and normalize to [0, 1]
letters = np.reshape(loaded_data, (-1, 64, 64), order='C').astype(np.float32)
letters = letters / 255.0

# Exclude designated letters reserved for comparative testing with the quantum autoencoder.
# These samples must not appear in training or validation sets.
if EXCLUDED_INDICES:
    letters = np.delete(letters, EXCLUDED_INDICES, axis=0)

# -------------------------
# Split into train/val/test
# -------------------------
train, val, test = split(letters)  # returns 4D tensors: (N, H, W, 1)

# Basic dataset stats
train_size = train.shape[0]
val_size = val.shape[0]
test_size = test.shape[0]

print(f"# training images: {train_size}")
print(f"# validation images: {val_size}")
print(f"# test images: {test_size}")

# -------------------------
# Build tf.data datasets
# -------------------------
train_ds = tune_training_ds(train, BATCH_SIZE)
val_ds = tune_validation_ds(val, BATCH_SIZE)
test_ds = tune_test_ds(test)

# -------------------------
# Model: build, compile
# -------------------------
autoencoder, model_name = get_simple_autoencoder(input_shape=[64, 64, 1])

optimizer = keras.optimizers.Adam(learning_rate=LR)
loss_fn = MeanSquaredError()

autoencoder.compile(optimizer=optimizer, loss=loss_fn)
autoencoder.summary()

# -------------------------
# Callbacks (checkpoint + early stopping)
# -------------------------
modelpath = './models_classic_autoencoders/' + model_name + '.keras'

model_save_callback = keras.callbacks.ModelCheckpoint(
    filepath=modelpath,
    monitor='val_loss',
    save_weights_only=False,
    mode='min',
    save_best_only=True
)

early_stopping_callback = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    mode='min',
    start_from_epoch=10
)

# -------------------------
# Train
# -------------------------
# Steps: number of full batches per epoch. Use max(1, ...) to avoid 0 if very small data.
steps_per_epoch = max(1, train_size // BATCH_SIZE)

start_time = time.time()
history = autoencoder.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=1,   # one validation step per epoch (adjust as needed)
    epochs=EPOCHS,
    callbacks=[model_save_callback, early_stopping_callback],
    verbose=1
)
end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")

# -------------------------
# Save training history
# -------------------------
plt.figure()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('MSE')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper left')
plt.savefig("training_history.png", dpi=300, bbox_inches='tight')


# -------------------------
# Load best model from checkpoint
# -------------------------
autoencoder = tf.keras.models.load_model(modelpath)

# -------------------------
# Visualize reconstructions on test set
# -------------------------def save_image(image_as_numpy_array, file_name):
# Take a fixed number of test samples from the repeated dataset.
num_samples = 160
test_list = list(test_ds.take(num_samples).as_numpy_iterator())

# Plot grid (16 rows x 10 columns) of input vs output pairs
plt.figure(figsize=(20, 20))
border = np.ones((64, 10), dtype=np.float32)  # narrow white border between input and output

for i in range(len(test_list)):
    ax = plt.subplot(16, 10, i + 1)
    image, reference = test_list[i]  # reference is same as image for autoencoders
    preds = autoencoder(image, training=False)  # forward pass

    # Remove batch/channel dims for display
    img = np.squeeze(image[0])
    rec = np.squeeze(preds.numpy()[0])

    # Concatenate input | border | reconstruction
    result = np.concatenate((img, border, rec), axis=1)
    plt.title("I - O")
    plt.imshow(result, cmap='gray')
    plt.axis("off")

plt.tight_layout()
plt.savefig("classic_autoencoder.png", dpi=300, bbox_inches='tight')


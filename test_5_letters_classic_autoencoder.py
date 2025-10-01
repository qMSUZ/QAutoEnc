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
import os, random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski

from skimage import io
from skimage import metrics as ski_metrics
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Cropping2D, MaxPooling2D, UpSampling2D, Activation, Dropout, Multiply, BatchNormalization, Dense, Flatten, concatenate
from tensorflow.keras.activations import softmax
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy, MeanSquaredError



def fidelity(img, pred):
    """
    Computes the fidelity (inner product similarity) between two images.

    This function flattens the input and predicted images into 1D vectors
    and computes their inner product using the complex conjugate dot product
    (`np.vdot`). This measure is often used in quantum-inspired contexts
    to quantify the similarity between two states (or images).

    Parameters
    ----------
    img : numpy.ndarray
        A NumPy array representing the reference image. Can be any shape
        (H, W) or (H, W, C).
    pred : numpy.ndarray
        A NumPy array representing the predicted image. Must have the same
        shape as `img`.

    Returns
    -------
    float or complex
        The fidelity value computed as the inner product:
        f = <img | pred>, where both arrays are flattened.
        If inputs are real-valued, the result is a float.

    Raises
    ------
    ValueError
        If `img` and `pred` do not have the same shape.

    Example
    -------
    >>> img = np.array([[0, 1], [1, 0]])
    >>> pred = np.array([[0.5, 0.5], [0.5, 0.5]])
    >>> fidelity(img, pred)
    (1+0j)
    """

    # Validate shapes
    if img.shape != pred.shape:
        raise ValueError(f"Input shapes must match! Got {img.shape} and {pred.shape}.")

    # Flatten both arrays into 1D vectors
    img = img.flatten()
    pred = pred.flatten()

    # Compute inner product (complex conjugate dot product)
    f = np.vdot(img, pred)

    return f





def save_image(image_as_numpy_array, file_name):
    """
    Saves a NumPy array as a grayscale image to a file.

    This function takes a NumPy array representing an image,
    displays it in grayscale using Matplotlib, removes axis
    decorations, and saves it to the specified file path.

    Parameters
    ----------
    image_as_numpy_array : numpy.ndarray
        A 2D array (H, W) or 3D array (H, W, 1) representing
        the image to save.
    file_name : str
        Path (including filename and extension) where the image
        will be saved. Supported formats depend on Matplotlib,
        e.g. 'png', 'jpg', 'pdf'.

    Returns
    -------
    None

    Example
    -------
    >>> img = np.random.rand(64, 64)
    >>> save_image(img, "random_image.png")
    # Image is saved as 'random_image.png' in the current directory.
    """

    # Clear any existing plots
    #plt.clf()

    # Create a new figure with no axes
    fig, ax = plt.subplots()
    ax.set_axis_off()

    # Display the image in grayscale, with intensity normalized
    ax.matshow(
        image_as_numpy_array,
        cmap='gray',
        vmin=0,
        vmax=np.max(image_as_numpy_array)
    )

    # Save the figure to file with tight bounding box (no extra whitespace)
    plt.savefig(file_name, bbox_inches='tight')

    # Close the figure to free resources
    plt.close()



def evaluate_autoencoders_on_selected_letters(
    model_folder,
    result_folder,
    data_path='pol_lett_ds.bin',
    labels_path='pol_lett_ds_labels.bin',
    selected_indices=None,
    selected_names=None,
):
    """
    Evaluate multiple autoencoder models on a fixed set of selected letters.

    This function:
      1) Loads all autoencoder models (*.keras) from the specified folder.
      2) Selects 5 predefined letters (images) from the dataset.
      3) For each model:
         - Reconstructs the selected samples,
         - Normalizes input and reconstruction (L2 norm, with safe division),
         - Computes evaluation metrics: MSE, SSIM, and fidelity,
         - Saves metrics into a text file,
         - Saves side-by-side images (original | reconstruction) for each letter,
         - Saves a summary figure with all results.

    Context
    -------
    The 5 letters/digits are specifically chosen for **comparative testing
    against a quantum autoencoder**. They must not be present in the training
    or validation sets.

    Parameters
    ----------
    model_folder : str
        Path to the folder containing autoencoder models (*.keras).
    result_folder : str
        Path to the folder where evaluation results and figures will be saved.
    data_path : str, optional
        Path to the binary file containing image data. Default is 'pol_lett_ds.bin'.
    labels_path : str, optional
        Path to the binary file containing labels. Default is 'pol_lett_ds_labels.bin'.
    selected_indices : list[int], optional
        Indices of selected test samples (default: [46, 23, 79, 4, 9] → Ą, k, Ż, 4, 9).
    selected_names : list[str], optional
        Names/labels corresponding to `selected_indices`
        (default: ['A_', 'k', 'Z_', '4', '9']).

    Returns
    -------
    None
        Results are saved to files in `result_folder`.

    Raises
    ------
    FileNotFoundError
        If the model folder does not exist or contains no *.keras files.
    ValueError
        If `selected_indices` and `selected_names` have different lengths.

    Example
    -------
    >>> evaluate_autoencoders_on_selected_letters(
    ...     model_folder='./models_01/',
    ...     result_folder='./results_02/'
    ... )
    """


    # Default set of 5 samples (Ą, k, Ż, 4, 9)
    if selected_indices is None:
        selected_indices = [46, 23, 79, 4, 9]
    if selected_names is None:
        selected_names = ['A_', 'k', 'Z_', '4', '9']

    if len(selected_indices) != len(selected_names):
        raise ValueError("selected_indices and selected_names must have the same length.")

    # Load dataset from binary files
    loaded_data, loaded_labels, labels_count = pld.load_pol_lett_ds_from_files(
        data_path,
        labels_path
    )

    # Reshape to (N, 64, 64) and normalize to [0, 1]
    letters = np.reshape(loaded_data, (-1, 64, 64), order='C').astype(np.float32)
    letters = letters / 255.0

    # Extract selected test samples
    images = [letters[idx, :] for idx in selected_indices]
    names = list(selected_names)

    # Collect model names from folder
    if not os.path.isdir(model_folder):
        raise FileNotFoundError(f"Model folder does not exist: {model_folder}")

    model_names = [
        os.path.splitext(f)[0]
        for f in sorted(os.listdir(model_folder))
        if f.endswith('.keras') and os.path.isfile(os.path.join(model_folder, f))
    ]

    if not model_names:
        raise FileNotFoundError(f"No *.keras files found in folder: {model_folder}")

    # Ensure result folder exists
    os.makedirs(result_folder, exist_ok=True)

    # Loop over all models in the folder
    for model_name in model_names:
        print(model_name)

        # Create folder for individual model results
        full_result_path = os.path.join(result_folder, model_name)
        os.makedirs(full_result_path, exist_ok=True)

        model_pathname = os.path.join(model_folder, model_name + '.keras')
        result_txt = os.path.join(result_folder, f"results_{model_name}.txt")

        # Load model
        autoencoder = tf.keras.models.load_model(model_pathname)

        # Create figure: 5 rows × 2 columns (input | reconstruction)
        plt.figure(figsize=(4, 10))

        with open(result_txt, "w", encoding="utf-8") as f:
            for i, (image, name) in enumerate(zip(images, names)):
                # Prepare input: expand to (1, H, W, 1)
                image_ext = np.expand_dims(image, axis=(0, -1))

                # Run model inference
                predictions = autoencoder(image_ext, training=False)
                predictions = np.squeeze(predictions.numpy()[0])  # shape (H, W)

                # Normalize input and output safely (avoid division by zero)
                img_norm = np.linalg.norm(image)
                image_norm = image / img_norm if img_norm > 0 else np.zeros_like(image)

                pred_norm = np.linalg.norm(predictions)
                predictions_norm = predictions / pred_norm if pred_norm > 0 else np.zeros_like(predictions)

                # Compute metrics
                mse = ski_metrics.mean_squared_error(image_norm, predictions_norm)
                ssim = ski_metrics.structural_similarity(
                    image_norm, predictions_norm, data_range=1.0
                )
                fid = fidelity(image_norm, predictions_norm)

                #Print results
                print(f"Letter {name}:")
                print(f"MSE = {mse:.8f}")
                print(f"SSIM = {ssim:.8f}")
                print(f"FID = {fid:.8f}\n")

                # Write results to text file
                f.write(f"Letter {name}:\n")
                f.write(f"MSE = {mse:.8f}\n")
                f.write(f"SSIM = {ssim:.8f}\n")
                f.write(f"FID = {fid:.8f}\n")

                # Plot original image
                ax = plt.subplot(5, 2, 2 * i + 1)
                plt.imshow(image, cmap='grey')
                plt.title(f"{name} (in)")
                plt.axis("off")

                # Plot reconstruction
                ax = plt.subplot(5, 2, 2 * i + 2)
                plt.imshow(predictions, cmap='grey')
                plt.title(f"{name} (out)")
                plt.axis("off")

                # Save individual reconstruction
                out_png = os.path.join(full_result_path, f"{model_name}_{name}.png")
                save_image(predictions, out_png)

            # Save combined figure for the model
            #plt.tight_layout()
            plt.savefig(os.path.join(result_folder, f"{model_name}.png"), dpi=300, bbox_inches='tight')
            #plt.close()




#main part
print(f"Tenesorflow version: {tf.__version__}")

evaluate_autoencoders_on_selected_letters(
    model_folder='./models_classic_autoencoders/',
    result_folder='./results_classic_autoencoders/experiment_1/',
    data_path='pol_lett_ds.bin',
    labels_path='pol_lett_ds_labels.bin',
    selected_indices=[46, 23, 79, 4, 9],   # Ą, k, Ż, 4, 9
    selected_names=['A_', 'k', 'Z_', '4', '9']
)

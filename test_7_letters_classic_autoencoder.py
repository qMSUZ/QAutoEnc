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
    Compute the fidelity (inner product similarity) between two images.

    This function flattens both the reference (`img`) and the predicted
    (`pred`) images into 1D vectors and computes their complex conjugate
    dot product using `np.vdot`. In quantum-inspired contexts, this
    corresponds to the inner product between two states.

    Parameters
    ----------
    img : numpy.ndarray
        A NumPy array representing the reference image.
        Can be 2D (H, W) or 3D (H, W, C).
    pred : numpy.ndarray
        A NumPy array representing the predicted image.
        Must have the same shape as `img`.

    Returns
    -------
    float or complex
        The fidelity value computed as the inner product <img | pred>.
        If inputs are real-valued, the result is a float.

    Raises
    ------
    ValueError
        If `img` and `pred` have different shapes.

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
    Save a NumPy array as a grayscale image to a file.

    This function takes a NumPy array representing an image,
    removes axes from the plot, displays it in grayscale, and
    saves the result to the specified file path.

    Parameters
    ----------
    image_as_numpy_array : numpy.ndarray
        A 2D array (H, W) or 3D array (H, W, 1) representing the image.
    file_name : str
        Path (including filename and extension) where the image will be saved.
        Supported formats depend on Matplotlib (e.g., 'png', 'jpg', 'pdf').

    Returns
    -------
    None

    Example
    -------
    >>> img = np.random.rand(64, 64)
    >>> save_image(img, "random_image.png")
    # Image is saved as 'random_image.png' in the current directory.
    """

    # Clear any previous plots
    plt.clf()

    # Create a new figure without axes
    fig, ax = plt.subplots()
    ax.set_axis_off()

    # Display the image in grayscale, scaling intensity to [0, max]
    ax.matshow(
        image_as_numpy_array,
        cmap='gray',
        vmin=0,
        vmax=np.max(image_as_numpy_array)
    )

    # Save the figure tightly (remove extra white margins)
    plt.savefig(file_name, bbox_inches='tight')

    # Close figure to free memory
    plt.close()






def evaluate_average_metrics_on_letters(
    model_folder: str,
    model_name: str,
    result_folder: str,
    data_path: str = "pol_lett_ds.bin",
    labels_path: str = "pol_lett_ds_labels.bin",
    letters: list[str] = None,
    names: list[str] = None,
    num_letters: list[int] = None,
):
    """
    Evaluate averaged reconstruction quality metrics (MSE, SSIM, Fidelity)
    for an autoencoder model across multiple instances of selected letters.

    The experiment:
      1) Loads image data and labels from binary files.
      2) Selects `num_letters[i]` instances for each target letter using `pld.select_letters`.
      3) Loads the specified autoencoder model (*.keras).
      4) For each letter instance:
         - Runs a forward pass to obtain a reconstruction,
         - Safely L2-normalizes input and reconstruction,
         - Computes MSE, SSIM (data_range=1.0), and Fidelity (inner product),
         - Aggregates metrics per letter.
      5) Computes per-letter averages and writes a text report to disk.

    Parameters
    ----------
    model_folder : str
        Path to the folder containing the target model (*.keras).
    model_name : str
        Base name of the model (without extension), e.g. 'autoencoder_I'.
    result_folder : str
        Path to the folder where results (a text file) will be saved.
    data_path : str, optional
        Path to the binary image data file. Default is 'pol_lett_ds.bin'.
    labels_path : str, optional
        Path to the binary labels file. Default is 'pol_lett_ds_labels.bin'.
    letters : list[str], optional
        List of characters to evaluate. Defaults to ['ć','Ę','k','L','3','7','Ź'].
    names : list[str], optional
        Friendly names used for display/reporting (same length as `letters`).
        Defaults to ['c_','E_','k','L','3','7','Z__'].
    num_letters : list[int], optional
        Number of instances per letter (same length as `letters`).
        Defaults to [52, 52, 52, 52, 52, 52, 52].

    Returns
    -------
    dict
        A dictionary with per-letter averages:
        {
          'mse': {letter: avg_mse, ...},
          'ssim': {letter: avg_ssim, ...},
          'fid': {letter: avg_fid, ...},
        }
        Results are also written to `<result_folder>/results_avg_<model_name>.txt`.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    ValueError
        If `letters`, `names`, and `num_letters` differ in length.

    Example
    -------
    >>> evaluate_average_metrics_on_letters(
    ...     model_folder='./models_classic_autoencoders/',
    ...     model_name='autoencoder_I',
    ...     result_folder='./results_classic_autoencoders/'
    ... )
    """
   

    # Defaults for the 7-character experiment
    if letters is None:
        letters = ['ć', 'Ę', 'k', 'L', '3', '7', 'Ź']
    if names is None:
        names = ['c_', 'E_', 'k', 'L', '3', '7', 'Z__']
    if num_letters is None:
        num_letters = [52] * len(letters)

    # Basic argument validation
    if not (len(letters) == len(names) == len(num_letters)):
        raise ValueError("`letters`, `names`, and `num_letters` must have the same length.")

    # Load dataset (images + labels)
    loaded_data, loaded_labels, labels_count = pld.load_pol_lett_ds_from_files(
        data_path,
        labels_path
    )

    # Reshape to (N, 64, 64) and normalize to [0, 1]
    letters_data = np.reshape(loaded_data, (-1, 64, 64), order='C').astype(np.float32)
    letters_data = letters_data / 255.0

    # Select up to `num_letters[i]` samples for each requested character
    sel_let, sel_lab = pld.select_letters(
        letters_data,
        loaded_labels,
        letters,
        num_letters,
        rand=False
    )

    # Prepare paths
    os.makedirs(result_folder, exist_ok=True)
    model_pathname = os.path.join(model_folder, f"{model_name}.keras")
    if not os.path.isfile(model_pathname):
        raise FileNotFoundError(f"Model file not found: {model_pathname}")

    # Load the model
    autoencoder = tf.keras.models.load_model(model_pathname)
    print(f"Model name: {model_name}")

    # Aggregators
    mse_agg: dict[str, float] = {letter: 0.0 for letter in letters}
    ssi_agg: dict[str, float] = {letter: 0.0 for letter in letters}
    fid_agg: dict[str, float] = {letter: 0.0 for letter in letters}

    # Evaluate each letter across its selected instances
    for letter, name, num in zip(letters, names, num_letters):
        for j in range(num):
            # Fetch j-th sample for this letter (shape (H, W))
            image = sel_let[letter][j]

            # Prepare model input: (1, H, W, 1)
            image_ext = np.expand_dims(image, axis=(0, -1))

            # Forward pass (inference)
            predictions = autoencoder(image_ext, training=False)
            predictions = np.squeeze(predictions.numpy()[0])  # (H, W)

            # Safe L2 normalization (avoid division by zero)
            img_norm = np.linalg.norm(image)
            image_norm = image / img_norm if img_norm > 0 else np.zeros_like(image)

            pred_norm = np.linalg.norm(predictions)
            predictions_norm = predictions / pred_norm if pred_norm > 0 else np.zeros_like(predictions)

            # Metrics
            mse = ski_metrics.mean_squared_error(image_norm, predictions_norm)
            ssi = ski_metrics.structural_similarity(
                image_norm, predictions_norm, data_range=1.0
            )
            fid = fidelity(image_norm, predictions_norm)

            # Aggregate
            mse_agg[letter] += mse
            ssi_agg[letter] += ssi
            fid_agg[letter] += fid

    # Compute per-letter averages
    mse_avg: dict[str, float] = {}
    ssi_avg: dict[str, float] = {}
    fid_avg: dict[str, float] = {}
    for letter, name, num in zip(letters, names, num_letters):
        mse_avg[letter] = mse_agg[letter] / num
        ssi_avg[letter] = ssi_agg[letter] / num
        fid_avg[letter] = fid_agg[letter] / num

    # Write report
    result_filename = os.path.join(result_folder, f"results_avg_{model_name}.txt")
    with open(result_filename, "w", encoding="utf-8") as f:
        for letter in letters:
            print(f"Mean MSE for {letter}:  {mse_avg[letter]:.8f}")
            print(f"Mean SSIM for {letter}: {ssi_avg[letter]:.8f}")
            print(f"Mean FID for {letter}:  {fid_avg[letter]:.8f}\n")
            f.write(f"Letter {letter}:\n")
            f.write(f"Mean MSE = {mse_avg[letter]:.8f}\n")
            f.write(f"Mean SSIM = {ssi_avg[letter]:.8f}\n")
            f.write(f"Mean FID = {fid_avg[letter]:.8f}\n")

    return {"mse": mse_avg, "ssim": ssi_avg, "fid": fid_avg}



#main part
print(f"Tenesorflow version: {tf.__version__}")

metrics_dict = evaluate_average_metrics_on_letters(
    model_folder='./models_classic_autoencoders/',
    model_name='autoencoder_I',
    result_folder='./results_classic_autoencoders/experiment_2/',
    data_path='pol_lett_ds.bin',
    labels_path='pol_lett_ds_labels.bin',
    letters=['ć', 'Ę', 'k', 'L', '3', '7', 'Ź'],
    names=['c_', 'E_', 'k', 'L', '3', '7', 'Z__'],
    num_letters=[52, 52, 52, 52, 52, 52, 52]
)

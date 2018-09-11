import numpy as np
import os
from skimage.util import view_as_windows
from . import hyper_params, logger


def pad_array(array):
    pad_width = (
        (hyper_params.context_length, hyper_params.context_length + 1),
        (0, 0)
    )
    return np.pad(array, pad_width, mode='constant', constant_values=0)


def get_sequences(melgram, _n_mels):
    """
    Transform 2 or 3D mel spectrogram into 3D array of windows

    :param melgram: 2 or 3D mel spectrogram (if 3D, squeezed to 2D)
    :param _n_mels: number of mel buckets
    :return: array of sequences. shape (n_frames, seq_len, n_mels)
    """
    if len(melgram.shape) == 3:
        melgram = np.squeeze(melgram)

    seq_len = hyper_params.sample_frames

    sequences = view_as_windows(melgram, window_shape=(seq_len, _n_mels))
    sequences = np.squeeze(sequences, 1)

    return sequences


def list_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if
            os.path.isdir(os.path.join(a_dir, name))]


def exists(path):
    return os.path.exists(path)


def log(message):
    logger.logger.info(message)

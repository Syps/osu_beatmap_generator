import librosa
import numpy as np

from . import errors, hyper_params


def get_duration(audio):
    duration = librosa.core.get_duration(audio)

    return duration


def get_audio(file_name):
    try:
        y, sr = librosa.load(file_name)
    except EOFError:
        raise errors.OsuError('EOFError while loading file', 'corrupt', file_name)
    return y


def get_mel_spectrogram(song, kwargs=hyper_params.get_stft_args()):
    hop_length = kwargs['hop_length']
    win_length = kwargs['win_length']
    n_fft = kwargs['n_fft']
    n_mels = kwargs['n_mels']
    pwr_spect = kwargs['power_spectrogram']
    fmin = kwargs['fmin']
    fmax = kwargs['fmax']

    if type(song) == str:
        y = get_audio(song)
    elif type(song) == np.ndarray:
        y = song
    else:
        raise ValueError('''
        `audio` must be either file path or np.ndarray. Received type {}
        '''.format(type(song)))

    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                            win_length=win_length)) ** pwr_spect
    S = librosa.feature.melspectrogram(S=D, n_fft=n_fft, n_mels=n_mels,
                                       fmin=fmin, fmax=fmax)
    dbs = librosa.core.power_to_db(S)

    spectrogram = np.squeeze(dbs)

    return spectrogram.T

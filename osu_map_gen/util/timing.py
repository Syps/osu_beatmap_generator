import collections
import pdb
from numbers import Number
from typing import List

import librosa

from osu_map_gen.util.hyper_params import n_fft, hop_length


def mls_to_seconds(times) -> List[float]:
    """
    Convert list milliseconds to list of seconds
    """

    if isinstance(times, collections.Iterable):
        seconds = [float(mls) / 1000.0 for mls in times]
    elif type(times) in [int, float]:
        seconds = float(times) / 1000.0
    else:
        raise ValueError(
            'expected either list or scalar value for times. Received {}'.format(
                times))

    return seconds


def mls_to_frames(times, **ft_args) -> list:
    """
    Convert list of milliseconds to list of frame indices
    """
    seconds = mls_to_seconds(times)
    frames = seconds_to_frames(seconds, **ft_args)

    if len(frames) < 1:
        pdb.set_trace()

    return frames


def seconds_to_frames(times, hop_length=hop_length, n_fft=n_fft):
    """
    Convert list of seconds to list of frame indices
    """

    if hop_length is None or hop_length <= 0:
        raise ValueError(
            'hop_length must be positive integer. Received {}'.format(
                hop_length))

    if n_fft is None or n_fft <= 0:
        raise ValueError(
            'n_fft must be positive integer. Received {}'.format(n_fft))

    if type(times) != list and type(times) not in [float, int]:
        raise ValueError(
            'times must be either list of floats or scalar positive number. Received {}'.format(
                times))

    if type(times) != list:
        times = [times]

    frames = librosa.time_to_frames(times, hop_length=hop_length, n_fft=n_fft)

    return [max(x, 0) for x in frames]


def frames_to_mls(frames, hop_length=hop_length, n_fft=n_fft):
    """
    Convert list of frame indices to list of milliseconds
    """
    if hop_length is None or hop_length <= 0:
        raise ValueError(
            'hop_length must be positive integer. Received {}'.format(
                hop_length))

    if n_fft is None or n_fft <= 0:
        raise ValueError(
            'n_fft must be positive integer. Received {}'.format(n_fft))

    if type(frames) != list and not isinstance(frames, Number):
        raise ValueError('frames must be either list of floats or a number. Received object {} of type {}'.format(frames, type(frames)))
    elif type(frames) == float:
        frames = [frames]

    times = librosa.frames_to_time(frames, hop_length=hop_length,
                                   n_fft=n_fft) * 1000

    return times


def bpm_to_beat_duration_mls(bpm):
    return (1 / (bpm / 60)) * 1000


def beat_duration_at_time(timing_points: List[dict], timestamp: float) -> float:
    """
    Return the beat duration (mls/beat) at the given timestamp
    """
    timestamp_reached = False

    for timing_point in reversed(timing_points):
        offset = float(timing_point['offset'])
        mls_per_beat = float(timing_point['mls_per_beat'])

        if not timestamp_reached:
            timestamp_reached = offset <= timestamp

        if timestamp_reached and mls_per_beat > 0:
            return mls_per_beat

    return float(timing_points[0]['mls_per_beat'])

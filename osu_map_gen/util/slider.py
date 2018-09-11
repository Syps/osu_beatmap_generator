import pdb
from typing import List

import numpy as np

from . import timing
from ..aisu_circles import position


def duration_mls(slider_multiplier: float, beat_duration_mls: float,
                 pixel_length: float) -> float:
    '''
    Get the duration (mls) of a slider with the given
    pixel length, beat duration.

    :param slider_multiplier
    :param beat_duration_mls
    :param pixel_length: pixel length of slider relative to
    512×384 virtual screen
    '''
    return pixel_length / (100.0 * slider_multiplier) * beat_duration_mls


def get_pixel_length(frame_duration, slider_multiplier, beat_duration_mls,
                     **kwargs):
    """
    Reverse solve for pixel length given duration_frames
    """
    mls_duration = timing.frames_to_mls(frame_duration, **kwargs)
    velocity = 100.0 * slider_multiplier
    n_beats = mls_duration / beat_duration_mls

    return n_beats * velocity


def duration_frames(slider_multiplier: float, beat_duration_mls: float,
                    pixel_length: float, **ft_args):
    duration = duration_mls(slider_multiplier, beat_duration_mls, pixel_length)
    len_frames = timing.mls_to_frames(duration, **ft_args)

    return len_frames


def start_end_mls(slider_event: dict, timing_points: List[dict],
                  slider_multiplier: float) -> tuple:
    start = float(slider_event['time'])
    pixel_length = float(slider_event['pixel_length'])
    beat_duration = timing.beat_duration_at_time(timing_points, start)
    end = start + duration_mls(slider_multiplier, beat_duration, pixel_length)

    return start, end


def start_end_frames(*args, **ft_args) -> list:
    """

    Params
    ------
    args = (
        slider_event: dict,
        timing_points: List[dict],
        slider_multiplier: float
    )

    ft_args {
        n_fft: int,
        hop_length: int
    }
    """
    _start_end_mls = start_end_mls(*args)
    _start_end_frames = timing.mls_to_frames(_start_end_mls, **ft_args)

    return _start_end_frames


def positions(slider_event, n_frames: int) -> np.ndarray:
    """
    Returns a list of positions covered by the slider
     path over the slider's frame duration
    """
    repeats = int(slider_event['repeat'])
    slider_type_and_path = slider_event['slider_type_curve_pts_str']
    slider_type = slider_type_and_path[0]
    path = slider_type_and_path[2:]
    start = int(slider_event['x']), int(slider_event['y'])

    points = [[int(x) for x in points.split(':')] for points in path.split('|')]

    if slider_type == 'L' or len(points) == 1:
        end = points[0]
        slider_positions = position.linear_positions(n_frames, start, end,
                                                     repeats)
    elif slider_type == 'P':
        try:
            pass_through, end = points
        except Exception:
            pdb.set_trace()
        slider_positions = position.circle_positions(n_frames, start,
                                                     pass_through, end, repeats)
    elif slider_type == 'B':
        points = [[int(slider_event['x']), int(slider_event['y'])]] + points
        slider_positions = position.bezier_positions(n_frames, points, repeats)
    else:
        raise Exception('Invalid slider type {}'.format(slider_type))

    return np.asarray(slider_positions)

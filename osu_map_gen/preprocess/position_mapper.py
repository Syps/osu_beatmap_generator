import math
import pdb
import numpy as np
from typing import List

from .convert import SimpleFrameConverter
from ..aisu_circles import position
from .. import timing


class StandardMapper:
    """
    Maps the position of all hit events over every song frame.
    Produces 2D array of shape (song_len, 2), w/ dimension 2 being xy coords.
    """
    def __init__(
            self,
            song_length: int,
            hit_events: dict,
            timing_points: List[dict],
            slider_multiplier,
            fill_method='none',
            converter=None
    ):
        self._song_len = song_length
        self._hit_events = hit_events
        self._timing_points = timing_points
        self._slider_multipler = float(slider_multiplier)
        self._positions = None
        self._converter = converter

        if fill_method not in ['none', 'repeat', 'interpolate']:
            raise ValueError('Fill method must be one of {{repeat, '
                             'interpolate}}. Received {}'.format(fill_method))

        self._fill_method = fill_method

    @property
    def hit_circles(self):
        _hit_circles = self._hit_events['hit_circles']
        _sorted = StandardMapper.sorted(_hit_circles)
        return _sorted

    @property
    def sliders(self):
        _sliders = self._hit_events['sliders']
        _sorted = StandardMapper.sorted(_sliders)
        return _sorted

    @property
    def spinners(self):
        _spinners = self._hit_events['spinners']
        _sorted = StandardMapper.sorted(_spinners)
        return _sorted

    @staticmethod
    def sorted(hit_events):
        sorted_hit_events = sorted(hit_events, key=lambda x: float(x['time']))

        return sorted_hit_events

    def _flatten(self, l):
        return [x for y in l for x in y]

    def get_positions(self) -> np.ndarray:
        """
        Returns
        -------
            self._positions: ndarray of shape (song_len, 2)
            Currently spinner events are not included.
        """
        if self._positions is not None:
            return self._positions

        self._positions = np.full((self._song_len, 2), -1)

        hit_circle_frames, hit_circle_positions = self._hit_circle_positions()
        slider_frames, slider_positions = self._slider_positions()
        spinner_frames, spinner_positions = self._spinner_positions()

        slider_frames = self._flatten(slider_frames)
        slider_positions = self._flatten(slider_positions)

        try:
            self._positions[hit_circle_frames, :] = hit_circle_positions
        except IndexError:
            pdb.set_trace()

        self._positions[slider_frames] = slider_positions
        # self._positions[spinner_frames] = spinner_positions

        self._fill_out_remaining_positions()

        return self._positions

    def _fill_out_remaining_positions(self):
        if self._fill_method == 'repeat':
            self._repeat_fill()

    def _repeat_fill(self):
        last_hit_position = np.asarray([512 / 2, 384 / 2])  # center

        for i, pos in enumerate(self._positions):
            if (pos[0], pos[1]) != (-1, -1):
                last_hit_position = pos
            else:
                self._positions[i] = last_hit_position

    def _interpolate_fill(self):
        pass

    def _hit_circle_positions(self):
        frame_indexes = []
        positions = np.empty((len(self.hit_circles), 2), dtype=np.float)

        for index, hit_circle in enumerate(self.hit_circles):
            frame_index = StandardMapper.frame_index(hit_circle)
            pos = np.asarray([hit_circle['x'], hit_circle['y']],
                                  dtype=np.float)

            frame_indexes.append(frame_index)
            positions[index] = pos

        return frame_indexes, positions

    def _slider_positions(self):
        if self._converter is None:
            '''
            hit_circles: List,
            sliders: List,
            spinners: List,
            timing_points: List,
            breaks: List,
            song_length: int,
            slider_multiplier: float,
            should_filter_breaks = True
            '''
            converter = SimpleFrameConverter(
                self.hit_circles,
                self.sliders,
                self.spinners,
                self._timing_points,
                [],
                self._song_len,
                self._slider_multipler,
                should_filter_breaks=False
            )
        else:
            converter = self._converter

        slider_frames = converter.slider_frames()

        positions = [self._single_slider_positions(ef) for ef in
                     zip(self.sliders, slider_frames)]

        return slider_frames, positions

    def _spinner_positions(self):
        return [], []  # stub

    def _single_slider_positions(self, args):
        """
        Return the positons at each frame for this slider

        :param args: (slider_event, array-like: duration_frames])
        :return: List[(x,y)] for representing positions for this
        slider over it's frame duration
        """
        event, frames = args
        repeats = int(event['repeat'])
        duration = len(frames)
        pixel_len = float(event['pixel_length'])
        slider_type_and_path = event['slider_type_curve_pts_str']
        slider_type = slider_type_and_path[0]
        path = slider_type_and_path[2:]
        start = int(event['x']), int(event['y'])

        points = [[int(x) for x in points.split(':')] for points in
                  path.split('|')]

        if slider_type == 'L' or len(points) == 1:
            end = points[0]
            positions = position.linear_positions(duration, start, end, repeats)
        elif slider_type == 'P':
            pass_through, end = points
            positions = position.circle_positions(pixel_len, duration, start,
                                                  pass_through,
                                                  end,
                                                  repeats)
        elif slider_type == 'B':
            points = [[int(event['x']), int(event['y'])]] + points
            positions = position.bezier_positions(duration, points, repeats)
        else:
            raise Exception('Invalid slider type {}'.format(slider_type))

        return np.asarray(positions)

    @staticmethod
    def frame_index(hit_event):
        timestamp = float(hit_event['time'])
        frame_index = timing.mls_to_frames(timestamp)[0]

        return frame_index

    @staticmethod
    def distance(x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


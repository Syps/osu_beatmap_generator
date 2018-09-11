from typing import List
import numpy as np
from .. import timing


def filter_breaks(breaks, frames: np.ndarray) -> np.ndarray:
    intervals = []

    for br in breaks:
        start = int(br['start'])
        end = int(br['end'])
        intervals.extend([start, end])

    subarrays = np.split(frames, intervals)
    return np.concatenate(subarrays[::2]) if len(subarrays) > 2 else subarrays[
        0]


class EventFrameConverter:
    def __init__(
            self,
            hit_circles: List,
            sliders: List,
            spinners: List,
            timing_points: List,
            breaks: List,
            song_length: int,  # in frames
            slider_multiplier: float,
            should_filter_breaks=True
    ):
        self._hit_circles = hit_circles
        self._sliders = sliders
        self._spinners = spinners
        self._timing_points = timing_points
        self._breaks = breaks
        self._song_length = song_length
        self._slider_multiplier = float(slider_multiplier) * 100.0
        self._filter_breaks = should_filter_breaks

    def convert(self):
        raise NotImplementedError

    def _convert_hit_circles(self):
        timestamps_mls = map(lambda x: x['time'], self._hit_circles)

        return timing.mls_to_frames(timestamps_mls)

    def _slider_duration_mls(self, beat_duration_mls, pixel_length) -> float:
        """
        Get the duration (mls) of a slider with the given pixel length,
         beat duration

        :param beat_duration_mls
        :param pixel_length: pixel length of slider relative
        to 512×384 virtual screen
        :param repeats number of times slider path is traced (1 == no repeat)
        """
        slider_len = pixel_length / self._slider_multiplier * beat_duration_mls

        return slider_len


class SimpleFrameConverter(EventFrameConverter):
    def convert(self):
        """
        Return ndarray of shape (self._song_length - total_break_time) where
        item at each index is either 1 for hit activity occurring, or 0
        """
        frames = np.zeros(self._song_length)
        hit_circle_frames = self._convert_hit_circles()
        slider_frames = self.slider_frames()

        if len(slider_frames) > 0:
            slider_frames = np.concatenate(slider_frames)
        else:
            slider_frames = np.asarray([], dtype=np.int32)

        spinner_frames = self._convert_spinners()

        hit_event_frames = np.concatenate(
            [hit_circle_frames, slider_frames, spinner_frames]
        )

        sorted_unique_frames = np.unique(np.sort(hit_event_frames))
        frames[sorted_unique_frames] = 1

        if self._filter_breaks:
            filtered_frames = filter_breaks(self._breaks, frames)
            return filtered_frames
        else:
            return frames

    def slider_frames(self) -> List:
        frames = list(map(self._convert_single_slider, self._sliders))

        return frames

    def _convert_single_slider(self, event):
        start_end_time_mls = self._slider_start_end(event)
        start_end_frames = timing.mls_to_frames(start_end_time_mls)
        duration_frames = np.arange(*start_end_frames)
        repeats = int(event['repeat'])

        # hack to make discrete frames/positions equivalent w/ repeats
        frames_repeats_diff = duration_frames.shape[0] % repeats
        if frames_repeats_diff == 0:
            duration_frames = duration_frames
        else:
            duration_frames = duration_frames[:-1 * frames_repeats_diff]

        return duration_frames

    def _convert_spinners(self):
        return np.asarray([], dtype=np.int32)

    def _slider_start_end(self, slider_event) -> tuple:
        start = float(slider_event['time'])
        pixel_length = float(slider_event['pixel_length'])
        beat_duration = self._beat_duration_at_time(start)
        end = start + self._slider_duration_mls(beat_duration, pixel_length)

        return start, end

    def _beat_duration_at_time(self, timestamp: float) -> float:
        timestamp_reached = False

        for timing_point in reversed(self._timing_points):
            offset = float(timing_point['offset'])
            mls_per_beat = float(timing_point['mls_per_beat'])

            if not timestamp_reached:
                timestamp_reached = offset <= timestamp

            if timestamp_reached and mls_per_beat > 0:
                return mls_per_beat

        return float(self._timing_points[0]['mls_per_beat'])


class EmptySliderFrameConverter(SimpleFrameConverter):
    def _convert_single_slider(self, event):
        start_end_time_mls = self._slider_start_end(event)
        start_end_frames = timing.mls_to_frames(start_end_time_mls)
        duration_frames = list(start_end_frames)
        distance = duration_frames[1] - duration_frames[0]
        repeats = int(event['repeat'])

        for repeat in range(1, repeats):
            additional_frame = start_end_frames[0] + repeat * distance
            if additional_frame < self._song_length:
                duration_frames.append(additional_frame)

        return duration_frames


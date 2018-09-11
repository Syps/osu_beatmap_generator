import pdb
from collections import deque

import joblib
import numpy as np
from keras.models import Model

from osu_map_gen.util import hyper_params


class HitModel:

    def __init__(self, model: Model, data: np.ndarray, scaler_path=None):
        self._model = model
        self._data = data
        self._scaler = joblib.load(scaler_path) if scaler_path else None

    def predict(self):
        raise NotImplementedError


HIT_CIRCLE = 0
SLIDER = 1
NONE = 1


class SingleDifficultyHitModel(HitModel):

    def _hit_val(self, predict_vals):
        """
        Transforms multi label array in binary encoded array
        """
        max_label = np.argmax(predict_vals)
        # if np.where(predict_vals > 0.7)[0].shape[0] == 0:
        #     return 0 if not in_slider else 1, in_slider

        if max_label == HIT_CIRCLE:  # hit circle
            if predict_vals[0, 0] > 0.8:
                hit_val = 1
            else:
                hit_val = 0
        elif max_label == NONE:  # none
            # hit_val = 0 if not in_slider else 1
            hit_val = 0
        elif max_label == SLIDER:  # slider
            # for now ignore repeats
            pdb.set_trace()
            hit_val = 1
        else:
            raise ValueError("unexpected label {}".format(max_label))

        return hit_val

    def predict(self, add_z_channel=False):
        n_seq, seq_len, n_features = self._data.shape
        ctx_len = hyper_params.context_length
        hit_events = np.zeros(n_seq)
        context_labels = deque(ctx_len * [0], ctx_len)

        for i in range(n_seq):
            sequence = self._data[i]
            context = np.zeros(hyper_params.sample_frames)
            c = int(ctx_len / 2)
            context[c: c + ctx_len] = np.asarray(context_labels)
            sequence[:, -1] = context

            inputs = [np.expand_dims(sequence, 0)]

            if add_z_channel:
                inputs = [np.expand_dims(inputs[0], 3)]

            predict_vals = self._model.predict(inputs)

            hit_val = self._hit_val(predict_vals)

            context_labels.append(hit_val)
            hit_events[i] = hit_val

        return hit_events


class MultipleDifficultiesModel(HitModel):

    def _hit_val(self, predict_vals, dist_since_last_hit):
        max_label = np.argmax(predict_vals)

        threshold = 0.6 if dist_since_last_hit > 7 else 0.8

        if max_label == 0 or predict_vals[0, max_label] < threshold:
            hit_val = 0
        else:
            hit_val = max_label

        return hit_val

    def predict(self, add_z_channel=False):
        n_seq, seq_len, n_features = self._data.shape
        ctx_len = hyper_params.context_length

        medium_hit_events = np.zeros(n_seq)
        hard_hit_events = np.zeros(n_seq)

        context_labels = deque(ctx_len * [0], ctx_len)

        print('predicting multiple difficulties...')

        dist_since_last_hit = 34

        for i in range(n_seq):
            sequence = self._data[i]
            context = np.zeros(hyper_params.sample_frames)
            c = int(ctx_len / 2)
            context[c: c + ctx_len] = np.asarray(context_labels)
            sequence[:, -1] = context

            inputs = [np.expand_dims(sequence, 0)]

            if add_z_channel:
                inputs = [np.expand_dims(inputs[0], 3)]

            predict_vals = self._model.predict(inputs)

            hit_val = self._hit_val(predict_vals, dist_since_last_hit)

            scaled_hit_val = (hit_val - self._scaler.mean_[-1]) / self._scaler.scale_[-1]
            context_labels.append(scaled_hit_val)

            medium_hit_events[i] = 1 if hit_val in [1, 3] else 0
            hard_hit_events[i] = 1 if hit_val in [2, 3] else 0

            if hit_val != 0:
                dist_since_last_hit = 1
            else:
                dist_since_last_hit += 1

        return {
            'medium': medium_hit_events,
            'hard': hard_hit_events
        }

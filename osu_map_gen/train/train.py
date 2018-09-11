import _pickle as cPickle
import collections
import operator
import os
import pdb
import random
from functools import reduce
from typing import List

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import view_as_windows
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Dense, Conv2D, Dropout, \
    Flatten, MaxPooling2D, BatchNormalization
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.optimizers import SGD

from ..preprocess.convert import EmptySliderFrameConverter
from .generator import HDF5DataGenerator
from .. import (
    utils,
    slider,
    timing,
    db,
    definitions,
    hyper_params
)

from . import data_utils

DIM_X = 69
DIM_Y = 41
DIM_Z = 1

seq_len = hyper_params.sample_frames
hidden_units = 128

n_mels = hyper_params.n_mels
n_fft = hyper_params.n_fft
hop_len = hyper_params.hop_length

dataset_file = 'train_inputs'
model_file = 'model_hits.test'
scaler_file = 'scaler_hits'


def flatten_array(t):
    for x in t:
        if type(x) == dict \
                or type(x) == tuple \
                or not isinstance(x, collections.Iterable):
            yield x
        else:
            yield from flatten_array(x)


def get_scaler(path):
    _scaler = joblib.load(path)

    return _scaler


def get_model(path):
    _model = load_model(path)

    return _model


def get_sequences(melgram, _n_mels):
    """
    Transform 2 or 3D mel spectrogram into 3D array of windows

    :param melgram: 2 or 3D mel spectrogram (if 3D, squeezed to 2D)
    :param _n_mels: number of mel buckets
    :return: array of sequences. shape (n_frames, seq_len, n_mels)
    """
    if len(melgram.shape) == 3:
        melgram = np.squeeze(melgram)

    sequences = view_as_windows(melgram, window_shape=(seq_len, _n_mels))
    sequences = np.squeeze(sequences, 1)

    return sequences


preprocess_args = {
    'cut_beginning': False,
    'seq_len': seq_len,
    'beatmap_id': '405096'
}

optimizer_args = {
    'lr': 1e-2,  # was 1e-3
    'decay': 1e-6,
    'momentum': 0.9,
    'nesterov': False
}

compile_args = {
    'loss': categorical_crossentropy,
    'optimizer': SGD(**optimizer_args),
    'metrics': ['accuracy']
}


def get_fit_args(model_save_path):
    return {
        'batch_size': 32,
        'epochs': 30,
        'shuffle': False,
        'callbacks': [ModelCheckpoint(model_save_path, save_best_only=False)]
    }


def get_data_paths(data_dir) -> dict:
    model_save_path = '{}/{}.h5'.format(data_dir, model_file)
    continued_model = '{}/{}.continued.h5'.format(data_dir, model_file)
    root_dir = '{}/train/model_data'.format(definitions.PROJECT_DIR)

    scaler_save_path = '{}/scaler_hits.save'.format(data_dir)
    all_table_path = '{}/all_train.hdf5'.format(data_dir)
    train_table_path = '{}/train.hdf5'.format(data_dir)
    hc_counts = '{}/hc_counts.npy'.format(data_dir)
    n_counts = '{}/n_counts.npy'.format(data_dir)
    val_table_all_path = '{}/all_validation.hdf5'.format(data_dir)
    validation_data_path = '{}/validation.hdf5'.format(data_dir)
    val_index_map_path = '{}/val_index_map.pkl'.format(data_dir)
    all_index_map_path = '{}/all_index_map.pkl'.format(data_dir)
    range_data_path = '{}/range_data.json'.format(data_dir)
    train_inputs_path = '{}/train_inputs'.format(root_dir)
    train_labels_path = '{}/train_labels'.format(root_dir)

    return {
        'model': model_save_path,
        'continued_model': continued_model,
        'range_data': range_data_path,
        'scaler': scaler_save_path,
        'all_table': all_table_path,
        'train_table': train_table_path,
        'train_inputs': train_inputs_path,
        'train_labels': train_labels_path,
        'val_table_all': val_table_all_path,
        'validation_data': validation_data_path,
        'val_index_map': val_index_map_path,
        'hc_counts': hc_counts,
        'n_counts': n_counts,
        'all_index_map': all_index_map_path
    }


def get_frame_indexes(objs, n_fft=hyper_params.n_fft,
                      hop_length=hyper_params.hop_length) -> List:
    frames = [timing.mls_to_frames(float(x['time']),
                                   n_fft=n_fft,
                                   hop_length=hop_length)[0]
              for x in objs]

    return list(map(lambda x: int(x), frames))


def get_input_shape(dim_1):
    return dim_1, DIM_X, DIM_Y


def build_inputs(spectrogram, indexes, bin_encoded, label):
    '''
    Returns np.array of training inputs for the events taking
    places at the given indexes

    params:
    -------
    bin_encoded: np.array (spectrogram.shape[0], d) where
     d is the number of difficulties

    returns:
    --------
    np.array: (len(indexes), sample_frames, mel_buckets + difficulties)
    '''

    if bin_encoded.shape[0] != spectrogram.shape[0]:
        raise ValueError('''
            Length of bin_encoded must match that of spectrogram. Found
            the following: bin_encoded.shape = {}, spectrogram.shape = {}.
        '''.format(bin_encoded.shape, spectrogram.shape))

    dim1 = len(indexes)
    labels = np.zeros((dim1, 4))
    labels[:, label] = 1
    inputs = np.empty(get_input_shape(dim1))
    ctx_len = hyper_params.context_length

    start_context_rows = -1

    for i, index in enumerate(indexes):
        context_labels = np.zeros((hyper_params.sample_frames, 1))
        c = int((hyper_params.sample_frames - ctx_len) / 2)
        context_labels[c:c + ctx_len] = bin_encoded[index - ctx_len: index]

        inputs[i, :, :start_context_rows] = spectrogram[
                                            index - ctx_len: index + ctx_len + 1]
        inputs[i, :, start_context_rows:] = context_labels

    return zip(inputs, labels)


def get_slider_points(beatmap_data) -> List[int]:
    sliders = beatmap_data['sliders']
    timing_points = beatmap_data['timing_points']
    slider_multiplier = float(beatmap_data['metadata']['slider_multiplier'])

    slider_points = []

    for s in sliders:
        repeats = int(s['repeat'])
        start, end = slider.start_end_frames(
            s,
            timing_points,
            slider_multiplier,
            n_fft=hyper_params.n_fft,
            hop_length=hyper_params.hop_length
        )
        slider_points.append(int(start))
        slider_points.append(int(end))

        duration = end - start
        for repeat in range(1, repeats):
            time = start + duration * (repeat + 1)
            slider_points.append(int(time))

    return slider_points


def get_bin_encoded(beatmap_data, song_length, pad=True):
    bin_encoded = EmptySliderFrameConverter(
        hit_circles=beatmap_data['hit_circles'],
        sliders=beatmap_data['sliders'],
        spinners=beatmap_data['spinners'],
        timing_points=beatmap_data['timing_points'],
        breaks=beatmap_data['breaks'],
        song_length=song_length,
        slider_multiplier=float(beatmap_data['metadata']['slider_multiplier']),
        should_filter_breaks=False
    ).convert()

    if pad:
        pad_width = (
            hyper_params.context_length,
            hyper_params.context_length + 1
        )
        bin_encoded = np.pad(
            bin_encoded,
            pad_width,
            mode='constant',
            constant_values=0
        )

    return bin_encoded


def get_hit_indexes(beatmap_data):
    return get_slider_points(beatmap_data) + \
           get_frame_indexes(beatmap_data['hit_circles'])


def get_hit_vals(bin_encoded, indexes_hit, indexes_none):
    '''
    Returns
    -------
        hit_vals: np.array (n_hits, 2 * n_difficulties)

        1-Hot vector:
        [
            0, - no hit
            0, - medium hit only
            0, - hard hit only
            0  - medium and hard hits
        ]
    '''
    encoder = OneHotEncoder(4, sparse=False)
    encoded = encoder.fit_transform(bin_encoded)

    return np.concatenate(
        (encoded[sorted(indexes_hit)], encoded[sorted(indexes_none)]))


def get_label_hits(hit_dict, pad=False):
    medium_hits = []
    hard_hits = []
    both_hits = []

    padding = hyper_params.context_length if pad else 0
    for key, l in hit_dict.items():
        key += padding
        if len(l) > 1:
            both_hits.append(key)
        elif l[0] == 0:
            medium_hits.append(key)
        else:
            hard_hits.append(key)

    return medium_hits, hard_hits, both_hits


def get_inputs(beatmap_data: List, spectrogram, limit_hc=None, limit_n=None,
               flatten=True):
    breaks = reduce(operator.add, [b['breaks'] for b in beatmap_data])
    song_len = spectrogram.shape[0]

    bin_encoded = np.zeros((song_len, 1))
    hit_dict = {}
    for i, beatmap in enumerate(beatmap_data):
        hits = sorted(get_hit_indexes(beatmap))
        bin_encoded[hits] = bin_encoded[hits] + i + 1
        for hit in hits:
            if hit not in hit_dict:
                hit_dict[hit] = [i]
            else:
                hit_dict[hit].append(i)

    medium_indexes, hard_indexes, both_indexes = get_label_hits(hit_dict,
                                                                pad=True)
    hit_indexes = medium_indexes + hard_indexes + both_indexes

    if len(set(hit_indexes)) != len(hit_indexes):
        pdb.set_trace()

    spectrogram = utils.pad_array(spectrogram)
    bin_encoded = utils.pad_array(bin_encoded)

    none_indexes = get_none_indexes(hit_indexes, spectrogram, breaks)

    none_inputs = build_inputs(spectrogram, none_indexes, bin_encoded, label=0)
    medium_inputs = build_inputs(spectrogram, medium_indexes, bin_encoded,
                                 label=1)
    hard_inputs = build_inputs(spectrogram, hard_indexes, bin_encoded, label=2)
    both_inputs = build_inputs(spectrogram, both_indexes, bin_encoded, label=3)

    none_groups = get_group_lists(none_inputs, group_size_limit=limit_n,
                                  label=0)
    medium_groups = get_group_lists(medium_inputs, group_size_limit=limit_hc,
                                    label=1)
    hard_groups = get_group_lists(hard_inputs, group_size_limit=limit_hc,
                                  label=2)
    both_groups = get_group_lists(both_inputs, group_size_limit=limit_hc,
                                  label=3)

    if flatten:
        all_inputs = flatten_array(
            none_groups + medium_groups + hard_groups + both_groups
        )
        # split into [(input, label), ...] and [(label, x-label, group), ...]
        input_labels, coords = zip(*all_inputs)
        # split into [input, ...] and [label, ...]
        all_inputs, all_labels = zip(*input_labels)

        return all_inputs, all_labels, coords

    return none_groups, medium_groups, hard_groups, both_groups


def get_counts(*args, limit_n=None):
    inputs = get_inputs(*args, limit_n=limit_n, flatten=False)

    counts = np.zeros((4, 3, 35))

    if inputs is None:
        return counts

    def to_sum_array(g):
        return [[len(l) for l in x] for x in g]

    g_n, g_m, g_h, g_b = [to_sum_array(i) for i in inputs]

    counts[0] = g_n
    counts[1] = g_m
    counts[2] = g_h
    counts[3] = g_b

    return counts


def get_none_indexes(event_indexes, spectrogram, breaks,
                     limit=None) -> np.ndarray:
    intervals = []
    for b in breaks:
        intervals.append(b['start'])
        intervals.append(b['end'])

    song_range = np.arange(
        hyper_params.context_length,
        spectrogram.shape[0] - hyper_params.context_length
    )
    subarrays = np.split(song_range, intervals)
    valid_indexes = np.concatenate(subarrays[::2]) if len(subarrays) > 2 else \
        subarrays[0]

    none_indexes = np.delete(valid_indexes, event_indexes)
    np.random.shuffle(none_indexes)

    if limit is None:
        return none_indexes
    else:
        return none_indexes[:limit]


def arg_nonzero(arr):
    """
    :param arr: 1D array
    :return: index of first non-zero element or -1 if no such elements exist
    """
    try:
        return np.flatnonzero(arr)[0]
    except IndexError:
        return -1


def get_ctx_group_number(sample):
    # go in reverse to even out group counts
    ctx_row = sample[:, -1][17:17 + 34][
              ::-1]  # last 34 frames before the inference frame
    a_max = arg_nonzero(ctx_row)
    if a_max >= 0:
        hit_val = ctx_row[a_max]
        group = ctx_row.shape[0] - a_max
    else:
        # 0 will be decremented to -1 and we'll end up storing
        # all of the empty-ctx-row values in one cross-label ("both")
        hit_val = 0
        group = 0  # group 0 corresponds to inputs with no preceeding hit events

    return group, hit_val


def get_group_lists(inputs, group_size_limit=None, label=0):
    """
    Returns
    -------
    group_lists: List[List[np.ndarray]]
        Each inner list contains all the inputs belonging to
        the group that corresponds to that index.
        An index or group here represents the frame number relative the sample.

        If group_size_limit is None and use_min_as_limit is false,
        simply returns all the groups across all the inputs.
        Otherwise, each group list is cut off by the given limit.
    """
    limit_type = type(group_size_limit)

    if group_size_limit is not None and limit_type != int:
        raise ValueError(
            "Group size limit must be int. Received type {}".format(limit_type)
        )

    n_groups = hyper_params.context_length + 1  # 35 groups
    n_cross_labels = 3  # medium, hard, both
    #  categorize input as one group of 35 in one
    #  of 3 different cross-label types
    group_lists = [[[] for _ in range(n_groups)] for _ in range(n_cross_labels)]

    for i in inputs:  # [(input, label), ...]
        group, hit_value = get_ctx_group_number(i[0])
        cross_label = int(hit_value - 1)  # any 0s will get grouped in "both"

        # Group lists is of type <((input, label), (label, cross-label, group))>
        group_lists[cross_label][int(group)].append(
            (i, (label, cross_label, group)))

    if group_size_limit is not None:
        #  take a random selection of size `group_size_limit`
        #  from each group in each label
        group_lists = [[random_selection(group, group_size_limit) for group in
                        label_groups] for label_groups in group_lists]

    return group_lists


def random_selection(inputs: list, selection: int) -> list:
    random.shuffle(inputs)
    return inputs[:selection]


def build_table(file_name, songs, inputs_shape, labels_shape, n_limit_per_song,
                save_index_path=None):
    table_groups = [
        {
            'name': 'inputs',
            'shape': inputs_shape,
            'dtype': np.float64
        },
        {
            'name': 'labels',
            'shape': labels_shape,
            'dtype': np.int
        }
    ]

    table = create_hdf5_table(file_name, *table_groups)
    inputs = table['inputs']
    labels = table['labels']

    fill_inputs_labels(inputs, labels, songs, n_limit_per_song,
                       save_index_path=save_index_path)

    print('table shape[0] was {}'.format(labels_shape[0]))

    return table


def build_empty_list_map():
    # shape -> (4, 3, 35, ?)
    return [[[[] for _ in range(35)] for _ in range(3)] for _ in range(4)]


def fill_inputs_labels(inputs_array, labels_array, songs, n_limit_per_song,
                       save_index_path=None):
    i = 0
    list_map = build_empty_list_map()

    for song_path, beatmap_ids in songs:
        print('iteration {}'.format(i))
        spectrogram = np.load(song_path)
        beatmap_data = [db.beatmap_data(beatmap_id) for beatmap_id in
                        beatmap_ids]
        try:
            inputs, labels, coords = get_inputs(
                beatmap_data,
                spectrogram,
                limit_n=n_limit_per_song
            )
        except:
            print('input_data was None for song_path {}. Delete!'.format(
                song_path))
            continue

        n = len(inputs)

        inputs_array[i:i + n] = inputs
        labels_array[i:i + n] = labels

        for index, coord in enumerate(coords):
            label, cross_label, group = coord
            # gather all the "all_table" indexes into coord map
            if save_index_path:
                list_map[label][cross_label][group].append(index + i)

        i += n

        print('total len is {}'.format(i))

    if save_index_path:
        print('saving list_map...')
        with open(save_index_path, 'wb') as f:
            cPickle.dump(list_map, f)
        print('list_map saved')


def rm_if_exists(path):
    if os.path.exists(path):
        os.remove(path)


def create_hdf5_table(file_name, *groups, rm_existing=True):
    if rm_existing:
        rm_if_exists(file_name)

    if os.path.exists(file_name):
        hdf5_file = h5py.File(file_name, mode='r+')
    else:
        hdf5_file = h5py.File(file_name, mode='w')
        for group in groups:
            hdf5_file.create_dataset(group['name'], group['shape'],
                                     group['dtype'])

    return hdf5_file


def build_filtered_table(file_name, all_h5, n_train_inputs, n_difficulties,
                         limit, index_map_path, scaler_path=None):
    train_inputs_shape = (n_train_inputs, DIM_X, DIM_Y)
    train_labels_shape = (n_train_inputs, 2 * n_difficulties)

    with open(index_map_path, 'rb') as f:
        list_map = cPickle.load(f)

    # to achieve even distribution of different song sample variants,
    # choose a random selection (min(len, n=limit)) for each group
    for i_l, label in enumerate(list_map):
        for i_cl, cross_label in enumerate(label):
            for i_g, group in enumerate(cross_label):
                group_inputs = list_map[i_l][i_cl][i_g]
                list_map[i_l][i_cl][i_g] = random_selection(group_inputs, limit)

    all_table_indexes = [x for x in flatten_array(list_map)]
    random.shuffle(all_table_indexes)

    groups = [
        {
            'name': 'inputs_unscaled',
            'shape': train_inputs_shape,
            'dtype': np.float64
        },
        {
            'name': 'inputs_scaled',
            'shape': train_inputs_shape,
            'dtype': np.float64
        },
        {
            'name': 'labels',
            'shape': train_labels_shape,
            'dtype': np.int
        }
    ]

    all_inputs = all_h5['inputs']
    all_labels = all_h5['labels']

    preexisting = os.path.exists(file_name)
    train_h5 = create_hdf5_table(file_name, *groups,
                                 rm_existing=preexisting)
    inputs_unscaled = train_h5['inputs_unscaled']
    inputs_scaled = train_h5['inputs_scaled']
    labels = train_h5['labels']

    if scaler_path:
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler()

    if not preexisting:
        for train_index, all_index in enumerate(all_table_indexes):

            if train_index % 100 == 0:
                print('shuffling {}/{}'.format(train_index, n_train_inputs))

            x = all_inputs[all_index]
            y = all_labels[all_index]

            inputs_unscaled[train_index] = x
            labels[train_index] = y

            if not scaler_path:
                scaler.partial_fit(x)

    indexes = np.arange(0, n_train_inputs)

    print('scaling inputs...')
    for index in indexes:
        if index % 100 == 0:
            print('scaling {}/{}'.format(index, n_train_inputs))
        inputs_scaled[index] = scaler.transform(inputs_unscaled[index])

    return train_h5, scaler


def build_val_data(paths, songs, limit, n_limit_per_group):
    '''
    Returns a generator that yields validation inputs and labels
    '''
    n_difficulties = len(songs[0][1])
    input_totals = compute_totals(songs, n_limit_per_group)
    total_val = np.sum(input_totals)
    total_capped = np.sum(
        np.fmin(input_totals, np.full(input_totals.shape, limit)))

    val_inputs_shape = get_input_shape(total_val)
    val_labels_shape = (total_val, 2 * n_difficulties)

    print('About to build val_table...')
    val_table_all = build_table(
        paths['val_table_all'],
        songs,
        val_inputs_shape,
        val_labels_shape,
        n_limit_per_group,
        save_index_path=paths['val_index_map']
    )

    build_filtered_table(
        paths['validation_data'],
        val_table_all,
        total_capped,
        n_difficulties,
        limit,
        paths['val_index_map'],
        scaler_path=paths['scaler']
    )


def build_training_data(songs, file_paths, limit, n_limit_per_group=None,
                        refresh_totals=True):
    n_difficulties = len(songs[1])

    if refresh_totals:
        totals = compute_totals(songs, n_limit_per_group,
                                save_file='hit_totals')
        np.save('hit_totals', totals)

    print('Totals Calculated!!')

    totals = np.load('hit_totals.npy')
    total_all = int(np.sum(totals))
    total_capped = int(np.sum(np.fmin(totals, np.full(totals.shape, limit))))

    all_table_path = file_paths['all_table']
    all_inputs_shape = get_input_shape(total_all)
    all_labels_shape = (total_all, 2 * n_difficulties)

    if not os.path.exists(all_table_path):
        print('About to build all_table...')
        all_h5 = build_table(all_table_path, songs, all_inputs_shape,
                             all_labels_shape, n_limit_per_group,
                             save_index_path=file_paths['all_index_map'])
    else:
        all_h5 = h5py.File(all_table_path, mode='r')

    print('About to build train_table...')
    train_h5, scaler = build_filtered_table(
        file_paths['train_table'],
        all_h5,
        total_capped,
        n_difficulties,
        limit,
        file_paths['all_index_map']
    )

    joblib.dump(scaler, file_paths['scaler'])


def get_generator(table, batch_size, first=None, last=None, n_difficulties=1):
    X = table['inputs_scaled']
    y = table['labels']

    return HDF5DataGenerator(batch_size, first=first, last=last,
                             n_difficulties=n_difficulties).generate(X, y)


def train(file_paths, n_difficulties=1, plot_history=False):
    batch_size = 32
    epochs = 25
    train_table = h5py.File(file_paths['train_table'], 'r')
    validation_table = h5py.File(file_paths['validation_data'], 'r')
    first = 1
    train_generator = get_generator(train_table, batch_size, first=first,
                                    n_difficulties=n_difficulties)
    validation_generator = get_generator(validation_table, batch_size,
                                         n_difficulties=n_difficulties)
    steps_per_epoch = int(
        first * train_table['inputs_scaled'].shape[0] // batch_size)
    validation_steps = int(
        validation_table['inputs_scaled'].shape[0] // batch_size)

    model = Sequential()
    model.add(Conv2D(128, kernel_size=(3, 3),
                     padding='same',
                     activation='relu',
                     input_shape=(DIM_X, DIM_Y, DIM_Z)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(
        Conv2D(128, padding='same', kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    # model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    fit_args = {
        'validation_data': validation_generator,
        'validation_steps': validation_steps,
        'steps_per_epoch': steps_per_epoch,
        'epochs': epochs,
        'callbacks': [ModelCheckpoint(file_paths['model'], save_best_only=True)]
    }

    model.compile(**compile_args)
    history = model.fit_generator(train_generator, **fit_args)

    if plot_history:
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


def compute_totals(songs, limit_n, save_file=None):
    """
    Return array of shape (4, 3, 35) representing counts for
    each group of each context type of each label
    """
    totals = np.zeros((4, 3, 35), dtype='int32')

    i = 0
    for song_path, beatmap_ids in songs:
        print('song {}'.format(i))
        spectrogram = np.load(song_path)
        beatmap_data = [db.beatmap_data(beatmap_id) for beatmap_id in
                        beatmap_ids]

        counts = get_counts(beatmap_data, spectrogram, limit_n=limit_n)
        totals[:] = totals + counts

        i += 1

    if save_file:
        np.save(save_file, totals)

    return totals


def get_multi_diff_song_data(song_paths_ids):
    ids_to_paths = {s[1]: s[0] for s in song_paths_ids}
    medium_beatmap_ids = [x.split('.')[0] for x in
                          os.listdir(definitions.XL_SONG_SET)]
    query = {
        'metadata.beatmap_id': {'$in': medium_beatmap_ids}
    }

    projection = {
        'metadata.beatmap_id': 1,
        'metadata.beatmapset_id': 1
    }

    beatmaps = db.beatmap_data(query=query, projection=projection, multi=True)

    DIFFICULTY_HARD = {'$gte': 3.2, '$lt': 6}
    beatmap_sets = {
        x['metadata']['beatmapset_id']: {'ids': [x['metadata']['beatmap_id']]}
        for x
        in beatmaps}

    for set_id, medium_id in beatmap_sets.items():
        beatmap_sets[set_id]['song_path'] = ids_to_paths[medium_id]

    query = {
        'metadata.beatmapset_id': {'$in': beatmap_sets},
        'metadata.difficultyrating': DIFFICULTY_HARD
    }

    projection = {
        'metadata.beatmap_id': 1,
        'metadata.beatmapset_id': 1
    }

    hard_beatmaps = db.beatmap_data(query=query, projection=projection,
                                    multi=True)

    for beatmap in hard_beatmaps:
        beatmap_sets[beatmap['metadata']['beatmapset_id']]['ids'].append(
            beatmap['metadata']['beatmap_id'])

    return [(s['song_path'], s['ids']) for s in beatmap_sets]


def main():
    data_dest = data_utils.get_data_dest(use_latest=True)
    paths = get_data_paths(data_dest)
    songs = data_utils.get_multi_diff_song_data()
    limit = 2500  # was 1000
    train_split = 700

    # build_training_data(
    #     songs=songs[:train_split],  # len = 864
    #     file_paths=paths,
    #     limit=limit,
    #     n_limit_per_group=10, # was 3
    #     refresh_totals=True
    # )
    #
    # build_val_data(paths, songs[train_split:], 100, 1)

    train(paths, n_difficulties=2, plot_history=True)

    # totals = compute_totals(songs, 1)
    # pdb.set_trace()

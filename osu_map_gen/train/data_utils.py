import _pickle as cPickle
import json
import os
import pdb
import shutil
from glob import glob
from typing import List
import numpy as np

from .. import (
    db,
    definitions,
    hyper_params,
    music
)

train_song = 'find_you'

base_path = '{}/train/songs'.format(definitions.PROJECT_DIR)
npy_template = '{}/{{}}.npy'.format(base_path)
base_data_dir = '{}/train/model_data'.format(definitions.PROJECT_DIR)

n_mels = hyper_params.n_mels
n_fft = hyper_params.n_fft
hop_len = hyper_params.hop_length

eval_song = {
    'name': 'ebb_and_flow',
    'beatmap_id': '352453'
}

eval_song_id = '352453'
eval_song_file = '{}/large_song_set/{}.npy'.format(definitions.OSU_DATA_DIR,
                                                   eval_song_id)

local_songs = [
    {
        'name': 'find_you',
        'beatmap_id': '405096',
    },
    {
        'name': 'dear_you',
        'beatmap_id': '136399',
    },
    {
        'name': 'lull',
        'beatmap_id': '338772',
    },
    {
        'name': 'wareta_ringo',
        'beatmap_id': '191796',
    },
    {
        'name': 'vidro_moyou',
        'beatmap_id': '308910',
    },
]


def get_local_song_paths_ids() -> List[tuple]:
    return [('{}/{}.npy'.format(base_path, x['name']), x['beatmap_id']) for x in
            local_songs]


def get_local_song_ids():
    return [x['beatmap_id'] for x in local_songs]


def get_beatmap_id(query, multi=True):
    if type(query) != dict:
        raise ValueError("query filters must be a dict, received {}"
                         .format(query))

    projection = {'metadata.beatmap_id': 1}
    result = db.beatmap_data(query=query, projection=projection, multi=multi)
    print('beatmap_found')
    return result['metadata']['beatmap_id'] if result is not None else None


def sam_favorite_beatmapset_ids():
    with open(definitions.SAM_FAVS, 'r') as f:
        lines = f.read().split('\n')
        beatmapset_ids = map(lambda l: l.split(' ')[0], lines)

    return list(beatmapset_ids)


def _update_db_entry(beatmap_id, beatmap_data):
    try:
        db.db.Beatmaps.update_one(
            {"metadata.beatmap_id": beatmap_id},
            {'$set': beatmap_data}
        )
    except Exception as e:
        print(str(e))


def update_favourites_type():
    file_pattern = '{}/*/*.mp3'.format(definitions.BEATMAPS_UNZIPPED)
    files = glob(file_pattern)
    beatmapset_ids = [f.split('/')[-2] for f in files]

    query = {
        'metadata.beatmapset_id': {'$in': beatmapset_ids},
    }

    projection = {
        'metadata.beatmap_id': 1,
        'metadata.favourite_count': 1
    }

    data = db.beatmap_data(beatmap_id=None, query=query, projection=projection,
                           multi=True)

    pdb.set_trace()

    count = 1
    for d in data:
        print('updating {}'.format(count))
        beatmap_id, fav_count = d['metadata']['beatmap_id'], \
                                d['metadata']['favourite_count']

        _update_db_entry(beatmap_id,
                         {'metadata.favourite_count': int(fav_count)})

        count += 1


def build_xl_song_set_spect_dir(reset=False, min_favs=250):
    song_dir = '{}/xl_song_set'.format(definitions.OSU_DATA_DIR)

    if reset and os.path.exists(song_dir):
        shutil.rmtree(song_dir)

    if not os.path.exists(song_dir):
        os.mkdir(song_dir)

    print('finding files...')

    song_path_ids_file = 'xl_song_set_path_ids.pkl'

    if not os.path.exists(song_path_ids_file):

        file_pattern = '{}/*/*.mp3'.format(definitions.BEATMAPS_UNZIPPED)
        files = glob(file_pattern)
        beatmapset_ids = [f.split('/')[-2] for f in files]

        query = {
            'metadata.beatmapset_id': {'$in': beatmapset_ids},
            'metadata.difficultyrating': {'$gt': 2.6, '$lt': 3.3},
            '$or': [
                {
                    'metadata.favourite_count': {'$gt': min_favs}
                },
                {
                    'metadata.beatmapset_id': {
                        '$in': sam_favorite_beatmapset_ids()
                    }
                }
            ]
        }

        beatmap_data = db.beatmap_data(
            beatmap_id=None,
            query=query,
            projection={
                'metadata.beatmap_id': 1,
                'metadata.beatmapset_id': 1,
                'metadata.audio_filename': 1
            },
            multi=True
        )

        song_root = '{}/{{}}/{{}}'.format(definitions.BEATMAPS_UNZIPPED)

        song_path_ids = list(map(
            lambda x: (
                song_root.format(
                    x['metadata']['beatmapset_id'],
                    x['metadata']['audio_filename']
                ),
                x['metadata']['beatmap_id']
            ),
            beatmap_data
        ))

        pdb.set_trace()

        with open(song_path_ids_file, 'wb') as f:
            cPickle.dump(song_path_ids, f)

    else:
        with open(song_path_ids_file, 'rb') as f:
            song_path_ids = cPickle.load(f)

    song_data = {}

    for path, beatmap_id in song_path_ids:
        save_file = '{}/{}'.format(song_dir, beatmap_id)

        if os.path.exists(save_file):
            continue
        try:

            spectrogram = music.get_mel_spectrogram(path)
            song_data[beatmap_id] = {'song_len': float(spectrogram.shape[0])}
            np.save(save_file, spectrogram)
        except Exception as e:
            print('unable to save beatmap {}'.format(beatmap_id))
            print(e)
            continue

    print('saving song_data.json')
    with open('song_data.json', 'w') as f:
        f.write(json.dumps(song_data))


def build_large_song_set_spect_dir(reset=False):
    song_dir = '{}/large_song_set'.format(definitions.OSU_DATA_DIR)

    if reset and os.path.exists(song_dir):
        shutil.rmtree(song_dir)

    if not os.path.exists(song_dir):
        os.mkdir(song_dir)

    print('finding files...')

    song_path_ids_file = 'large_song_set_path_ids.pkl'

    if not os.path.exists(song_path_ids_file):

        file_pattern = '{}/*/*.mp3'.format(definitions.BEATMAPS_UNZIPPED)
        files = glob(file_pattern)
        beatmapset_ids = [f.split('/')[-2] for f in files]
        pdb.set_trace()
        query = {
            'metadata.beatmapset_id': {'$in': beatmapset_ids},
            'metadata.difficultyrating': {'$gt': 2.6, '$lt': 3.3}}
        beatmap_ids = get_beatmap_id(query)

        song_path_ids = list(zip(files, beatmap_ids))
        song_path_ids = [x for x in song_path_ids if x[1] is not None]

        with open(song_path_ids_file, 'wb') as f:
            cPickle.dump(song_path_ids, f)

    else:
        with open(song_path_ids_file, 'rb') as f:
            song_path_ids = cPickle.load(f)

    pdb.set_trace()

    for path, beatmap_id in song_path_ids:
        save_file = '{}/{}'.format(song_dir, beatmap_id)
        spectrogram = music.get_mel_spectrogram(path)
        np.save(save_file, spectrogram)


def assert_song_limit(limit):
    if limit is not None and type(limit) != int:
        raise ValueError(
            'Song set limit must be an int. received {}'.format(type(limit)))


def get_song_set(song_set_path, limit=None, skip=''):
    assert_song_limit(limit)

    file_pattern = '{}/*.npy'.format(song_set_path)
    files = glob(file_pattern)
    beatmap_ids = [f.split('/')[-1].split('.')[0] for f in files]
    song_path_ids = list(zip(files, beatmap_ids))

    if limit is None:
        result = song_path_ids
    else:
        result = song_path_ids[:limit]

    return [x for x in result if x[1][0] != skip]


def get_large_song_set(**kwargs):
    return get_song_set(definitions.LARGE_SONG_SET, **kwargs)


def get_xl_song_set(**kwargs):
    return get_song_set(definitions.XL_SONG_SET, **kwargs)


def list_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if
            os.path.isdir(os.path.join(a_dir, name))]


def get_data_dest(use_latest=False):
    subdirs = list_subdirectories(base_data_dir)
    versions = [float(v[1:]) for v in subdirs if v[0] == 'v']
    versions.sort()
    latest = versions[-1] if len(versions) > 0 else 0
    new_version = round(latest + 0.1, 1) if not use_latest else latest
    data_package = '{}/v{}'.format(base_data_dir, new_version)

    if not os.path.exists(data_package):
        os.makedirs(data_package)

    return data_package


def get_multi_diff_song_data():
    if os.path.exists(definitions.SONGS_V2_FILE):
        with open(definitions.SONGS_V2_FILE, 'rb') as f:
            song_path_ids = cPickle.load(f)
        return song_path_ids

    song_paths_ids = get_xl_song_set(limit=None, skip=eval_song_id)
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

    for set_id, data in beatmap_sets.items():
        song_path = ids_to_paths[data['ids'][0]]
        beatmap_sets[set_id]['song_path'] = song_path

    query = {
        'metadata.beatmapset_id': {'$in': list(beatmap_sets.keys())},
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

    result = [(beatmap_sets[s]['song_path'], beatmap_sets[s]['ids'][:2]) for s
              in beatmap_sets.keys()]

    result = list(filter(lambda x: len(x[1]) > 1, result))

    with open(definitions.SONGS_V2_FILE, 'wb') as f:
        cPickle.dump(result, f)

    return result

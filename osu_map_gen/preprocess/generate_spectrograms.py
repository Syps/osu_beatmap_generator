# coding: utf-8
import os
import pdb
import sys
import time
from glob import glob
import numpy as np

from .. import (
    errors,
    db,
    hyper_params,
    definitions,
    music
)

MIN_SONG_LEN = 25.0  # seconds
MAX_SONG_LEN = 340.0  # seconds
OUTPUT_DIR = definitions.FULL_SPECT_DIR
SONG_FILE_PATTERN = '{}/beatmaps/unzipped/*/*.mp3'.format(
    definitions.OSU_DATA_DIR)
START_INDEX = 0
BRIGHTNESS_FILTER = 115

WRITE_METADATA_TO_DB = False

STFT_VALS = hyper_params.get_stft_args()


def get_song_files_with_set_ids(mp3_file_query):
    files = glob(mp3_file_query)
    ids = map(lambda x: x.split('/')[-2], files)
    return zip(ids, files)


def write_song_data_to_db(upsert=False, **kwargs):
    '''
    Writes following to db:
    {
        beatmapset_id,
        duration,
        audio_file
        spectrogram_file
    }
    '''
    if kwargs.get('beatmapset_id', None) is None:
        raise ValueError('song model_data must include a beatmapset_id')

    db.Songs.update_one({"beatmapset_id": kwargs['beatmapset_id']},
                        {'$set': kwargs}, upsert=upsert)


def save_spectrogram(dbs, filename):
    np.save(filename, dbs)


def exists_in_db(beatmapset_id):
    return db.Songs.find_one(
        {'beatmapset_id': beatmapset_id, 'spectrogram_file': {'$exists': True}})


STORE_MEDIUM = 'medium'
STORE_HARD = 'hard'

DIFFICULTY_MEDIUM = {'$gt': 2.6, '$lt': 3.2}
DIFFICULTY_HARD = {'$gte': 3.2, '$lt': 5}


def get_sam_beatmap_ids(difficulty):
    with open(definitions.SAM_FAVS, 'r') as f:
        sam_favs = f.readlines()

    sam_ids = [line.split(' ')[0] for line in sam_favs]

    sam_beatmaps = db.Beatmaps.find(
        {
            'metadata.beatmapset_id': {'$in': sam_ids},
            'metadata.difficultyrating': difficulty
        },
        {
            'metadata.beatmap_id': 1
        }
    )
    sam_beatmaps = list(sam_beatmaps)

    return [b['metadata']['beatmap_id'] for b in sam_beatmaps]


def build_beatmap_store(difficulty, store_name, min_favs=250):
    sam_b_ids = get_sam_beatmap_ids(difficulty)
    high_favs = list(db.Beatmaps.find(
        {
            'metadata.difficultyrating': difficulty,
            'hit_circles': {'$exists': True, '$not': {'$size': 0}}
        },
        {
            'metadata.favourite_count': 1,
            'metadata.beatmapset_id': 1,
            'metadata.beatmap_id': 1
        }
    ))

    high_favs_set_ids = [item['metadata']['beatmapset_id'] for item in high_favs
            if
            int(item['metadata']['favourite_count']) > min_favs
            or item['metadata']['beatmap_id'] in sam_b_ids]

    def song_filter(song_data):
        beatmapset_id, audio_path = song_data
        return beatmapset_id in high_favs_set_ids or beatmapset_id in sam_b_ids

    def add_beatmap_ids(songs):
        songs_with_ids = []

        for song in songs:
            set_id = song[0]
            try:
                id = next((d['metadata']['beatmap_id'] for d in high_favs if
                           d['metadata']['beatmapset_id'] == set_id))
            except StopIteration:
                pdb.set_trace()

            songs_with_ids.append((set_id, id, song[1]))

        return songs_with_ids

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print('getting song files...')
    song_files = get_song_files_with_set_ids(SONG_FILE_PATTERN)
    song_files = list(filter(song_filter, song_files))
    song_files.sort()
    song_files = add_beatmap_ids(song_files)

    len_sf = len(song_files)
    print('found {} mp3 files (includes sound files)'.format(len_sf))

    index = START_INDEX
    for beatmapset_id, _id, audio_file_path in song_files[index:]:
        if index % 10 == 0:
            print('save count = {} / {}'.format(index, len_sf))

        if WRITE_METADATA_TO_DB and exists_in_db(beatmapset_id):
            print('exists. skipping...')
            continue

        song_data = {
            'beatmapset_id': beatmapset_id,
            'audio_file': audio_file_path
        }

        try:
            audio = music.get_audio(audio_file_path)
            duration = music.get_duration(audio)
            if duration < MIN_SONG_LEN:
                raise errors.OsuError('Song is too short', 'short', audio_file_path)
            elif duration > MAX_SONG_LEN:
                raise errors.OsuError('Song is too long', 'long', audio_file_path)
            spectrogram = music.get_mel_spectrogram(audio, **STFT_VALS)

            output_file = '{}/{}/{}_{}'.format(OUTPUT_DIR, store_name, beatmapset_id, _id)
            save_spectrogram(spectrogram, output_file)

            if WRITE_METADATA_TO_DB:
                song_data.update({
                    'duration': duration,
                    'spectrogram_file': output_file,
                })

                write_song_data_to_db(upsert=True, **song_data)

            index = index + 1
            time.sleep(5)
        except errors.OsuError as e:
            if WRITE_METADATA_TO_DB:
                song_data.update({
                    'invalid': True,
                    'invalid_reason': e.error_type
                })
                write_song_data_to_db(upsert=True, **song_data)
        except KeyboardInterrupt:
            sys.exit()


if __name__ == '__main__':
    build_beatmap_store(DIFFICULTY_HARD, STORE_HARD)

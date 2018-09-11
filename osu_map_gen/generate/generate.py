import os
import pdb
import joblib
import shutil
import numpy as np
import json
import time

from tensorflow.python.keras.models import load_model

from . import (
    circles,
    definitions,
    hyper_params,
    music,
    utils
)

from .hit_model import MultipleDifficultiesModel

gen_dir = '{}/generated'.format(definitions.ROOT_DIR)
model_location = '{}/train/model_data'.format(definitions.PROJECT_DIR)
prob_mat_path = '{}/data/area_path_map.json'.format(definitions.ROOT_DIR)

n_features = hyper_params.n_mels + 1


def latest_model_version():
    subdirs = utils.list_subdirectories(model_location)
    versions = [float(v[1:]) for v in subdirs if v[0] == 'v']
    versions.sort()
    latest = versions[-1] if len(versions) > 0 else 0

    return latest


def load_hits(audio_file, data_dest, save_path):
    model_path = '{}/model_hits.h5'.format(data_dest)
    scaler_path = '{}/scaler_hits.save'.format(data_dest)

    utils.log('loading model from {}'.format(model_path))
    hit_keras_model = load_model(model_path)
    utils.log('loading scaler from {}'.format(scaler_path))
    mel_scaler = joblib.load(scaler_path)
    utils.log('loading audio from {}'.format(audio_file))
    mel_spectrogram = music.get_mel_spectrogram(audio_file)

    x = mel_spectrogram
    mel_spectrogram = np.zeros((x.shape[0], n_features))
    mel_spectrogram[:, :-1] = x

    mel_nrm = mel_scaler.transform(mel_spectrogram)
    mel_nrm = utils.pad_array(mel_nrm)
    sequences = utils.get_sequences(mel_nrm, n_features)

    hit_model = MultipleDifficultiesModel(hit_keras_model, sequences,
                                          scaler_path=scaler_path)
    utils.log('predicting hit activity... ')
    start = time.time()
    hit_predictions = hit_model.predict(add_z_channel=True)
    end = time.time()
    utils.log('predict elapsed time: {}'.format(end - start))

    with open(save_path, 'w') as f:
        json.dump({key: i.tolist() for key, i in hit_predictions.items()}, f)

    return hit_predictions


def get_hit_events(audio_file, data_dest, save_path, refresh=True):
    if utils.exists(save_path) and not refresh:
        utils.log('Loading hits from {}'.format(save_path))
        with open(save_path, 'r') as f:
            events = json.loads(f.read())
    else:
        utils.log('Predicting hits...')
        events = load_hits(audio_file, data_dest, save_path)

    return events


def create_new_package(song_path, song_name, img_path):
    osz_hash = str(hash(time.time()))[-6:]

    osz_location = '{}/{}_{}'.format(gen_dir, song_name, osz_hash)
    os.makedirs(osz_location)
    os.makedirs('{}/osz_files'.format(osz_location))

    shutil.copy(img_path,
                '{}/osz_files/BG.jpg'.format(osz_location))

    shutil.copy(song_path,
                '{}/osz_files/{}.mp3'.format(osz_location, song_name))

    return osz_location, osz_hash


def generate(gen_dest, hit_events, osz_hash, song_path, song_name, bpm):
    osu_gen_path = '{}/osz_files/generated_{}_{{}}.osu'.format(gen_dest,
                                                               osz_hash)
    osz_file = '{}/{}.osz'.format(gen_dest, osz_hash)

    try:
        circles.build_osu_files(
            hit_events,
            bpm,
            song_name,
            song_path,
            osu_gen_path
        )
    except BaseException as e:
        shutil.rmtree(gen_dest)
        raise e

    osu_files_dir = '{}/osz_files'.format(gen_dest)
    circles.zip_osz_file(osz_file, osu_files_dir)

    return osz_file


def main(song_path, song_name, bpm, image_path, model_version):
    if model_version == 'latest':
        latest_model = latest_model_version()
        model_dir = '{}/v{}'.format(model_location, latest_model)
    else:
        model_dir = '{}/v{}'.format(model_location, model_version)

    osz_location, osz_hash = create_new_package(
        song_path=song_path,
        song_name=song_name,
        img_path=image_path
    )

    print('Selected model_version: {}'.format(model_version))

    relative_song_path = song_path.split('/')[-1]
    hits_save_path = '{}/predicted_hits_{}.json'.format(model_dir,
                                                        song_name)
    hit_events = get_hit_events(song_path, model_dir, hits_save_path,
                                refresh=False)

    return generate(
        osz_location,
        hit_events,
        osz_hash,
        relative_song_path,
        song_name,
        bpm
    )

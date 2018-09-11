import os
from pathlib import Path

dir_path = os.path.dirname(os.path.realpath(__file__))

PROJECT_DIR = Path(dir_path).parent
ROOT_DIR = PROJECT_DIR.parent
OSU_DATA_DIR = '{}/osu_data'.format(ROOT_DIR.parent)
MODULE_PATH = '{}/osu_map_gen'.format(PROJECT_DIR)
MONGO_URL = 'localhost:27017'
BEATMAPS_UNZIPPED = '{}/unzipped'.format(OSU_DATA_DIR)
LARGE_SONG_SET = '{}/large_song_set'.format(OSU_DATA_DIR)
XL_SONG_SET = '{}/xl_song_set'.format(OSU_DATA_DIR)
XL_SONG_SET_V2 = '{}/xl_song_set_v2'.format(OSU_DATA_DIR)
FULL_SPECT_DIR = '{}/spect_npz_512'.format(OSU_DATA_DIR)
FULL_SPECT_PATTERN = '{}/*.npy'.format(FULL_SPECT_DIR)
SAM_FAVS = '{}/model_data/sam_favorites.txt'.format(PROJECT_DIR)


HDF5_FILE = '{}/model/frames.hdf5'.format(PROJECT_DIR)
BEATMAP_METADATA = '{}/model_data/beatmap_metadata.json'.format(PROJECT_DIR)
LOG_FILE = '{}/logs/osu-map-generator.log'.format(ROOT_DIR)
STD_ERR = '{}/logs/osu-map-generator.stderr.txt'.format(ROOT_DIR)

SUBMODULE_LOCATION = '{}/aisu_circles'.format(MODULE_PATH)

SONGS_V2_FILE = '{}/songs_v2.pkl'.format(PROJECT_DIR)

DEFAULT_BG = '{}/train/static/BG.jpg'.format(PROJECT_DIR)

# HDF5

ALL_FRAMES = 'model/frames.hdf5'
DATASETS = 'model/datasets.hdf5'
LABELS = 'model/labels.hdf5'

# TEST

# ALL_FRAMES = 'model/frames_test.hdf5'
# DATASETS = 'model/datasets_test.hdf5'
# LABELS = 'model/labels_test.hdf5'


MODEL_PATH = '{}/model_best.h5'.format(PROJECT_DIR)
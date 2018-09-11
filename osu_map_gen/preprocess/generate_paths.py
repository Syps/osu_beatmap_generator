import json
import os
import pdb
import numpy as np
from typing import Dict, List
from skimage.util import view_as_windows

from osu_map_gen.train import model_data
from .. import (
    db,
    definitions,
)
from . import markov
from .position_mapper import StandardMapper

path_len = 2

map_save_path = '{}/{}/area_path_map.json'.format(
    definitions.PROJECT_DIR, 'train/model_data'
)


def build_map(areas: List[tuple]):
    position_lists = []

    current_list = []
    for area in areas:
        if area == -1:
            position_lists.append(current_list)
            current_list = []
        else:
            current_list.append(area)

    if len(current_list) > 0:
        position_lists.append(current_list)

    pos_map = {}
    for l in position_lists:
        pos_map = path_map(l, pos_map)

    return pos_map


def path_map(areas: List[int], pos_map=None) -> Dict[str, List[int]]:
    if pos_map is None:
        next_area_map: Dict[str, List[int]] = {}
    else:
        next_area_map = pos_map

    if len(areas) < path_len + 1:
        return next_area_map

    paths = view_as_windows(np.asarray(areas), path_len)

    for index in range(1, paths.shape[0]):

        path = list(paths[index])

        prev_path = area_key(*list(paths[index - 1]))
        next_area = int(path[-1])

        if next_area != -1:
            if prev_path in next_area_map:
                next_area_map[prev_path].append(next_area)
            else:
                next_area_map[prev_path] = [next_area]

    return next_area_map


def area_key(*areas: int) -> str:
    """
    Params
    ------
    areas: List of area numbers. Length is equal to len_path
    """
    return '_'.join(map(str, areas))


def filter_valid_paths(pm: dict) -> Dict[str, List[int]]:
    return {key: value for key, value in pm.items() if '-' not in
            key}


def positions_for_song(song_length, beatmap_id) -> np.ndarray:
    beatmap_data = db.beatmap_data(beatmap_id)

    if beatmap_id == '338772':
        pdb.set_trace()

    positions = StandardMapper(
        song_length=song_length,
        hit_events={
            'hit_circles': beatmap_data['hit_circles'],
            'sliders': beatmap_data['sliders'],
            'spinners': beatmap_data['spinners']
        },
        timing_points=beatmap_data['timing_points'],
        slider_multiplier=float(beatmap_data['metadata']['slider_multiplier']),
        fill_method='none',
    ).get_positions()

    return positions


def load_path_map(replace=False):
    if os.path.exists(map_save_path) and not replace:
        with open(map_save_path, 'r') as f:
            return json.loads(f.read())
    else:
        full_path_map = {}
        s_args = {
            'limit': 100,
            'skip': model_data.eval_song_id,
        }
        for song_file, beatmap_id in model_data.get_large_song_set(**s_args):
            song_length = np.load(song_file).shape[0]
            try:
                positions = positions_for_song(song_length, beatmap_id)
            except ValueError:
                print('Skipping beatmap {}'.format(beatmap_id))
                continue

            positions_at_interval = markov.interval_positions(positions)
            areas = markov.positions_to_areas(positions_at_interval)

            full_path_map = filter_valid_paths(path_map(areas, full_path_map))

        with open(map_save_path, 'w') as f:
            json.dump(full_path_map, f)

        return full_path_map


if __name__ == '__main__':
    load_path_map(replace=True)

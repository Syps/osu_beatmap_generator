import csv
import json
import os
import pdb

import numpy as np
from .. import (
    db,
    definitions,
    timing
)

'''
MONGO DOCUMENT EXAMPLE FORMAT
--------------

{
    audio_file_name: Closer.mp3,
    audio_lead_in: 1500,
    preview_time: 49660,
    countdown: 0,
    sample_set: Soft,
    stack_leniency: 0.5,
    mode: 0,
    letterbox_in_breaks: 1,
    version: Easy,
    slider_multiplier: 0.99999,
    slider_tick_rate: 2,
    // ... other metadata from api
    // contains v0.1 set id and v0.1 id
    timing_points: [
        {
        Offset,
        Milliseconds per Beat,
        Meter,
        Sample Type,
        Sample Set,
        Volume,
        Inherited,
        Kiai Mode
        }
    ],
    colors: [
        {
            r: 255,
            g: 255,
            b: 255
        }
    ],
    hit_objects: {
        hit_circles: [
            {
                x: 0,
                y: 0,
                time: 0,
                type: 1,
                // Bit 0 (1) = circle, bit 1 (2) = slider, bit 2 (4) = new combo,
                // bit 3 (8) = spinner. Bits 4-6 (16, 32, 64) form a 3-bit number (0-7)
                // that chooses how many combo colors to skip.
                // Bit 7 (128) is for an osu!mania hold note. Circles, sliders,
                // and spinners can be OR'd with new combos and the combo skip value,
                // but not with each other.
                hit_sound: 1
            }
        ],
        sliders: [
            {
                 x: 0,
                 y: 0,
                 time: 0,
                 type: 1,
                 hit_sound: 0,
                 curve_points: [
                    {
                        x: 0,
                        y: 0
                    }
                 ],
                 repeat: 1,
                 pixel_length: 120,
                 edge_hit_sounds: [2,1,3]
                 edge_additions: [
                    '...'
                 ],
                 addition: '...'

            }
        ],
        spinners: [
            {
                 x: 0,
                 y: 0,
                 time: 0,
                 type: 1,
                 hit_sound: 0,
                 end_time: ...,
                 addition: ...,
            }
        ]
    }
}


'''

beatmap_dir = definitions.BEATMAPS_UNZIPPED
beatmap_folders = ['{}/{}'.format(definitions.BEATMAPS_UNZIPPED, beatmap) for
                   beatmap in os.listdir(beatmap_dir)]

csv_header = ['beatmap_id', 'src_file', 'offset', 'tempo']

with open(definitions.BEATMAP_METADATA, 'r') as api_metadata_file:
    api_metadata = json.loads(api_metadata_file.read())


def first_osu_file(folder):
    files = os.listdir(folder)
    file_name = next((f for f in files if f[-4:] == '.osu'), None)
    return '{}/{}'.format(folder, file_name)


def get_timing_data(osu_file):
    with open(osu_file, 'r') as f:
        lines = f.readlines()

    timing_index = next(((i, e) for i, e in enumerate(lines) if 'Timing' in e),
                        None)
    timing_data = lines[timing_index[0] + 1].strip().split(',')

    return timing_data[:2]


def _parse_obj(str_rep, keys):
    obj = {}
    values = str_rep.split(',')
    for index, key in enumerate(keys):
        if index == len(values):
            break
        obj[key] = values[index]

    return obj


def _add_to_db(beatmap_data):
    try:
        db.Beatmaps.replace_one(
            {"metadata.beatmap_id": beatmap_data["metadata"]['beatmap_id']},
            beatmap_data, upsert=True)
    except Exception as e:
        print(str(e))


def _update_db_entry(beatmap_id, beatmap_data):
    try:
        db.Beatmaps.update_one({"metadata.beatmap_id": beatmap_id},
                               {'$set': beatmap_data})
    except Exception as e:
        print(str(e))


def parse_timing_point(timing_str):
    keys = ['offset', 'mls_per_beat', 'meter', 'sample_type',
            'sample_set', 'volume', 'inherited', 'kiai_mode']
    return _parse_obj(timing_str, keys)


def parse_color(color_str):
    keys = ['r', 'g', 'b']
    try:
        color_str = color_str.split(':')[-1].strip()
    except ValueError:
        pdb.set_trace()
    return _parse_obj(color_str, keys)


def parse_hit_circle(str_rep):
    keys = ['x', 'y', 'time', 'type', 'hit_sound', 'addition']
    hit_obj = _parse_obj(str_rep, keys)
    hit_obj['type'] = 'hit_circle'
    return hit_obj


def parse_slider(str_rep):
    keys = ['x', 'y', 'time', 'type', 'hit_sound', 'slider_type_curve_pts_str',
            'repeat', 'pixel_length',
            'edge_hitsounds', 'edge_additions', 'addition']
    hit_obj = _parse_obj(str_rep, keys)
    hit_obj['type'] = 'slider'
    return hit_obj


def parse_spinner(str_rep):
    keys = ['x', 'y', 'time', 'type', 'hit_sound', 'end_time', 'addition']
    hit_obj = _parse_obj(str_rep, keys)
    hit_obj['type'] = 'spinner'
    return hit_obj


def parse_hit_object(str_rep):
    values = str_rep.split(',')
    type_bitmap = '{0:b}'.format(int(values[3]))
    if type_bitmap[-1] == '1':
        return parse_hit_circle(str_rep)
    if type_bitmap[-2] == '1':
        return parse_slider(str_rep)
    if type_bitmap[-4] == '1':
        return parse_spinner(str_rep)

    raise ValueError('Unexpected hit object type -> {}'.format(type_bitmap))


def parse_break_line(str_rep):
    start, end = str_rep.split(',')[1:]
    start_end_seconds = [float(start) / 1000.0, float(end) / 1000.0]
    frames = timing.seconds_to_frames(start_end_seconds)

    return {'start': int(frames[0]), 'end': int(frames[1])}


def parse_breaks_section(file_contents):
    section_header = '//Break Periods'
    lines = list(map(lambda x: x.replace('\r', '').replace('\n', '').strip(),
                     file_contents))
    try:
        timing_index = lines.index(section_header) + 1
    except ValueError:
        if section_header != '[Colours]':
            return None
        else:
            return []

    end_index = timing_index

    while len(lines[end_index]) > 2 and lines[end_index][:2] != '//':
        end_index += 1

    if end_index == timing_index:
        return []

    return list(map(parse_break_line, lines[timing_index:end_index]))


def parse_section(file_contents, section_header, parsing_func):
    lines = list(map(lambda x: x.replace('\r', '').replace('\n', '').strip(),
                     file_contents))
    try:
        timing_index = lines.index(section_header) + 1
    except ValueError:
        if section_header != '[Colours]':
            return None
        else:
            return []
    proceeding_lines = lines[timing_index:]
    try:
        end_index = timing_index + (
            proceeding_lines.index('') if '' in proceeding_lines else len(
                proceeding_lines))
        next_section_index = timing_index + next(
            (i for i, e in enumerate(proceeding_lines) if
             e != '' and e[0] == '['), 0)
        end_index = min(end_index,
                        next_section_index) if next_section_index > timing_index else end_index
    except ValueError:
        pdb.set_trace()

    return list(map(parsing_func, lines[timing_index:end_index]))


def get_timing_points(file_contents):
    return parse_section(file_contents, '[TimingPoints]', parse_timing_point)


def get_breaks(file_contents):
    return parse_breaks_section(file_contents)


def get_events(file_contents):
    return parse_section(file_contents, '[Events]')


def get_colors(file_contents):
    return parse_section(file_contents, '[Colours]', parse_color)


def get_hit_objects(file_contents):
    try:
        hit_objects = parse_section(file_contents, '[HitObjects]',
                                    parse_hit_object)
    except ValueError as e:
        print(
            'Beatmap contains invalid hit objects. ValueError =>\n' + e.message)
        hit_objects = []
    return hit_objects


def get_hit_circles(hit_objects):
    return filter(lambda x: x['type'] == 'hit_circle', hit_objects)


def get_sliders(hit_objects):
    return filter(lambda x: x['type'] == 'slider', hit_objects)


def get_spinners(hit_objects):
    return filter(lambda x: x['type'] == 'spinner', hit_objects)


def get_api_metadata(beatmap_set_id, version):
    # match to beatmap_metadata.json w/ v0.1 set id and version name
    return next((x for x in api_metadata if
                 x['beatmapset_id'] == beatmap_set_id and x[
                     'version'] == version))


def parse_metadata_value(str_rep):
    try:
        divide = str_rep.index(':')
    except ValueError:
        pdb.set_trace()
    key = str_rep[:divide].strip()
    value = str_rep[divide + 1:].strip()
    key = key[0].lower() + key[1:]
    key = ''.join([x if x.islower() else '_{}'.format(x.lower()) for x in key])
    return key, value


def get_file_metadata(file_contents):
    '''
    audio_file_name: Closer.mp3,
    audio_lead_in: 1500,
    preview_time: 49660,
    countdown: 0,
    sample_set: Soft,
    stack_leniency: 0.5,
    mode: 0,
    letterbox_in_breaks: 1,
    slider_multiplier: 0.99999,
    slider_tick_rate: 2,
    '''
    metadata = parse_section(file_contents, '[General]', parse_metadata_value)
    if metadata is None:
        return None
    metadata.extend(
        parse_section(file_contents, '[Metadata]', parse_metadata_value))
    difficulty_data = parse_section(file_contents, '[Difficulty]',
                                    parse_metadata_value)
    file_metadata = {d[0]: d[1] for d in metadata}
    for diff_item in difficulty_data:
        if diff_item[0] in ['slider_multiplier', 'slider_tickrate']:
            file_metadata.update({diff_item[0]: diff_item[1]})

    return file_metadata


def get_timing_point_data(beatmap_file, beatmap_set_id):
    with open(beatmap_file, 'r') as bf:
        try:
            file_contents = bf.readlines()
        except Exception:
            print(
                'Unable to retrieve timing points for {}\n....Skipping....\n'.format(
                    beatmap_file))
            return None

    data = {'timing_points': get_timing_points(file_contents)}
    beatmap_id = get_beatmap_id(beatmap_set_id, file_contents)

    if beatmap_id is None:
        print(
            'Unable to retrieve v0.1 id for file {} with set id {}. Skipping'.format(
                beatmap_file, beatmap_set_id
            ))
        return None

    return data, beatmap_id


def get_breaks_data(beatmap_file, beatmap_set_id):
    with open(beatmap_file, 'r') as bf:
        try:
            file_contents = bf.readlines()
        except Exception:
            print(
                'Unable to retrieve breaks for {}\n....Skipping....\n'.format(
                    beatmap_file))
            return None

    data = {'breaks': get_breaks(file_contents)}
    beatmap_id = get_beatmap_id(beatmap_set_id, file_contents)

    if beatmap_id is None:
        print(
            'Unable to retrieve v0.1 id for file {} with set id {}. Skipping'.format(
                beatmap_file, beatmap_set_id
            ))
        return None

    return data, beatmap_id


def get_favourites_data(beatmap_file, beatmap_set_id):
    with open(beatmap_file, 'r') as bf:
        try:
            file_contents = bf.readlines()
        except Exception:
            print(
                'Unable to retrieve breaks for {}\n....Skipping....\n'.format(
                    beatmap_file))
            return None

    data = {'metadata.favourite_count': get_breaks(file_contents)}
    beatmap_id = get_beatmap_id(beatmap_set_id, file_contents)

    if beatmap_id is None:
        print(
            'Unable to retrieve v0.1 id for file {} with set id {}. Skipping'.format(
                beatmap_file, beatmap_set_id
            ))
        return None

    return data, beatmap_id


def get_beatmap_id(beatmap_set_id, file_contents):
    file_metadata = get_file_metadata(file_contents)

    if file_metadata is None:
        return None

    version = file_metadata.get('version', None)

    if version is None:
        return None

    try:
        api_data = get_api_metadata(beatmap_set_id, version)
    except StopIteration:
        return None

    return api_data['beatmap_id']


def get_beatmap_data(beatmap_set_id, beatmap_file):
    with open(beatmap_file, 'r') as bf:
        file_contents = bf.readlines()

    data = {}
    file_metadata = get_file_metadata(file_contents)

    if file_metadata is None:
        print('Unable to retrieve metadata for {}\n....Skipping....\n'.format(
            beatmap_file))
        return None

    version = file_metadata.get('version', None)

    if not version:
        print('No version for v0.1 file: {}\n....Skipping....\n'.format(
            beatmap_file))
        return None

    api_data = get_api_metadata(beatmap_set_id, version)
    file_metadata.update(api_data)

    if file_metadata['mode'] != '0':
        return None

    hit_objects = get_hit_objects(file_contents)

    if len(hit_objects) == 0:
        return None

    data['metadata'] = file_metadata
    data['metadata']['difficultyrating'] = float(
        data['metadata']['difficultyrating'])
    try:
        data['colors'] = get_colors(file_contents)
    except ValueError:
        data['colors'] = []
    data['hit_circles'] = get_hit_circles(hit_objects)
    data['sliders'] = get_sliders(hit_objects)
    data['spinners'] = get_spinners(hit_objects)
    data['timing_points'] = get_timing_points(file_contents)
    data['breaks'] = get_breaks(file_contents)

    if len(data['hit_circles']) == 0:
        return None

    return data


def refresh_beatmap_db():
    for index, beatmap_folder in enumerate(beatmap_folders):
        files = os.listdir(beatmap_folder)
        file_names = [f for f in files if f[-4:] == '.osu']

        for file_name in file_names:
            beatmap_set_id = beatmap_folder.split('/')[-1]
            full_path = '{}/{}'.format(beatmap_folder, file_name)

            beatmap_data = get_beatmap_data(beatmap_set_id, full_path)
            if beatmap_data is not None:
                _add_to_db(beatmap_data)

        if index > 0 and index % 100 == 0:
            print('row count = {}'.format(index))


def add_timing_points_to_db():
    for index, beatmap_folder in enumerate(beatmap_folders):
        files = os.listdir(beatmap_folder)
        file_names = [f for f in files if f[-4:] == '.osu']

        for file_name in file_names:
            beatmap_set_id = beatmap_folder.split('/')[-1]
            full_path = '{}/{}'.format(beatmap_folder, file_name)
            timing_data = get_timing_point_data(full_path, beatmap_set_id)

            if timing_data is None:
                continue

            beatmap_data, beatmap_id = timing_data

            if beatmap_data is not None:
                _update_db_entry(beatmap_id, beatmap_data)

        if index > 0 and index % 100 == 0:
            print(
                'row count = {}. last v0.1 = {}'.format(index, beatmap_set_id))


def add_breaks_to_db():
    for index, beatmap_folder in enumerate(beatmap_folders):
        files = os.listdir(beatmap_folder)
        file_names = [f for f in files if f[-4:] == '.osu']

        for file_name in file_names:
            beatmap_set_id = beatmap_folder.split('/')[-1]
            full_path = '{}/{}'.format(beatmap_folder, file_name)
            breaks_data = get_breaks_data(full_path, beatmap_set_id)

            if breaks_data is None:
                continue

            beatmap_data, beatmap_id = breaks_data

            if beatmap_data is not None:
                _update_db_entry(beatmap_id, beatmap_data)

        if index > 0 and index % 100 == 0:
            print(
                'row count = {}. last v0.1 = {}'.format(index, beatmap_set_id))


def convert_favourites_to_int():
    for index, beatmap_folder in enumerate(beatmap_folders):
        files = os.listdir(beatmap_folder)
        file_names = [f for f in files if f[-4:] == '.osu']

        for file_name in file_names:
            beatmap_set_id = beatmap_folder.split('/')[-1]
            full_path = '{}/{}'.format(beatmap_folder, file_name)
            breaks_data = get_breaks_data(full_path, beatmap_set_id)

            if breaks_data is None:
                continue

            beatmap_data, beatmap_id = breaks_data

            if beatmap_data is not None:
                _update_db_entry(beatmap_id, beatmap_data)

        if index > 0 and index % 100 == 0:
            print(
                'row count = {}. last v0.1 = {}'.format(index, beatmap_set_id))


def write_timing_data_csv():
    with open('timing_data.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(csv_header)

        print('writing to csv...')

        for index, beatmap_set in enumerate(beatmap_folders):
            osu = first_osu_file(beatmap_set)
            data = [beatmap_set, osu]
            data.extend(get_timing_data(osu))

            writer.writerow(data)

            if index > 0 and index % 100 == 0:
                print('row count = {}'.format(index))

        print('all done :)')


def frame_indexes_to_hit_circles(predictions, n_frames=1024.0, sr=22050.0):
    indexes = []

    predictions = np.rot90(predictions, k=3)
    hit_frames = np.where(predictions[0] < predictions[1])[0]
    pdb.set_trace()

    x_range = np.arange(0, 512, 10)
    x_pos_index = 0
    y_pos = 192

    hit_objects = []

    for frame_number in indexes:
        ms_per_frame = n_frames / (sr / 1000)

        timestamp = int(ms_per_frame * frame_number)
        hit_obj_template = '{x},{y},{time},1,1,0:0:0:0:'

        if x_pos_index > x_range.shape[0]:
            x_pos_index = 0
        x_pos = x_range[x_pos_index]

        hit_object = hit_obj_template.format(x=x_pos, y=y_pos, time=timestamp)
        hit_objects.append(hit_object)
        x_pos_index += 1

    return hit_objects


if __name__ == '__main__':
    # add_timing_points_to_db()
    add_breaks_to_db()

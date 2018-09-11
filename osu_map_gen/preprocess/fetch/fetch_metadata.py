import json
import os

import requests

from osu_map_gen.preprocess.fetch.auth import api_key

url_template = "https://osu.ppy.sh/api/get_beatmaps?k={k}&s={s}"
metadata_file = 'beatmap_metadata.json'
cached_sets = []

if os.path.exists(metadata_file):
    with open('beatmap_metadata.json', 'r') as f:
        beatmap_data = json.loads(f.read())
    cached_sets = set([x['beatmapset_id'] for x in beatmap_data])

print('{} sets exist in cache'.format(len(cached_sets)))

with open('beatmap_ids.txt', 'r') as f:
    beatmap_set_ids = f.readlines()
beatmap_set_ids = map(lambda x: x.replace('\n', ''), beatmap_set_ids)
beatmap_set_ids = [_id for _id in beatmap_set_ids if _id not in cached_sets]


print('fetching model_data...')
for index, beatmap_set_id in enumerate(beatmap_set_ids):
    url = url_template.format(k=api_key, s=beatmap_set_id)
    attempts = 0
    res = None
    while attempts < 5:
        try:
            res = requests.get(url)
        except requests.exceptions.ChunkedEncodingError:
            attempts += 1
            continue
        break

    if not res:
        print('Could not get model_data for v0.1 set {}. Skipping...'.format(beatmap_set_id))
        continue

    beatmaps = json.loads(res.text)
    beatmap_data.extend(beatmaps)

    if index % 250 == 0:
        print('Current count => {}'.format(index))

print('writing to json file...')
with open(metadata_file, 'w') as f:
    f.write(json.dumps(beatmap_data))

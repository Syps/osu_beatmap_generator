from datetime import datetime

from bson.objectid import ObjectId
from pymongo import MongoClient
from .utils import log

from osu_map_gen.util import definitions

client = MongoClient(definitions.MONGO_URL)
db = client.Osu

# job statuses
STATUS_PENDING = 0
STATUS_FAILED = 1
STATUS_GENERATED = 2


def beatmap_data(beatmap_id=None, query=None, projection=None,
                 multi=False) -> dict:
    if query is None and beatmap_id is None:
        raise ValueError(
            'If no query provided, must include beatmap_id (was None)'
        )

    if projection is not None and type(projection) != dict:
        raise ValueError(
            'Projection must be type dict. Was {}'.format(projection)
        )

    if query is None:
        query = {
            'metadata.beatmap_id': beatmap_id
        }

    if projection is None:
        projection = {
            'timing_points': 1,
            'metadata.slider_multiplier': 1,
            'hit_circles': 1,
            'sliders': 1,
            'spinners': 1,
            'metadata.beatmapset_id': 1,
            'metadata.beatmap_i_d': 1,
            'breaks': 1
        }

    if multi:
        return db.Beatmaps.find(query, projection)
    else:
        return db.Beatmaps.find_one(query, projection)


def update_job(job_id, osz_path=None, error_message=None):
    status = STATUS_GENERATED if osz_path else STATUS_FAILED
    log('updating job in db. job_id={}, path={}, error_message={}'.format(
        job_id, osz_path, error_message))
    db.Jobs.update(
        {'_id': ObjectId(job_id)},
        {
            'oszPath': osz_path,
            'status': status,
            'finishedAt': datetime.utcnow(),
            'errorMessage': error_message,
        },
        upsert=False
    )

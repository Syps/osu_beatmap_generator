# _Osu!_ Beatmap Generator

Read the full blog post [here](https://www.nicksypteras.com/projects/generating-osu-beatmaps).

See example beatmaps [here](https://www.youtube.com/playlist?list=PLXHqh2k-tZQeTHQ_s_kbrfhYQaYajfbpg).

## Requirements
- [ffmpeg](https://www.ffmpeg.org/)

## Steps to generate beatmaps:
_The repo comes with a trained model so you can start generating right away_ 😄 _(`osu_map_gen/train/model_data/v1.1`)_

1. Init [submodule](https://github.com/Syps/aisu_circles)
  - `git submodule update --init`
2. Install dependencies
  - `pipenv --three install`
 
3. Generate a beatmap
  - `pipenv run python -m osu_map_gen -g /path/to/my/song.mp3 120 "My Favorite Song"`
  
  
## Steps to train:

_Coming soon_

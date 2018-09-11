import sys
from . import definitions, generate, train


err_msg = 'Must use 1 of following flags: [-g, -t]'

if len(sys.argv) == 1:
    raise ValueError(err_msg)

flag = sys.argv[1]


def run_generate(args):
    gen_err_msg = 'See generate#__main__ for required args.'
    assets_dir = 'osu_map_gen/generate/static'
    model_version = None

    if len(args) == 0:
        raise ValueError(gen_err_msg)

    # flag for quickly testing model updates end-to-end
    if args[0] == '-test':
        model_version = 'latest'
        song_path = '{}/ebb_and_flow.mp3'.format(assets_dir)
        song_name = 'ebb_and_flow'
        img_path = '{}/BG.jpg'.format(assets_dir)
        bpm = 106  # ebb and flow
    else:

        if len(args) == 5:
            model_version = float(args.pop())

        if len(args) == 4:
            img_path = args.pop()
        else:
            img_path = '{}/BG.jpg'.format(assets_dir)

        if model_version is None:
            model_version = 'latest'

        song_path, bpm, song_name = args

    generate(song_path, song_name, bpm, img_path, model_version)


def run_train():
    train()


if flag == '-g':
    run_generate(sys.argv[2:])
elif flag == '-t':
    run_train()
else:
    raise ValueError(err_msg)

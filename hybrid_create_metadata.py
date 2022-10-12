'''
Used to be a much larger test file used for exploration.
Now used simply to generate metadata for narrow_down_hybrid.py
'''
from helpers import NumpyEncoder
import os
import json
import pathlib



epsilon = 1e-6
do_diff = False
base_dirname = './outputs/hybrid-perfect/'
init_size_factor = 1
init_size = 'n'

seeds_to_run = [11650, 31650]

results = {
            11650: {'m': 1000000,
                    'n': 50,
                    'prob_type': 'low',
                    'init_method': 'random'},
            31650: {'m': 1000000,
                    'n': 50,
                    'prob_type': 'very',
                    'init_method': 'qr'}
}

for seed in seeds_to_run:

    # check for existing data
    dirname = base_dirname + str(seed) + '/'
    if os.path.exists(dirname):
        print('Skipping %d...' % seed)
        continue

    m = results[seed]['m']
    n = results[seed]['n']
    prob_type = results[seed]['prob_type']
    init_method = results[seed]['init_method']

    # save results to file
    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
    metadata_dict = {'seed': seed, 'epsilon': epsilon,
                     'm': m, 'n': n, 'init_method': init_method,
                     'init_size': init_size, 'prob_type': prob_type,
                     'init_size_factor': init_size_factor,
                     'do_diff': str(do_diff)}
    with open(dirname + 'metadata.json', 'w') as f:
        json.dump(metadata_dict, f, indent=4, cls=NumpyEncoder)

# vim: foldmethod=marker

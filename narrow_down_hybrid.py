'''
Given a few decent ideas for when to transition in a given problem,
try all of them and see which is best, how different they are
'''
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from mvee import mvee2
from mvee import initialize_qr, initialize_random, initialize_norm
from mvee import kurtosis
from parameters import Param
from helpers import NumpyEncoder, change_dict_to_integer_keys
from plotting import applySettings
import time
import json
import os
import argparse
from pathlib import Path



parser = argparse.ArgumentParser(description="Generate hybrid-related results")
parser.add_argument("-p", "--plot", action="store_true", help="Plot")
args = parser.parse_args()

epsilon = 1e-6
newt_epsilon = 1e-14
plotter = plt.plot
base_dirname = './outputs/hybrid-narrow/'
base_metadata_dirname = './outputs/hybrid-perfect/'

if args.plot:
    mode = 'display'
else:
    mode = 'run_only'

display_seed = [11650, 31650]
seeds_to_run = display_seed
plot_no_change_lines = False
plot_individually = True
interactive_plot = True
if not interactive_plot:
    raise Exception('Are you sure you want to overwrite?')

# {{{ Results from perfect_hybrid.py
# key is seed; value is list of likely transition points
results = {
    11510: {'m': 100000, 'n': 10,
            'switch': [5, 20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]},
    12510: {'m': 100000, 'n': 10,
            'switch': [5, 20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]},
    13510: {'m': 100000, 'n': 10,
            'switch': [5, 20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]},
    11610: {'m': 1000000, 'n': 10,
            'switch': [5, 20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]},
    12610: {'m': 1000000, 'n': 10,
            'switch': [5, 20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]},
    13610: {'m': 1000000, 'n': 10,
            'switch': [5, 20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]},
    11520: {'m': 100000, 'n': 20,
            'switch': [100, 500, 700, 1000, 1500]},
    12520: {'m': 100000, 'n': 20,
            'switch': [275, 350, 500, 1000]},
    13520: {'m': 100000, 'n': 20,
            'switch': [200, 400, 500, 800]},  # stopped here
    11530: {'m': 100000, 'n': 30,
            'switch': [100, 300, 500, 800]},
    12530: {'m': 100000, 'n': 30,
            'switch': [100, 300, 500, 800]},
    13530: {'m': 100000, 'n': 30,
            'switch': [100, 300, 500, 800]},
    11540: {'m': 100000, 'n': 40,
            'switch': [100, 300, 500, 800, 1000, 1200, 1400, 1800]},
    12540: {'m': 100000, 'n': 40,
            'switch': [100, 300, 500, 800, 1000, 1200, 1400, 1800]},
    13540: {'m': 100000, 'n': 40,
            'switch': [100, 300, 500, 800, 1000, 1200, 1400, 1800]},
    11550: {'m': 100000, 'n': 50,
            'switch': [100, 300, 500, 800, 1000, 1200, 1400, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3500, 4000, 4500]},
    12550: {'m': 100000, 'n': 50,
            'switch': [100, 300, 500, 800, 1000, 1200, 1400, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3500, 4000, 4500]},
    13550: {'m': 100000, 'n': 50,
            'switch': [100, 300, 500, 800, 1000, 1200, 1400, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3500, 4000, 4500]},
    11650: {'m': 100000, 'n': 50,
            'switch': [100, 300, 500, 800, 1000, 1200, 1400, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3500, 4000, 4500]},
    12650: {'m': 100000, 'n': 50,
            'switch': [100, 300, 500, 800, 1000, 1200, 1400, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3500, 4000, 4500]},
    13650: {'m': 100000, 'n': 50,
            'switch': [100, 300, 500, 800, 1000, 1200, 1400, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3500, 4000, 4500]},
    11560: {'m': 100000, 'n': 60,
            'switch': [100, 300, 500, 800, 1000, 1200, 1400, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3500, 4000, 4500]},
    12560: {'m': 100000, 'n': 60,
            'switch': [100, 300, 500, 800, 1000, 1200, 1400, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3500, 4000, 4500]},
    13560: {'m': 100000, 'n': 60,
            'switch': [100, 300, 500, 800, 1000, 1200, 1400, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3500, 4000, 4500]},
    31510: {'m': 100000, 'n': 10,
            'switch': [10, 20, 30, 40, 60, 80, 100, 120]},
    32510: {'m': 100000, 'n': 10,
            'switch': [10, 20, 30, 40, 60, 80, 100, 120]},
    33510: {'m': 100000, 'n': 10,
            'switch': [10, 20, 30, 40, 60, 80, 100, 120]},
    31550: {'m': 100000, 'n': 50,
            'switch': [10, 20, 30, 40, 60, 80, 100, 120]},
    32550: {'m': 100000, 'n': 50,
            'switch': [10, 20, 30, 40, 60, 80, 100, 120]},
    33550: {'m': 100000, 'n': 50,
            'switch': [10, 20, 30, 40, 60, 80, 100, 120]},
    31610: {'m': 100000, 'n': 10,
            'switch': [10, 20, 30, 40, 60, 80, 100, 120]},
    32610: {'m': 100000, 'n': 10,
            'switch': [10, 20, 30, 40, 60, 80, 100, 120]},
    33610: {'m': 100000, 'n': 10,
            'switch': [10, 20, 30, 40, 60, 80, 100, 120]},
    31650: {'m': 100000, 'n': 50,
            'switch': [10, 20, 30, 40, 60, 80, 100, 120]},
    32650: {'m': 100000, 'n': 50,
            'switch': [10, 20, 30, 40, 60, 80, 100, 120]},
    33650: {'m': 100000, 'n': 50,
            'switch': [10, 20, 30, 40, 60, 80, 100, 120]},
    41510: {'m': 100000, 'n': 10,
            'switch': [10, 25, 50, 100, 150, 200, 250, 300, 350, 400]},
    42510: {'m': 100000, 'n': 10,
            'switch': [10, 25, 50, 100, 150, 200, 250, 300, 350, 400]},
    43510: {'m': 100000, 'n': 10,
            'switch': [10, 25, 50, 100, 150, 200, 250, 300, 350, 400]},
    41520: {'m': 100000, 'n': 20,
            'switch': [10, 25, 50, 100, 150, 200, 400]},
    42520: {'m': 100000, 'n': 20,
            'switch': [10, 25, 50, 100, 150, 200, 400]},
    43520: {'m': 100000, 'n': 20,
            'switch': [10, 25, 50, 100, 150, 200, 400]},
    41530: {'m': 100000, 'n': 30,
            'switch': [10, 25, 50, 100, 150, 200, 400]},
    42530: {'m': 100000, 'n': 30,
            'switch': [10, 25, 50, 100, 150, 200, 400]},
    43530: {'m': 100000, 'n': 30,
            'switch': [10, 25, 50, 100, 150, 200, 400]},
    41540: {'m': 100000, 'n': 40,
            'switch': [10, 25, 50, 100, 150, 200, 400]},
    42540: {'m': 100000, 'n': 40,
            'switch': [10, 25, 50, 100, 150, 200, 400]},
    43540: {'m': 100000, 'n': 40,
            'switch': [10, 25, 50, 100, 150, 200, 400]},
    41550: {'m': 100000, 'n': 50,
            'switch': [10, 25, 50, 100, 150, 200, 250, 300, 350, 400]},
    42550: {'m': 100000, 'n': 50,
            'switch': [10, 25, 50, 100, 150, 200, 250, 300, 350, 400]},
    43550: {'m': 100000, 'n': 50,
            'switch': [10, 25, 50, 100, 150, 200, 250, 300, 350, 400]},
    41610: {'m': 100000, 'n': 10,
            'switch': [10, 25, 50, 100, 150, 200, 250, 300, 350, 400]},
    42610: {'m': 100000, 'n': 10,
            'switch': [10, 25, 50, 100, 150, 200, 250, 300, 350, 400]},
    43610: {'m': 100000, 'n': 10,
            'switch': [10, 25, 50, 100, 150, 200, 250, 300, 350, 400]},
    41650: {'m': 100000, 'n': 50,
            'switch': [10, 25, 50, 100, 150, 200, 250, 300, 350, 400]},
    42650: {'m': 100000, 'n': 50,
            'switch': [10, 25, 50, 100, 150, 200, 250, 300, 350, 400]},
    43650: {'m': 100000, 'n': 50,
            'switch': [10, 25, 50, 100, 150, 200, 250, 300, 350, 400]},
    41560: {'m': 100000, 'n': 60,
            'switch': [10, 25, 50, 100, 150, 200, 400]},
    42560: {'m': 100000, 'n': 60,
            'switch': [10, 25, 50, 100, 150, 200, 400]},
    43560: {'m': 100000, 'n': 60,
            'switch': [10, 25, 50, 100, 150, 200, 400]}
}
# }}}

# Define method-specific parameters
defaults = {'verbose': True, 'max_iter': 4000, 'drop_every': 50,
            'converged': True}
param = Param(defaults, ['todd', 'newton'])
param.set_param('todd', 'max_iter', 50000)

if mode not in ['display', 'display_all']:
    for seed in seeds_to_run:

        m = results[seed]['m']
        n = results[seed]['n']
        switches = results[seed]['switch']
        do_newton = True
        do_todd = True
        times = {}

        # check for existing data
        dirname = base_dirname + str(seed) + '/'
        if os.path.exists(dirname):
            with open(dirname + 'output.json', 'r') as f:
                output_dict = json.load(f)
            times = change_dict_to_integer_keys(output_dict['times'])

            if 'base_newt' in output_dict.keys():
                base_newt = output_dict['base_newt']
                do_newton = False
            if 'base_todd' in output_dict.keys():
                base_todd = output_dict['base_todd']
                do_todd = False

            already_done = set(switches).issubset(set(times.keys()))
            if already_done and not do_todd and not do_newton:
                print('Skipping %d...' % seed)
                continue
            else:
                print('Appending to %d...' % seed)

        print('----------------------')
        print('Running with seed: %d' % seed)

        # last 4 digits of seed specify actual seed to run
        # any leading digits specify that different run configurations were
        # used
        run_seed = seed % 10000

        # Generate test problem
        np.random.seed(run_seed)
        X = np.random.randn(n, m)

        if seed < 10000:
            # init_X is used to account for upproject
            init_X = np.vstack([X, np.ones(X.shape[1])])
            t1 = time.time()
            init = initialize_qr(init_X, n_desired=int(n**1.5))
            t2 = time.time()
            t_init = t2 - t1
        elif seed < 20000:
            init_X = np.vstack([X, np.ones(X.shape[1])])
            t1 = time.time()
            init = initialize_random(init_X, n_desired=(n + 1))
            t2 = time.time()
            t_init = t2 - t1
        elif seed < 40000:
            X = X*np.random.standard_cauchy(m)
            init_X = np.vstack([X, np.ones(X.shape[1])])
            t1 = time.time()
            init = initialize_qr(init_X, n_desired=(n + 1))
            t2 = time.time()
            t_init = t2 - t1
        elif seed < 50000:
            X = np.random.lognormal(size=(n, m))
            init_X = np.vstack([X, np.ones(X.shape[1])])
            t1 = time.time()
            init = initialize_qr(init_X, n_desired=(n + 1))
            t2 = time.time()
            t_init = t2 - t1

        # Always use best inits for Newton and CA
        if seed < 20000:
            # Gaussian data
            if do_newton:
                init_X = np.vstack([X, np.ones(X.shape[1])])
                t1 = time.time()
                init_newton = initialize_norm(init_X, p=2,
                                              n_desired=int(n**1.5))
                t2 = time.time()
                t_init_newton = t2 - t1
            if do_todd:
                if seed < 10000:
                    init_X = np.vstack([X, np.ones(X.shape[1])])
                    t1 = time.time()
                    init_todd = initialize_random(init_X, n_desired=(n + 1))
                    t2 = time.time()
                    t_init_todd = t2 - t1
                else:
                    init_todd = init
                    t_init_todd = t_init
        elif seed < 50000:
            # High- or Very-high-kurtosis data
            init_newton = init
            t_init_newton = t_init
            init_todd = init
            t_init_todd = t_init

        for transition in switches:

            # skip any runs already done
            if transition in times.keys():
                continue

            hybrid_dict = {'method': 'newton', 'step_count': transition}
            param.set_param('todd', 'hybrid', hybrid_dict)

            t1 = time.time()
            obj = mvee2(X, initialize='given', full_output=False,
                        epsilon=epsilon, update_data=init,
                        **param.get_dict('todd'))
            t2 = time.time()
            print('%d done: %f seconds' % (transition, (t2 - t1)))
            if obj['converged']:
                times[transition] = t2 - t1 + t_init
            else:
                times[transition] = np.inf

        if do_newton:
            t1 = time.time()
            obj = mvee2(X, initialize='given', full_output=False,
                        epsilon=epsilon, update_data=init_newton,
                        **param.get_dict('newton'))
            t2 = time.time()
            print('%s done: %f seconds' % ('Newton', (t2 - t1)))
            if obj['converged']:
                base_newt = t2 - t1 + t_init_newton
            else:
                base_newt = np.inf

        if do_todd:
            param.set_param('todd', 'hybrid', None)
            t1 = time.time()
            obj = mvee2(X, initialize='given', full_output=False,
                        epsilon=epsilon, update_data=init_todd,
                        **param.get_dict('todd'))
            t2 = time.time()
            print('%s done: %f seconds' % ('CA', (t2 - t1)))
            if obj['converged']:
                base_todd = t2 - t1 + t_init_todd
            else:
                base_todd = np.inf

        # save results to file
        output_dict = {'times': times, 'base_newt': base_newt,
                       'base_todd': base_todd}
        os.makedirs(dirname, exist_ok=True)
        with open(dirname + 'output.json', 'w') as f:
            json.dump(output_dict, f, indent=4, cls=NumpyEncoder)

all_seeds = [seeds_to_run]
if mode == 'display':
    all_seeds = list(display_seed)
elif mode == 'run_only':
    all_seeds = []
elif mode == 'run_and_display':
    all_seeds = [seed]
elif mode == 'display_all':
    all_seeds = all_seeds[0]

if plot_individually:
    for seed in all_seeds:

        # read metadata
        metadata_dirname = base_metadata_dirname + str(seed) + '/'
        with open(metadata_dirname + 'metadata.json', 'r') as f:
            info_dict = json.load(f)
        epsilon = float(info_dict['epsilon'])
        prob_type = info_dict['prob_type']
        m = int(info_dict['m'])
        n = int(info_dict['n'])
        init_method = info_dict['init_method']
        init_size = info_dict['init_size']
        init_size_factor = info_dict['init_size_factor']
        do_diff = info_dict['do_diff']

        # read timing results
        dirname = base_dirname + str(seed) + '/'
        with open(dirname + 'output.json', 'r') as f:
            output_dict = json.load(f)
        all_times = change_dict_to_integer_keys(output_dict['times'])
        switches = list(all_times.keys())
        yval = list(all_times.values())
        base_newt = output_dict['base_newt']
        base_todd = output_dict['base_todd']

        for scale in ['linear']:
            plt.figure()
            img_dirname = base_dirname + 'images-model/' + str(seed) + '/'
            os.makedirs(img_dirname, exist_ok=True)

            if prob_type == 'low':
                prob_str = 'Gaussian'
            elif prob_type == 'very':
                prob_str = 'Cauchy'
            elif prob_type == 'high':
                prob_str = 'Lognormal'

            title_extra_info = '\nm = %d, n = %d\n%s data' % (m, n, prob_str)

            if scale == 'linear':
                divide_scale = 1
                xlabel_suffix = ''
            elif scale == 'n sqrt(n)':
                divide_scale = n**1.5
                xlabel_suffix = '/n**1.5'
            switches_scaled = np.array(switches)/divide_scale
            plt.plot(switches_scaled, yval, 'o')
            plt.title('Transition points%s' % title_extra_info)
            plt.xlabel('transition iteration%s' % xlabel_suffix)
            plt.ylabel('time (s)')

            plt.plot([min(switches_scaled), max(switches_scaled)], [base_newt]*2,
                     '--', label='newton')
            plt.plot([min(switches_scaled), max(switches_scaled)], [base_todd]*2,
                     '--', label='CA')
            if plot_no_change_lines:
                with open(metadata_dirname + 'output.json', 'r') as f:
                    sizes_dict = json.load(f)
                core_sizes = np.array(sizes_dict['core_sizes'])

                run_seed = seed % 10000

                # Generate test problem
                np.random.seed(run_seed)
                X = np.random.randn(n, m)

                if seed < 30000:
                    pass
                elif seed < 40000:
                    X = X*np.random.standard_cauchy(m)
                elif seed < 50000:
                    X = np.random.lognormal(size=(n, m))
                kur = kurtosis(X)
                if kur <= 3:
                    pred_core_size = 1.5*n**1.5
                elif kur < 30:
                    pred_core_size = (np.log10(kur) - np.log10(3))*(1.2*n - 1.5*n**1.5) + \
                                      1.5*n**1.5
                else:
                    pred_core_size = 1.2*n

                # fallback to other criteria
                try:
                    pred_iter = np.where(core_sizes >= pred_core_size)[0].min()
                except:
                    pred_iter = np.inf

                no_change = (core_sizes[1:] == core_sizes[:-1]).astype(int)
                for max_no_change_core_set in [10, 20, 30, 40, 50]:
                    sum_no_change = np.convolve(np.ones(max_no_change_core_set), no_change)
                    fallback_iters = np.argmax(sum_no_change == max_no_change_core_set)

                    plt.plot([fallback_iters/divide_scale]*2,
                             [min(yval), max(base_newt, base_todd)], '--',
                             label='no change for %d' % max_no_change_core_set)
            plt.legend()

if mode == 'display' or mode == 'display_all':
    plt.show()
else:
    plt.close('all')

# vim:foldmethod=marker

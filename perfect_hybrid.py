'''
Run to find all constraint changes
'''
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from mvee import mvee2
from mvee import initialize_qr, initialize_random, initialize_norm
from parameters import Param
from helpers import NumpyEncoder
import time
import os
import json



def generate_perfect_init(X):
    uco = mvee2(X, epsilon=1e-8, verbose=False, max_iter=5000,
                full_output=True, method='newton', drop_every=np.inf)['u']
    u_compare = uco.copy()
    nonzeros = uco > 1e-10
    nonzero_inds = np.where(uco > 1e-10)[0]
    u_order = np.argsort(u_compare)[::-1]
    u_order = u_order[:nonzeros.sum()]
    uco = np.zeros(m)
    uco[nonzeros] = 1.0/nonzeros.sum()
    return uco


epsilon = 1e-6
newt_epsilon = 1e-14
plotter = plt.plot
do_diff = False
base_dirname = './outputs/hybrid-perfect/'
mode = 'run_only'
#mode = 'display_all'
display_seed = 31510
#seeds_to_run = [11510, 11550]  # CA; Gaussian; rand(n)
#seeds_to_run = [121510, 121550]  # CA; Gaussian; 2norm(n)
#seeds_to_run = [111510, 111550]  # CA; Gaussian; 2norm(n root n)
#seeds_to_run = [151510, 151550]  # CA; Gaussian; 2norm(1.3 n root n)
#seeds_to_run = [161510, 161550]  # CA; Gaussian; 2norm(1.5 n root n)
#seeds_to_run = [171510, 171550]  # CA; Gaussian; perfect
#seeds_to_run = [101510, 101550]  # Newton; Gaussian; rand(n)
#seeds_to_run = [91510, 91550]  # Newton; Gaussian; 2norm(n)
#seeds_to_run = [81510, 81550]  # Newton; Gaussian; 2norm(n root n)
#seeds_to_run = [141510, 141550]  # Newton; Gaussian; 2norm(1.3 n root n)
#seeds_to_run = [131510, 131550]  # Newton; Gaussian; 2norm(1.5 n root n)
#seeds_to_run = [181510, 181550]  # Newton; Gaussian; perfect
seeds_to_run = [101550, 91550, 81550, 131550, 141550, 181550, 11550, 151550, 121550, 171550, 161550, 111550]

# Define method-specific parameters
defaults = {'verbose': True, 'max_iter': 4000, 'drop_every': 50}
param = Param(defaults, ['todd', 'newton'])
param.set_param('todd', 'max_iter', 15000)

# Save results here so that we don't rerun every time
# key is seed; value is list of likely transition points
# this won't actually be used in the program -- it is for record keeping
# {{{ results
results = {
    1234: {'m': 300000, 'n': 40,
            'switch': [1300, 1400, 2300, 2500, 2800, 3300, 3700, 4000]},
    1111: {'m': 300000, 'n': 10,
            'switch': [250, 400, 550, 1300]},
    1510: {'m': 100000, 'n': 10,
            'switch': [100, 500, 700, 1000, 1500]},
    2510: {'m': 100000, 'n': 10,
            'switch': [150, 200, 400, 600, 1000, 1250]},
    3510: {'m': 100000, 'n': 10,
            'switch': [100, 200, 500, 700, 900]},
    4510: {'m': 100000, 'n': 10,
            'switch': [100, 100, 300, 400]},
    5510: {'m': 100000, 'n': 10,
            'switch': [150, 200, 400, 600, 800]},
    1520: {'m': 100000, 'n': 20,
            'switch': [100, 500, 700, 1000, 1500]},
    2520: {'m': 100000, 'n': 20,
            'switch': [275, 350, 500, 1000]},
    3520: {'m': 100000, 'n': 20,
            'switch': [200, 400, 500, 800]},  # stopped here
    4520: {'m': 100000, 'n': 20,
            'switch': [100, 300, 500, 800]},
    5520: {'m': 100000, 'n': 20,
            'switch': [100, 300, 500, 800]},
    1530: {'m': 100000, 'n': 30,
            'switch': [100, 300, 500, 800]},
    2530: {'m': 100000, 'n': 30,
            'switch': [100, 300, 500, 800]},
    3530: {'m': 100000, 'n': 30,
            'switch': [100, 300, 500, 800]},
    4530: {'m': 100000, 'n': 30,
            'switch': [100, 300, 500, 800]},
    5530: {'m': 100000, 'n': 30,
            'switch': [100, 300, 500, 800]},
    1540: {'m': 100000, 'n': 40,
            'switch': [100, 300, 500, 800]},
    2540: {'m': 100000, 'n': 40,
            'switch': [100, 300, 500, 800]},
    3540: {'m': 100000, 'n': 40,
            'switch': [100, 300, 500, 800]},
    4540: {'m': 100000, 'n': 40,
            'switch': [100, 300, 500, 800]},
    5540: {'m': 100000, 'n': 40,
            'switch': [100, 300, 500, 800]},
    1550: {'m': 100000, 'n': 50,
            'switch': [100, 300, 500, 800]},
    2550: {'m': 100000, 'n': 50,
            'switch': [100, 300, 500, 800]},
    3550: {'m': 100000, 'n': 50,
            'switch': [100, 300, 500, 800]},
    4550: {'m': 100000, 'n': 50,
            'switch': [100, 300, 500, 800]},
    5550: {'m': 100000, 'n': 50,
            'switch': [100, 300, 500, 800]},
    1560: {'m': 100000, 'n': 60,
            'switch': [100, 300, 500, 800]},
    2560: {'m': 100000, 'n': 60,
            'switch': [100, 300, 500, 800]},
    3560: {'m': 100000, 'n': 60,
            'switch': [100, 300, 500, 800]},
    4560: {'m': 100000, 'n': 60,
            'switch': [100, 300, 500, 800]},
    5560: {'m': 100000, 'n': 60,
            'switch': [100, 300, 500, 800]},
    11510: {'m': 100000, 'n': 10,
            'switch': []},
    12510: {'m': 100000, 'n': 10,
            'switch': []},
    13510: {'m': 100000, 'n': 10,
            'switch': []},
    11520: {'m': 100000, 'n': 20,
            'switch': []},
    12520: {'m': 100000, 'n': 20,
            'switch': []},
    13520: {'m': 100000, 'n': 20,
            'switch': []},
    11530: {'m': 100000, 'n': 30,
            'switch': []},
    12530: {'m': 100000, 'n': 30,
            'switch': []},
    13530: {'m': 100000, 'n': 30,
            'switch': []},
    11540: {'m': 100000, 'n': 40,
            'switch': []},
    12540: {'m': 100000, 'n': 40,
            'switch': []},
    13540: {'m': 100000, 'n': 40,
            'switch': []},
    11550: {'m': 100000, 'n': 50,
            'switch': []},
    12550: {'m': 100000, 'n': 50,
            'switch': []},
    13550: {'m': 100000, 'n': 50,
            'switch': []},
    11560: {'m': 100000, 'n': 60,
            'switch': []},
    12560: {'m': 100000, 'n': 60,
            'switch': []},
    13560: {'m': 100000, 'n': 60,
            'switch': []},
    11610: {'m': 1000000, 'n': 10,
            'switch': []},
    12610: {'m': 1000000, 'n': 10,
            'switch': []},
    13610: {'m': 1000000, 'n': 10,
            'switch': []},
    11650: {'m': 1000000, 'n': 50,
            'switch': []},
    12650: {'m': 1000000, 'n': 50,
            'switch': []},
    13650: {'m': 1000000, 'n': 50,
            'switch': []},
    21510: {'m': 100000, 'n': 10,
            'switch': []},
    22510: {'m': 100000, 'n': 10,
            'switch': []},
    23510: {'m': 100000, 'n': 10,
            'switch': []},
    21520: {'m': 100000, 'n': 20,
            'switch': []},
    22520: {'m': 100000, 'n': 20,
            'switch': []},
    23520: {'m': 100000, 'n': 20,
            'switch': []},
    21530: {'m': 100000, 'n': 30,
            'switch': []},
    22530: {'m': 100000, 'n': 30,
            'switch': []},
    23530: {'m': 100000, 'n': 30,
            'switch': []},
    21540: {'m': 100000, 'n': 40,
            'switch': []},
    22540: {'m': 100000, 'n': 40,
            'switch': []},
    23540: {'m': 100000, 'n': 40,
            'switch': []},
    21550: {'m': 100000, 'n': 50,
            'switch': []},
    22550: {'m': 100000, 'n': 50,
            'switch': []},
    23550: {'m': 100000, 'n': 50,
            'switch': []},
    21560: {'m': 100000, 'n': 60,
            'switch': []},
    22560: {'m': 100000, 'n': 60,
            'switch': []},
    23560: {'m': 100000, 'n': 60,
            'switch': []},
    31510: {'m': 100000, 'n': 10,
            'switch': []},
    32510: {'m': 100000, 'n': 10,
            'switch': []},
    33510: {'m': 100000, 'n': 10,
            'switch': []},
    31520: {'m': 100000, 'n': 20,
            'switch': []},
    32520: {'m': 100000, 'n': 20,
            'switch': []},
    33520: {'m': 100000, 'n': 20,
            'switch': []},
    31530: {'m': 100000, 'n': 30,
            'switch': []},
    32530: {'m': 100000, 'n': 30,
            'switch': []},
    33530: {'m': 100000, 'n': 30,
            'switch': []},
    31540: {'m': 100000, 'n': 40,
            'switch': []},
    32540: {'m': 100000, 'n': 40,
            'switch': []},
    33540: {'m': 100000, 'n': 40,
            'switch': []},
    31550: {'m': 100000, 'n': 50,
            'switch': []},
    32550: {'m': 100000, 'n': 50,
            'switch': []},
    33550: {'m': 100000, 'n': 50,
            'switch': []},
    31560: {'m': 100000, 'n': 60,
            'switch': []},
    32560: {'m': 100000, 'n': 60,
            'switch': []},
    33560: {'m': 100000, 'n': 60,
            'switch': []},
    31610: {'m': 1000000, 'n': 10,
            'switch': []},
    32610: {'m': 1000000, 'n': 10,
            'switch': []},
    33610: {'m': 1000000, 'n': 10,
            'switch': []},
    31650: {'m': 1000000, 'n': 50,
            'switch': []},
    32650: {'m': 1000000, 'n': 50,
            'switch': []},
    33650: {'m': 1000000, 'n': 50,
            'switch': []},
    41510: {'m': 100000, 'n': 10,
            'switch': []},
    42510: {'m': 100000, 'n': 10,
            'switch': []},
    43510: {'m': 100000, 'n': 10,
            'switch': []},
    41520: {'m': 100000, 'n': 20,
            'switch': []},
    42520: {'m': 100000, 'n': 20,
            'switch': []},
    43520: {'m': 100000, 'n': 20,
            'switch': []},
    41530: {'m': 100000, 'n': 30,
            'switch': []},
    42530: {'m': 100000, 'n': 30,
            'switch': []},
    43530: {'m': 100000, 'n': 30,
            'switch': []},
    41540: {'m': 100000, 'n': 40,
            'switch': []},
    42540: {'m': 100000, 'n': 40,
            'switch': []},
    43540: {'m': 100000, 'n': 40,
            'switch': []},
    41550: {'m': 100000, 'n': 50,
            'switch': []},
    42550: {'m': 100000, 'n': 50,
            'switch': []},
    43550: {'m': 100000, 'n': 50,
            'switch': []},
    41560: {'m': 100000, 'n': 60,
            'switch': []},
    42560: {'m': 100000, 'n': 60,
            'switch': []},
    43560: {'m': 100000, 'n': 60,
            'switch': []},
    41610: {'m': 1000000, 'n': 10,
            'switch': []},
    42610: {'m': 1000000, 'n': 10,
            'switch': []},
    43610: {'m': 1000000, 'n': 10,
            'switch': []},
    41650: {'m': 1000000, 'n': 50,
            'switch': []},
    42650: {'m': 1000000, 'n': 50,
            'switch': []},
    43650: {'m': 1000000, 'n': 50,
            'switch': []},
    51510: {'m': 100000, 'n': 10,
            'switch': []},
    52510: {'m': 100000, 'n': 10,
            'switch': []},
    53510: {'m': 100000, 'n': 10,
            'switch': []},
    51520: {'m': 100000, 'n': 20,
            'switch': []},
    52520: {'m': 100000, 'n': 20,
            'switch': []},
    53520: {'m': 100000, 'n': 20,
            'switch': []},
    51530: {'m': 100000, 'n': 30,
            'switch': []},
    52530: {'m': 100000, 'n': 30,
            'switch': []},
    53530: {'m': 100000, 'n': 30,
            'switch': []},
    51540: {'m': 100000, 'n': 40,
            'switch': []},
    52540: {'m': 100000, 'n': 40,
            'switch': []},
    53540: {'m': 100000, 'n': 40,
            'switch': []},
    51550: {'m': 100000, 'n': 50,
            'switch': []},
    52550: {'m': 100000, 'n': 50,
            'switch': []},
    53550: {'m': 100000, 'n': 50,
            'switch': []},
    51560: {'m': 100000, 'n': 60,
            'switch': []},
    52560: {'m': 100000, 'n': 60,
            'switch': []},
    53560: {'m': 100000, 'n': 60,
            'switch': []},
    61510: {'m': 100000, 'n': 10,
            'switch': []},
    61550: {'m': 100000, 'n': 50,
            'switch': []},
    71510: {'m': 100000, 'n': 10,
            'switch': []},
    71550: {'m': 100000, 'n': 50,
            'switch': []},
    81510: {'m': 100000, 'n': 10,
            'switch': [100, 500, 700, 1000, 1500]},
    81550: {'m': 100000, 'n': 50,
            'switch': [100, 500, 700, 1000, 1500]},
    91510: {'m': 100000, 'n': 10,
            'switch': [100, 500, 700, 1000, 1500]},
    91550: {'m': 100000, 'n': 50,
            'switch': [100, 300, 500, 800]},
    101510: {'m': 100000, 'n': 10,
             'switch': [100, 500, 700, 1000, 1500]},
    101550: {'m': 100000, 'n': 50,
             'switch': [100, 300, 500, 800]},
    111510: {'m': 100000, 'n': 10,
             'switch': [100, 500, 700, 1000, 1500]},
    111550: {'m': 100000, 'n': 50,
             'switch': [100, 300, 500, 800]},
    121510: {'m': 100000, 'n': 10,
             'switch': [100, 500, 700, 1000, 1500]},
    121550: {'m': 100000, 'n': 50,
             'switch': [100, 300, 500, 800]},
    131510: {'m': 100000, 'n': 10,
            'switch': []},
    131550: {'m': 100000, 'n': 50,
            'switch': []},
    141510: {'m': 100000, 'n': 10,
            'switch': []},
    141550: {'m': 100000, 'n': 50,
            'switch': []},
    151510: {'m': 100000, 'n': 10,
            'switch': []},
    151550: {'m': 100000, 'n': 50,
            'switch': []},
    161510: {'m': 100000, 'n': 10,
            'switch': []},
    161550: {'m': 100000, 'n': 50,
            'switch': []},
    171510: {'m': 100000, 'n': 10,
            'switch': []},
    171550: {'m': 100000, 'n': 50,
            'switch': []},
    181510: {'m': 100000, 'n': 10,
            'switch': []},
    181550: {'m': 100000, 'n': 50,
            'switch': []}
}
# }}}

if mode not in ['display', 'display_all']:
    #for seed in results.keys():
    for seed in seeds_to_run:

        # check for existing data
        dirname = base_dirname + str(seed) + '/'
        if os.path.exists(dirname):
            print('Skipping %d...' % seed)
            continue

        # last 4 digits of seed specify actual seed to run
        # any leading digits specify that different run configurations were
        # used
        run_seed = seed % 10000

        np.random.seed(run_seed)
        m = results[seed]['m']
        n = results[seed]['n']

        # Generate test problem
        X = np.random.randn(n, m)

        #true_sol = mvee2(X, initialize='qr', epsilon=newt_epsilon,
                            #max_iter=1000, full_output=True, method='newton',
                            #verbose=True, drop_every=np.inf)['u']

        # run coordinate ascent
        method = 'todd'
        prob_type = 'low'
        # {{{ Run types
        if seed < 10000:
            init_X = np.vstack([X, np.ones(X.shape[1])])
            init = initialize_qr(init_X, n_desired=int(n**1.5))
            init_method = 'qr'
            init_size = 'n*sqrt(n)'
            init_size_factor = 1
        elif seed < 20000:
            init_X = np.vstack([X, np.ones(X.shape[1])])
            init = initialize_random(init_X, n_desired=(n + 1))
            init_method = 'random'
            init_size = 'n'
            init_size_factor = 1
        elif seed < 30000:
            init_X = np.vstack([X, np.ones(X.shape[1])])
            init = initialize_qr(init_X, n_desired=int(n**1.5))
            init_method = 'qr'
            init_size = 'n*sqrt(n)'
            init_size_factor = 1
            method = 'newton'
        elif seed < 40000:
            X = X*np.random.standard_cauchy(m)
            init_X = np.vstack([X, np.ones(X.shape[1])])
            init = initialize_qr(init_X, n_desired=(n + 1))
            init_method = 'qr'
            init_size = 'n'
            init_size_factor = 1
            prob_type = 'very'
        elif seed < 50000:
            X = np.random.lognormal(size=(n, m))
            init_X = np.vstack([X, np.ones(X.shape[1])])
            init = initialize_qr(init_X, n_desired=(n + 1))
            init_method = 'qr'
            init_size = 'n'
            init_size_factor = 1
            prob_type = 'high'
        elif seed < 60000:
            init_X = np.vstack([X, np.ones(X.shape[1])])
            init = initialize_qr(init_X, n_desired=(n + 1))
            init_method = 'qr'
            init_size = 'n'
            init_size_factor = 1
            method = 'newton'
        elif seed < 70000:
            X = X*np.random.standard_cauchy(m)
            init_X = np.vstack([X, np.ones(X.shape[1])])
            init = initialize_qr(init_X, n_desired=(n + 1))
            init_method = 'qr'
            init_size = 'n'
            init_size_factor = 1
            prob_type = 'very'
            method = 'newton'
        elif seed < 80000:
            X = np.random.lognormal(size=(n, m))
            init_X = np.vstack([X, np.ones(X.shape[1])])
            init = initialize_qr(init_X, n_desired=(n + 1))
            init_method = 'qr'
            init_size = 'n'
            init_size_factor = 1
            prob_type = 'high'
            method = 'newton'
        elif seed < 90000:
            init_X = np.vstack([X, np.ones(X.shape[1])])
            init = initialize_norm(init_X, p=2, n_desired=int(n**1.5))
            init_method = '2norm'
            init_size = 'n*sqrt(n)'
            init_size_factor = 1
            method = 'newton'
        elif seed < 100000:
            init_X = np.vstack([X, np.ones(X.shape[1])])
            init = initialize_norm(init_X, p=2, n_desired=(n + 1))
            init_method = '2norm'
            init_size = 'n'
            init_size_factor = 1
            method = 'newton'
        elif seed < 110000:
            init_X = np.vstack([X, np.ones(X.shape[1])])
            init = initialize_random(init_X, n_desired=(n + 1))
            init_method = 'random'
            init_size = 'n'
            init_size_factor = 1
            method = 'newton'
        elif seed < 120000:
            init_X = np.vstack([X, np.ones(X.shape[1])])
            init = initialize_norm(init_X, p=2, n_desired=int(n**1.5))
            init_method = '2norm'
            init_size = 'n*sqrt(n)'
            init_size_factor = 1
        elif seed < 130000:
            init_X = np.vstack([X, np.ones(X.shape[1])])
            init = initialize_norm(init_X, p=2, n_desired=(n + 1))
            init_method = '2norm'
            init_size = 'n'
            init_size_factor = 1
        elif seed < 140000:
            init_X = np.vstack([X, np.ones(X.shape[1])])
            init = initialize_norm(init_X, p=2, n_desired=int(1.5*n**1.5))
            init_method = '2norm'
            init_size = 'n*sqrt(n)'
            init_size_factor = 1.5
            method = 'newton'
        elif seed < 150000:
            init_X = np.vstack([X, np.ones(X.shape[1])])
            init = initialize_norm(init_X, p=2, n_desired=int(1.3*n**1.5))
            init_method = '2norm'
            init_size = 'n*sqrt(n)'
            init_size_factor = 1.3
            method = 'newton'
        elif seed < 160000:
            init_X = np.vstack([X, np.ones(X.shape[1])])
            init = initialize_norm(init_X, p=2, n_desired=int(1.3*n**1.5))
            init_method = '2norm'
            init_size = 'n*sqrt(n)'
            init_size_factor = 1.3
        elif seed < 170000:
            init_X = np.vstack([X, np.ones(X.shape[1])])
            init = initialize_norm(init_X, p=2, n_desired=int(1.5*n**1.5))
            init_method = '2norm'
            init_size = 'n*sqrt(n)'
            init_size_factor = 1.5
        elif seed < 180000:
            init = generate_perfect_init(X)
            init_method = 'perfect'
            init_size = 'exact'
            init_size_factor = 1
        elif seed < 190000:
            init = generate_perfect_init(X)
            init_method = 'perfect'
            init_size = 'exact'
            init_size_factor = 1
            method = 'newton'
        # }}}

        #import ipdb; ipdb.set_trace()
        t1 = time.time()
        ret = mvee2(X, initialize='given', full_output=False,
                    epsilon=epsilon, **param.get_dict(method),
                    constraint_add_inds=True, constraint_rem_inds=True,
                    track_differences_obj=do_diff, update_data=init,
                    track_core_set_size=True)

        c_a, c_r = ret['constraint_add_inds'], ret['constraint_rem_inds']
        core_sizes = ret['core_set_sizes']
        if do_diff:
            diff = ret['diffs']
        else:
            diff = None

        t2 = time.time()
        print('%s done: %f seconds' % (method, (t2 - t1)))

        # save results to file
        os.mkdir(dirname)
        metadata_dict = {'seed': seed, 'epsilon': epsilon,
                         'm': m, 'n': n, 'init_method': init_method,
                         'init_size': init_size, 'prob_type': prob_type,
                         'init_size_factor': init_size_factor,
                         'do_diff': str(do_diff)}
        with open(dirname + 'metadata.json', 'w') as f:
            json.dump(metadata_dict, f, indent=4, cls=NumpyEncoder)

        all_output = {'core_sizes': core_sizes, 'c_a': c_a, 'c_r': c_r,
                      'diff': diff}
        with open(dirname + 'output.json', 'w') as f:
            json.dump(all_output, f, indent=4, cls=NumpyEncoder)

all_seeds = results.keys()
if mode == 'display':
    all_seeds = [display_seed]
elif mode == 'run_only':
    all_seeds = []
elif mode == 'run_and_display':
    all_seeds = [seed]

for seed in all_seeds:

    # reading previous results
    dirname = base_dirname + str(seed) + '/'

    with open(dirname + 'metadata.json', 'r') as f:
        info_dict = json.load(f)
    seed = int(info_dict['seed'])
    epsilon = float(info_dict['epsilon'])
    prob_type = info_dict['prob_type']
    m = int(info_dict['m'])
    n = int(info_dict['n'])
    init_method = info_dict['init_method']
    init_size = info_dict['init_size']
    init_size_factor = info_dict['init_size_factor']
    do_diff = info_dict['do_diff']

    with open(dirname + 'output.json', 'r') as f:
        all_output = json.load(f)
    core_sizes = all_output['core_sizes']
    c_a = all_output['c_a']
    c_r = all_output['c_r']
    diff = all_output['diff']

    if prob_type == 'low':
        prob_str = 'Gaussian'
    elif prob_type == 'very':
        prob_str = 'Cauchy'
    elif prob_type == 'high':
        prob_str = 'Lognormal'

    diff = np.array(diff)
    core_sizes = np.array(core_sizes)
    #title_extra_info = '\nm = %d, n = %d\n%s data (seed=%d)' % \
                       #(m, n, prob_str, seed)
    #title_extra_info = '\n%s data' % (prob_str)
    title_extra_info = '\nm = %d, n = %d\n%s data' % (m, n, prob_str)

    img_dirname = base_dirname + 'images/' + str(seed) + '/'
    os.makedirs(img_dirname, exist_ok=True)

    '''
    # plot constraint changes
    plt.figure()
    plt.title('Ratio of objective decreases')
    plt.xlabel('iteration')
    plt.ylabel('ratio')
    plt.plot(diff, 'o')
    plt.plot(c_a, diff[c_a], 'o', label='add')
    plt.plot(c_r, diff[c_r], 'o', label='rem')
    plt.legend()
    '''

    # plot constraint changes
    plt.figure()
    plt.title('Constraint Changes%s' % title_extra_info)
    plt.xlabel('iteration')
    plt.ylabel('iteration')
    plt.plot(c_a, np.arange(len(core_sizes))[c_a], 'o', label='add constraint')
    plt.plot(c_r, np.arange(len(core_sizes))[c_r], 'o', label='remove constraint')
    plt.legend()
    plt.savefig(img_dirname + 'constraint_changes.png')

    # plot core set size
    plt.figure()
    plt.title('Core set size%s' % title_extra_info)
    plt.xlabel('iteration')
    plt.ylabel('size')
    plt.plot(core_sizes, 'o', label='core set')
    #plt.legend()
    plt.savefig(img_dirname + 'core_set_size.png')

    # plot core set size relative to n
    plt.figure()
    plt.title('Core set size\niterations relative to n%s' % title_extra_info)
    plt.xlabel('iteration/n')
    plt.ylabel('size')
    plt.plot(np.arange(len(core_sizes))/n, core_sizes, 'o', label='core set')
    plt.legend()
    plt.savefig(img_dirname + 'core_set_size_n.png')

    # plot core set size relative to n**1.5
    plt.figure()
    plt.title('Core set size\niterations relative to n**1.5%s' %
              title_extra_info)
    plt.xlabel('iteration/n**1.5')
    plt.ylabel('size')
    plt.plot(np.arange(len(core_sizes))/n**1.5, core_sizes, 'o', label='core set')
    plt.legend()
    plt.savefig(img_dirname + 'core_set_size_n_sqrtn.png')

    '''
    # plot iters since core set size changed
    since_last_change = np.zeros(len(core_sizes))
    for i in range(len(since_last_change)):
        count = 0
        j = i - 1
        while j >= 0 and core_sizes[j] == core_sizes[i]:
            count += 1
            j -= 1
        since_last_change[i] = count
    plt.figure()
    plt.title('Iterations since Constraint Change%s' % title_extra_info)
    plt.xlabel('iteration')
    plt.ylabel('iters since last change')
    plt.plot(since_last_change, 'o')
    '''

    '''
    # plot smoothed ratios
    plt.figure()
    plt.title('Ratio of objective decreases')
    plt.xlabel('iteration')
    plt.ylabel('ratio')
    plt.plot(diff, 'o', label='1')
    end = len(diff)
    for window in [5, 10, 15, 20]:
        start = window
        smooth = np.array([np.average(diff[i-window:i]) for i in range(start, end)])
        plt.plot(range(start, end), smooth, 'o', label=str(window))
    plt.legend(title='Window')
    '''

    #plt.show()
    if mode != 'display_all':
        plt.show()
    else:
        plt.close('all')


# vim: foldmethod=marker

'''
How does time grow as n and m increase?

Several files will display the results of these runs.
The most up-to-date (as of Nov 24, 2021) is display_scaling.py.
'''
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
from mvee import mvee2
from parameters import Param
import time
import os
import pathlib



def read_all(dirname, fnames):
    # {{{
    '''
    For each method in fnames, read its times in from a file and store them
    as a dictionary.

    If a name does not exist, create an empty dictionary for it.
    '''
    timess = {}
    for fname in fnames:
        try:
            ms, ns, arr = read_array(dirname, fname)
            timess[fname] = fill_dict(arr, ms, ns)
        except OSError:
            timess[fname] = {}
    return timess
    # }}}


def read_array(dirname, fname):
    # {{{
    '''
    Read timing information from fname into a 2d array. Assumes that the
    file starts with a line of `ns` and each row starts with the `m` value
    used in the timings.
    '''
    big = np.loadtxt(dirname + '/' + fname)
    ms = big[1:, 0]
    ns = big[0, 1:]
    data = big[1:, 1:]
    return ms, ns, data
    # }}}


def fill_dict(timesarr, ms, ns):
    # {{{
    '''
    Given a 2d array of timing data and the `ms` and `ns` used to generate
    the data, convert the array to a dictionary.
    '''
    times = {}
    for n in ns:
        times[n] = {}
    for j, n in enumerate(ns):
        for i, m in enumerate(ms):
            times[n][m] = timesarr[i, j]
    return times
    # }}}


def fill_array(times, ms=None, ns=None):
    # {{{
    '''
    Given a dictionary corresonding to the timings for a single method,
    convert the dictionary to a 2d array.
    
    If `ms` and `ns` are given, assume we are working with an incomplete
    dictionary and fill the missing values with -1.
    '''
    if ms is not None and ns is not None:
        xvals = ns
        nx = len(ns)
        yvals = ms
        ny = len(ms)
    else:
        xvals = sorted(times.keys())
        nx = len(xvals)
        yvals = sorted(times[xvals[0]].keys())
        ny = len(yvals)
    timesarr = np.zeros((ny, nx))
    for j, n in enumerate(xvals):
        for i, m in enumerate(yvals):
            try:
                timesarr[i, j] = times[n][m]
            except KeyError:
                timesarr[i, j] = -1
    return timesarr
    # }}}


def save_array(times_dict, ms, ns, dirname, label):
    # {{{
    '''
    Given a dict of timings for a single method, write the timings to
    disk as a 2d array.

    If the dict is incomplete, the empty spots will be filled with -1.
    '''
    times = fill_array(times_dict, ms, ns)
    nx = len(ns)
    ny = len(ms)
    ret = np.zeros((ny + 1, nx + 1))
    ret[1:, 0] = ms
    ret[0, 1:] = ns
    ret[1:, 1:] = times

    fname = dirname + '/' + label
    np.savetxt(fname, ret)
    # }}}


#ns = np.logspace(4, 7, num=4, base=2, dtype=int)
ms = np.logspace(3, 6, num=8, base=10, dtype=int)
ns = [100]
#base_dirname = 'outputs/scaling-runs/instance-%d/%s-vary-n'
base_dirname = 'outputs/scaling-runs/instance-%d/%s-vary-m'
epsilon = 1e-6
prob_type = 'randn*cauchy'
#prob_type = 'lognormal'
#prob_type = 'randn'
read_only = False

methods = ['newton', 'todd', 'hybrid']

# Choose appropriate init for non-CA (and non-hybrid) methods
if prob_type == 'lognormal' or prob_type == 'randn*cauchy':
    init = 'qr'
elif prob_type == 'randn':
    init = '2norm'

# Define method-specific parameters
defaults = {'verbose': True, 'max_iter': 10000, 'drop_every': 50,
            'initialize': init}
param = Param(defaults, methods)

if 'todd' in methods:
    param.set_param('todd', 'max_iter', 500000)
    param.set_param('todd', 'drop_every', 50)
if 'hybrid' in methods:
    step_count = 5000
    hybrid = {'method': 'newton', 'step_count': step_count}
    param.set_param('hybrid', 'method', 'todd')
    param.set_param('hybrid', 'hybrid', hybrid)

disp_prob_type = prob_type
if prob_type == 'randn':
    disp_prob_type = 'gaussian'
elif prob_type == 'randn*cauchy':
    disp_prob_type = 'cauchy'

for run_count in range(1, 6):
    dirname = base_dirname % (run_count, disp_prob_type)

    if os.path.exists(dirname):
        print('Directory ' + dirname + ' already exists. Appending...')
        with open(dirname + '/' + 'otherinfo.txt', 'r') as f:
            info_list = f.readlines()
        info_dict = dict([map(str.strip, s.split(':')) for s in info_list])
        seed = int(info_dict['seed'])
        epsilon = float(info_dict['epsilon'])
        prob_type = info_dict['prob_type']
        init = info_dict['initialize']
        timess = read_all(dirname, methods)
    else:
        seed = int(time.time())
        timess = {}
        pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
        with open(dirname + '/' + 'otherinfo.txt', 'w') as f:
            f.write('seed: %d\nepsilon: %g\n' % (seed, epsilon))
            f.write('prob_type: %s\n' % prob_type)
            f.write('max_iter: %d\n' % defaults['max_iter'])
            if 'lbfgs' in methods:
                f.write('lbfgs_n_vecs: %d\n' % param.get_param('lbfgs',
                                                               'lbfgs_n_vecs'))
            f.write('drop_every: %d\n' % defaults['drop_every'])
            f.write('initialize: %s\n' % defaults['initialize'])

    if read_only:
        raise Exception('Data is loaded into timess')

    for method in methods:
        label = param.get_label(method)

        if method not in timess:
            timess[method] = {}
        for n in ns:
            if n not in timess[method]:
                timess[method][n] = {}
            for m in ms:
                #if m not in timess[method][n] or timess[method][n][m] == -1 or \
                  #timess[method][n][m] == np.inf:  # To rerun ones that failed
                if m not in timess[method][n] or timess[method][n][m] == -1:

                    param.set_param(method, 'init_size', n + 1)

                    np.random.seed(seed)
                    if prob_type == 'randn':
                        X = np.random.randn(n, m)
                        #if method not in ('hybrid', 'todd'):
                            #param.set_param(method, 'init_size', int(n**1.5))
                        param.set_param(method, 'init_size', int(n**1.5))
                    elif prob_type == 'randn*cauchy':
                        mat = np.random.randn(n, m)
                        cauchy = np.random.standard_cauchy(m)
                        X = mat*cauchy
                    elif prob_type == 'lognormal':
                        X = np.random.lognormal(size=(n, m))
                    else:
                        raise NotImplementedError('Problem "%s" does not exist'
                                                  % prob_type)

                    t1 = time.time()
                    conv = mvee2(X, full_output=False, epsilon=epsilon,
                                 **param.get_dict(method))['converged']
                    t2 = time.time()
                    if conv:
                        timess[method][n][m] = t2 - t1
                    else:
                        timess[method][n][m] = np.inf
                    save_array(timess[method], ms, ns, dirname, method)
                    print('%s, %d, %3g, done: %f seconds' %
                          (label, n, m, (t2 - t1)))
                    if not conv:
                        break


    # vim: foldmethod=marker

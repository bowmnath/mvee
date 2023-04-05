'''
Plot results of scaling_m_n.py under new running scheme
'''
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
from mvee import mvee2
from plotting import applySettings
from parameters import Param
import time
import os
import argparse


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


parser = argparse.ArgumentParser(description="Plot timing results.")
parser.add_argument("changing_variable",
                    choices=['m', 'n'],
                    help='variable that will not remain fixed')
args = parser.parse_args()

good_labels = {'cg': 'CG', 'lbfgs': 'L-BFGS', 'todd': 'Coord. Ascent',
               'gradient': 'Gradient Ascent', 'bfgsrb': 'BFGS (R)',
               'truncated': 'Trunc. Newton', 'newton': 'Newton',
               'truncated_fd': 'Trunc. Newton (FD)', 'bfgsub': 'BFGS (U)',
               'hybrid': 'Hybrid'}
kur_to_fname = {'Low': 'low', 'High': 'high', 'Very high': 'very'}

methods = ['newton', 'todd', 'hybrid']
run_types = ['gaussian', 'cauchy', 'lognormal']

vary_m = (args.changing_variable == 'm')

n_instances = 5
if vary_m:
    base_dirname = 'outputs/scaling-runs/instance-%d/%s-vary-m/'
else:
    base_dirname = 'outputs/scaling-runs/instance-%d/%s-vary-n/'

results = {}
for run_type in run_types:

    for i in range(1, n_instances + 1):
        dirname = base_dirname % (i, run_type)
        timess = read_all(dirname, methods)

        if not vary_m:
            # flip structure of timess so m comes first as key
            new_timess = {}
            for key in timess:
                new_timess[key] = {}
                largest_n = max(timess[key].keys())
                m = max(timess[key][largest_n].keys())
                new_timess[key][m] = {}
                for n in timess[key]:
                    new_timess[key][m][n] = timess[key][n][m]
            timess = new_timess

        for method in methods:
            non_vary = max(timess[method].keys())

            if method not in results.keys():
                varies = list(timess[method][non_vary].keys())
                results[method] = np.zeros((n_instances, len(varies)))

            results[method][i - 1] = list(timess[method][non_vary].values())

    plt.figure()

    for method in methods:
        non_vary = max(timess[method].keys())
        varies = list(timess[method][non_vary].keys())

        medvals = np.median(results[method], axis=0)
        minvals = np.min(results[method], axis=0)
        maxvals = np.max(results[method], axis=0)

        plt.loglog(varies, medvals, '--o', label=good_labels[method])

    if vary_m:
        xlabel = 'm'
        other_label = 'n'
        plt.title('%s data\nn = %d' % (run_type, non_vary))
    else:
        xlabel = 'n'
        other_label = 'm'
        plt.title('%s data\nm = %d' % (run_type, non_vary))
    applySettings(xlabel, 'time (sec)', None, True)

plt.show()

# vim: foldmethod=marker

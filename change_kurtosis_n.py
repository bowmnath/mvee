'''
Same as change_kurtosis_only.py, but with varying values of n.
'''
import itertools
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.stats
import sklearn.metrics
import argparse

from mvee import mvee2
from mvee import kurtosis
from plotting import applySettings



parser = argparse.ArgumentParser(description="Generate figure 4")
parser.add_argument("method", help="{todd|newton}")
args = parser.parse_args()

if args.method not in ['todd', 'newton']:
    raise Exception('Run with method %s. Acceptable options are '
                    '"todd" and "newton"' % args.method)

m = 5000
ns = [10, 12, 15, 18]
epsilon = 1e-6
max_iter = 1000
n_repeats = 2
method = args.method

kur_params = np.logspace(-1, 2, num=40)
skew_param = 0

if method == 'newton':
    plot_title = 'Newton'
    todd_suffix = ''
elif method == 'todd':
    plot_title = 'Coordinate Ascent'
    max_iter = 10000
    todd_suffix = '-todd'

data = {}
for n in ns:
    kurs = []
    iters = []
    cores = []
    data[n] = {}
    data[n]['kurs'] = kurs
    data[n]['iters'] = iters
    data[n]['cores'] = cores
    for kur_param in kur_params:
        normal = np.random.normal(size=(n, m))
        X = np.sinh(1.0/kur_param*(np.arcsinh(normal) - skew_param))
        kur = kurtosis(X)

        retvals = mvee2(X, initialize='qr', epsilon=epsilon, verbose=False,
                        method=method, max_iter=max_iter, drop_every=50,
                        full_output=True, track_count=True)
        converged, iter_count = [retvals[v] for v in ['converged', 'iter_count']]
        core_set_size = (retvals['u'] > 1e-6).sum()

        if not converged:
            print('failed for kurtosis: ', kur)
        else:
            print('kurtosis, iters: ', kur, iter_count)
            kurs.append(kur)
            iters.append(iter_count)
            cores.append(core_set_size)

plt.figure()
for n in ns:
    kurs = data[n]['kurs']
    iters = data[n]['iters']
    cores = data[n]['cores']
    if False:
        plt.figure(figsize=(3, 3))
        applySettings('log(kurtosis)', 'iterations')
        plt.scatter(np.log10(kurs), iters)
        plot_base_title = 'Sinh-arcsinh transformed data\n%s\nm = %d'
        plt.title(plot_base_title % (plot_title, m))

        plt.figure(figsize=(3, 3))
        applySettings('log(kurtosis)', 'core set size')
        plt.scatter(np.log10(kurs), cores)
        plot_base_title = 'Sinh-arcsinh transformed data\n%s\nm = %d'
        plt.title(plot_base_title % (plot_title, m))

    #plt.figure(figsize=(3, 3))
    applySettings('core set size', 'iterations')
    plt.plot(cores, iters, 'o', label='%d' % n)
    plot_base_title = 'Sinh-arcsinh transformed data\n%s\nm = %d'
    plt.title(plot_base_title % (plot_title, m))

plt.legend(title="n", loc='upper left')
plt.show()

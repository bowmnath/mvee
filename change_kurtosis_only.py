'''
Use sample kurtosis to predict difficulty of problem.
Compare only normal-like distributions with varying kurtosis.
Distributions generated using sinh-arcsinh distributions:
https://stats.stackexchange.com/questions/43482/transformation-to-increase-kurtosis-and-skewness-of-normal-r-v
'''
import itertools
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.stats
import sklearn.metrics
import time

from mvee import mvee2
from mvee import kurtosis
from plotting import applySettings


m = 5000
n = 50
epsilon = 1e-6
max_iter = 1000
n_repeats = 5
method = 'todd'
#method = 'newton'

kur_params = np.logspace(-1, 2, num=40)
skew_param = 0

if method == 'newton':
    plot_title = 'Newton'
    todd_suffix = ''
    fname_str = ''
elif method == 'BFGS':
    plot_title = 'BFGS'
    todd_suffix = ''
elif method == 'todd':
    plot_title = 'Coordinate Ascent'
    max_iter = 10000
    todd_suffix = '-todd'
    fname_str = '-coord-ascent'

interactive_plot = False
#if not interactive_plot:
    #raise Exception('Are you sure you want to overwrite?')

kurs = []
iters = []
cores = []
times = []
for kur_param in kur_params:
    normal = np.random.normal(size=(n, m))
    X = np.sinh(1.0/kur_param*(np.arcsinh(normal) - skew_param))
    kur = kurtosis(X)

    t1 = time.time()
    retvals = mvee2(X, initialize='qr', epsilon=epsilon, verbose=False,
                    method=method, max_iter=max_iter, drop_every=50,
                    full_output=True, track_count=True)
    t2 = time.time()
    converged, iter_count = [retvals[v] for v in ['converged', 'iter_count']]
    core_set_size = (retvals['u'] > 1e-6).sum()

    if not converged:
        print('failed for kurtosis: ', kur)
    else:
        print('kurtosis, iters: ', kur, iter_count)
        kurs.append(kur)
        iters.append(iter_count)
        cores.append(core_set_size)
        times.append(t2 - t1)

basedir = 'outputs/kur-tests%s/kur-test-%d/' % (fname_str, n)
np.savetxt(basedir + 'change-kurtosis-only-kurs', kurs)
np.savetxt(basedir + 'change-kurtosis-only-cores', cores)
np.savetxt(basedir + 'change-kurtosis-only-n', np.array([n]))
np.savetxt(basedir + 'change-kurtosis-only-times', times)
np.savetxt(basedir + 'change-kurtosis-only-iters', iters)

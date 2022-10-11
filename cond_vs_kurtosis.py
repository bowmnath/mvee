'''
Determine how kurtosis affects:
    condition number of test matrix
    condition number of Hessian
    size of core set

Most of this code was copied from prediction_not_beta.py.

Warning -- if any of the runs do not finish, the labels in the legend will
be wrong
'''
import itertools
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.stats
import sklearn.metrics
import time

from mvee import mvee2, projected_Hessian, kurtosis
from plotting import applySettings


m = 5000
n = 50
epsilon = 1e-6
n_repeats = 5
#method = 'todd'
method = 'newton'
interactive_plot = False

if method == 'newton':
    fname_str = ''
elif method == 'todd':
    fname_str = '-coord-ascent'

#if not interactive_plot:
    #raise Exception('Are you sure you want to overwrite?')

distros = []
for i in range(n_repeats):
    mat = np.random.randn(n, m)
    cauchy = np.random.standard_cauchy(m)
    distros.append(mat)
    distros.append(np.random.randint(-1000000, 100001, size=(n, m)))
    distros.append(np.random.beta(0.99, 0.5, size=(n, m)))
    distros.append(np.random.beta(1.9, 0.25, size=(n, m)))
    distros.append(np.random.beta(2.1, 0.2, size=(n, m)))
    distros.append(np.random.binomial(50, 0.125, size=(n, m)))
    distros.append(np.random.chisquare(70000, size=(n, m)))
    distros.append(np.random.chisquare(2, size=(n, m)))
    distros.append(np.random.laplace(size=(n, m)))
    distros.append(np.random.exponential(size=(n, m)))
    distros.append(np.random.gumbel(size=(n, m)))
    distros.append(np.random.logistic(size=(n, m)))
    distros.append(np.random.lognormal(size=(n, m)))
    distros.append(np.random.poisson(size=(n, m)))
    distros.append(np.random.rayleigh(size=(n, m)))
    distros.append(np.random.uniform(-1, 1, size=(n, m)))
    distros.append(scipy.stats.gamma(6/250).rvs(size=(n, m)))
    distros.append(scipy.stats.gamma(6/80).rvs(size=(n, m)))
    distros.append(scipy.stats.gamma(6/5).rvs(size=(n, m)))

labels = []
labels.append('randn')
labels.append('randint')
labels.append('beta(0.99, 0.5)')
labels.append('beta(1.9, 0.25)')
labels.append('beta(2.1, 0.2)')
labels.append('binomial(50, 0.125)')
labels.append('chisquare(70000)')
labels.append('chisquare(2)')
labels.append('laplace')
labels.append('exponential')
labels.append('gumbel')
labels.append('logistic')
labels.append('lognormal')
labels.append('poisson')
labels.append('rayleigh')
labels.append('uniform')
labels.append('gamma(6/250)')
labels.append('gamma(6/80)')
labels.append('gamma(6/5)')

assert len(labels) == len(distros)/n_repeats

kurs = []
iters = []
cores = []
times = []
for dist in distros:
    X = dist
    kur = kurtosis(X)

    t1 = time.time()
    retvals = mvee2(X, initialize='qr', epsilon=epsilon, verbose=False,
                    method=method,
                    large_output=True,
                    max_iter=20000, drop_every=50, track_count=True,
                    full_output=True,
                    upproject=True)
    t2 = time.time()
    converged, iter_count = [retvals[v] for v in ['converged', 'iter_count']]

    u = retvals['u']
    core = np.sum(u > 1e-6)

    if not converged:
        print('failed for kurtosis: ', kur)
    else:
        print('kurtosis, iters: ', kur, iter_count)
        kurs.append(kur)
        iters.append(iter_count)
        cores.append(core)
        times.append(t2 - t1)

basedir = 'outputs/kur-tests%s/kur-test-%d/' % (fname_str, n)
np.savetxt(basedir + 'cond-vs-kurtosis-kurs', kurs)
np.savetxt(basedir + 'cond-vs-kurtosis-cores', cores)
np.savetxt(basedir + 'cond-vs-kurtosis-n', np.array([n]))
np.savetxt(basedir + 'cond-vs-kurtosis-times', times)
np.savetxt(basedir + 'cond-vs-kurtosis-iters', iters)

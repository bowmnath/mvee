'''
Track the error of methods over iterations. Plot them in a way designed to
demonstrate superlinear convergence.

Determine which iterations change the core set vs the values.
'''
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from mvee import mvee2, mvee_eig
from mvee import initialize_ky, initialize_qr
from parameters import Param
from plotting import applySettings
import time

n = 10
m = 5000
epsilon = 1e-7
newt_epsilon = 1e-14
plotter = plt.semilogy
dist = 'low'

interactive_plot = True
if not interactive_plot:
    raise Exception('Are you sure you want to overwrite this file?')

methods = ['todd', 'newton']

# Define method-specific parameters
defaults = {'verbose': True, 'max_iter': 1000}
param = Param(defaults, methods)

param.set_param('todd', 'max_iter', 50000)

# Generate test problem
mat = np.random.randn(n, m)
cauchy = np.random.standard_cauchy(m)
if dist == 'high':
    X = mat*cauchy
elif dist == 'low':
    X = mat
else:
    raise Exception
#X = np.random.lognormal(size=(n, m))

true_sol = mvee2(X, initialize='ky', epsilon=newt_epsilon, verbose=True,
                 max_iter=1000, full_output=True, method='newton',
                 drop_every=np.inf)['u']

init = initialize_qr(X, n_desired=len(X) + 1)
#init = initialize_qr(X, 3*n)

#plt.figure(figsize=(6.4, 4.8))

for method in methods:
    plt.figure()
    label = param.get_label(method)

    t1 = time.time()
    ret = mvee2(X, initialize='given', full_output=False, track_all_iters=True,
                epsilon=epsilon, drop_every=np.inf, update_data=init,
                constraint_add_inds=True, constraint_rem_inds=True,
                track_stepsizes=True,
                **param.get_dict(method))
    obj = ret['us']
    c_a, c_r = ret['constraint_add_inds'], ret['constraint_rem_inds']
    #aks = ret['stepsizes']
    t2 = time.time()
    print('%s done: %f seconds' % (label, (t2 - t1)))
    err = np.array([la.norm(z - true_sol) for z in obj])
    #plotter(err, '-', label=label)

    plotter(err, 'b-', label=label)
    non_c = [i for i in range(len(obj)) if i not in c_a and i not in c_r]
    plotter(non_c, [err[c] for c in non_c], 'bo', markersize=9)
    plotter(c_a, [err[c] for c in c_a], 'r^', label='add constraint',
            markersize=9)
    plotter(c_r, [err[c] for c in c_r], 'k^', label='remove constraint',
            markersize=9)

    applySettings('iteration', '||error||', None, True)

    plt.title('Convergence of Methods\nm = %d; n = %d' % (m, n))
    plt.show()

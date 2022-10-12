'''
Determine the correct "fudge factor" for the step model of
kurtosis vs core set size by plotting the median of each clump for various
values of n.
'''
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from plotting import applySettings



fnames = {'general': 'cond-vs-kurtosis', 'specific': 'change-kurtosis-only'}
base_basedir = 'outputs/kur-tests/kur-test-%d/'
plotter = plt.loglog
breakpoints = [3, 30]

ns = [10, 20, 30, 40, 50, 60]
maxs = defaultdict(list)
mins = defaultdict(list)
low_meds = defaultdict(list)
high_meds = defaultdict(list)
for base in ns:
    basedir = base_basedir % base
    for name in fnames.keys():
        fname = fnames[name]
        cores = np.loadtxt(basedir + fname + '-cores')
        kurs = np.loadtxt(basedir + fname + '-kurs')
        n = int(np.loadtxt(basedir + fname + '-n'))
        maxs[name].append(np.max(cores))
        mins[name].append(np.min(cores))
        low_meds[name].append(np.median(cores[kurs < breakpoints[0]]))
        high_meds[name].append(np.median(cores[kurs > breakpoints[1]]))

ns = np.array(ns)

# Plot maxima and minima (left-hand side of thesis figure)
plt.figure()
for fname in fnames.keys():
    fmaxs = np.array(maxs[fname])
    fmins = np.array(mins[fname])
    plotter(ns, fmaxs, 'o-', label='%s max' % fname)
    plotter(ns, fmins, 'o-', label='%s min' % fname)
plotter(ns, ns*(ns + 1)/2, '--', label='n(n + 1)/2')
plotter(ns, ns*np.sqrt(ns), '--', label='n sqrt(n)')
plotter(ns, ns, '--', label='n')
applySettings('n', 'core set size', None, True)
plt.title('Maximum and minimum core set sizes')

labels = plt.gca().get_xticklabels(minor=True)
for ind in [16, 18, 19]:
    labels[ind].set_visible(False)

# Plot medians (right-hand side of thesis figure)
plt.figure()
for fname in fnames.keys():
    fmaxs = np.array(low_meds[fname])
    fmins = np.array(high_meds[fname])
    plotter(ns, fmaxs, 'o-', label='%s low' % fname)
    plotter(ns, fmins, 'o-', label='%s very high' % fname)
plotter(ns, 1.3*ns*np.sqrt(ns), '--', label='1.3*n sqrt(n)')
plotter(ns, 1.4*ns, '--', label='1.4*n')
applySettings('n', 'core set size', None, True)
plt.title('Median core set sizes')

labels = plt.gca().get_xticklabels(minor=True)
for ind in [16, 18, 19]:
    labels[ind].set_visible(False)

plt.show()

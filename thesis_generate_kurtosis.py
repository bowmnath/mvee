'''
Compare core set size vs log(kurtosis) for two families of distributions.
'''
import numpy as np
import matplotlib.pyplot as plt
from plotting import applySettings



do_todd = False
do_iters = False
varied_data = False
n = 50
m = 5000
if do_todd:
    basedir = 'outputs/kur-tests-coord-ascent/kur-test-%d/' % n
else:
    basedir = 'outputs/kur-tests/kur-test-%d/' % n

if varied_data:
    kurs = np.loadtxt(basedir + 'cond-vs-kurtosis-kurs')
    cores = np.loadtxt(basedir + 'cond-vs-kurtosis-cores')
    iters = np.loadtxt(basedir + 'cond-vs-kurtosis-iters')
    plt.title('Various Distributions\nm = %d; n = %d' % (m, n))
else:
    kurs = np.loadtxt(basedir + 'change-kurtosis-only-kurs')
    cores = np.loadtxt(basedir + 'change-kurtosis-only-cores')
    iters = np.loadtxt(basedir + 'change-kurtosis-only-iters')
    plt.title('sinh-arcsinh-Transformed Data\nm = %d; n = %d' % (m, n))

if do_iters:
    plt.scatter(np.log10(kurs), iters, label='simulated')
else:
    plt.scatter(np.log10(kurs), cores, label='simulated')

if do_iters:
    applySettings('log(kurtosis)', 'iterations', None, False)
else:
    applySettings('log(kurtosis)', 'core set size', None, False)

if not do_iters:
    # plot lines to show different possible core set sizes
    #xmax = np.log10(np.max(kurs)) + .1
    xmax = 6
    #plt.plot([-0.25, xmax], [n*(n + 1)/2]*2, '--r', label=r'$n(n + 1)/2$')
    plt.plot([-0.25, xmax], [n*np.sqrt(n)]*2, '--c', label=r'$n \sqrt{n}$')
    plt.plot([-0.25, xmax], [n]*2, '--k', label=r'$n$')

#savedxlim = plt.xlim()
#plt.xlim(0.327689, 4.855427)
#plt.xlim(-0.23175889, 4.8680236)
#plt.xlim(0.032769, 4.855427)
plt.xlim(0.032769, 6.2)

plt.show()

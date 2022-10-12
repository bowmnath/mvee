'''
Compare core-set size or iters vs log(kurtosis) for various data.
'''
import numpy as np
import matplotlib.pyplot as plt
import argparse
from plotting import applySettings



parser = argparse.ArgumentParser(description="Plot kurtosis-related results")
parser.add_argument("method", help="{todd|newton} Does not matter if "
                                   "plotting core set size, but must still "
                                   " be passed")
parser.add_argument("n", type=int)
parser.add_argument("-i", "--iters", action="store_true",
                    help="Plot iterations (instead of core set size)")
parser.add_argument("-v", "--varied", action="store_true",
                    help="Use varied date (instead of sinh-arcsinh)")
parser.add_argument("-u", "--upper_bound", action="store_true",
                    help="Plot n**2 bound (only if plotting core set size)")
parser.add_argument("-r", "--real_data", action="store_true",
                    help="Plot BoW and MNIST data")
args = parser.parse_args()

if args.method not in ['todd', 'newton']:
    raise Exception('Run with argument %s. Acceptable options are '
                    '"todd" and "newton"' % args.method)

if args.n < 2:
    raise Exception('Run with n = %d. Must be an integer greater than 1'
                    % args.n)

if args.real_data and args.n != 50:
    raise Exception('Run with n = %d. When run with --real_data, '
                    'n must be 50' % args.n)

do_todd = (args.method == 'todd')
do_iters = args.iters
varied_data = args.varied
show_real_data = args.real_data
n = args.n
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

if show_real_data:
    if do_todd:
        base_mnist_dir = 'outputs/mnist/coord-ascent/'
        col_ind = 0
        max_iter = 10000
    else:
        base_mnist_dir = 'outputs/mnist/newton/'
        col_ind = 0
        max_iter = 200
    kurs_m = np.load(base_mnist_dir + 'kurtosis.npy')[:, col_ind]  # 49
    cores_m = np.load(base_mnist_dir + 'cores.npy')[:, col_ind]  # 49
    iters_m = np.load(base_mnist_dir + 'iters.npy')[:, col_ind]  # 49

    if do_todd:
        base_docs_dir = 'outputs/doc-dumps/coord-ascent/%s/'
    else:
        base_docs_dir = 'outputs/doc-dumps/%s/'
    docs_files = ['enron', 'kos', 'nips', 'nytimes']

    kurs_docs = np.array([])
    cores_docs = np.array([])
    iters_docs = np.array([])
    for doc_dump in docs_files:
        kurs_new = np.load(base_docs_dir % doc_dump + 'kurtosis.npy')
        cores_new = np.load(base_docs_dir % doc_dump + 'cores.npy')
        iters_new = np.load(base_docs_dir % doc_dump + 'iters.npy')
        iters_new = iters_new[iters_new < max_iter]
        '''
        # One per document dump (rather than per subset sampled)
        kurs_new = np.median(kurs_new)
        cores_new = np.median(cores_new)
        iters_new = np.median(iters_new)
        '''
        kurs_docs = np.hstack([kurs_docs, kurs_new])
        cores_docs = np.hstack([cores_docs, cores_new])
        iters_docs = np.hstack([iters_docs, iters_new])

    '''
    # Plot combined data
    kurs_e = np.hstack([kurs_m, kurs_docs])
    cores_e = np.hstack([cores_m, cores_docs])
    iters_e = np.hstack([iters_m, iters_docs])
    plt.scatter(np.log10(kurs_e), cores_e, marker='x',
                label='empirical data')
    '''

    if do_iters:
        plt.scatter(np.log10(kurs_m), iters_m, marker='x', color='green',
                    label='MNIST')
        plt.scatter(np.log10(kurs_docs), iters_docs, marker='x', color='orange',
                    label='BoW')
    else:
        plt.scatter(np.log10(kurs_m), cores_m, marker='x', color='green',
                    label='MNIST')
        plt.scatter(np.log10(kurs_docs), cores_docs, marker='x', color='orange',
                    label='BoW')

if not do_iters:
    # plot lines to show different possible core set sizes
    #xmax = np.log10(np.max(kurs)) + .1
    xmax = 6
    if args.upper_bound:
        plt.plot([-0.25, xmax], [n*(n + 1)/2]*2, '--r', label=r'$n(n + 1)/2$')
    plt.plot([-0.25, xmax], [n*np.sqrt(n)]*2, '--c', label=r'$n \sqrt{n}$')
    plt.plot([-0.25, xmax], [n]*2, '--k', label=r'$n$')

#savedxlim = plt.xlim()
#plt.xlim(0.327689, 4.855427)
#plt.xlim(-0.23175889, 4.8680236)
#plt.xlim(0.032769, 4.855427)
plt.xlim(0.032769, 6.2)

plt.show()

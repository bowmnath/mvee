'''
Display various runs from perfect_hybrid.py together in one plot
'''
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from parameters import Param
from plotting import applySettings
from helpers import NumpyEncoder
import time
import os
import json



plotter = plt.plot
base_dirname = './outputs/hybrid-perfect/'

# {{{ Adding perfect to other 5
#all_seeds = [11550, 151550, 121550, 171550, 161550, 111550]  # CA; Gaussian; n=50
all_seeds = [101550, 91550, 81550, 131550, 141550, 181550]  # Newton; Gaussian; n=50

colors = ['green', 'orange', 'blue', 'brown', 'yellow', 'red']  # Newton
order = [0, 1, 2, 4, 3, 5]  # Newton

#colors = ['green', 'yellow', 'orange', 'red', 'brown', 'blue']  # CA 50
#order = [0, 2, 5, 1, 4, 3]  # CA 50

ca_or_newt = 'newton'
data_type = 'gaussian'
num = 50
# }}}

for seed_num, seed in enumerate(all_seeds):

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
    init_size_factor = str(info_dict['init_size_factor'])

    with open(dirname + 'output.json', 'r') as f:
        all_output = json.load(f)
    core_sizes = all_output['core_sizes']
    diff = all_output['diff']

    if prob_type == 'low':
        prob_str = 'Gaussian'
    elif prob_type == 'very':
        prob_str = 'Cauchy'
    elif prob_type == 'high':
        prob_str = 'Lognormal'

    diff = np.array(diff)
    core_sizes = np.array(core_sizes)
    title_extra_info = '\nm = %d, n = %d\n%s data' % (m, n, prob_str)

    img_dirname = base_dirname + 'images/' + str(seed) + '/'
    os.makedirs(img_dirname, exist_ok=True)

    # plot core set size
    #plt.figure()
    total_size_init = init_size
    if init_size_factor != '1':
        total_size_init = init_size_factor + '*' + total_size_init

    if init_method != 'perfect':
        label = '%s(%s)' % (init_method, total_size_init)
    else:
        label = init_method

    plt.plot(core_sizes, 'o', color=colors[seed_num], label=label)
    print('%s: %d' % (label, len(core_sizes)))
    # random jitter to avoid overlap
    #noise = np.random.uniform(-0.8, 0.8, len(core_sizes))
    #plt.plot(core_sizes + noise, 'o', markersize=1, label='%s(%s)' % (init_method, init_size))
    #plt.legend(loc='lower right')

    #plt.savefig(img_dirname + 'core_set_size.png')

handles, labels = plt.gca().get_legend_handles_labels()
lgd = plt.legend([handles[idx] for idx in order],
           [labels[idx] for idx in order],
           loc='lower right', labelspacing=0.85)
plt.setp(lgd.get_texts(), fontsize='9')
plt.title('Core set size%s' % title_extra_info)
applySettings('iteration', 'size', None, False)

plt.show()

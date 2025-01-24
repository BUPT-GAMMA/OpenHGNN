import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import os
import json
import scipy
import scipy.stats

jian_file = 'result2'
grid_file = 'result2'
datasets = ['HGBn-ACM', 'HGBn-DBLP', 'HGBn-IMDB', 'HNE-PubMed', 'HGBn-Freebase', 'HGBn-ACM']
xL = [[[0, 1], [0, 1], [0, 1],[0, 1] ,[0, 1],[0, 1],],
      [[0.8, 0.95], [0.7, 0.95], [0.5, 0.65], [0.1, 0.6], [0.2, 0.5], [0.2, 0.5]],]
yL = [[[0, 1], [0, 1], [0, 1],[0, 1], [0, 1],[0, 1],],
      [[0.6, 1], [0.55, 1], [0.6, 1],[0.6, 1], [0.6, 1],[0.6, 1]]]

# jian_file = 'result2'
# grid_file = 'result2'
# datasets = ['HGBl-ACM', 'HGBl-DBLP', 'HGBl-IMDB', 'HGBl-PubMed', 'HGBl-amazon', 'HGBl-LastFM']
# xL = [[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
#       [[0.8, 1], [0.6, 1], [0.5, 1], [0.7, 1], [0.8, 1],[0.8, 1]]]
# yL = [[[0,1], [0,1], [0,1],[0,1], [0,1],[0,1],],
#       [[0.6, 1], [0.6, 1], [0.6, 1],[0.6, 1], [0.6, 1],[0.6, 1],]]

score = 'score'
dim = 'model_family'

num_data = len(datasets)
# Detectron colors
_COLORS = np.array([
    0.000, 0.447, 0.741,
    0.850, 0.325, 0.098,
    0.929, 0.694, 0.125,
    0.494, 0.184, 0.556,
    0.466, 0.674, 0.188
]).astype(np.float32).reshape((-1, 3))

# Random number generator seed
_RNG_SEED = 1

# Fix RNG seeds
random.seed(_RNG_SEED)
np.random.seed(_RNG_SEED)

# Directory where sweep summaries are stored
_DATA_DIR = '.'

def load_sweep(sweep_name):
    """Loads a sweep summary."""
    summary_path = os.path.join(_DATA_DIR, '{}.csv'.format(sweep_name))
    with open(summary_path, 'r') as f:
        sweep_summary = pd.read_csv(f, sep=',')
    return sweep_summary

# Load ResNet sweep
results1 = load_sweep('{}'.format(jian_file))
results2 = load_sweep('{}'.format(grid_file))

def draw( i, j, ax, has_y=True, has_x=True):
    if i == 0:
        results = results1
    else:
        results = results2
    dataset = datasets[j]
    homo = results[(results[dim] == 'homo') & (results['dataset'] == dataset)]
    homo = set(homo[score].values.tolist())

    relation = results[(results[dim] == 'relation') & (results['dataset'] == dataset)]
    relation = set(relation[score].values.tolist())

    mp = results[(results[dim] == 'metapath') & (results['dataset'] == dataset)]
    mp = set(mp[score].values.tolist())

    mix = results[(results[dim] == 'mixed') & (results['dataset'] == dataset)]
    mix = set(mix[score].values.tolist())
    # Text experiment, point estimates
    random.seed(_RNG_SEED)

    num_trials = 5000
    N_mp = len(mp)
    N_relation = len(relation)
    N_homo = len(homo)
    N_mix = len(mix)

    random.seed(_RNG_SEED)

    err_homo = sorted([j for j in homo])
    err_mp = sorted([j for j in mp])
    err_relation = sorted([j for j in relation])
    err_mix = sorted([j for j in mix])



    edf_homo = np.arange(N_homo) / float(N_homo - 1)
    edf_relation = np.arange(N_relation) / float(N_relation - 1)
    edf_mp = np.arange(N_mp) / float(N_mp - 1)
    edf_mix = np.arange(N_mix) / float(N_mix)


    ax.plot(
        err_homo, edf_homo, color=_COLORS[1], linewidth=2, alpha=0.8,
        zorder=1, label='{}=homo'.format(dim)
    )


    ax.plot(
        err_relation, edf_relation, color=_COLORS[0], linewidth=2, alpha=0.8,
        zorder=0, label='{}=relation'.format(dim)
    )

    ax.plot(
        err_mp, edf_mp, color=_COLORS[2], linewidth=2, alpha=0.8,
        zorder=1, label='{}=metapath'.format(dim)
    )

    # ax.plot(
    #     err_mix, edf_mix, color=_COLORS[3], linewidth=2, alpha=0.8,
    #     zorder=0, label='{}=mixed'.format(dim)
    # )

    #ax.set_xlim([4.5, 13.5])
    ax.set_xlim(xL[i][j])
    ax.set_ylim(yL[i][j])
    #ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    if not has_x:
        ax.set_xlabel('', fontsize=20)
    else:
        ax.set_xlabel('{}'.format(dataset), fontsize=20)
    if not has_y:
        ax.set_ylabel('', fontsize=20)
    else:
        ax.set_ylabel('cumulative prob.', fontsize=20)
    ax.grid(alpha=0.4)
    #ax.legend(loc='upper left', prop={'size': 14})


r, c = 2, num_data
l_w, l_h = 4, 3
r_w, r_h = 4, 3
fig, axes = plt.subplots(
    nrows=r, ncols=c,
    figsize=(22, 6),
    gridspec_kw = {'width_ratios': [2] * num_data}
)

for i in range(2):
    for j in range(len(datasets)):
        draw(i, j, axes[i, j], has_x = i==1, has_y= j == 0)

plt.tight_layout()
#plt.subplots_adjust(left=0.1, bottom=0.2, right=0.85, top=0.9, hspace=0.4, wspace=0.5)
plt.subplots_adjust(left=0.05, bottom=0.2, right=0.97, top=0.9, hspace=0.3, wspace=0.25)
lines, labels = fig.axes[-1].get_legend_handles_labels()

fig.legend(lines, labels, loc='center right', title_fontsize= 'large', )

path = 'figs/1112'
if not os.path.exists(path):
    os.makedirs(path)
plt.savefig('{}/all_{}_node_1112.png'.format(path, dim), dpi=300)

plt.show()
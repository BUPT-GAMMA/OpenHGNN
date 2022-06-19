import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import os

sns.set(style='ticks',context='poster')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

from scipy.stats import rankdata, ttest_ind, f_oneway
from matplotlib.ticker import MaxNLocator

results_file_path = 'result.csv'
bias = 0.01
df = pd.read_csv(results_file_path)
dataset = 'all'
model = 'all'
df=df[df['model_family'] != 'mixed']
if dataset != 'all':
    df = df[df['dataset'] == dataset]
if model != 'all':
    df = df[df['model'] == model]
name_mapping = { 'has_bn':'Batch Normalization', 'has_l2norm':'L2 Normalization', 'dropout':'Dropout', 'aggregation':'Aggregation',
                'layers_gnn':'Message passing layers', 'layers_pre_mp': 'Pre-process layers', 'layers_post_mp': 'Post-process layers', 'stage_type': 'Layer connectivity',
                'lr': 'Learning rate', 'max_epoch':'Training epochs', 'model': 'Model', 'macro_func' : 'Macro function', 'activation': 'Activation', 'num_heads' : 'Number of heads', 'hidden_dim':'Hidden dimension', 'optimizer': 'Optimizer'}


column = df.columns.values.tolist()
score = 'score'
column.remove(score)
column.remove('value')
column.remove('epoch')
column.remove('time')
column.remove('parameter')

def get_acc(df, name, ax, metric='acc', has_y=True):
    print(name)
    df_selected = df[df['key'] == name].copy()
    df_selected[column] = df_selected[column].fillna('Nan')
    # if(name == 'macro_func'):
    #     df_selected = df_selected[df_selected['value'] != 'homo_GNN']
    #     if model == 'homo_GNN':
    #         return
    # df_selected[column] = df_selected[column].fillna('Nan')
    # if(name == 'num_heads'):
    #     df_selected['num_heads'] = df_selected['value']
    #     df_selected = df_selected[df_selected['gnn_type'] == 'gatconv']
    # if(name == 'optimizer'):
    #     df_selected['optimizer'] = df_selected['value']
    # if(name == 'activation'):
    #     df_selected['activation'] = df_selected['value']
    #     df_selected = df_selected[(df_selected['activation'] != 'relu6') & (df_selected['activation'] != 'sigmoid')]
    # if (name == 'lr'):
    #     #df_selected['activation'] = df_selected['value']
    #     df_selected = df_selected[(df_selected['lr'] != 0.0001)]
    # if (name == 'max_epoch'):
    #
    #     df_selected = df_selected[(df_selected['max_epoch'] != 50) & (df_selected['max_epoch'] != 300)]

    # column_temp = copy.deepcopy(column)
    # column_temp.remove(name)
    column_temp = ['key', 'dataset', 'model_family', 'gnn_type', 'times']
    #df_selected['num_heads'] = df_selected['value']2
    df_pivot = pd.pivot_table(df_selected, values=score, index=column_temp, columns=[name], aggfunc=np.mean)
    accs_np = df_pivot.fillna(df_pivot.min()).values.round(5)
    options = df_pivot.columns.values

    ranks_raw = {'Model ID': [], 'Accuracy': [], 'Acc. Ranking': [], name_mapping[name]: []}

    rank_np = np.zeros((accs_np.shape[0], accs_np.shape[1]))
    for i, row in enumerate(accs_np):
        # (1) rank is asceneding, so we neg the row; (2) rank start with 1 so we minus 1
        rank_base = -row
        med = np.median(rank_base)

        for j in range(len(rank_base)):
            if abs(rank_base[j] - med) <= bias:
                rank_base[j] = med
        rank = rankdata(rank_base, method='min')


        for j in range(len(rank)):
            ranks_raw['Model ID'].append(i)
            ranks_raw['Accuracy'].append(accs_np[i, j])
            ranks_raw['Acc. Ranking'].append(rank[j])
            ranks_raw[name_mapping[name]].append(options[j])
            rank_np[i, j] = rank[j]

    #     if metric == 'rank_bar':
    #         p_ffffc value = f_oneway(*[rank_np[:,i] for i in range(rank_np.shape[1])])[1]
    #         print(name, p_value, p_value<0.05, p_value<0.05/12)

    ranks_raw = pd.DataFrame(data=ranks_raw)
    with sns.color_palette("muted"):
        if metric == 'acc':
            splot = sns.violinplot(x=name_mapping[name], y="Accuracy", inner="box", data=ranks_raw, cut=0, ax=ax)
        elif metric == 'rank_bar':
            splot = sns.barplot(x=name_mapping[name], y="Acc. Ranking", data=ranks_raw, ax=ax)
            ax.set_ylim(bottom=1)
            ax.set_yticks([1, 2])
            ax.set_xlabel('', fontsize=48)
            if not has_y:
                ax.set_ylabel('', fontsize=48)
            else:
                ax.set_ylabel('Average', fontsize=48)
        elif metric == 'rank_violin':
            sns.violinplot(x=name_mapping[name], y="Acc. Ranking", inner="box", data=ranks_raw, cut=0, ax=ax)
            ax.set_ylim(bottom=1)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            if not has_y:
                ax.set_ylabel('', fontsize=48)
            else:
                ax.set_ylabel('Distribution', fontsize=48)
        ax.xaxis.label.set_size(48)
        ax.yaxis.label.set_size(48)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(40)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(40)

count = 3
for names in [['has_bn', 'dropout', 'activation', 'has_l2norm', 'layers_pre_mp',  'layers_post_mp',], [ 'layers_gnn', 'stage_type', 'hidden_dim', 'optimizer',  'lr',  'max_epoch', ]]:
#for names in [['lr', 'dropout', 'activation', 'has_l2norm', 'layers_pre_mp',  'layers_post_mp',], [ 'layers_gnn', 'stage_type', 'hidden_dim', 'optimizer',  'lr',  'max_epoch', ]]:
#for names in [['num_heads', 'macro_func', ],['num_heads', 'macro_func',]]:
    col = 6
    row = 2
    f, axes = plt.subplots(nrows=row, ncols=col, figsize=(55, 10))
    #f, axes = plt.subplots(nrows=row, ncols=col, figsize=(20, 10))

    for j, metric in enumerate(['rank_bar', 'rank_violin']):
        for i, name in enumerate(names):
            get_acc(df, name, axes[j, i], metric, has_y=i == 0)
    f.text(0.02, 0.5, 'Score Ranking', ha='center', va='center', rotation='vertical', fontsize=48)
    plt.tight_layout()
    #plt.subplots_adjust(wspace=0.2, hspace=0.2)

    plt.subplots_adjust(left=0.05, bottom=0.2, right=0.97, top=0.9, hspace=0.3, wspace=0.2)
    #plt.subplots_adjust(left=0.15, bottom=0.2, right=0.97, top=0.9, hspace=0.3, wspace=0.2)
    path = 'figs/1112'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig('{}/{}_all_{}_{}.png'.format(path, dataset, count, bias), dpi=300)
    count += 1
    plt.show()
import random
import numpy as np
from scipy.stats import rankdata
import subprocess
import logging
import math


def cal_ranks(scores, labels, filters):
    scores = scores - np.min(scores, axis=1, keepdims=True)
    full_rank = rankdata(-scores, method='ordinal', axis=1)
    filter_scores = scores * filters
    filter_rank = rankdata(-filter_scores, method='ordinal', axis=1)
    ranks = (full_rank - filter_rank + 1) * labels
    ranks = ranks[np.nonzero(ranks)]
    return list(ranks)


def cal_performance(ranks, masks):
    mrr = (1. / ranks).sum() / len(ranks)
    m_r = sum(ranks) * 1.0 / len(ranks)
    h_1 = sum(ranks <= 1) * 1.0 / len(ranks)
    h_3 = sum(ranks <= 3) * 1.0 / len(ranks)
    h_10 = sum(ranks <= 10) * 1.0 / len(ranks)
    h_10_50 = []
    for i, rank in enumerate(ranks):
        num_sample = 50
        threshold = 10
        score = 0
        fp_rate = (rank - 1) / masks[i]
        for i in range(threshold):
            num_comb = math.factorial(num_sample) / math.factorial(i) / math.factorial(num_sample - i)
            score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i))
        h_10_50.append(score)
    h_10_50 = np.mean(h_10_50)

    return mrr, m_r, h_1, h_3, h_10, h_10_50


def select_gpu():
    nvidia_info = subprocess.run('nvidia-smi', stdout=subprocess.PIPE)
    gpu_info = False
    gpu_info_line = 0
    proc_info = False
    gpu_mem = []
    gpu_occupied = set()
    i = 0
    for line in nvidia_info.stdout.split(b'\n'):
        line = line.decode().strip()

        if gpu_info:
            gpu_info_line += 1
            if line == '':
                gpu_info = False
                continue
            if not ('RTX' in line or 'GTX' in line or ''):
                try:
                    mem_info = line.split('|')[2]
                    used_mem_mb = int(mem_info.strip().split()[0][:-3])
                    gpu_mem.append(used_mem_mb)
                except:
                    continue
        if proc_info:
            if line == '|  No running processes found                                                 |':
                continue
            if line == '+-----------------------------------------------------------------------------+':
                proc_info = False
                continue
            proc_gpu = int(line.split()[1])
            gpu_occupied.add(proc_gpu)

        if line == '|===============================+======================+======================|':
            gpu_info = True
        if line == '|=============================================================================|':
            proc_info = True
        i += 1

    for i in range(0, len(gpu_mem)):
        if i not in gpu_occupied:
            # print('Automatically selected GPU Np.{} because it is vacant.'.format(i))
            return i

    for i in range(0, len(gpu_mem)):
        if gpu_mem[i] == min(gpu_mem):
            # print('All GPUs are occupied. Automatically selected GPU No.{} because it has the most free memory.'.format(i))
            return i


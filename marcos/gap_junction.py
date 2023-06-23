#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 19:12:53 2023

@author: marcos
"""


from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np
from scipy import interpolate

from utils import contenidos, find_point_by_value, calc_mode, sort_by
import analysis_utils as au

import pyabf

#%% Visualize one dataset

BASE_DIR = '/media/marcos/DATA/marcos/FloClock_data/gap_junctions/'

pair_type = 3
run_nr = 0

files = contenidos(BASE_DIR)
runs = contenidos(files[pair_type])
file = contenidos(runs[run_nr], filter_ext='.abf', sort='age')[-1]

# Load data
abf = pyabf.ABF(file)

fig, axarr = plt.subplots(abf.sweepCount*2, sharex=True, figsize=[11, 8])

# Set sweep and channel, extract tmes and data

for i in abf.sweepList:
    abf.setSweep(sweepNumber=i, channel=0)
    times = abf.sweepX
    data = abf.sweepY
    perturbation = abf.sweepC
    
    abf.setSweep(sweepNumber=i, channel=1)
    data2 = abf.sweepY
    
    ax1 = axarr[i*2]
    ax2 = axarr[i*2+1]
    
    ax1.plot(times, data)
    ax2.plot(times, data2, c='C2')
    ax1.plot(times, perturbation+data.mean(), c='C1')
    ax1.grid()
    ax2.grid()
    
    for p1 in abf.sweepEpochs.p1s:
        ax1.axvline(times[p1], color='k', ls='--', alpha=.5)

ax2.set_xlabel('time [sec]')
ax2.set_xlim(times.min(), times.max())

fig.suptitle(runs[0].parent.name)
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
from scipy import interpolate, signal, stats

from utils import contenidos, find_point_by_value, calc_mode, sort_by, enzip
import analysis_utils as au

import pyabf

def lowpass_filter(times, data, filter_order, frequency_cutoff):
    
    # some filter methods leave behind nans in the data, that raises a LinAlgError
    nan_locs = np.isnan(data)
            
    sampling_rate = 1/(times[1]-times[0])
    sos = signal.butter(filter_order, frequency_cutoff, btype='lowpass', output='sos', fs=sampling_rate)
    filtered = signal.sosfiltfilt(sos, data[~nan_locs])
    
    return filtered
    

def highpass_filter(times, data, filter_order, frequency_cutoff):
    
    # some filter methods leave behind nans in the data, that raises a LinAlgError
    nan_locs = np.isnan(data)
            
    sampling_rate = 1/(times[1]-times[0])
    sos = signal.butter(filter_order, frequency_cutoff, btype='highpass', output='sos', fs=sampling_rate)
    filtered = signal.sosfiltfilt(sos, data[~nan_locs])
    
    return filtered


scolor = '#d06c9eff'
lcolor = '#006680ff'
pcolor = '#1db17eff'

scolor_alt = '#e2a6c4ff'
lcolor_alt = '#6db2c4ff'
pcolor_alt = '#7ad6b7ff'

#%% Visualize one dataset

BASE_DIR = '/media/marcos/DATA/marcos/FloClock_data/gap_junctions/'

pair_type = 3 # 0: l-l, 1: l-r, 2: s-l, 3:s-s
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

#%% Quantify and see the quantification of single runs

BASE_DIR = '/media/marcos/DATA/marcos/FloClock_data/gap_junctions/'

pair_type = 2 # 0: l-l, 1: l-r, 2: s-l, 3:s-s
run_nr = 0

files = contenidos(BASE_DIR)
runs = contenidos(files[pair_type])
file = contenidos(runs[run_nr], filter_ext='.abf', sort='age')[-1]

# Load data
abf = pyabf.ABF(file)

fig, axarr = plt.subplots(abf.sweepCount*2, sharex=True, figsize=[11, 8])
fig_extra, (axx1, axx2) = plt.subplots(2, sharex=True, sharey=True, figsize=[11, 3])
# Set sweep and channel, extract tmes and data

for i in abf.sweepList[::-1]:
    abf.setSweep(sweepNumber=i, channel=0)
    times = abf.sweepX
    data = abf.sweepY
    perturbation = abf.sweepC
    
    abf.setSweep(sweepNumber=i, channel=1)
    data2 = abf.sweepY
    
    ax1 = axarr[i*2]
    ax2 = axarr[i*2+1]
    
    ax1.plot(times, data)
    ax2.plot(times, data2, c='C1')
    ax1.plot(times, perturbation+data.mean(), c='C3')
    ax1.grid()
    ax2.grid()
    
    if i==2:
        axx1.plot(times, data, c=scolor if pair_type>1 else lcolor)
        axx2.plot(times, data2, c=scolor if pair_type==3 else lcolor)
        axx1.plot(times, perturbation+data.mean(), c='C3')
    
    # quantify the perturbation
    if i!=0:
        # save the position from previous runs for the one without pert
        start, end = np.where(np.diff(perturbation)!=0)[0]
    
    for ax, d, axx in zip((ax1, ax2), (data, data2), (axx1, axx2)):
        min_inx, _ = signal.find_peaks(-lowpass_filter(times, d, 2, frequency_cutoff=8))
        minima = d[min_inx]
        ax.plot(times[min_inx], minima, '.', c='k')
        
        # separate before and during perturbation
        for pstart, pend in zip([0, start], [start, end]):
            # select the minima in the relevant range
            min_inx_pice = min_inx[ np.logical_and(min_inx>pstart, min_inx<pend) ]
            
            # find the median of the minima
            minima = d[min_inx_pice]
            min_median = np.median(minima)
            
            # keep only the minima under the median
            min_inx_pice = min_inx_pice[minima<=min_median]
            minima = d[min_inx_pice]
            
            # plot filtered minima and trend line
            ax.plot(times[min_inx_pice], minima, '.', c='r')
            baseline = minima.mean()
            ax.hlines(baseline, times[pstart], times[pend], color='k')
            
            # plot confidence interval
            baseline_err = minima.std()
            
            if i==2:
                axx.hlines(baseline, times[pstart], times[pend], color='k')
                axx.fill_between(times[pstart:pend], baseline-baseline_err, baseline+baseline_err, facecolor='0.7')

    for p1 in abf.sweepEpochs.p1s:
        ax1.axvline(times[p1], color='k', ls='--', alpha=.5)

ax = axarr[-1]
ax.set_xlabel('time [sec]')
ax.set_xlim(times.min(), times.max())

axarr[0].set_title(runs[0].parent.name)

axx1.set_xlim(0, 90)
axx1.set_ylim(-135, -15)
au.make_scalebar(axx2)


#%% Extract the values from all runs

BASE_DIR = '/media/marcos/DATA/marcos/FloClock_data/gap_junctions/'

output = {
    'file' : [],
    'dirnr': [],
    'type' : [],
    'ch1'  : [],
    'ch2'  : [],
    
    'ch1_pre_0': [],
    'ch2_pre_0': [],
    'ch1_pre_3': [],
    'ch2_pre_3': [],
    'ch1_pre_6': [],
    'ch2_pre_6': [],
    
    'ch1_dur_0': [],
    'ch2_dur_0': [],
    'ch1_dur_3': [],
    'ch2_dur_3': [],
    'ch1_dur_6': [],
    'ch2_dur_6': [],
    }

files = contenidos(BASE_DIR)

for pair_dir in files:
    runs = contenidos(pair_dir)
    
    runtype = pair_dir.name
    print('Running', runtype)
    
    # skip the runs with random for this
    if 'random' in runtype:
        continue
    
    # save cell type data
    ch1, ch2 = runtype.split('-')
    
    for run in runs:
        file = contenidos(run, filter_ext='.abf', sort='age')[-1] # the -1 is there because there's also a baseline recording in the folder
        
        output['type'].append(runtype)
        output['ch1'].append(ch1)
        output['ch2'].append(ch2)            
        output['file'].append(file.stem)
        output['dirnr'].append(run.stem)
        
        # Load data
        abf = pyabf.ABF(file)
        
        # Set sweep and channel, extract tmes and data
        
        for i in abf.sweepList[::-1]:
            abf.setSweep(sweepNumber=i, channel=0)
            times = abf.sweepX
            data = abf.sweepY
            perturbation = abf.sweepC
            
            abf.setSweep(sweepNumber=i, channel=1)
            data2 = abf.sweepY
            
            # quantify the perturbation
            if i!=0:
                # save the position from previous runs for the one without pert
                start, end = np.where(np.diff(perturbation)!=0)[0]
                
            pert_magnitude = np.abs(perturbation.min())
            
            for i, (ax, d) in enzip((ax1, ax2), (data, data2)):
                min_inx, _ = signal.find_peaks(-lowpass_filter(times, d, 2, frequency_cutoff=8))
                minima = d[min_inx]
                ax.plot(times[min_inx], minima, '.', c='k')
                
                # separate before and during perturbation
                pair = []
                for pstart, pend in zip([0, start], [start, end]):
                    # select the minima in the relevant range
                    min_inx_pice = min_inx[ np.logical_and(min_inx>pstart, min_inx<pend) ]
                    
                    # find the median of the minima
                    minima = d[min_inx_pice]
                    min_median = np.median(minima)
                    base = np.mean( minima[minima<min_median] )
                    
                    pair.append(base)
                
                output[f'ch{i+1}_pre_{int(pert_magnitude)}'].append(pair[0])
                output[f'ch{i+1}_dur_{int(pert_magnitude)}'].append(pair[1])
                # print(f'ch{i+1}_{int(pert_magnitude)}') 
    # break
                                     
output_df = pd.DataFrame(output)

#%% Analize membrane potential during

fig, (ax, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(7, 4.8), sharey=True)

lines = []

# First, cell on which the current was applied

colors_dict = {'small': scolor_alt, 'large': lcolor_alt}
handles = {}
for _, row in output_df.iterrows():
    ctype = row.ch1
    l, = ax.plot([0, 3, 6], row[[ch for ch in output_df.columns if 'ch1_dur' in ch]], 
                 ls = '-', marker='.', color=colors_dict[ctype], label=ctype)
    
    handles[ctype] = l

for i, (celltype, grouped) in enumerate(output_df.groupby('ch1')):
    ax.errorbar([0, 3, 6], 
            grouped[[ch for ch in output_df.columns if 'ch1_dur' in ch]].mean(),
            grouped[[ch for ch in output_df.columns if 'ch1_dur' in ch]].std(),
            fmt='o', label=celltype, c=f'C{i}')

ax.legend(handles = handles.values())
    
# Second, other cell

colors_dict = {'small-small': scolor_alt, 'large-large': lcolor_alt, 'small-large': pcolor_alt, }
handles = {}
for _, row in output_df.iterrows():
    ctype = row.type
    l, = ax2.plot([0, 3, 6], row[[ch for ch in output_df.columns if 'ch2_dur' in ch]], 
                 ls = '-', marker='.', color=colors_dict[ctype], label=ctype)
    
    handles[ctype] = l

for i, (celltype, grouped) in enumerate(output_df.groupby('type')):
    ax2.errorbar([0, 3, 6], 
            grouped[[ch for ch in output_df.columns if 'ch2_dur' in ch]].mean(),
            grouped[[ch for ch in output_df.columns if 'ch2_dur' in ch]].std(),
            fmt='o', label=celltype, c=f'C{i}')
    

ax2.legend(handles = handles.values())

ax.set_ylabel('membrane potential [mV]')
ax.set_xlabel('applied current [-pA]')
ax.set_xticks([0, 3, 6])

ax2.set_xlabel('applied current [-pA]')
ax2.set_xticks([0, 3, 6])

ax.set_title('Cell with applied current')
ax2.set_title('Other cell (second in the pair)')

#%% Analize membrane potential difference

fig, (ax, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(7, 4.8), sharey=True)

# First, cell on which the current was applied

colors_dict = {'small': scolor_alt, 'large': lcolor_alt}
handles = {}
for _, row in output_df.iterrows():
    ctype = row.ch1
    vals = row[[ch for ch in output_df.columns if 'ch1_dur' in ch]].values - row[[ch for ch in output_df.columns if 'ch1_pre' in ch]].values
    l, = ax.plot([0, 3, 6], vals, 
                 ls = '-', marker='.', color=colors_dict[ctype], label=ctype)
    
    handles[ctype] = l

color_order = [lcolor, scolor]
for i, (celltype, grouped) in enumerate(output_df.groupby('ch1')):
    val_pre = grouped[[ch for ch in output_df.columns if 'ch1_pre' in ch]].values
    val_dur = grouped[[ch for ch in output_df.columns if 'ch1_dur' in ch]].values
    val = np.mean(val_dur - val_pre, axis=0)
    err = np.std(val_dur - val_pre, axis=0)
    ax.errorbar([0, 3, 6], val, err, fmt='o', label=celltype, c=color_order[i])

ax.legend(handles = handles.values())
    
# Second, other cell

colors_dict = {'small-small': scolor_alt, 'large-large': lcolor_alt, 'small-large': pcolor_alt, }
handles = {}
for _, row in output_df.iterrows():
    ctype = row.type
    vals = row[[ch for ch in output_df.columns if 'ch2_dur' in ch]].values - row[[ch for ch in output_df.columns if 'ch2_pre' in ch]].values
    l, = ax2.plot([0, 3, 6], vals, 
                 ls = '-', marker='.', color=colors_dict[ctype], label=ctype)
    
    handles[ctype] = l

color_order = [lcolor, pcolor, scolor]
for i, (celltype, grouped) in enumerate(output_df.groupby('type')):
    val_pre = grouped[[ch for ch in output_df.columns if 'ch2_pre' in ch]].values
    val_dur = grouped[[ch for ch in output_df.columns if 'ch2_dur' in ch]].values
    val = np.mean(val_dur - val_pre, axis=0)
    err = np.std(val_dur - val_pre, axis=0)
    ax2.errorbar([0, 3, 6], val, err, fmt='o', label=celltype, c=color_order[i])
    

ax2.legend(handles = handles.values())

ax.set_ylabel('membrane potential difference [mV]')
ax.set_xlabel('applied current [-pA]')
ax.set_xticks([0, 3, 6])

ax2.set_xlabel('applied current [-pA]')
ax2.set_xticks([0, 3, 6])

ax.set_title('Cell with applied current')
ax2.set_title('Other cell (second in the pair)')


#%% Analize membrane potential fold change

fig, (ax, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(7, 4.8), sharey=True)

lines = []

# First, cell on which the current was applied

# all points
colors_dict = {'small': scolor_alt, 'large': lcolor_alt}
handles = {}
same_vals = {'small':[], 'large':[]} # baseline values in the cell the current was applied
for _, row in output_df.iterrows():
    ctype = row.ch1
    vals = (row[[ch for ch in output_df.columns if 'ch1_dur' in ch]].values / row[[ch for ch in output_df.columns if 'ch1_pre' in ch]].values).astype(float)
    l, = ax.plot([0, 3, 6], vals,
                 ls = '-', marker='.', color=colors_dict[ctype], label=ctype)
    
    same_vals[ctype].append(vals)
    handles[ctype] = l

# average
color_order = [lcolor, scolor]
for i, (celltype, grouped) in enumerate(output_df.groupby('ch1')):
    val_pre = grouped[[ch for ch in output_df.columns if 'ch1_pre' in ch]].values
    val_dur = grouped[[ch for ch in output_df.columns if 'ch1_dur' in ch]].values
    val = np.mean(val_dur / val_pre, axis=0)
    err = np.std(val_dur / val_pre, axis=0)
    ax.errorbar([0, 3, 6], val, err, fmt='o', label=celltype, c=color_order[i])

ax.legend(handles = handles.values())
    
# Second, other cell

# all points
colors_dict = {'small-small': scolor_alt, 'large-large': lcolor_alt, 'small-large': pcolor_alt, }
handles = {}
diff_vals = {'small-small':[], 'large-large':[], 'small-large':[]} # baseline values in the cell the current was not applied
for _, row in output_df.iterrows():
    ctype = row.type
    vals = (row[[ch for ch in output_df.columns if 'ch2_dur' in ch]].values / row[[ch for ch in output_df.columns if 'ch2_pre' in ch]].values).astype(float)
    l, = ax2.plot([0, 3, 6], vals, 
                 ls = '-', marker='.', color=colors_dict[ctype], label=ctype)
    
    diff_vals[ctype].append(vals)
    handles[ctype] = l

# average
color_order = [lcolor, pcolor, scolor]
for i, (celltype, grouped) in enumerate(output_df.groupby('type')):
    val_pre = grouped[[ch for ch in output_df.columns if 'ch2_pre' in ch]].values
    val_dur = grouped[[ch for ch in output_df.columns if 'ch2_dur' in ch]].values
    val = np.mean(val_dur / val_pre, axis=0)
    err = np.std(val_dur / val_pre, axis=0)
    ax2.errorbar([0, 3, 6], val, err, fmt='o', label=celltype, c=color_order[i])
    

ax2.legend(handles = handles.values())

ax.set_ylabel('membrane potential fold change')
ax.set_xlabel('applied current [-pA]')
ax.set_xticks([0, 3, 6])

ax2.set_xlabel('applied current [-pA]')
ax2.set_xticks([0, 3, 6])

ax.set_title('Cell with applied current')
ax2.set_title('Other cell (second in the pair)')

# ax.set_yscale('log', base=2)
# ax2.set_yscale('log', base=2)

# pairwise statistical tests
names = 'In the same cell', 'In the other cell'

print('\n Pariwise tests: p_t = students-t | p_w : mann-whitney')
print(' Global test: Kruscak-Wallis\n')
for name, vals in zip(names, (same_vals, diff_vals)):
    print(name)
    for kind, values in vals.items():
        print('\t', kind)
        values = np.asarray(values)
        
        # unpack the three data gropus with different applied currents
        current_0, current_3, current_6 = values.T
        current_values = '3pA', '6pA'
        
        for curr_val, data in zip(current_values, (current_3, current_6)):
            print(f'\t\t {curr_val}', end=' ')
            
            # get p-values
            pval_t = stats.ttest_ind(current_0, data).pvalue
            pval_w = stats.mannwhitneyu(current_0, data).pvalue
                
            print(f' :: p_t:{pval_t:.2e} \tp_w:{pval_w:.2e} \t (n={data.size})')

        res_k = stats.kruskal(current_0, current_3, current_6)
        print('\t\t global :: pval =', f'{res_k.pvalue:.3e}')

    print()

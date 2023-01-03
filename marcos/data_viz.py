#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 13:17:52 2023

@author: marcos
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils import contenidos
import analysis_utils as au
    
#%% Visualize one run

# Load data
data_dir = '/data/marcos/FloClock_data/data'
file_inx = 6

data_files = contenidos(data_dir, filter_ext='.abf')
pair_guide_file = contenidos(data_dir, filter_ext='.xlsx').pop()

# Process data a bit
data = au.load_data(data_files[file_inx], gauss_filter=True, override_raw=False)
data.process.lowpass_filter(filter_order=2, frequency_cutoff=10, keep_og=True)
data = data.process.downsample()
data.process.poly_detrend(degree=5, keep_og=True, channels='gfilt')

# Plot data

mosaic_layout = """
                ax
                by
                cz
                """
fig, ax_dict = plt.subplot_mosaic(mosaic_layout, constrained_layout=True, figsize=(9.4, 5.4))

raw, = ax_dict['a'].plot(data.times, data.ch1, label='raw data')
ax_dict['b'].plot(data.times, data.ch2)

low, = ax_dict['a'].plot(data.times, data.ch1_lpfilt, label='low pass')
ax_dict['b'].plot(data.times, data.ch2_lpfilt)

gaus, = ax_dict['a'].plot(data.times, data.ch1_gfilt, label='gaussian')
ax_dict['x'].plot(data.times, data.ch1_gfilt, 'C02')
ax_dict['b'].plot(data.times, data.ch2_gfilt)
ax_dict['y'].plot(data.times, data.ch2_gfilt, 'C02')

# ax1.plot(data.times, data.ch1_gfilt2, '--')
# ax2.plot(data.times, data.ch2_gfilt2, '--')

trend, = ax_dict['a'].plot(data.times, data.process.get_trend(1), label='trend')
ax_dict['x'].plot(data.times, data.process.get_trend(1), 'C03')
ax_dict['b'].plot(data.times, data.process.get_trend(2))
ax_dict['y'].plot(data.times, data.process.get_trend(2), 'C03')

ax_dict['c'].plot(data.times, data.ch1_gfilt_detrend, label=f'ch1 ({data.metadata.ch1})')
ax_dict['c'].plot(data.times, data.ch2_gfilt_detrend)
ax_dict['z'].plot(data.times, data.ch1_gfilt_detrend, label=f'ch2 ({data.metadata.ch2})')
ax_dict['z'].plot(data.times, data.ch2_gfilt_detrend, alpha=0.6)

fig.legend(handles=[raw, low, gaus, trend], ncol=4, loc='upper center', bbox_to_anchor=(0.5, 0.98))
ax_dict['c'].legend()


for ax in 'abc':
    ax_dict[ax].set_xlim(30, 40)
for ax in 'xyz':
    ax_dict[ax].set_xlim(data.times.min(), data.times.max())
for ax in 'abxy':
    ax_dict[ax].set_xticklabels([])

fig.suptitle(data.metadata.file.stem + '\n')
for ax in 'ax':
    ax_dict[ax].set_title('Channel 1')
for ax in 'by':
    ax_dict[ax].set_title('Channel 2')
for ax in 'cz':
    ax_dict[ax].set_title('Both channels, detrended')
    ax_dict[ax].set_xlabel('Time [seconds]')
    
#%% Plot and save all runs

# Load data
data_dir = '/data/marcos/FloClock_data/data'
save_dir = '/data/marcos/FloClock pics/Trends'

data_files = contenidos(data_dir, filter_ext='.abf')

for i, file in enumerate(data_files):
    print(f'Running {file.stem}: {i+1}/{len(data_files)}')
    
    # Process data a bit
    data = au.load_data(file, gauss_filter=True, override_raw=False)
    data.process.lowpass_filter(filter_order=2, frequency_cutoff=10, keep_og=True)
    data = data.process.downsample()
    data.process.poly_detrend(degree=5, keep_og=True, channels='gfilt')
    
    # Plot data
    
    mosaic_layout = """
                    ax
                    by
                    cz
                    """
    fig, ax_dict = plt.subplot_mosaic(mosaic_layout, constrained_layout=True, figsize=(9.4, 5.4))
    
    raw, = ax_dict['a'].plot(data.times, data.ch1, label='raw data')
    ax_dict['b'].plot(data.times, data.ch2)
    
    low, = ax_dict['a'].plot(data.times, data.ch1_lpfilt, label='low pass')
    ax_dict['b'].plot(data.times, data.ch2_lpfilt)
    
    gaus, = ax_dict['a'].plot(data.times, data.ch1_gfilt, label='gaussian')
    ax_dict['x'].plot(data.times, data.ch1_gfilt, 'C02')
    ax_dict['b'].plot(data.times, data.ch2_gfilt)
    ax_dict['y'].plot(data.times, data.ch2_gfilt, 'C02')
    
    # ax1.plot(data.times, data.ch1_gfilt2, '--')
    # ax2.plot(data.times, data.ch2_gfilt2, '--')
    
    trend, = ax_dict['a'].plot(data.times, data.process.get_trend(1), label='trend')
    ax_dict['x'].plot(data.times, data.process.get_trend(1), 'C03')
    ax_dict['b'].plot(data.times, data.process.get_trend(2))
    ax_dict['y'].plot(data.times, data.process.get_trend(2), 'C03')
    
    ax_dict['c'].plot(data.times, data.ch1_gfilt_detrend, label=f'ch1 ({data.metadata.ch1})')
    ax_dict['c'].plot(data.times, data.ch2_gfilt_detrend)
    ax_dict['z'].plot(data.times, data.ch1_gfilt_detrend, label=f'ch2 ({data.metadata.ch2})')
    ax_dict['z'].plot(data.times, data.ch2_gfilt_detrend, alpha=0.6)
    
    fig.legend(handles=[raw, low, gaus, trend], ncol=4, loc='upper center', bbox_to_anchor=(0.5, 0.98))
    ax_dict['c'].legend()
    
    
    for ax in 'abc':
        ax_dict[ax].set_xlim(30, 40)
    for ax in 'xyz':
        ax_dict[ax].set_xlim(data.times.min(), data.times.max())
    for ax in 'abxy':
        ax_dict[ax].set_xticklabels([])
    
    fig.suptitle(data.metadata.file.stem + '\n')
    for ax in 'ax':
        ax_dict[ax].set_title('Channel 1')
    for ax in 'by':
        ax_dict[ax].set_title('Channel 2')
    for ax in 'cz':
        ax_dict[ax].set_title('Both channels, detrended')
        ax_dict[ax].set_xlabel('Time [seconds]')
        
    plt.savefig(save_dir + f'/{data.metadata.file.stem}.png')
    plt.close(fig)

#%% Single Lissajous figures

# Load data
data_dir = '/data/marcos/FloClock_data/data'
file_inx = -2

data_files = contenidos(data_dir, filter_ext='.abf')
pair_guide_file = contenidos(data_dir, filter_ext='.xlsx').pop()

# Process data a bit
data = au.load_data(data_files[file_inx], gauss_filter=True, override_raw=False)
data = data.process.downsample()
data.process.poly_detrend(degree=5, keep_og=True, channels='gfilt')

plt.plot(data.ch1, data.ch2)
plt.plot(data.ch1_gfilt, data.ch2_gfilt)
plt.plot(data.ch1_gfilt_detrend, data.ch2_gfilt_detrend)

#%% All Lissajous figures
# Load data
data_dir = '/data/marcos/FloClock_data/data'
save_dir = '/data/marcos/FloClock pics/Lissajous'

data_files = contenidos(data_dir, filter_ext='.abf')
upstrokes_stat_file = Path(data_dir) / '../output' / 'upstroke_delay_stats.dat'
upstrokes_stat_file.resolve()
upstroke_stats = pd.read_csv(upstrokes_stat_file, sep=r'\s+')

for i, file in enumerate(data_files):
    print(f'Running {file.stem}: {i+1}/{len(data_files)}')
    
    # Process data a bit
    data = au.load_data(file, gauss_filter=True, override_raw=True)
    data = data.process.downsample()
    data.process.poly_detrend(degree=5, keep_og=False)
    
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_aspect('equal')
    
    ax.plot(data.ch1, data.ch2)
    
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    
    ax.plot(xlims, ylims, '--', c='0.3')
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    ax.set_xlabel(f'Channel 1 ({data.metadata.ch1})')
    ax.set_ylabel(f'Channel 2 ({data.metadata.ch2})')
    
    uss_row = upstroke_stats[upstroke_stats['#rec'] == data.metadata.file.stem]
    uss_row = next(uss_row.itertuples())
    ax.set_title(f'{data.metadata.file.stem} | upstroke lag = {uss_row.mean:.2f}±{uss_row.std:.2f}')
    
    plt.savefig(save_dir + f'/{data.metadata.file.stem}.png')
    plt.close(fig)
    
#%% Hilbert Transform

# Load data
data_dir = '/data/marcos/FloClock_data/data'
file_inx = 0

data_files = contenidos(data_dir, filter_ext='.abf')
pair_guide_file = contenidos(data_dir, filter_ext='.xlsx').pop()

# Process data a bit
data = au.load_data(data_files[file_inx], gauss_filter=True, override_raw=True)
data = data.process.downsample()
data.process.poly_detrend(degree=5)
data.process.gaussian_filter(sigma_ms=100)
data.process.calc_phase_difference()

plt.plot(data.times, data.K)
plt.axhline(data.process.phase_difference, color='k')
plt.title(data.process.phase_difference)

#%% Calculate all phase differences

# Load data
data_dir = '/data/marcos/FloClock_data/data'

data_files = contenidos(data_dir, filter_ext='.abf')
pair_guide_file = contenidos(data_dir, filter_ext='.xlsx').pop()

write_file = Path(data_dir) / '../output' / 'phase_differences.dat'
write_file.resolve()

with open(write_file, 'w') as writefile:
    
    for i, file in enumerate(data_files):
        # print(f'Running {file.stem}: {i+1}/{len(data_files)}')
        
        # Process data a bit
        data = au.load_data(file, gauss_filter=True, override_raw=True)
        data = data.process.downsample()
        data.process.poly_detrend(degree=5)
        data.process.gaussian_filter(sigma_ms=100)
        data.process.calc_phase_difference()
        
        print(file.stem, data.process.phase_difference, file=writefile)

#%% Plot phase differences vs upstroke lags

data_dir = Path('/data/marcos/FloClock_data/output')
upstroke_file = data_dir / 'upstroke_delay_stats.dat'
phase_diff_file = data_dir / 'phase_differences.dat'

upstroke_stats = pd.read_csv(upstroke_file, sep=r'\s+').sort_values('#rec').reset_index(drop=True)
phase_diff = pd.read_csv(phase_diff_file, sep=r'\s+').sort_values('#name').reset_index(drop=True)

plt.plot(upstroke_stats['mean'], phase_diff.phase_diff, 'o')

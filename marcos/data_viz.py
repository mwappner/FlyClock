#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 13:17:52 2023

@author: marcos
"""
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import interpolate


from utils import contenidos, enzip, find_point_by_value
import analysis_utils as au
    
#%% Visualize one run

# Load data
# data_dir = '/data/marcos/FloClock_data/data'
data_dir = '/data/marcos/FloClock_data/data - mecamilamina'
file_inx = 9

data_files = contenidos(data_dir, filter_ext='.abf')
pair_guide_file = contenidos(data_dir, filter_ext='.xlsx').pop()

# Process data a bit
data = au.load_data(data_files[file_inx], gauss_filter=True, override_raw=False)
data.process.lowpass_filter(filter_order=2, frequency_cutoff=10, keep_og=True)
data = data.process.downsample()
data.process.poly_detrend(degree=5, keep_og=True, channels='gfilt')
data.process.gaussian_filter(sigma_ms=100, keep_og=True, channels='gfilt')

# Plot data

mosaic_layout = """
                ax
                by
                cz
                """
fig, ax_dict = plt.subplot_mosaic(mosaic_layout, constrained_layout=True, figsize=[10.54,  6.84])

# plot raw data
raw, = ax_dict['a'].plot(data.times, data.ch1, label='raw data')
ax_dict['b'].plot(data.times, data.ch2)

# plot data filtered with lowpass filter
low, = ax_dict['a'].plot(data.times, data.ch1_lpfilt, label='low pass')
ax_dict['b'].plot(data.times, data.ch2_lpfilt)

# plot data filtered with built-in gaussian filter
gaus, = ax_dict['a'].plot(data.times, data.ch1_gfilt, label='gaussian')
ax_dict['x'].plot(data.times, data.ch1_gfilt, 'C02')
ax_dict['b'].plot(data.times, data.ch2_gfilt)
ax_dict['y'].plot(data.times, data.ch2_gfilt, 'C02')

# # plot data filtered with gaussina filter with big kernel
# gaus2, = ax_dict['a'].plot(data.times, data.ch1_gfilt_gfilt2, 'C04', label='gauss (big kernel)')
# ax_dict['b'].plot(data.times, data.ch2_gfilt_gfilt2, 'C04')

# plot trends
trend, = ax_dict['a'].plot(data.times, data.process.get_trend(1), label='trend')
ax_dict['x'].plot(data.times, data.process.get_trend(1), 'C03')
ax_dict['b'].plot(data.times, data.process.get_trend(2))
ax_dict['y'].plot(data.times, data.process.get_trend(2), 'C03')

# plot detrended data
ax_dict['c'].plot(data.times, data.ch1_gfilt_pdetrend, label=f'ch1 ({data.metadata.ch1})')
ax_dict['c'].plot(data.times, data.ch2_gfilt_pdetrend, label=f'ch2 ({data.metadata.ch2})')
ax_dict['z'].plot(data.times, data.ch1_gfilt_pdetrend)
ax_dict['z'].plot(data.times, data.ch2_gfilt_pdetrend, alpha=0.6)

# plot detrended very smoothed data
ax_dict['c'].plot(data.times, data.ch1_gfilt_gfilt2 - data.process.get_trend(1), '#89bce0')
ax_dict['c'].plot(data.times, data.ch2_gfilt_gfilt2 - data.process.get_trend(2), '#fab67a')

fig.legend(handles=[raw, low, gaus, trend], ncol=6, loc='upper center', bbox_to_anchor=(0.5, 0.97))
ax_dict['c'].legend()

# plot mec start line, if needed
if hasattr(data.metadata, 'mec_start_sec'):
    for ax in 'xyz':
        mec_line = ax_dict[ax].axvline(data.metadata.mec_start_sec, ls='--', c='k', label='mec start')
    
    fig.legend(handles=[mec_line], ncol=6, loc='upper right', bbox_to_anchor=(1, 0.97))    
    
for ax in 'abc':
    ax_dict[ax].set_xlim(30, 40)
    # ax_dict[ax].set_xlim(data.metadata.mec_start_sec, data.metadata.mec_start_sec+100)
for ax in 'xyz':
    ax_dict[ax].set_xlim(data.times.min(), data.times.max())
for ax in 'abxy':
    ax_dict[ax].set_xticklabels([])

fig.suptitle(data.metadata.file.stem + '\n\n')
for ax in 'ax':
    ax_dict[ax].set_title('Channel 1')
for ax in 'by':
    ax_dict[ax].set_title('Channel 2')
for ax in 'cz':
    ax_dict[ax].set_title('Both channels, detrended')
    ax_dict[ax].set_xlabel('Time [seconds]')
    
#%% Plot and save all runs

# Load data
data_dir = '/data/marcos/FloClock_data/data - mecamilamina'
save_dir = '/data/marcos/FloClock pics/mecamilamina/Trends'

data_files = contenidos(data_dir, filter_ext='.abf')

for i, file in enumerate(data_files):
    print(f'Running {file.stem}: {i+1}/{len(data_files)}')
    
    # Process data a bit
    data = au.load_data(file, gauss_filter=True, override_raw=False)
    data.process.lowpass_filter(filter_order=2, frequency_cutoff=10, keep_og=True)
    data = data.process.downsample()
    data.process.poly_detrend(degree=5, keep_og=True, channels='gfilt')
    data.process.gaussian_filter(sigma_ms=100, keep_og=True, channels='gfilt')

    # Plot data

    mosaic_layout = """
                    ax
                    by
                    cz
                    """
    fig, ax_dict = plt.subplot_mosaic(mosaic_layout, constrained_layout=True, figsize=[10.54,  6.84])

    # plot raw data
    raw, = ax_dict['a'].plot(data.times, data.ch1, label='raw data')
    ax_dict['b'].plot(data.times, data.ch2)

    # plot data filtered with lowpass filter
    low, = ax_dict['a'].plot(data.times, data.ch1_lpfilt, label='low pass')
    ax_dict['b'].plot(data.times, data.ch2_lpfilt)

    # plot data filtered with built-in gaussian filter
    gaus, = ax_dict['a'].plot(data.times, data.ch1_gfilt, label='gaussian')
    ax_dict['x'].plot(data.times, data.ch1_gfilt, 'C02')
    ax_dict['b'].plot(data.times, data.ch2_gfilt)
    ax_dict['y'].plot(data.times, data.ch2_gfilt, 'C02')

    # # plot data filtered with gaussina filter with big kernel
    # gaus2, = ax_dict['a'].plot(data.times, data.ch1_gfilt_gfilt2, 'C04', label='gauss (big kernel)')
    # ax_dict['b'].plot(data.times, data.ch2_gfilt_gfilt2, 'C04')

    # plot trends
    trend, = ax_dict['a'].plot(data.times, data.process.get_trend(1), label='trend')
    ax_dict['x'].plot(data.times, data.process.get_trend(1), 'C03')
    ax_dict['b'].plot(data.times, data.process.get_trend(2))
    ax_dict['y'].plot(data.times, data.process.get_trend(2), 'C03')
    
    # plot detrended data
    ax_dict['c'].plot(data.times, data.ch1_gfilt_pdetrend, label=f'ch1 ({data.metadata.ch1})')
    ax_dict['c'].plot(data.times, data.ch2_gfilt_pdetrend, label=f'ch2 ({data.metadata.ch2})')
    ax_dict['z'].plot(data.times, data.ch1_gfilt_pdetrend)
    ax_dict['z'].plot(data.times, data.ch2_gfilt_pdetrend, alpha=0.6)
    
    # plot detrended very smoothed data
    ax_dict['c'].plot(data.times, data.ch1_gfilt_gfilt2 - data.process.get_trend(1), '#89bce0')
    ax_dict['c'].plot(data.times, data.ch2_gfilt_gfilt2 - data.process.get_trend(2), '#fab67a')
    
    fig.legend(handles=[raw, low, gaus, trend], ncol=6, loc='upper center', bbox_to_anchor=(0.5, 0.97))
    ax_dict['c'].legend()
    
    # plot mec start line, if needed
    if hasattr(data.metadata, 'mec_start_sec'):
        for ax in 'xyz':
            mec_line = ax_dict[ax].axvline(data.metadata.mec_start_sec, ls='--', c='k', label='mec start')
        
        fig.legend(handles=[mec_line], ncol=6, loc='upper right', bbox_to_anchor=(1, 0.97))    


    for ax in 'abc':
        ax_dict[ax].set_xlim(30, 40)
    for ax in 'xyz':
        ax_dict[ax].set_xlim(data.times.min(), data.times.max())
    for ax in 'abxy':
        ax_dict[ax].set_xticklabels([])

    fig.suptitle(data.metadata.file.stem + '\n\n')
    for ax in 'ax':
        ax_dict[ax].set_title('Channel 1')
    for ax in 'by':
        ax_dict[ax].set_title('Channel 2')
    for ax in 'cz':
        ax_dict[ax].set_title('Both channels, detrended')
        ax_dict[ax].set_xlabel('Time [seconds]')
        
    plt.savefig(save_dir + f'/{data.metadata.file.stem}.png')
    plt.close(fig)
   
    
#%% Plot trendlines

# Load data
data_dir = '/data/marcos/FloClock_data/data - mecamilamina'
save_dir = '/data/marcos/FloClock pics/mecamilamina/Trends'
out_dir = '/data/marcos/FloClock_data/data - mecamilamina/output/polynomial_trends'

file_inx = 0
plot_all = False

data_files = contenidos(data_dir, filter_ext='.abf')
pair_guide_file = contenidos(data_dir, filter_ext='.xlsx').pop()

iter_over = data_files if plot_all else (data_files[file_inx], )
for i, file in enumerate(iter_over):
    print(f'Running {file.stem}: {i+1}/{len(data_files)}')
    
    # Process data a bit
    data = au.load_data(file, gauss_filter=True, override_raw=False)
    data.process.lowpass_filter(filter_order=2, frequency_cutoff=10, keep_og=True)
    data = data.process.downsample()
    data.process.poly_detrend(degree=5, keep_og=True, channels='gfilt')
    data.process.gaussian_filter(sigma_ms=100, keep_og=True, channels='gfilt')

    # Plot data

    # fig, axarr = plt.subplots(2, 1, constrained_layout=True, figsize=[10.54,  6.84], sharex=True)
    # ax1, ax2 = axarr
    
    fig = plt.figure(figsize=[15,  6.5], tight_layout=True)
    gs = fig.add_gridspec(2,2, height_ratios=[1, 4])
    
    ax1, ax2 = [fig.add_subplot(gs[0, i]) for i in range(2)]
    
    gs2 = gs[1, :].subgridspec(2,1)
    ax = fig.add_subplot(gs2[0])
    ax_trend = fig.add_subplot(gs2[1], sharex=ax)
    
    axes = [ax1, ax2, ax, ax_trend]
    
    # plot data and trendlines
    ax1.plot(data.times, data.ch1_gfilt_gfilt2 )
    ax1.plot(data.times, data.process.get_trend(1), c='C03')
    ax2.plot(data.times, data.ch2_gfilt_gfilt2 , c='C01')
    ax2.plot(data.times, data.process.get_trend(2), c='C03')
    
    # plot detrended very smoothed data
    ax.plot(data.times, data.ch1_gfilt_gfilt2 - data.process.get_trend(1))
    ax.plot(data.times, data.ch2_gfilt_gfilt2 - data.process.get_trend(2))
    
    # plot trends
    ch1_line, = ax_trend.plot(data.times, data.process.get_trend(1), label=f'ch1 ({data.metadata.ch1})')
    ch2_line, = ax_trend.plot(data.times, data.process.get_trend(2), label=f'ch2 ({data.metadata.ch2})')
    
    
    # Plot mec start line, if needed
    if hasattr(data.metadata, 'mec_start_sec'):
        for axi in axes:
            mec_line = axi.axvline(data.metadata.mec_start_sec, ls='--', c='k', label='mec start')
        
        fig.legend(handles=[mec_line], loc='upper right', bbox_to_anchor=(.95, 0.97))

    # Format plots
    for axi in axes:
        axi.set_xlim(data.times.min(), data.times.max())

    fig.suptitle(data.metadata.file.stem + '\n\n')
    fig.legend(handles=[ch1_line, ch2_line], ncol=2, loc='upper left', bbox_to_anchor=(.05, 0.97))

    ax_trend.set_xlabel('Time [seconds]')
    ax_trend.set_title('Trendline')
    ax.set_title('Detrended data')
    
    # ax2.legend(loc='upper right')
        
    # Save and close
    if plot_all:
        plt.savefig(save_dir + f'/{data.metadata.file.stem}.png')
        plt.close(fig)
        
        for ch in (1,2):
            with open(out_dir + f'/{data.metadata.file.stem}_ch{ch}.pickle', 'wb') as f:
                poly = data.process.get_trend_poly(ch)
                pickle.dump(poly, f)
    
#%% Analize trendlines

data_dir = '/data/marcos/FloClock_data/data - mecamilamina'
poly_dir = Path(data_dir) / 'output' / 'polynomial_trends'

data_files = contenidos(poly_dir, filter_ext='.pickle')
pair_guide_file = contenidos(data_dir, filter_ext='.xlsx').pop()

fig, ax = plt.subplots()

# load trends
polys = []
for f in data_files:
    with open(f, 'rb') as f:
        poly = pickle.load(f)
        polys.append(poly)
    

# pair guide
pair_guide = pd.read_excel(pair_guide_file).sort_values('name', ignore_index=True)

# construct a daraframe containing the duration of each run
durations = pd.concat([
    pair_guide.loc[:, ['ch1', 'duration(min)', 'name']].rename(columns={'ch1':'ch'}), 
    pair_guide.loc[:, ['ch2', 'duration(min)', 'name']].rename(columns={'ch2':'ch'})
    ]).sort_values('name', ignore_index=True)
    #]).sort_values(['ch', 'name'], ascending=(True, True), ignore_index=True)
durations['duration(sec)'] = durations['duration(min)'] * 60

interpolators = []
for i, (poly, dur, name) in enzip(polys, durations['duration(sec)'], durations.name):
    # if i != 0:
    #     continue
    
    mec_start = pair_guide[pair_guide.name==name]['mec_start(sec)'].values[0]

    t = np.arange(0, dur, 0.01)
    trend = poly(t) / np.abs(poly(mec_start))
    ax.plot(t-mec_start, trend)
    
    interpolators.append(
        interpolate.interp1d(t-mec_start, trend, bounds_error=False, fill_value=np.nan)
        )

time_range = -350, 2000
average_times = np.linspace(*time_range, 1000)
average = np.nanmean( [interp(average_times) for interp in interpolators], axis=0 )

ax.plot(average_times, average, 'k', lw=2)


#%% Single Lissajous figures

# Load data
# data_dir = '/data/marcos/FloClock_data/data'
data_dir = '/home/user/Documents/Doctorado/Fly clock/FlyClock_data/data/'
file_inx = 0

data_files = contenidos(data_dir, filter_ext='.abf')
pair_guide_file = contenidos(data_dir, filter_ext='.xlsx').pop()

# Process data a bit
data = au.load_data(data_files[file_inx], gauss_filter=True, override_raw=False)
data = data.process.downsample()
data.process.poly_detrend(degree=5, keep_og=True, channels='gfilt')

# Find direction of the blob
P = np.polynomial.Polynomial
nanlocs = np.isnan(data.ch1_gfilt_pdetrend) | np.isnan(data.ch2_gfilt_pdetrend)
fit_poly = P.fit(data.ch1_gfilt_pdetrend[~nanlocs], data.ch2_gfilt_pdetrend[~nanlocs], deg=1)
slope = fit_poly.convert().coef[1]

# PLot data
plt.plot(data.ch1, data.ch2)
plt.plot(data.ch1_gfilt, data.ch2_gfilt)
plt.plot(data.ch1_gfilt_pdetrend, data.ch2_gfilt_pdetrend)

plt.plot(data.ch1_gfilt_pdetrend, fit_poly(data.ch1_gfilt_pdetrend))

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
file_inx = 2

data_files = contenidos(data_dir, filter_ext='.abf')
pair_guide_file = contenidos(data_dir, filter_ext='.xlsx').pop()

# Process data a bit
data = au.load_data(data_files[file_inx], gauss_filter=True, override_raw=True)
data = data.process.downsample()
data.process.poly_detrend(degree=5)
data.process.gaussian_filter(sigma_ms=100)
data.process.calc_phase_difference()
data.process.magnitude_detrend(keep_og=True)
data.process.calc_phase_difference(channels='mdetrend')

plt.figure(figsize=(18, 5))

ax1 = plt.subplot(2,2,1)
plt.plot(data.times, data.ch1, label='data')
plt.plot(data.times, data.ch1_magnitude, label='envelope')
plt.plot(data.times, data.ch1_mdetrend, label='data/envelope')
plt.title('Channel 1')
plt.legend()

plt.subplot(2,2,3, sharex=ax1)
plt.plot(data.times, data.ch2)
plt.plot(data.times, data.ch2_magnitude)
plt.plot(data.times, data.ch2_mdetrend)
plt.title('Channel 2')
plt.xlim(data.times.min(), data.times.max())

plt.subplot(2,2,2, sharex=ax1)
plt.plot(data.times, data.K)
plt.axhline(data.process.get_phase_difference(channels=''), color='k')
plt.title(f"Phase diff for raw channel = {data.process.get_phase_difference(channels=''):f}")
plt.xlim(data.times.min(), data.times.max())
plt.ylim(-1, 1)

plt.subplot(2,2,4, sharex=ax1)
plt.plot(data.times, data.K)
plt.axhline(data.process.get_phase_difference(channels='mdetrend'), color='k')
plt.title(f"Phase diff for mdetrend = {data.process.get_phase_difference(channels='mdetrend'):f}")
plt.xlim(data.times.min(), data.times.max())
plt.ylim(-1, 1)

plt.suptitle(data.metadata.file.stem)

plt.tight_layout()
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

#%% Findpeaks tests

from scipy import signal
from skimage import filters

from utils import sort_by

# Load data
# data_dir = '/data/marcos/FloClock_data/data'
data_dir = '/data/marcos/FloClock_data/data - mecamilamina'
file_inx = 0
ch_inx = 2 # channel index

data_files = contenidos(data_dir, filter_ext='.abf')

# Process data a bit
data = au.load_data(data_files[file_inx], gauss_filter=True, override_raw=False)
data = data.process.downsample()
data.process.poly_detrend(degree=5, channels='gfilt')
first_trend = data.process.get_trend(ch_inx)
data.process.gaussian_filter(sigma_ms=100, keep_og=True, channels='gfilt')
data.process.poly_detrend(degree=15, channels='gfilt_gfilt2')

fallback_threshold = 4
ch = data[f'ch{ch_inx}_gfilt_gfilt2']
times = data.times

# first pass, with threshold 0mv
p_inx, _ = signal.find_peaks(ch)#, height=0)
peaks = ch.values[p_inx]

# second pass, with threshold given by otsu
threshold = filters.threshold_otsu(peaks)
# we will only accept otsu's threshold if we cna detect two peaks in the 
# distribution and the threshold falls between them
counts, bins = np.histogram(peaks, bins='auto')
bin_centers = bins[:-1] + np.diff(bins) / 2
maxima, _ = signal.find_peaks(counts)
# make sure the other peak is not the first point of the distribution
if all(counts[0]>c for c in counts[1:3]):
    maxima = np.array( (0, *maxima) )

if len(maxima) < 2:
    # if only one maximum was detected, we fallback
    threshold = fallback_threshold
else:
    if len(maxima) > 2:
        # if too many maxima were found, keep the largest two
        maxima = sort_by(maxima, counts[maxima])[-2:]
        maxima.sort()    

    # if at least two maxima were found, accept threshold only if it lies between them
    if not( bin_centers[maxima[0]] < threshold < bin_centers[maxima[1]]):
        threshold = fallback_threshold   

# p_inx, _ = signal.find_peaks(ch, height=threshold)
peaks = ch.values[p_inx]
t_peaks = times.values[p_inx]

# third pass, with minimum distance between peaks
counts, bins = np.histogram(np.diff(t_peaks))
bin_centers = bins[:-1] + np.diff(bins) / 2
period_mode = bin_centers[ np.argmax(counts) ]
distance_points = int(period_mode * 0.3 / (times[1] - times[0]))
# p_inx, _ = signal.find_peaks(ch, height=threshold, distance=distance_points)
p_inx, props = signal.find_peaks(ch, prominence=1)#, distance=distance_points)
        

# Plot data and peaks
fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True, sharex=True)

trend = data.process.get_trend(ch_inx)
ax.plot(data.times, (data[f'ch{ch_inx}'] - first_trend - trend), color='0.8')
# ax.plot(data.times, data[f'ch{ch_inx}_gfilt'] - trend)
ax.plot(data.times, data[f'ch{ch_inx}_gfilt_gfilt2'])

peak_pos = times[p_inx]
peak_val = ch[p_inx]
ax.plot(peak_pos, peak_val, 'o')
ax.set_xlim(data.times.values[0], data.times.values[-1])

ax.plot(times[props['left_bases']], ch[props['left_bases']], 'ro', mfc='none')
ax.plot(times[props['right_bases']], ch[props['right_bases']], 'bx')
for pp, pv, prominence in zip(peak_pos, peak_val, props['prominences']):
    ax.text(pp, pv, f'{prominence:.1f}', horizontalalignment='center', verticalalignment='bottom')

#%% Plot peaks of a run

# Load data
# data_dir = '/data/marcos/FloClock_data/data'
data_dir = '/data/marcos/FloClock_data/data - mecamilamina'
file_inx = 0

data_files = contenidos(data_dir, filter_ext='.abf')

# Process data a bit
data = au.load_data(data_files[file_inx], gauss_filter=True, override_raw=False)
data = data.process.downsample()
data.process.poly_detrend(degree=5, channels='gfilt')
data.process.gaussian_filter(sigma_ms=100, keep_og=True, channels='gfilt')

data.process.find_peaks(channels='gfilt_gfilt2', period_percent=0.4, prominence=3)

# Plot data and peaks

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True, sharex=True)

for ax, ch in zip((ax1, ax2), (1, 2)):
    ax.plot(data.times, data[f'ch{ch}'] - data.process.get_trend(ch), color='0.6')
    ax.plot(data.times, data[f'ch{ch}_gfilt'])
    ax.plot(data.times, data[f'ch{ch}_gfilt_gfilt2'])
    
    peak_pos = data.process.get_peak_pos(ch)
    peak_val = data.process.get_peak_values(ch)
    ax.plot(peak_pos, peak_val, 'o')
    ax.set_xlim(data.times.values[0], data.times.values[-1])

    if hasattr(data.metadata, 'mec_start_sec'):
        mec_line = ax.axvline(data.metadata.mec_start_sec, ls='--', c='k', label='mec start')

fig.suptitle(data.metadata.file.stem)
    

""" Some missed peaks:
    LS13 (21): ch1 around t=165
               both around t=132 
    LS15 (23): ch2 around t:120 (multiple)
"""

print('Running', data.metadata.file.stem)

#%% Plot and save all peaks of a run

# Load data
data_dir = '/data/marcos/FloClock_data/data - mecamilamina'
savedir = Path('/data/marcos/FloClock pics/mecamilamina/Peaks')
periods_per_panel = 12
panel_height = 1.5

data_files = contenidos(data_dir, filter_ext='.abf')

for i, file in enumerate(data_files):
    print(f'Running {file.stem}: {i+1}/{len(data_files)}')
    
    # Process data a bit
    data = au.load_data(file, gauss_filter=True, override_raw=False)
    data = data.process.downsample()
    data.process.poly_detrend(degree=5, channels='gfilt')
    data.process.gaussian_filter(sigma_ms=100, keep_og=True, channels='gfilt')
    
    data.process.find_peaks(channels='gfilt_gfilt2')
    
    # Plot data and peaks
    
    avg_period = data.process.get_avg_period()
    if np.isnan(avg_period):
        avg_period = 3 # an estiamte
    points_per_panel = int(avg_period * data.metadata.sampling_rate * periods_per_panel)
    period_count = data.times.values[-1] / avg_period
    panel_count = np.ceil(period_count / periods_per_panel).astype(int)
    
    fig = plt.figure(figsize=(12, panel_height * panel_count * 2))
    gs = fig.add_gridspec(panel_count, 1)
    
    for panel_number in range(panel_count):
        
        sub_gs = gs[panel_number].subgridspec(2,1, hspace=0)
        
        ax1 = fig.add_subplot(sub_gs[0])
        ax2 = fig.add_subplot(sub_gs[1])
        ax1.set_xticks([])
        
        panel_point_range = panel_number * points_per_panel, (panel_number+1) * points_per_panel
        plot_slice = slice(*panel_point_range)
        panel_time_interval = [ppr / data.metadata.sampling_rate for ppr in panel_point_range]
        
        # for each channel, plot everything
        for ax, ch in zip((ax1, ax2), (1, 2)):
            ax.plot(data.times[plot_slice], 
                    (data[f'ch{ch}'] - data.process.get_trend(ch))[plot_slice], color='0.6')
            ax.plot(data.times[plot_slice], data[f'ch{ch}_gfilt'][plot_slice])
            ax.plot(data.times[plot_slice], data[f'ch{ch}_gfilt_gfilt2'][plot_slice])
            
            # mark the peaks with dots
            peak_pos = data.process.get_peak_pos(ch)
            peak_val = data.process.get_peak_values(ch)
            ax.plot(peak_pos, peak_val, 'o')
            ax.set_xlim(panel_time_interval)
            
            # plot mec line, if needed
            if hasattr(data.metadata, 'mec_start_sec'):
                mec_time = data.metadata.mec_start_sec
                panel_time = data.times[plot_slice]
                if panel_time.min() <= mec_time <= panel_time.max():
                    ax.axvline(mec_time, ls='--', c='k')
            
    plt.tight_layout()
        
    plt.savefig(savedir / 'pngs' / f'{data.metadata.file.stem}.png')
    plt.savefig(savedir / 'pdfs' / f'{data.metadata.file.stem}.pdf')
    plt.close()

#%% Time dependent period


# Load data
data_dir = '/data/marcos/FloClock_data/data'
# data_dir = '/data/marcos/FloClock_data/data - mecamilamina'
file_inx = 10
outlier_mode_proportion = 1.8 #1.8 for normal runs

data_files = contenidos(data_dir, filter_ext='.abf')

# Process data a bit
data = au.load_data(data_files[file_inx], gauss_filter=True, override_raw=False)
data = data.process.downsample()
data.process.poly_detrend(degree=5, channels='gfilt')
data.process.gaussian_filter(sigma_ms=100, keep_og=True, channels='gfilt')

data.process.find_peaks(channels='gfilt_gfilt2', period_percent=0.4, prominence=3)

# Plot data and peaks

fig, (ax1, ax3, ax2, ax4) = plt.subplots(4, 1, figsize=(12, 8), constrained_layout=True, sharex=True, height_ratios=[2,1,2,1])

modes = []
trends = []
P = np.polynomial.Polynomial
for ax, ax_p, ch in zip((ax1, ax2), (ax3, ax4), (1, 2)):
    ax.plot(data.times, data[f'ch{ch}'] - data.process.get_trend(ch), color='0.6')
    ax.plot(data.times, data[f'ch{ch}_gfilt'])
    ax.plot(data.times, data[f'ch{ch}_gfilt_gfilt2'])

    # plot timeseries    
    peak_pos = data.process.get_peak_pos(ch)
    peak_val = data.process.get_peak_values(ch)
    ax.plot(peak_pos, peak_val, 'o')
    ax.set_xlim(data.times.values[0], data.times.values[-1])

    # plot periods
    period_times, periods = data.process.get_periods(ch)
    counts, bins = np.histogram(periods)
    bin_centers = bins[:-1] + np.diff(bins) / 2
    period_mode = bin_centers[np.argmax(counts)]
    ax_p.plot(period_times, periods, 'o', color='C04')
    ax_p.plot(period_times[periods > outlier_mode_proportion*period_mode], periods[periods > outlier_mode_proportion*period_mode], 'ro')
    
    # plot mode, mean and trend line
    valid_period_inxs = periods < outlier_mode_proportion*period_mode
    valid_periods = periods[valid_period_inxs]
    period_mean = np.mean(valid_periods)
    modeline = ax_p.axhline(period_mode, color='0.3', linestyle='--', zorder=1)
    meanline = ax_p.axhline(period_mean, color='0.3', linestyle=':', zorder=1)
    
    trend_poly = P.fit(period_times[valid_period_inxs], valid_periods, 1)
    trendline, = ax_p.plot(data.times, trend_poly(data.times), 'C00', zorder=0.9)
    trends.append(trend_poly.convert().coef[1])
    
    modes.append(period_mode)

if hasattr(data.metadata, 'mec_start_sec'):
    for ax in (ax1, ax3, ax2, ax4):
        mec_line = ax.axvline(data.metadata.mec_start_sec, ls='--', c='k', label='mec start')
    
    fig.legend(handles=[mec_line], ncol=6, loc='center right', bbox_to_anchor=(1, 0.97))    

fig.suptitle(data.metadata.file.stem)
ax4.set_xlabel('time (s)')

ax1.set_ylabel('mV')
ax2.set_ylabel('mV')
ax3.set_ylabel('period (s)')
ax4.set_ylabel('period (s)')

ax1.set_title('Channel 1')
ax2.set_title('Channel 2')
ax3.set_title(f'Channel 1 period | mode = {modes[0]:.2f} sec')
ax4.set_title(f'Channel 2 period | mode = {modes[1]:.2f} sec')

ax3.legend(handles=[modeline, meanline, trendline], loc='upper left',
           labels=['mode', 'mean (no outliers)', f'slope={trends[0]*1000:.1f}e3'])
ax4.legend(handles=[trendline], loc='upper left', labels=[f'slope={trends[1]*1000:.1f}e3'])
    
print('Running', data.metadata.file.stem)

#%% Plot and save all time dependent periods

# Load data
data_dir = '/data/marcos/FloClock_data/data'
savedir = Path('/data/marcos/FloClock pics/Periods')
outputdir = Path('/data/marcos/FloClock_data/output')

data_files = contenidos(data_dir, filter_ext='.abf')

data_files = contenidos(data_dir, filter_ext='.abf')

with open(outputdir / 'period_info2.csv', 'w') as writefile:
    run_stats = 'mean, std, valid_mean, valid_std, mode, slope'
    writefile.write('name,')
    writefile.write(','.join([f'ch{ch}_{what}' for ch in [1,2] for what in run_stats.split(', ')]))
    writefile.write('\n')

for i, file in enumerate(data_files):
    print(f'Running {file.stem}: {i+1}/{len(data_files)}')

    # Process data a bit
    data = au.load_data(file, gauss_filter=True, override_raw=False)
    data = data.process.downsample()
    data.process.poly_detrend(degree=5, channels='gfilt')
    data.process.gaussian_filter(sigma_ms=100, keep_og=True, channels='gfilt')
    
    data.process.find_peaks(channels='gfilt_gfilt2', period_percent=0.4, prominence=3)
    
    # Plot data and peaks
    
    fig, (ax1, ax3, ax2, ax4) = plt.subplots(4, 1, figsize=(12, 8), constrained_layout=True, sharex=True, height_ratios=[2,1,2,1])
    
    modes = []
    trends = []
    run_stats = []
    P = np.polynomial.Polynomial
    for ax, ax_p, ch in zip((ax1, ax2), (ax3, ax4), (1, 2)):
        ax.plot(data.times, data[f'ch{ch}'] - data.process.get_trend(ch), color='0.6')
        ax.plot(data.times, data[f'ch{ch}_gfilt'])
        ax.plot(data.times, data[f'ch{ch}_gfilt_gfilt2'])
    
        # plot timeseries    
        peak_pos = data.process.get_peak_pos(ch)
        peak_val = data.process.get_peak_values(ch)
        ax.plot(peak_pos, peak_val, 'o')
        ax.set_xlim(data.times.values[0], data.times.values[-1])
    
        # plot periods
        period_times, periods = data.process.get_periods(ch)
        counts, bins = np.histogram(periods)
        bin_centers = bins[:-1] + np.diff(bins) / 2
        period_mode = bin_centers[np.argmax(counts)]
        ax_p.plot(period_times, periods, 'o', color='C04')
        ax_p.plot(period_times[periods>1.8*period_mode], periods[periods>1.8*period_mode], 'ro')
        
        # plot mode, mean and trend line
        valid_period_inxs = periods<1.8*period_mode
        valid_periods = periods[valid_period_inxs]
        period_mean = np.mean(valid_periods)
        modeline = ax_p.axhline(period_mode, color='0.3', linestyle='--', zorder=1)
        meanline = ax_p.axhline(period_mean, color='0.3', linestyle=':', zorder=1)
        
        trend_poly = P.fit(period_times[valid_period_inxs], valid_periods, 1)
        trendline, = ax_p.plot(data.times, trend_poly(data.times), 'C00', zorder=0.9)
        trends.append(trend_poly.convert().coef[1])
    
        modes.append(period_mode)
        run_stats += [np.mean(periods), np.std(periods),
                    period_mean, np.std(valid_periods),
                    period_mode, trends[-1]]
    
    fig.suptitle(data.metadata.file.stem)
    ax4.set_xlabel('time (s)')
    
    ax1.set_ylabel('mV')
    ax2.set_ylabel('mV')
    ax3.set_ylabel('period (s)')
    ax4.set_ylabel('period (s)')
    
    ax1.set_title('Channel 1')
    ax2.set_title('Channel 2')
    ax3.set_title(f'Channel 1 period | mode = {modes[0]:.2f} sec')
    ax4.set_title(f'Channel 2 period | mode = {modes[1]:.2f} sec')
    
    ax3.legend(handles=[modeline, meanline, trendline], loc='upper left',
               labels=['mode', 'mean (no outliers)', f'slope={trends[0]*1000:.1f}e3'])
    ax4.legend(handles=[trendline], loc='upper left', labels=[f'slope={trends[1]*1000:.1f}e3'])
    
    with open(outputdir / 'period_info.csv', 'a') as writefile:
        writefile.write(data.metadata.file.stem + ',')
        writefile.write(','.join(map(str, run_stats)))
        writefile.write('\n')
    
    plt.savefig(savedir / f'{data.metadata.file.stem}.png')
    plt.close()


#%% Rolling average detrending

# Load data
# data_dir = '/data/marcos/FloClock_data/data'
# data_dir = '/home/user/Documents/Doctorado/Fly clock/FlyClock_data/data/'
data_dir = '/data/marcos/FloClock_data/data - mecamilamina'
file_inx = 10
outlier_mode_proportion = 5 # 1.8 for normla runs

data_files = contenidos(data_dir, filter_ext='.abf')
pair_guide_file = contenidos(data_dir, filter_ext='.xlsx').pop()

# Process data a bit
data = au.load_data(data_files[file_inx], gauss_filter=True, override_raw=True)
data = data.process.downsample()
data.process.poly_detrend(degree=5)
data.process.gaussian_filter(sigma_ms=100)
data.process.find_peaks(period_percent=0.6, prominence=5)
data.process.average_detrend(outlier_mode_proportion=1.8, keep_og=True)

plt.figure(figsize=(18, 5))

ax1 = plt.subplot(3,1,1)
plt.plot(data.times, data.ch1, 'C00')
plt.plot(data.times, data.ch1_average, 'C03')
plt.title('Channel 1')
# plt.gca().set_xticklabels([])

plt.subplot(3,1,2, sharex=ax1)
plt.plot(data.times, data.ch2, 'C01')
plt.plot(data.times, data.ch2_average, 'C03')
plt.title('Channel 2')
# plt.gca().set_xticklabels([])

plt.subplot(3,1,3, sharex=ax1)
plt.plot(data.times, data.ch1_adetrend)
plt.plot(data.times, data.ch2_adetrend)
plt.title('Detrended')
plt.xlabel('time (s)')

plt.xlim(data.times.min(), data.times.max())

plt.tight_layout()

if hasattr(data.metadata, 'mec_start_sec'):
    for ax in plt.gcf().get_axes():
        mec_line = ax.axvline(data.metadata.mec_start_sec, ls='--', c='k', label='mec start')
    
    plt.gcf().legend(handles=[mec_line], ncol=6, loc='center right', bbox_to_anchor=(1, 0.97))    

#%% Plot and save all rolling average trend lines

# Load data
data_dir = '/data/marcos/FloClock_data/data'
savedir = Path('/data/marcos/FloClock pics/Rolling average')
ch = 1
outlier_mode_proportion = 1.8

data_files = contenidos(data_dir, filter_ext='.abf')
pair_guide_file = contenidos(data_dir, filter_ext='.xlsx').pop()

for i, file in enumerate(data_files):
    print(f'Running {file.stem}: {i+1}/{len(data_files)}')
    
    # Process data a bit
    data = au.load_data(file, gauss_filter=True, override_raw=True)
    data = data.process.downsample()
    data.process.poly_detrend(degree=5)
    data.process.gaussian_filter(sigma_ms=100)
    data.process.find_peaks(period_percent=0.6, prominence=5)
    data.process.average_detrend(outlier_mode_proportion=1.8, keep_og=True)
    
    plt.figure(figsize=(18, 5))
    
    ax1 = plt.subplot(3,1,1)
    plt.plot(data.times, data.ch1, 'C00')
    plt.plot(data.times, data.ch1_average, 'C03')
    plt.title('Channel 1')
    # plt.gca().set_xticklabels([])
    
    plt.subplot(3,1,2, sharex=ax1)
    plt.plot(data.times, data.ch2, 'C01')
    plt.plot(data.times, data.ch2_average, 'C03')
    plt.title('Channel 2')
    # plt.gca().set_xticklabels([])
    
    plt.subplot(3,1,3, sharex=ax1)
    plt.plot(data.times, data.ch1_adetrend)
    plt.plot(data.times, data.ch2_adetrend)
    plt.title('Detrended')
    plt.xlabel('time (s)')
    
    plt.xlim(data.times.min(), data.times.max())
    
    plt.tight_layout()

    plt.savefig(savedir / f'{data.metadata.file.stem}.png')
    plt.close()


#%% All Lissajous figures with detrended data
# Load data
data_dir = '/data/marcos/FloClock_data/data'
save_dir = '/data/marcos/FloClock pics/Lissajous'
# data_dir = '/home/user/Documents/Doctorado/Fly clock/FlyClock_data/data/'
# save_dir = '/home/user/Documents/Doctorado/Fly clock/FlyClock pics/Lissajous detrended'

data_files = contenidos(data_dir, filter_ext='.abf')
upstrokes_stat_file = Path(data_dir) / '../output' / 'upstroke_delay_stats.dat'
upstrokes_stat_file.resolve()
upstroke_stats = pd.read_csv(upstrokes_stat_file, sep=r'\s+')

for i, file in enumerate(data_files):
    print(f'Running {file.stem}: {i+1}/{len(data_files)}')
        
    # Process data a bit
    data = au.load_data(file, gauss_filter=True, override_raw=False)
    data = data.process.downsample()
    data.process.poly_detrend(degree=5, channels='gfilt')
    data.process.gaussian_filter(sigma_ms=100, channels='gfilt')
    data.process.find_peaks(period_percent=0.6, prominence=5, channels='gfilt')
    data.process.average_detrend(outlier_mode_proportion=1.8, channels='gfilt')
    
    # Find direction of the blob
    P = np.polynomial.Polynomial
    nanlocs = np.isnan(data.ch1_gfilt) | np.isnan(data.ch2_gfilt)
    fit_poly = P.fit(data.ch1_gfilt[~nanlocs], data.ch2_gfilt[~nanlocs], deg=1)
    slope = fit_poly.convert().coef[1]

    # Plot data
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_aspect('equal')
    
    # Plot Lissajous
    ax.plot(data.ch1 - data.process.get_trend(1) - data.ch1_gfilt_average, 
            data.ch2 - data.process.get_trend(2) - data.ch2_gfilt_average,
            color='0.8')
    ax.plot(data.ch1_gfilt, data.ch2_gfilt)
    
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    
    # Plot blob trend line
    ax.plot(data.ch1_gfilt, fit_poly(data.ch1_gfilt), ':', c='xkcd:salmon')
    
    ax.plot(xlims, xlims, '--', c='0.3')
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    ax.set_xlabel(f'Channel 1 ({data.metadata.ch1})')
    ax.set_ylabel(f'Channel 2 ({data.metadata.ch2})')
    
    uss_row = upstroke_stats[upstroke_stats['#rec'] == data.metadata.file.stem]
    uss_row = next(uss_row.itertuples())
    ax.set_title(f'{data.metadata.file.stem} | slope = {slope:.2f} | upstroke lag = {uss_row.mean:.2f}±{uss_row.std:.2f}')
    
    plt.savefig(save_dir + f'/{data.metadata.file.stem}.png')
    plt.close(fig)



#%% Hilbert and phase difference on rolling average detrended data

from scipy.signal import find_peaks

def calc_stats(x, **hist_kwargs):
    """ Calculate the mode, mean and standard deviation of the data in x. Since
    the mode of a continuos quantity requires binning, hist_kwargs is passed to
    numpy.hitogram to customize the binning."""
    x = x[~np.isnan(x)]
    counts, bins = np.histogram(x, **hist_kwargs)
    bin_centers = bins[:-1] + np.diff(bins)/2
    
    remove = 10 # ignore a few points from each end of the histogram, assuming 
    # the mode we are itnerested in is somewhat close to the center 
    mode = bin_centers[ remove + np.argmax(counts[remove:-remove]) ]
    mean = np.mean(x)
    std = np.std(x)
    
    return mode, mean, std

def calc_thresholds(x, has_full_turn):
    
    x = x[~np.isnan(x)]
    
    if not has_full_turn:
        thresholds = x.min(), x.max()
    else:
                
        counts, bins = np.histogram(x, bins=100, density=True)
        plt.axvline(mode, c='k')
        max_loc = 10 + np.argmax(counts[10:-10])
        bin_centers = bins[:-1] + np.diff(bins)/2
        
        P = np.polynomial.Polynomial
        left_counts = counts[:max_loc]
        left_bins = bin_centers[:max_loc]
        right_counts = counts[max_loc:]
        right_bins = bin_centers[max_loc:]
        
        left = P.fit(left_bins, left_counts, deg=2)
        right = P.fit(right_bins, right_counts, deg=2)
        
        b, a = left.convert().coef[1:]
        t1 = -b/(2*a)
        if a<0 or not left_bins.min() < t1 < left_bins.max():
            t1 = left_bins[ np.argmin(left_counts) ]
                
        b, a = right.convert().coef[1:]
        t2 = -b/(2*a)
        if a<0 or not right_bins.min() < t2 < right_bins.max():
            t2 = right_bins[ np.argmin(right_counts) ]
        
        # plt.figure()
        # counts, bins, _ = plt.hist(x, bins=100, density=True)
        
        # plt.plot(bin_centers[:max_loc], counts[:max_loc], 'r')
        # plt.plot(bin_centers[max_loc:], counts[max_loc:], 'g')
        
        # plt.plot(bins, left(bins))
        # plt.plot(bins, right(bins))
        
        # plt.ylim(-0.1, max(counts)*1.1)
        
        # plt.axvline(t1)
        # plt.axvline(t2)
        
        thresholds = t1, t2
        
    return thresholds

def find_slips(data):
    
    t = data.times
    k = data.K
    x = data.ch1_phase - data.ch2_phase
    
    inx_max, _ = find_peaks(k, height=1-1e-2)
    inx_min, _ = find_peaks(-k, height=1-1e-2)

    max_df = pd.DataFrame({'pos': inx_max, 'is_max': np.full(inx_max.shape, True)})
    min_df = pd.DataFrame({'pos': inx_min, 'is_max': np.full(inx_min.shape, False)})
    
    extrema = pd.concat([max_df, min_df]).sort_values('pos').reset_index(drop=True)
       
    # find points that may be jumps: 
    #   - if one is a minumum the next is a maximum or viceversa
    #   - one point and the next one are no more than 3 seconds apart
    #   - the value o K monotonically changes between the points
    starts = []
    ends = []
    i = 0
    while i<len(extrema)-1:
        
        ismax = extrema.is_max.values[i]
        pos1 = extrema.pos[i]
        pos2 = extrema.pos[i+1]
        # if the current point is different form the next one, add both and skip one
        if ismax != extrema.is_max.values[i+1] and t[pos2]-t[pos1] < 3:
            
            jump_sign = np.sign(k[pos1] - k[pos2])
            
            if all( s==jump_sign for s in np.sign(np.diff(extrema[pos1:pos2])) ):
                starts.append(i)
                ends.append(i+1)
        i += 1
    
    # store in a dataframe the locations of the starts and ends of the jumps
    jumps = extrema.loc[starts].rename(columns={'pos':'starts'}).reset_index(drop=True)
    jumps['ends'] = extrema.loc[ ends, 'pos'].reset_index(drop=True)
    jumps['duration'] = jumps.ends - jumps.starts
    # add to the dataframe columns that make the jump a bit wider in either direction
    jumps['pre_start'] = jumps.starts - (jumps.duration/2).astype(int)
    jumps.loc[jumps.pre_start<0, 'pre_start'] = 0
    jumps['post_end'] = jumps.ends + (jumps.duration/2).astype(int)
    jumps.loc[jumps.post_end>=len(t), 'post_end'] = len(t) - 1
    
    # make a boolean array stat stores the locations where a jump is occurring
    invalid_locs = np.full(t.shape, False)
    for start, end in zip(jumps.pre_start, jumps.post_end):
        invalid_locs[start:end] = True
    
    # plt.figure()
    # ax1 = plt.subplot(2,1,1)
    # plt.plot(t, k, 'k')
    # plt.scatter(t[extrema.pos], k[extrema.pos], c=extrema.is_max, cmap='bwr', zorder=3)
    # for start, end in zip(jumps.starts, jumps.ends):
    #     plt.plot(t[start:end], k[start:end], 'm')    
    
    # plt.subplot(2,1,2, sharex=ax1)
    # plt.plot(t, x, 'k')
    # plt.scatter(t[extrema.pos], x[extrema.pos], c=extrema.is_max, cmap='bwr', zorder=3)
    # # plt.scatter(t[consecutive.pos], x[consecutive.pos], c=consecutive.is_max, cmap='PiYG', zorder=3.1, marker='x')
    # for start, end in zip(jumps.starts, jumps.ends):
    #     plt.plot(t[start:end], x[start:end], 'm')    
        
    
    return jumps, invalid_locs
    
# Load data
data_dir = '/data/marcos/FloClock_data/data'
save_dir = '/data/marcos/FloClock pics/Hilbert phase outliers'
# data_dir = '/home/user/Documents/Doctorado/Fly clock/FlyClock_data/data/'
# save_dir = '/home/user/Documents/Doctorado/Fly clock/FlyClock pics/Hilbert phase'
file_inx = 12
plot_all = True

data_files = contenidos(data_dir, filter_ext='.abf')

upstrokes_stat_file = Path(data_dir) / '../output' / 'upstroke_delay_stats.dat'
upstrokes_stat_file.resolve()
upstroke_stats = pd.read_csv(upstrokes_stat_file, sep=r'\s+')

write_file = Path(data_dir) / '../output' / 'phase_differences_outliers.csv'
write_file.resolve()

if plot_all:
    with open(write_file, 'w') as writefile:
        writefile.write('name,K,Kstd,hasturn,modelag,meanlag,stdlag\n')

iter_over = data_files if plot_all else (data_files[file_inx], )
for i, file in enumerate(iter_over):
    print(f'Running {file.stem}: {i+1}/{len(data_files)}')
    
    # Process data a bit
    data = au.load_data(file, gauss_filter=True, override_raw=True)
    data = data.process.downsample()
    data.process.poly_detrend(degree=5)
    data.process.gaussian_filter(sigma_ms=100)
    data.process.find_peaks(period_percent=0.6, prominence=5)
    data.process.average_detrend(outlier_mode_proportion=1.8, keep_og=False)
    data.process.calc_phase_difference()
    
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(4,4)
    
    ax_timeseries = plt.subplot(gs[:2, :3])
    ax_Khist = plt.subplot(gs[0, 3])
    ax_laghist = plt.subplot(gs[1, 3])
    ax_zoom = plt.subplot(gs[2, :2])
    ax_phase = plt.subplot(gs[2, 2:], sharex=ax_timeseries)
    ax_K = plt.subplot(gs[3, :2], sharex=ax_timeseries)
    ax_diff = plt.subplot(gs[3, 2:], sharex=ax_timeseries)
    
    # Plot data
    ax_timeseries.plot(data.times, data.ch1, label=f'Ch1 ({data.metadata.ch1})')
    ax_timeseries.plot(data.times, data.ch2, label=f'Ch1 ({data.metadata.ch1})')
    ax_timeseries.set_xlabel('time (s)')
    ax_timeseries.set_title('Data')
    
    # Plot zoom of data
    ax_zoom.plot(data.times, data.ch1)
    ax_zoom.plot(data.times, data.ch2)
    ax_zoom.set_title('Zoom in')
    ax_zoom.set_xlabel('time (s)')
    ax_zoom.set_xlim(10, 30)
    
    # Plot phase
    ax_phase.plot(data.times, data.ch1_phase)
    ax_phase.plot(data.times, data.ch2_phase)
    # ax_phase.plot(data.times[:-1], np.diff(data.ch1_phase - data.ch2_phase))
    ax_phase.set_title('Phase')
    ax_phase.set_xlabel('time (s)')
    
    # Plot sine of phase difference
    ax_K.plot(data.times, data.K)
    ax_K.axhline(0, color='0.6')
    ax_K.set_ylim(-1.05, 1.05)
    ax_K.set_xlabel('time (s)')
    
    uss_row = upstroke_stats[upstroke_stats['#rec'] == data.metadata.file.stem]
    uss_row = next(uss_row.itertuples())
    ax_K.set_title(f'Sine phase difference | K = {np.mean(data.K):.2f} ± {np.std(data.K):.2f} | upstroke lag = ({uss_row.mean:.2f} ± {uss_row.std:.2f})s')
    
    # Plot phase difference
    ax_diff.plot(data.times, data.ch1_phase - data.ch2_phase)
    ax_diff.axhline(0, color='0.6')
    has_full_turn = np.nanmax(np.abs(
            np.diff( (data.ch1_phase - data.ch2_phase)[::int(data.metadata.sampling_rate)] )
                    )) > 3
    ax_diff.set_title('Phase difference | Has full turn: ' + ('No', 'Yes')[int(has_full_turn)])
    ax_diff.set_xlabel('time (s)')
    ax_diff.set_xlim(data.times.min(), data.times.max())
        
    # Sine phase difference histogram
    ax_Khist.hist(data.K, bins=100, density=True)
    mode, *_ = calc_stats(data.K.values, bins=100)
    
    # thresholds = calc_thresholds(data.K, has_full_turn)
    # outside_thresholds = np.logical_or( data.K.values<thresholds[0], data.K.values>thresholds[1] )    
    # for thresh in thresholds:
    #     ax_Khist.axvline(thresh, c='r')
    _, invalid_locs = find_slips(data)    
    
    ax_Khist.set_xlabel('K')
    ax_Khist.set_ylabel('density')
    ax_Khist.set_title('Distribution of sine phase difference')
    
    # Lag histogram
    period = np.mean([data.process.get_instant_period(ch) for ch in (1,2)], axis=0)
    lag = np.arcsin(data.K) * period / (2*np.pi)
    # lag = lag[~outside_thresholds]
    lag = lag[~invalid_locs]
    ax_laghist.hist(lag, bins=100, density=True)
    mode_lag, mean_lag, std_lag = calc_stats(lag, bins=100)
    ax_laghist.set_xlabel(r'lag = $\frac{T}{2\pi}\,asin(K)$ (s)')
    ax_laghist.set_ylabel('density')    
    ax_laghist.set_title(f'Distribution of lag | mode=({mode_lag:.2f}±{std_lag:.2f})s')
    
    # Mark outlier data
    outlier_K = data.K.copy()
    # outlier_K[~outside_thresholds] = np.nan
    outlier_K[~invalid_locs] = np.nan
    ax_K.plot(data.times, outlier_K, 'r')

    outlier_diff = data.ch1_phase - data.ch2_phase
    # outlier_diff[~outside_thresholds] = np.nan
    outlier_diff[~invalid_locs] = np.nan
    ax_diff.plot(data.times, outlier_diff , 'r')

    # Format
    fig.suptitle(data.metadata.file.stem)
    plt.tight_layout()
    
    if plot_all:
        # Save and close
        fig.savefig(save_dir + f'/{data.metadata.file.stem}.png')
        plt.close(fig)
        
        # Store data    
        with open(write_file, 'a') as writefile:
            print(file.stem, np.mean(data.K), np.std(data.K), has_full_turn, mode_lag, mean_lag, std_lag,
                  file=writefile, sep=',')
    

#%% Plot phase differences vs upstroke lags

# from matplotlib import lines

data_dir = Path('/data/marcos/FloClock_data/output')
# data_dir = Path('/home/user/Documents/Doctorado/Fly clock/FlyClock_data/output/')
upstroke_file = data_dir / 'upstroke_delay_stats.dat'
phase_diff_file = data_dir / 'phase_differences.csv'

#load data and save it all to a single dataframe
upstroke_stats = pd.read_csv(upstroke_file, sep=r'\s+').sort_values('#rec').reset_index(drop=True)
phase_diff = pd.read_csv(phase_diff_file).sort_values('name').reset_index(drop=True)
stats = pd.concat( [phase_diff, upstroke_stats.loc[:, [ '1', '2', 'mean', 'std']].rename(columns={'mean': 'upstroke_mean', 'std':'upstroke_std'})], axis='columns')

names_dict = {
    'upstroke' : {'name': 'upstroke_mean', 'error': 'upstroke_std', 'units': True, 'label': 'Upstroke lag'},
    'mean' : {'name': 'meanlag', 'error': 'stdlag', 'units': True, 'label': 'Avg. sine phase diff. lag'},
    'mode' : {'name': 'modelag', 'error': 'stdlag', 'units': True, 'label': 'Sine phase diff. lag mode'},
    'K' : {'name': 'K', 'error': 'Kstd', 'units': False, 'label': 'Sine phase difference'},
    }

pair = 'upstroke', 'K'


# add a few new columns
stats['type'] = [n[:2] for n in stats['name']]
stats['well_ordered'] = [ch1+ch2==t for ch1, ch2, t in zip(stats['1'], stats['2'], stats.type)]

for name in names_dict.keys():
    stats.loc[stats['well_ordered'], names_dict[name]['name'] ] *= -1


plt.figure(figsize=(13, 6))

ax1 = plt.subplot(1,2,1)
plt.errorbar(stats[names_dict[pair[0]]['name']], stats[names_dict[pair[1]]['name']],
             xerr = stats[names_dict[pair[0]]['error']], yerr = stats[names_dict[pair[1]]['error']],
             fmt='.', color='0.7')
plt.xlabel(names_dict[pair[0]]['label'])
plt.ylabel(names_dict[pair[1]]['label'])
# plt.title('Upstroke lag - Sine phase difference correlation')
plt.title('Correlation')

both_categories = []
for i, name in enumerate(pair):
        
    grouped = stats.groupby('type')
    categories = {key:df[names_dict[name]['name']] for key, df in sorted(grouped)}
        
    categories['LL+SS'] = [*categories['LL'], *categories['SS']]
    categories['LR+SR'] = [*categories['LR'], *categories['SR']]

    plt.subplot(2,2,2*(i+1))
    plt.boxplot(categories.values())
    plt.gca().set_xticklabels(categories.keys())
    plt.title(names_dict[name]['label'])
    
    if names_dict[name]['units']:
        plt.ylabel('lag (s)')
    
    both_categories.append(categories)
    
styles = {'LL' : ('o', 'C00'), 
          'SS' : ('^', 'C00'),
          'LS' : ('s', 'C01'), 
          'LR' : ('o', 'C03'), 
          'SR' : ('^', 'C03'),
          }

for key in categories.keys():
    if key not in styles:
        continue
    cat1, cat2 = both_categories
    ax1.plot(cat1[key], cat2[key], styles[key][0], color=styles[key][1], zorder=3, label=key)

ax1.legend()
plt.tight_layout()

plt.savefig(f'/data/marcos/FloClock pics/{pair[0]} vs {pair[1]}')


#%% Find peak density

from scipy import signal
from skimage import filters

from utils import sort_by

# Load data
# data_dir = '/data/marcos/FloClock_data/data'
data_dir = '/data/marcos/FloClock_data/data - mecamilamina'
file_inx = 10
ch_inx = 1 # channel index

data_files = contenidos(data_dir, filter_ext='.abf')

# Process data a bit
data = au.load_data(data_files[file_inx], gauss_filter=True, override_raw=False)
data = data.process.downsample()
data.process.poly_detrend(degree=5, channels='gfilt')
first_trend = data.process.get_trend(ch_inx)
data.process.gaussian_filter(sigma_ms=100, keep_og=True, channels='gfilt')
data.process.poly_detrend(degree=15, channels='gfilt_gfilt2')

ch = data[f'ch{ch_inx}_gfilt_gfilt2']
times = data.times

# first pass, with threshold 0mv
p_inx, _ = signal.find_peaks(ch)#, height=0)
peaks = ch.values[p_inx]
t_peaks = times.values[p_inx]

# second pass: deleted

# third pass, with minimum distance between peaks
counts, bins = np.histogram(np.diff(t_peaks))
bin_centers = bins[:-1] + np.diff(bins) / 2
period_mode = bin_centers[ np.argmax(counts) ]
distance_points = int(period_mode * 0.3 / (times[1] - times[0]))
# p_inx, _ = signal.find_peaks(ch, height=threshold, distance=distance_points)
p_inx, props = signal.find_peaks(ch, prominence=5)#, distance=distance_points)
        

# Plot data and peaks
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True, 
                              sharex=True, height_ratios=[3,1])

trend = data.process.get_trend(ch_inx)
ax.plot(data.times, (data[f'ch{ch_inx}'] - first_trend - trend), color='0.8')
# ax.plot(data.times, data[f'ch{ch_inx}_gfilt'] - trend)
ax.plot(data.times, data[f'ch{ch_inx}_gfilt_gfilt2'])

peak_pos = times[p_inx]
peak_val = ch[p_inx]
ax.plot(peak_pos, peak_val, 'o')
ax.set_xlim(data.times.values[0], data.times.values[-1])

ax.plot(times[props['left_bases']], ch[props['left_bases']], 'ro', mfc='none')
ax.plot(times[props['right_bases']], ch[props['right_bases']], 'bx')
for pp, pv, prominence in zip(peak_pos, peak_val, props['prominences']):
    ax.text(pp, pv, f'{prominence:.1f}', horizontalalignment='center', verticalalignment='bottom')


# plot mec application lines
ax.axvline(data.metadata.mec_start_sec, ls='--', c='k', label='mec start')
ax2.axvline(data.metadata.mec_start_sec, ls='--', c='k', label='mec start')

### Find peak density

def peak_prop_density(times, peak_output, prop, box_size=60, box_overlap=0.3):
    """Calculate the density of some property of the peaks (count, prominences,
    heights, etc) by averaging in boxes of box_size in seconds. Each subsequent 
    box has an overlap given by box__overlap with the previous ones. Boxes that
    include stuff outside the data range are cropped. """

    halfbox = box_size / 2
    box_positions = np.arange(times.min(), times.max(), box_size * (1-box_overlap))
    sizes = np.asarray([ min(times.max(), bp+halfbox) - max(times.min(), bp-halfbox) for bp in box_positions])
    
    # Find the peak positions
    peak_inx, props = peak_output
    peak_pos = times[peak_inx]
    
    # calculate the density of counts or peak property
    if prop == 'counts':
        counts = np.asarray([np.sum( np.logical_and(peak_pos > bp - halfbox, peak_pos < bp + halfbox)) for bp in box_positions])
    else:
        counts = np.asarray([np.sum( props[prop][np.logical_and(peak_pos > bp - halfbox, peak_pos < bp + halfbox)]) for bp in box_positions])
    
    count_densities = counts / sizes
    
    return box_positions, count_densities

# Calculate a few props
box_positions, peak_count_densities = peak_prop_density(times, (p_inx, props), 'counts')
_, prom_densities = peak_prop_density(times, (p_inx, props), 'prominences')


### Find burst density

raw = data[f'ch{ch_inx}'] - first_trend - trend
# p_inx, props = signal.find_peaks(raw, height=5, distance=int(0.05 * data.metadata.sampling_rate))
height_threshold = 5
peak_out = p_inx, props = signal.find_peaks(raw, height=height_threshold, prominence=1)

_, burst_count_densities = peak_prop_density(times, peak_out, 'counts')
_, height_densities = peak_prop_density(times, peak_out, 'peak_heights')

# Normalize everything to the first value
for x in [peak_count_densities, prom_densities, burst_count_densities, height_densities]:
    x /= x[0]

# Plot peaks properties
ax2.plot(box_positions, peak_count_densities, label='Peak count density')
ax2.plot(box_positions, prom_densities, label='Peak prominence (window-averaged)')
ax2.plot(box_positions, burst_count_densities, c='0.4', label='Burst count density')
ax2.plot(box_positions, height_densities, label=f'Burst height density (<{height_threshold}mV)')

# Plot burst peaks
peak_pos = times[p_inx]
peak_val = raw[p_inx]
ax.plot(peak_pos, peak_val, '.', c='0.4')

# ax2.plot(box_positions, prom_densities, label='Peak prominence (window-averaged)')

ax2.legend()
channels = data.metadata.ch1, data.metadata.ch2
fig.suptitle(f'{data.metadata.file.stem} (ch {ch_inx} : {channels[ch_inx-1]})')


#%% Plot and save all peak property densities

from scipy import signal
from skimage import filters

from utils import sort_by


def peak_prop_density(times, peak_output, prop, box_size=60, box_overlap=0.3):
    """Calculate the density of some property of the peaks (count, prominences,
    heights, etc) by averaging in boxes of box_size in seconds. Each subsequent 
    box has an overlap given by box__overlap with the previous ones. Boxes that
    include stuff outside the data range are cropped. """

    halfbox = box_size / 2
    box_positions = np.arange(times.min(), times.max(), box_size * (1-box_overlap))
    sizes = np.asarray([ min(times.max(), bp+halfbox) - max(times.min(), bp-halfbox) for bp in box_positions])
    
    # Find the peak positions
    peak_inx, props = peak_output
    peak_pos = times[peak_inx]
    
    # calculate the density of counts or peak property
    if prop == 'counts':
        counts = np.asarray([np.sum( np.logical_and(peak_pos > bp - halfbox, peak_pos < bp + halfbox)) for bp in box_positions])
    else:
        counts = np.asarray([np.sum( props[prop][np.logical_and(peak_pos > bp - halfbox, peak_pos < bp + halfbox)]) for bp in box_positions])
    
    count_densities = counts / sizes
    
    return box_positions, count_densities

# Load data
# data_dir = '/data/marcos/FloClock_data/data'
data_dir = '/data/marcos/FloClock_data/data - mecamilamina'
save_dir = '/data/marcos/FloClock pics/mecamilamina/Peak prop densities'
output_dir = Path(data_dir) / 'output' / 'peak_property_densities' 

data_files = contenidos(data_dir, filter_ext='.abf')

with open(output_dir / 'stats.csv', 'w') as statsfile:
    statsfile.write('output_file,data_file,pair,channel,mec_start\n')

for i, file in enumerate(data_files):
    print(f'Running {file.stem}: {i+1}/{len(data_files)}')
    
    for ch_inx in [1,2]:
        # Process data a bit
        data = au.load_data(file, gauss_filter=True, override_raw=False)
        data = data.process.downsample()
        data.process.poly_detrend(degree=5, channels='gfilt')
        first_trend = data.process.get_trend(ch_inx)
        data.process.gaussian_filter(sigma_ms=100, keep_og=True, channels='gfilt')
        data.process.poly_detrend(degree=15, channels='gfilt_gfilt2')
        
        fallback_threshold = 4
        ch = data[f'ch{ch_inx}_gfilt_gfilt2']
        times = data.times
        
        # first pass, with threshold 0mv
        p_inx, _ = signal.find_peaks(ch)#, height=0)
        peaks = ch.values[p_inx]
        
        # second pass: deleted
        
        # third pass, with minimum distance between peaks
        counts, bins = np.histogram(np.diff(t_peaks))
        bin_centers = bins[:-1] + np.diff(bins) / 2
        period_mode = bin_centers[ np.argmax(counts) ]
        distance_points = int(period_mode * 0.3 / (times[1] - times[0]))
        # p_inx, _ = signal.find_peaks(ch, height=threshold, distance=distance_points)
        p_inx, props = signal.find_peaks(ch, prominence=5)#, distance=distance_points)
                
        
        # Plot data and peaks
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True, 
                                      sharex=True, height_ratios=[3,1])
        
        trend = data.process.get_trend(ch_inx)
        ax.plot(data.times, (data[f'ch{ch_inx}'] - first_trend - trend), color='0.8')
        # ax.plot(data.times, data[f'ch{ch_inx}_gfilt'] - trend)
        ax.plot(data.times, data[f'ch{ch_inx}_gfilt_gfilt2'])
        
        peak_pos = times[p_inx]
        peak_val = ch[p_inx]
        ax.plot(peak_pos, peak_val, 'o')
        ax.set_xlim(data.times.values[0], data.times.values[-1])
        
        ax.plot(times[props['left_bases']], ch[props['left_bases']], 'ro', mfc='none')
        ax.plot(times[props['right_bases']], ch[props['right_bases']], 'bx')
        for pp, pv, prominence in zip(peak_pos, peak_val, props['prominences']):
            ax.text(pp, pv, f'{prominence:.1f}', horizontalalignment='center', verticalalignment='bottom')
        
        
        # plot mec application lines
        ax.axvline(data.metadata.mec_start_sec, ls='--', c='k', label='mec start')
        ax2.axvline(data.metadata.mec_start_sec, ls='--', c='k', label='mec start')
        
        ### Find peak density
        
        # Calculate a few props
        box_positions, peak_count_densities = peak_prop_density(times, (p_inx, props), 'counts')
        _, prom_densities = peak_prop_density(times, (p_inx, props), 'prominences')
        
        
        ### Find burst density
        
        raw = data[f'ch{ch_inx}'] - first_trend - trend
        # p_inx, props = signal.find_peaks(raw, height=5, distance=int(0.05 * data.metadata.sampling_rate))
        height_threshold = 5
        peak_out = p_inx, props = signal.find_peaks(raw, height=height_threshold, prominence=1)
        
        _, burst_count_densities = peak_prop_density(times, peak_out, 'counts')
        _, height_densities = peak_prop_density(times, peak_out, 'peak_heights')
        
        # Normalize everything to the first value and plot
        all_densities = [peak_count_densities, prom_densities, burst_count_densities, height_densities]
        all_labels = 'Peak count density', 'Peak prominence (window-averaged)', 'Burst count density',f'Burst height density (<{height_threshold}mV)'
        for x, label in zip(all_densities, all_labels):
            ax2.plot(box_positions, x / x[0], label=label)
        
        # Plot burst peaks
        peak_pos = times[p_inx]
        peak_val = raw[p_inx]
        ax.plot(peak_pos, peak_val, '.', c='0.4')
        
        ax2.legend()
        channels = data.metadata.ch1, data.metadata.ch2
        fig.suptitle(f'{data.metadata.file.stem} (ch {ch_inx} : {channels[ch_inx-1]})')
        
        ### Save data
        
        out_filename = f'{data.metadata.file.stem}_ch{ch_inx}'
        
        # save figure
        plt.savefig(Path(save_dir) / f'{out_filename}.png')
        plt.close()
        
        # Save data
        df = pd.DataFrame({'box_pos': box_positions, 
                            'peak_count': peak_count_densities, 
                            'peak_prominence': prom_densities, 
                            'burst_count': burst_count_densities, 
                            'burst_height': height_densities})
        
        df.to_csv(output_dir / f'{out_filename}.csv', index=False)
        
        with open(output_dir / 'stats.csv', 'a') as statsfile:
            statsfile.write(','.join( (
                out_filename,
                data.metadata.file.stem,
                data.metadata.pair,
                channels[ch_inx-1],
                str(data.metadata.mec_start_sec)
                )) )
            statsfile.write('\n')
            
#%% Explore all peak prop densities

data_dir = Path('/data/marcos/FloClock_data/data - mecamilamina')
peak_prop_dir = data_dir / 'output' / 'peak_property_densities' 
save_dir = '/data/marcos/FloClock pics/mecamilamina'

stats_file = peak_prop_dir / 'stats.csv'
data_files = contenidos(peak_prop_dir)
pair_guide_file = data_dir / 'par_guide.xlsx'

# load data
data = {file.stem: pd.read_csv(file) for file in data_files if 'stats' not in file.stem}
stats = pd.read_csv(stats_file)
pair_guide = pd.read_excel(pair_guide_file)
data_example = data[stats.output_file[0]]

# construct a daraframe containing the duration of each run
durations = pd.concat([
    pair_guide.loc[:, ['ch1', 'duration(min)', 'name']].rename(columns={'ch1':'ch'}), 
    pair_guide.loc[:, ['ch2', 'duration(min)', 'name']].rename(columns={'ch2':'ch'})
    ]).sort_values(['ch', 'duration(min)'], ascending=(False, False), ignore_index=True)
durations['duration(sec)'] = durations['duration(min)'] * 60

# plot stuff
fig, axarr = plt.subplots(4,1, sharex=True, figsize=(12, 8), constrained_layout=True)
fig_d, (ax_d1, ax_d2) = plt.subplots(2,1, sharex=True, figsize=(12, 8), constrained_layout=True)

colors = {'R': '0.3', 'L': 'C0', 'S': 'C1'}
averages = {col:{ch:[] for ch in 'RLS'} for col in data_example.columns[1:]}
for i, row in stats.iterrows():
    df = data[row.output_file]
    mec_start = row.mec_start
    
    times = df.box_pos - mec_start
    zero_index = np.argmin(np.abs(times))
    for j, (ax, col) in enzip(axarr, df.columns[1:]):
        x = df[col].values
    
        # plot the line        
        c = colors[row.channel]
        ax.plot(times, x/x[zero_index], color=c, alpha=0.6)       
        ax.set_title(col)
        
        # save data interpolator for later averaging
        averages[col][row.channel].append(
            interpolate.interp1d(times, x/x[zero_index], bounds_error=False, fill_value=np.nan))
        
    for j, (ch, _, _, duration) in durations[durations.name==row.data_file].iterrows():
        ax_d2.plot([times[0], times[0]+duration], [j, j], color=colors[ch])

## Figure for averages
fig2, axarr2 = plt.subplots(4,1, sharex=True, figsize=(12, 8), constrained_layout=True)

average_line_handles = {ch:None for ch in 'RLS'}
for i, (ax, col) in enzip(axarr2, df.columns[1:]):
    
    # plot averages
    time_range = -350, 2000
    average_times = np.linspace(*time_range, 1000)
    
    for ch, interpolators in averages[col].items():
        average = np.nanmean( [interp(average_times) for interp in interpolators], axis=0 )
        deviation = np.nanstd( [interp(average_times) for interp in interpolators], axis=0 )
        
        line, = ax.plot(average_times, average, color=colors[ch], label=ch)
        ax.fill_between(average_times, average-deviation, average+deviation, color=colors[ch], alpha=0.2)
        ax.set_title(col)
        
        if i == 0:
            ax_d1.plot(average_times, average, color=colors[ch], label=ch)
            ax_d1.fill_between(average_times, average-deviation, average+deviation, color=colors[ch], alpha=0.2)
            ax_d1.set_title(col)
            
        average_line_handles[ch] = line

# format plots
for ax in (*axarr.flat, *axarr2.flat, ax_d1, ax_d2):
    # ax.set_title(col)
    ax.set_ylim(-0.1, 2)
    ax.set_xlim(time_range)
    ax.axvline(0, color='0.5')
    
ax_d2.set_ylim(-1, len(durations))

for ax in (axarr[0], axarr2[0], ax_d1):
    ax.legend(handles=average_line_handles.values(), labels=average_line_handles.keys())

# Save figures
save_dir = Path(save_dir)

# fig.savefig(save_dir / 'Peak property density.png')
# fig2.savefig(save_dir / 'Peak propery densiy averages.png')
# fig_d.savefig(save_dir / 'Peak propery rec durations.png')

#%% Plot mec runs just after mec

# Load data
# data_dir = '/data/marcos/FloClock_data/data'
data_dir = '/data/marcos/FloClock_data/data - mecamilamina'
save_dir = '/data/marcos/FloClock pics/mecamilamina/After mec/raw'

file_inx = 6
plot_all = True
extension = 200 # in seconds after mec

data_files = contenidos(data_dir, filter_ext='.abf')
pair_guide_file = contenidos(data_dir, filter_ext='.xlsx').pop()

iter_over = data_files if plot_all else (data_files[file_inx], )
for i, file in enumerate(iter_over):
    print(f'Running {file.stem}: {i+1}/{len(data_files)}')
    
    # Process data a bit
    data = au.load_data(file, gauss_filter=True, override_raw=False)
    data = data.process.downsample()
    data.process.poly_detrend(degree=5, channels='gfilt')
    trends = [ data.process.get_trend(ch_inx) for ch_inx in (1,2) ]
    data.process.gaussian_filter(sigma_ms=100, channels='gfilt')
    data.process.poly_detrend(degree=15, channels='gfilt')
    
    # # Plot data
    # mosaic_layout = """
    #                 ab
    #                 xx
    #                 yy
    #                 """
    # fig, ax_dict = plt.subplot_mosaic(mosaic_layout, tight_layout=True, 
    #                                   figsize=[15,  6.5], height_ratios=[1,3,1])
    # comment = data.metadata.comment
    # fig.suptitle(data.metadata.file.stem + '\n' + (comment if isinstance(comment, str) else ''))
    
    # ax1 = ax_dict['a']
    # ax2 = ax_dict['b']
    # ax = ax_dict['x']
    # ax_trend = ax_dict['y']
    
    fig = plt.figure(figsize=[15,  6.5], tight_layout=True)
    
    gs = fig.add_gridspec(2,2, height_ratios=[1,4])
    
    ax1, ax2 = [fig.add_subplot(gs[0, i]) for i in range(2)]
    
    gs2 = gs[1, :].subgridspec(2,1,height_ratios=[3,1])
    ax = fig.add_subplot(gs2[0])
    ax_trend = fig.add_subplot(gs2[1], sharex=ax)
    
    # channel 1
    ax1.plot(data.times, data.ch1 - trends[0] - data.process.get_trend(1), c='0.7')
    ax1.plot(data.times, data.ch1_gfilt, c='C0')
    ax1.set_title(f'ch1: {data.metadata.ch1}')
    
    # channel 2
    ax2.plot(data.times, data.ch2 - trends[1] - data.process.get_trend(2), c='0.7')
    ax2.plot(data.times, data.ch2_gfilt, c='C1')
    ax2.set_title(f'ch2: {data.metadata.ch2}')
    
    # combined
    mec_start = data.metadata.mec_start_sec
    start = find_point_by_value(data.times, mec_start)
    end = find_point_by_value(data.times, mec_start + extension)
    plot_range = slice(start, end)
    short_time = data.times[plot_range] - mec_start
    
    ax.plot(short_time, (data.ch1 - trends[0] - data.process.get_trend(1))[plot_range], c='0.7')
    ax.plot(short_time, (data.ch2 - trends[1] - data.process.get_trend(2))[plot_range], c='0.7')
    ax.plot(short_time, data.ch1_gfilt[plot_range])
    ax.plot(short_time, data.ch2_gfilt[plot_range], alpha=0.6)
    ax.set_ylim( max(ax.get_ylim()[0], -20), min( 40, ax.get_ylim()[1]) )
    
    ax_trend.set_xlim( 0, extension)
    ax_trend.set_xlabel('time sinsce mec start (sec)')
    
    ax.set_title('Both channels after mec start')

    ax_trend.plot(short_time, (trends[0] + data.process.get_trend(1))[plot_range])
    ax_trend.plot(short_time, (trends[1] + data.process.get_trend(2))[plot_range])

    for axi in [ax1, ax2]:
        mec_line = axi.axvline(mec_start, ls='--', c='k', label='mec start')
        axi.set_xlabel('time (sec)')
        axi.set_xlim(data.times.min(), data.times.max())
    
    if plot_all:
        plt.savefig(save_dir + f'/{data.metadata.file.stem}.png')
        plt.close(fig)
        
    
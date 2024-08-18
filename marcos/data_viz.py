#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 13:17:52 2023

@author: marcos
"""
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import colors
import pandas as pd
import numpy as np
from scipy import interpolate, stats

from utils import contenidos, enzip, find_point_by_value, find_closest_value
from utils import find_numbers, smooth, sort_by, calc_mode
import analysis_utils as au


scolor = '#d06c9eff'
scoloralt = '#d699b7ff'
lcolor = '#006680ff'
lcoloalt = '#1fa0c1ff'
pcolor = '#1db17eff'
#%% Visualzie raw data

import pyabf

# Load data
data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
# data_dir = '/media/marcos/DATA//marcos/FloClock_data/data - mecamilamina'
file_inx = 36

data_files = contenidos(data_dir, filter_ext='.abf')

file = data_files[file_inx]
abf = pyabf.ABF(file)


abf.setSweep(0, channel=0)
times = abf.sweepX
ch1 = abf.sweepY

abf.setSweep(0, channel=1)
ch2 = abf.sweepY

fig, (top, bot) = plt.subplots(2,2, figsize=[17, 6], sharex='col', width_ratios=[3,1], sharey=True)
ax1, ax3 = top
ax2, ax4 = bot

ax1.plot(times, ch1)
ax2.plot(times, ch2)

plt_slice = slice(220000, 320000)
ax3.plot(times[plt_slice], ch1[plt_slice])
ax4.plot(times[plt_slice], ch2[plt_slice])

# au.make_scalebar(ax1)
# au.make_scalebar(ax3)

# ax1.axvline(times[220000])
# ax1.axvline(times[320000])

fig.suptitle(file.name)

print('file:', file.stem)
print('duration (min):', f'{times[-1]/60:.3f}')
print('sampling rate:', abf.sampleRate)

#%% Visualize one run

# Load data
data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
# data_dir = '/media/marcos/DATA//marcos/FloClock_data/data - mecamilamina'
file_inx = 17

data_files = contenidos(data_dir, filter_ext='.abf')
pair_guide_file = contenidos(data_dir, filter_ext='.xlsx').pop()

# Process data a bit
data = au.load_data(data_files[file_inx], gauss_filter=True, override_raw=False)
data.process.lowpass_filter(filter_order=2, frequency_cutoff=10, keep_og=True)
data = data.process.downsample()
data.process.highpass_filter(filter_order=2, frequency_cutoff=0.1, keep_og=True, channels='lpfilt')
# data.process.poly_detrend(degree=5, keep_og=True, channels='gfilt')
# data.process.gaussian_filter(sigma_ms=100, keep_og=True, channels='lpfilt') #apply it on the non-detrended data because we will detrned manually later when plotting
data.process.lowpass_filter(filter_order=2, frequency_cutoff=2, keep_og=True, channels='lpfilt') #apply it on the non-detrended data because we will detrned manually later when plotting

   
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# # plot detrended very smoothed data

# ax1.plot(data.times, data.ch1_lpfilt - data.process.get_hptrend(1), '0.5', label='raw')
# ax2.plot(data.times, data.ch2_lpfilt - data.process.get_hptrend(2), '0.5')

# ax1.plot(data.times, data.ch1_lpfilt_gfilt2 - data.process.get_hptrend(1), label='gauss')
# ax2.plot(data.times, data.ch2_lpfilt_gfilt2 - data.process.get_hptrend(2), )

# ax1.plot(data.times, data.ch1_lpfilt_lpfilt - data.process.get_hptrend(1), label='lowpass')
# ax2.plot(data.times, data.ch2_lpfilt_lpfilt - data.process.get_hptrend(2), )

# ax1.legend()

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
# trend, = ax_dict['a'].plot(data.times, data.process.get_trend(1), label='trend')
trend, = ax_dict['a'].plot(data.times, data.process.get_hptrend(1), label='trend')
ax_dict['x'].plot(data.times, data.process.get_hptrend(1), 'C03')
ax_dict['b'].plot(data.times, data.process.get_hptrend(2))
ax_dict['y'].plot(data.times, data.process.get_hptrend(2), 'C03')

# plot detrended data
ax_dict['c'].plot(data.times, data.ch1_lpfilt_hpfilt, label=f'ch1 ({data.metadata.ch1})')
ax_dict['c'].plot(data.times, data.ch2_lpfilt_hpfilt, label=f'ch2 ({data.metadata.ch2})')
ax_dict['z'].plot(data.times, data.ch1_lpfilt_hpfilt)
ax_dict['z'].plot(data.times, data.ch2_lpfilt_hpfilt, alpha=0.6)

# plot detrended very smoothed data
ax_dict['c'].plot(data.times, data.ch1_lpfilt_lpfilt - data.process.get_hptrend(1), '#89bce0')
ax_dict['c'].plot(data.times, data.ch2_lpfilt_lpfilt - data.process.get_hptrend(2), '#fab67a')

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

#%% Single channel plots for figures

# Load data
data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
data_dir = '/media/marcos/DATA//marcos/FloClock_data/data - mecamilamina'
file_inx = 4

data_files = contenidos(data_dir, filter_ext='.abf')
pair_guide_file = contenidos(data_dir, filter_ext='.xlsx').pop()

# Process data a bit
data = au.load_data(data_files[file_inx], gauss_filter=True, override_raw=False)
data.process.lowpass_filter(filter_order=2, frequency_cutoff=10, keep_og=True)
data = data.process.downsample()
# data.process.highpass_filter(filter_order=2, frequency_cutoff=0.1, keep_og=True, channels='lpfilt')
data.process.lowpass_filter(filter_order=2, frequency_cutoff=2, keep_og=True, channels='lpfilt') #apply it on the non-detrended data because we will detrned manually later when plotting
data = data.process.downsample(2)

fig, axdict = plt.subplot_mosaic([['A', 'A'], ['B', 'C']], constrained_layout=True)

ax = axdict['A']
ax1 = axdict['B']
ax2 = axdict['C']

ax.plot(data.times, data.ch2)
# ax.plot(data.times, data.ch2_lpfilt)
ax.plot(data.times, data.ch2_lpfilt_lpfilt)

start_times = [170, 285]
end_times = [185, 300]
for startt, endt, axz in zip(start_times, end_times, (ax1, ax2)):
    start = find_point_by_value(data.times, startt)
    end = find_point_by_value(data.times, endt)
    
    axz.plot(data.times[start:end], data.ch2.values[start:end])
    axz.plot(data.times[start:end], data.ch2_lpfilt.values[start:end])
    # axz.plot(data.times[start:end], data.ch2_lpfilt_lpfilt.values[start:end])
    
    # mark the zoomed in areas
    ax.fill_betweenx(ax.get_ylim(), startt, endt, color='0.5')

# plot mec start line, if needed
if hasattr(data.metadata, 'mec_start_sec'):
    
    mec_line = ax.axvline(data.metadata.mec_start_sec, ls='--', c='k', label='mec start')
    fig.legend(handles=[mec_line], ncol=6, loc='upper right', bbox_to_anchor=(1, 0.97))    

fig.suptitle(data.metadata.file.stem)
ax.set_xlabel('Time [seconds]')

# # set ax limits
# for axi in axdict.values():
#     axi.set_ylim(-60, -8)
# ax.set_xlim(150, 700)

# make scale bar
for axi in axdict.values():
    au.make_scalebar(axi)


#%% Dual channel plots for figures

# Load data
# data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
data_dir = '/media/marcos/DATA//marcos/FloClock_data/data - mecamilamina'
file_inx = 4 # 0, 11, 22, 37
plot_interval = [125, 155]
# plot_interval = [548, 561]

data_files = contenidos(data_dir, filter_ext='.abf')
pair_guide_file = contenidos(data_dir, filter_ext='.xlsx').pop()

# Process data a bit
data = au.load_data(data_files[file_inx], gauss_filter=False, override_raw=False)
data.process.lowpass_filter(filter_order=2, frequency_cutoff=10, keep_og=True)
data = data.process.downsample()
# data.process.highpass_filter(filter_order=2, frequency_cutoff=0.1, channels='lpfilt')
data.process.lowpass_filter(filter_order=2, frequency_cutoff=2, keep_og=True, channels='lpfilt') #apply it on the non-detrended data because we will detrned manually later when plotting
data = data.process.downsample(2)

fig, axarr = plt.subplots(2,2, figsize=[17, 6], sharex='col', width_ratios=[3,1], sharey=True)

for (axl, axr), ch in zip(axarr, (1,2)):
 
    # left axis
    axl.plot(data.times, data[f'ch{ch}'])
    axl.plot(data.times, data[f'ch{ch}_lpfilt'])
    axl.plot(data.times, data[f'ch{ch}_lpfilt_lpfilt'])
    
    start = find_point_by_value(data.times, plot_interval[0])
    end = find_point_by_value(data.times, plot_interval[1])
    
    # right axis
    axr.plot(data.times[start:end], data[f'ch{ch}'].values[start:end])
    axr.plot(data.times[start:end], data[f'ch{ch}_lpfilt'].values[start:end])
    axr.plot(data.times[start:end], data[f'ch{ch}_lpfilt_lpfilt'].values[start:end])
    
    # mark the zoomed in areas
    axl.fill_betweenx(axl.get_ylim(), *plot_interval, color='0.5')

    # plot mec start line, if needed
    if hasattr(data.metadata, 'mec_start_sec'):
        
        mec_line = axl.axvline(data.metadata.mec_start_sec, ls='--', c='k', label='mec start')
        fig.legend(handles=[mec_line], ncol=6, loc='upper right', bbox_to_anchor=(1, 0.97))    

fig.suptitle(data.metadata.file.stem)
axarr[-1, 0].set_xlabel('Time [seconds]')

# axarr[0, 0].set_ylim(-70, -10)
# axarr[0, 0].set_ylim(-10, 30)
# axarr[0, 0].set_xlim(330, 650)

# make scale bar
for axi in axarr.flat:
    au.make_scalebar(axi)


#%% Dual channel superimposed plots

# Load data
data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
# data_dir = '/media/marcos/DATA//marcos/FloClock_data/data - mecamilamina'
file_inx = 18 # 0, 11, 22, 37
plot_interval = [
    # 5.5, 20.5 #1: LL
    83, 99, #18: LS
    # 427, 443 #42: SS
    ]

data_files = contenidos(data_dir, filter_ext='.abf')
pair_guide_file = contenidos(data_dir, filter_ext='.xlsx').pop()

# Process data a bit
data = au.load_data(data_files[file_inx], gauss_filter=True, override_raw=False)
data.process.lowpass_filter(filter_order=2, frequency_cutoff=10, keep_og=True)
data = data.process.downsample()
data.process.highpass_filter(filter_order=2, frequency_cutoff=0.1, channels='lpfilt', keep_og=True)
data.process.lowpass_filter(filter_order=2, frequency_cutoff=2, keep_og=True, channels='lpfilt_hpfilt') #apply it on the non-detrended data because we will detrned manually later when plotting
data = data.process.downsample(2)

fig, ax = plt.subplots(figsize=[17, 6])
    
start = find_point_by_value(data.times, plot_interval[0])
end = find_point_by_value(data.times, plot_interval[1])

for ch in (1,2):
    
    ax.plot(data.times[start:end], data[f'ch{ch}_lpfilt_hpfilt_lpfilt'][start:end])
    ax.plot(data.times[start:end], data[f'ch{ch}'][start:end] - data.process.get_hptrend(ch)[start:end], c=f'C{ch-1}', alpha=0.3, zorder=1)
    
# plot mec start line, if needed
if hasattr(data.metadata, 'mec_start_sec'):
    
    mec_line = axl.axvline(data.metadata.mec_start_sec, ls='--', c='k', label='mec start')
    fig.legend(handles=[mec_line], ncol=6, loc='upper right', bbox_to_anchor=(1, 0.97))    

fig.suptitle(data.metadata.file.stem)
ax.set_xlabel('Time [seconds]')

au.make_scalebar(ax)
# axarr[0, 0].set_ylim(-70, -10)
# axarr[0, 0].set_ylim(-10, 30)


#%% Step by step analysis


# Load data
data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
# data_dir = '/media/marcos/DATA//marcos/FloClock_data/data - mecamilamina'
file_inx = 17
channel = 2 # 1 or 2
tmax = 1020 # in sec

# for the zoomed-in plot
plot_seconds = 9 # duration
t0 = 83 # where to start in seconds

data_files = contenidos(data_dir, filter_ext='.abf')
pair_guide_file = contenidos(data_dir, filter_ext='.xlsx').pop()

fig = plt.figure(constrained_layout=True, figsize=[11, 9])
# figs, axarr = plt.subplots(4,2, constrained_layout=True, sharex='col')
figs = fig.subfigures(5)

sup_axs = figs[4].subplots(1, 2,  width_ratios=[3,1], sharey=True)

ch = f'ch{channel}'

data = au.load_data(data_files[file_inx], gauss_filter=False, override_raw=False)
data = data.process.downsample(2)

# construct step slices
t0_points = find_point_by_value(data.times, t0)
tf_points = find_point_by_value(data.times, t0+plot_seconds)
tmax_points = find_point_by_value(data.times, tmax)

step_full = slice(None, tmax_points, 100) # full range, 100× downsampling
step_ins = slice(t0_points, tf_points, 10) # short frame, 10× downsampling

# Raw data
for ax, step in zip(figs[0].subplots(1, 2,  width_ratios=[3,1], sharey=True), (step_full, step_ins)):
    ax.plot(data.times[step], data[ch][step], '0.6')
    ax.set_ylabel('mV')
    ax.set_xlim(data.times.min(), data.times.max())    
    
ax.set_xlim(t0, t0+plot_seconds)
figs[0].suptitle('Raw data')

# Highpassed (detrended)
data.process.highpass_filter(filter_order=2, frequency_cutoff=0.1)
for ax, step, sup_ax in zip(figs[1].subplots(1, 2,  width_ratios=[3,1], sharey=True), 
                            (step_full, step_ins), sup_axs):
    ax.plot(data.times[step], data[ch][step], '0.6')
    ax.set_ylabel('mV')
    ax.set_xlim(data.times.min(), data.times.max()) 
    
    sup_ax.plot(data.times[step], data[ch][step], c='0.6')
    
ax.set_xlim(t0, t0+plot_seconds)
ax.set_ylim(-9, 19.5)
figs[1].suptitle('Highpass filter (botterworth filter with 0.1Hz cutoff frequency) [sacamos el trend global, enderezamos]')

# Lowpassed (denosied)
data.process.lowpass_filter(filter_order=2, frequency_cutoff=10)
for ax, step, sup_ax in zip(figs[2].subplots(1, 2,  width_ratios=[3,1], sharey=True),
                    (step_full, step_ins), sup_axs):
    ax.plot(data.times[step], data[ch][step], '0.6')
    ax.set_ylabel('mV')
    ax.set_xlim(data.times.min(), data.times.max())
    
    sup_ax.plot(data.times[step], data[ch][step], c='k')
    
ax.set_xlim(t0, t0+plot_seconds)
ax.set_ylim(-9, 19.5)
figs[2].suptitle('Lowpass filter (botterworth filter with 10Hz cutoff frequency) [limpiamos spikes y un poco de ruido]')

# Lowpassed (smoothed)
# data = data.process.downsample()
data.process.lowpass_filter(filter_order=2, frequency_cutoff=2)
data.process.find_peaks(period_percent=0.4, prominence=3)
for ax, step, sup_ax in zip(figs[3].subplots(1, 2,  width_ratios=[3,1], sharey=True),
                    (step_full, step_ins), sup_axs):
    ax.plot(data.times[step], data[ch][step], '0.6')
    ax.set_ylabel('mV')
    ax.set_xlim(data.times.min(), data.times.max())   
    
    sup_ax.plot(data.times[step], data[ch][step], c=lcolor)
    sup_ax.plot(data.process.get_peak_pos(channel), data.process.get_peak_values(inx=channel), '.', c=lcoloalt)

ax.set_xlim(t0, t0+plot_seconds)
ax.set_ylim(-9, 19.5)
figs[3].suptitle('Lowpass filter (botterworth filter with 2Hz cutoff frequency) [alisamos las curvas par aalgunos de los análisis]')

# Format axis with superimposed data
for sup_ax in sup_axs:
    sup_ax.set_xlabel('time [sec]')
    sup_ax.set_ylabel('mV')
    sup_ax.set_xlim(data.times.min(), data.times.max())
    
sup_ax.set_xlim(t0, t0+plot_seconds)
sup_ax.set_ylim(-9, 19.5)

figs[4].suptitle('Superimposed data with all fitlers applied')
    

fig.suptitle(f'{data.metadata.file.stem} (Ch{channel})')

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
data_dir = '/media/marcos/DATA/marcos/FloClock_data/data - mecamilamina'
save_dir = '/media/marcos/DATA/marcos/FloClock pics/mecamilamina/Trends'
out_dir = '/media/marcos/DATA/marcos/FloClock_data/data - mecamilamina/output/polynomial_trends'

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

data_dir = '/media/marcos/DATA//marcos/FloClock_data/data - mecamilamina'
poly_dir = Path(data_dir) / 'output' / 'polynomial_trends'

data_files = contenidos(poly_dir, filter_ext='.pickle')
pair_guide_file = Path(data_dir) / 'par_guide.xlsx' 

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

#%% Test polynomial detrending vs highpass filter

# Load data
data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
# data_dir = '/media/marcos/DATA//marcos/FloClock_data/data - mecamilamina'
file_inx = 2

data_files = contenidos(data_dir, filter_ext='.abf')
pair_guide_file = contenidos(data_dir, filter_ext='.xlsx').pop()

# create figure
fig, (top, mid, bot) = plt.subplots(3,2, sharex='col')

ax01, ax02 = top
ax1, ax2 = mid
ax3, ax4 = bot

# Process data a bit
data = au.load_data(data_files[file_inx], gauss_filter=False)
data.process.lowpass_filter(filter_order=2, frequency_cutoff=10, keep_og=True)

ax01.plot(data.times, data.ch1, c='0.7')
ax02.plot(data.times, data.ch2, c='0.7')
ax01.plot(data.times, data.ch1_lpfilt, c='C0')
ax02.plot(data.times, data.ch2_lpfilt, c='C1')

data = data.process.downsample()
data.process.poly_detrend(degree=5, keep_og=True, channels='lpfilt')
data.process.highpass_filter(filter_order=2, frequency_cutoff=0.1, keep_og=True, channels='lpfilt')
# data.process.gaussian_filter(sigma_ms=100, keep_og=True, channels='gfilt')


ax1.plot(data.times, data.ch1_lpfilt_pdetrend)
ax2.plot(data.times, data.ch2_lpfilt_pdetrend, c='C1')

ax3.plot(data.times, data.ch1_lpfilt_hpfilt)
ax4.plot(data.times, data.ch2_lpfilt_hpfilt, c='C1')

# plot trends
ax01.plot(data.times, data.process.get_hptrend(1), c='C3')
ax02.plot(data.times, data.process.get_hptrend(2), c='C3')

ax01.plot(data.times, data.process.get_trend(1), c='C4')
ax02.plot(data.times, data.process.get_trend(2), c='C4')


ax01.set_title('CH1')
ax02.set_title('CH2')

ax01.set_ylabel('raw')
ax1.set_ylabel('polynomial detrend')
ax3.set_ylabel('highpass')

#%% Single Lissajous figures

# Load data
data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
# data_dir = '/home/user/Documents/Doctorado/Fly clock/FlyClock_data/data/'
file_inx = 4

data_files = contenidos(data_dir, filter_ext='.abf')
pair_guide_file = contenidos(data_dir, filter_ext='.xlsx').pop()

# Process data a bit
data = au.load_data(data_files[file_inx], gauss_filter=False)
data.process.lowpass_filter(filter_order=2, frequency_cutoff=10, keep_og=True)
data = data.process.downsample()
# data.process.poly_detrend(degree=5, keep_og=True, channels='gfilt')
data.process.highpass_filter(filter_order=2, frequency_cutoff=0.1, keep_og=True, channels='lpfilt')

# Find direction of the blob
P = np.polynomial.Polynomial
nanlocs = np.isnan(data.ch1_lpfilt_hpfilt) | np.isnan(data.ch2_lpfilt_hpfilt)
fit_poly = P.fit(data.ch1_lpfilt_hpfilt[~nanlocs], data.ch2_lpfilt_hpfilt[~nanlocs], deg=1)
slope = fit_poly.convert().coef[1]

# PLot data
plt.plot(data.ch1, data.ch2)
plt.plot(data.ch1_lpfilt, data.ch2_lpfilt)
plt.plot(data.ch1_lpfilt_hpfilt, data.ch2_lpfilt_hpfilt)

plt.plot(data.ch1_lpfilt_hpfilt, fit_poly(data.ch1_lpfilt_hpfilt))

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
data_dir = '/media/marcos/DATA//marcos/FloClock_data/data'
file_inx = 2

data_files = contenidos(data_dir, filter_ext='.abf')
pair_guide_file = contenidos(data_dir, filter_ext='.xlsx').pop()

# Process data a bit
data = au.load_data(data_files[file_inx], gauss_filter=False)
data.process.lowpass_filter()
data = data.process.downsample()
# data.process.poly_detrend(degree=5)
# data.process.gaussian_filter(sigma_ms=100)
data.process.highpass_filter()
data.process.lowpass_filter(frequency_cutoff=2)
data.process.calc_phase_difference()
data.process.magnitude_detrend(keep_og=True)
data.process.calc_phase_difference(channels='mdetrend')

plt.figure(figsize=(18, 5))

ax1 = plt.subplot(2,2,1)
plt.plot(data.times, data.ch1, label='data')
plt.plot(data.times, data.ch1_magnitude, label='envelope')
plt.plot(data.times, data.ch1_mdetrend, label='data÷envelope')
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
data_dir = '/media/marcos/DATA//marcos/FloClock_data/data'
# data_dir = '/media/marcos/DATA//marcos/FloClock_data/data - mecamilamina'
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
data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
# data_dir = '/media/marcos/DATA/marcos/FloClock_data/data - mecamilamina'
file_inx = 4
outlier_mode_proportion = 1.8 #1.8 for normal runs

data_files = contenidos(data_dir, filter_ext='.abf')

# Process data a bit
data = au.load_data(data_files[file_inx], gauss_filter=False)
data.process.lowpass_filter(frequency_cutoff=10, keep_og=True)
data = data.process.downsample()
# data.process.poly_detrend(degree=5, channels='gfilt')
# data.process.gaussian_filter(sigma_ms=100, keep_og=True, channels='gfilt')
data.process.highpass_filter(frequency_cutoff=0.1, keep_og=True, channels='lpfilt')
data.process.lowpass_filter(frequency_cutoff=2, keep_og=True, channels='lpfilt_hpfilt')

data.process.find_peaks(channels='lpfilt_hpfilt_lpfilt', period_percent=0.4, prominence=3)

# Plot data and peaks

fig, (ax1, ax3, ax2, ax4) = plt.subplots(4, 1, figsize=(12, 8), constrained_layout=True, sharex=True, height_ratios=[2,1,2,1])

modes = []
trends = []
P = np.polynomial.Polynomial
for ax, ax_p, ch in zip((ax1, ax2), (ax3, ax4), (1, 2)):
    ax.plot(data.times, data[f'ch{ch}'] - data.process.get_hptrend(ch), color='0.6')
    ax.plot(data.times, data[f'ch{ch}_lpfilt_hpfilt'])
    ax.plot(data.times, data[f'ch{ch}_lpfilt_hpfilt_lpfilt'])

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
    ax_p.plot(period_times, periods, 'o', color='C02')
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
    
    # add period calculated through threshold crossings
    rising = data.process.get_crossings(ch, 'rising', 5, 0.5)
    falling = data.process.get_crossings(ch, 'falling', 5, 0.5)
    
    ax.plot(data.times.values[rising], data[f'ch{ch}_lpfilt_hpfilt_lpfilt'][rising], 'o', c='C3')
    ax.plot(data.times.values[falling], data[f'ch{ch}_lpfilt_hpfilt_lpfilt'][falling], 'o', c='C4')
    
    rising_times, rising_periods = data.process.get_edge_periods(ch, 'rising')
    falling_times, falling_periods = data.process.get_edge_periods(ch, 'falling')
    ax_p.plot(rising_times, rising_periods, 'x', c='C3')
    ax_p.plot(falling_times, falling_periods, '*', c='C4')

    # calculate average differences between period calculation types
    
    # since rising and falling edge sometimes skip cycles, we must find the peak that corresponds to each edge detected
    corresponding_pairs = []
    for i, ptime in enumerate(rising_times):
        distances = np.abs(period_times - ptime)
        closest_inx = np.argmin( distances )
        
        # average distance between peaks is in the order of 3
        # average distance between rising edge and peak is usually less than 1
        if distances[closest_inx] < 1:
            corresponding_pairs.append([i, closest_inx])
    
    # rising and falling edge will always have the same corresponding peak
    arp = [] # abs difference between rising and peaks
    afp = [] # abs difference between falling and peaks
    for edge_inx, peak_inx in corresponding_pairs:
        arp.append( np.abs(rising_periods[edge_inx] - periods[peak_inx] ))
        afp.append( np.abs(falling_periods[edge_inx] - periods[peak_inx] ))
    arp = np.mean(arp)
    afp = np.mean(afp)
    
    # falling and rising edge periods always have corresponding points
    afr = np.mean( np.abs(rising_periods - falling_periods))
    print(f'Ch{ch}')
    print(f'\t{afr = :.3f}')
    print(f'\t{arp = :.3f}')
    print(f'\t{afp = :.3f}')


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

ax1.set_title(f'Channel 1: {data.metadata.ch1}')
ax2.set_title(f'Channel 2: {data.metadata.ch2}')
ax3.set_title(f'Channel 1 period | mode = {modes[0]:.2f} sec')
ax4.set_title(f'Channel 2 period | mode = {modes[1]:.2f} sec')

ax3.legend(handles=[modeline, meanline, trendline], loc='upper left',
           labels=['mode', 'mean (no outliers)', f'slope={trends[0]*1000:.1f}e3'])
ax4.legend(handles=[trendline], loc='upper left', labels=[f'slope={trends[1]*1000:.1f}e3'])
    
print('Running', data.metadata.file.stem)


#%% Error of the time dependent period

# Load data
data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
# data_dir = '/media/marcos/DATA/marcos/FloClock_data/data - mecamilamina'
file_inx = 4
outlier_mode_proportion = 1.8 #1.8 for normal runs

data_files = contenidos(data_dir, filter_ext='.abf')

# Process data a bit
data = au.load_data(data_files[file_inx], gauss_filter=False)
data.process.lowpass_filter(frequency_cutoff=10, keep_og=True)
data = data.process.downsample()
# data.process.poly_detrend(degree=5, channels='gfilt')
# data.process.gaussian_filter(sigma_ms=100, keep_og=True, channels='gfilt')
data.process.highpass_filter(frequency_cutoff=0.1, keep_og=True, channels='lpfilt')
data.process.lowpass_filter(frequency_cutoff=2, keep_og=True, channels='lpfilt_hpfilt')

data.process.find_peaks(channels='lpfilt_hpfilt_lpfilt', period_percent=0.4, prominence=3)

# Plot data and peaks

fig, (ax1, ax3, ax2, ax4) = plt.subplots(4, 1, figsize=(12, 8), constrained_layout=True, sharex=True, height_ratios=[2,1,2,1])

modes = []
trends = []
P = np.polynomial.Polynomial
for ax, ax_p, ch in zip((ax1, ax2), (ax3, ax4), (1, 2)):
    ax.plot(data.times, data[f'ch{ch}'] - data.process.get_hptrend(ch), color='0.6')
    ax.plot(data.times, data[f'ch{ch}_lpfilt_hpfilt'])
    ax.plot(data.times, data[f'ch{ch}_lpfilt_hpfilt_lpfilt'])

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
    ax_p.plot(period_times, periods, 'o', color='C02')
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
    
    # add period calculated through threshold crossings
    # rising = data.process.get_crossings(ch, 'rising', 5, 0.5)
    falling = data.process.get_crossings(ch, 'falling', 5, 0.5)
    
    rising, multi_rising = data.process.get_multi_crossings(ch, 'rising', 
                                                            threshold=5, threshold_var=5, 
                                                            peak_min_distance=0.5)
    
    rr = multi_rising.flat[~np.isnan(multi_rising.flat)].astype(int)
    ax.plot(data.times.values[rr], data[f'ch{ch}_lpfilt_hpfilt_lpfilt'][rr], '.', c='k', ms=3)
        
    ax.plot(data.times.values[rising], data[f'ch{ch}_lpfilt_hpfilt_lpfilt'][rising], 'o', c='C3')
    ax.plot(data.times.values[falling], data[f'ch{ch}_lpfilt_hpfilt_lpfilt'][falling], 'o', c='C4')
    
    rising_times, rising_periods = data.process.get_edge_periods(ch, 'rising')
    falling_times, falling_periods = data.process.get_edge_periods(ch, 'falling')
    ax_p.plot(rising_times, rising_periods, 'x', c='C3')
    ax_p.plot(falling_times, falling_periods, '*', c='C4')
    
    mrising_out = data.process.get_multi_edge_periods(1, 'rising', threshold_var=5)
    mrising_times, mrising_periods, mrising_errors, mrising_ptp = mrising_out
    ax_p.errorbar(mrising_times, mrising_periods, mrising_errors, fmt='.', color='0.5')

    # calculate average differences between period calculation types
    
    # since rising and falling edge sometimes skip cycles, we must find the peak that corresponds to each edge detected
    corresponding_pairs = []
    for i, ptime in enumerate(rising_times):
        distances = np.abs(period_times - ptime)
        closest_inx = np.argmin( distances )
        
        # average distance between peaks is in the order of 3
        # average distance between rising edge and peak is usually less than 1
        if distances[closest_inx] < 1:
            corresponding_pairs.append([i, closest_inx])
    
    # rising and falling edge will always have the same corresponding peak
    arp = [] # abs difference between rising and peaks
    afp = [] # abs difference between falling and peaks
    for edge_inx, peak_inx in corresponding_pairs:
        arp.append( np.abs(rising_periods[edge_inx] - periods[peak_inx] ))
        afp.append( np.abs(falling_periods[edge_inx] - periods[peak_inx] ))
    arp = np.mean(arp)
    afp = np.mean(afp)
    
    # falling and rising edge periods always have corresponding points
    afr = np.mean( np.abs(rising_periods - falling_periods))
    print(f'Ch{ch}')
    print(f'\t{afr = :.3f}')
    print(f'\t{arp = :.3f}')
    print(f'\t{afp = :.3f}')


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

ax1.set_title(f'Channel 1: {data.metadata.ch1}')
ax2.set_title(f'Channel 2: {data.metadata.ch2}')
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
data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
# data_dir = '/home/user/Documents/Doctorado/Fly clock/FlyClock_data/data/'
# data_dir = '/media/marcos/DATA/marcos/FloClock_data/data - mecamilamina'
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

data_dir = Path('/media/marcos/DATA/marcos/FloClock_data/output')
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

# plt.savefig(f'/data/marcos/FloClock pics/{pair[0]} vs {pair[1]}')


#%% Correlate with distance between points

""" Correlates the distance between cells (Extracted from pictures) with the 
delay between the oscillations (the lags) as calculated by phase differences."""

base_dir = Path('/media/marcos/DATA/marcos/FloClock_data')
dist_dir = base_dir / 'todos los registros/pics'
output_dir = base_dir / 'output'
data_dir = base_dir / 'data'

dist_file = dist_dir / 'distances.csv'
phase_diff_file = output_dir / 'phase_differences.csv'
upstroke_file = output_dir / 'upstroke_delay_stats.dat'
pair_guide_file = data_dir / 'par_guide.xlsx'


#load data and save it all to a single dataframe
upstroke_stats = pd.read_csv(upstroke_file, sep=r'\s+').sort_values('#rec').reset_index(drop=True)
phase_diff = pd.read_csv(phase_diff_file).sort_values('name').reset_index(drop=True)
pair_guide = pd.read_excel(pair_guide_file, index_col=0)
distances  = pd.read_csv(dist_file)
stats = pd.concat( [phase_diff, upstroke_stats.loc[:, [ '1', '2', 'mean', 'std']].rename(columns={'mean': 'upstroke_mean', 'std':'upstroke_std'})], axis='columns')

# remove empty rows
for i, d in enumerate(distances.Distance):
    if np.isnan(d):
        distances.drop(labels=i, inplace=True)

# add the name of the files to the distances
distances = distances.astype({'Registro':int})
distances['name'] = [pair_guide.loc[rec]['name'] for rec in distances.Registro]


# add a few new columns
stats['type'] = [n[:2] for n in stats['name']]
stats['well_ordered'] = [ch1+ch2==t for ch1, ch2, t in zip(stats['1'], stats['2'], stats.type)]

for name in ['upstroke_mean', 'meanlag', 'modelag', 'K']:
    stats.loc[stats['well_ordered'], name ] *= -1
    
# add the info I want to plot to my distances df
distances['lag'] = [lag for lag, name in zip(stats.meanlag, stats['name']) if name in distances['name'].values]
distances['type'] = [n[:2] for n in distances['name']]

grouped = distances.groupby('type')
for key, group in grouped:
    plt.plot(group.lag, group['Scaled distance'], 'o', label=key)

plt.legend()
plt.xlabel('phase difference lag [sec]')
plt.ylabel('distance [soma radii]')

#%% Find peak density

from scipy import signal
from skimage import filters

from utils import sort_by

# Load data
# data_dir = '/data/marcos/FloClock_data/data'
data_dir = '/media/marcos/DATA/marcos/FloClock_data/data - mecamilamina'
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
data_dir = '/media/marcos/DATA/marcos/FloClock_data/data - mecamilamina'
save_dir = '/media/marcos/DATA/marcos/FloClock pics/mecamilamina/Peak prop densities'
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

data_dir = Path('/media/marcos/DATA/marcos/FloClock_data/data - mecamilamina')
peak_prop_dir = data_dir / 'output' / 'peak_property_densities' 
save_dir = '/media/marcos/DATA/marcos/FloClock pics/mecamilamina'

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
data_dir = '/media/marcos/DATA/marcos/FloClock_data/data - mecamilamina'
save_dir = '/media/marcos/DATA/marcos/FloClock pics/mecamilamina/After mec/raw'

file_inx = 6
plot_all = False # this means plot save and close
extension = 200 # in seconds after mec

data_files = contenidos(data_dir, filter_ext='.abf')
pair_guide_file = contenidos(data_dir, filter_ext='.xlsx').pop()

iter_over = data_files if plot_all else (data_files[file_inx], )
for i, file in enumerate(iter_over):
    print(f'Running {file.stem}', end='')
    if plot_all:
        print(f': {i+1}/{len(data_files)}')
    else:
        print('')
    
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
        

#%%Plot and save mec runs baselines

# Load data
# data_dir = '/data/marcos/FloClock_data/data'
data_dir = '/media/marcos/DATA/marcos/FloClock_data/data - mecamilamina'
save_dir = '/media/marcos/DATA/marcos/FloClock pics/mecamilamina/After mec/baselines'

file_inx = 3
plot_all = True # this means plot save and close
extension = 200 # in seconds after mec

data_files = contenidos(data_dir, filter_ext='.abf')
pair_guide_file = contenidos(data_dir, filter_ext='.xlsx').pop()

iter_over = data_files if plot_all else (data_files[file_inx], )
for i, file in enumerate(iter_over):
    print(f'Running {file.stem}', end='')
    if plot_all:
        print(f': {i+1}/{len(data_files)}')
    else:
        print('')
    
    # Process data a bit
    data = au.load_data(file, gauss_filter=False, override_raw=False)   
    data.process.lowpass_filter(frequency_cutoff=10, keep_og=True)
    data = data.process.downsample()
    data.process.highpass_filter(frequency_cutoff=0.1, keep_og=True, channels='lpfilt')
    
    # mininx, minima, fmininx, fminima = data.process.baseline_in_one_channel(data[f'ch{ch}_lpfilt'], drop_quantile=0.3)
    btimes, *baselines = data.process.multi_baseline('lpfilt', drop_quantile=0.3, length=20)
    
    fig = plt.figure(figsize=[15,  6.5], tight_layout=True)
    
    gs = fig.add_gridspec(2,2, height_ratios=[1,4])
    
    ax1, ax2 = [fig.add_subplot(gs[0, i]) for i in range(2)]
    
    gs2 = gs[1, :].subgridspec(2,1,height_ratios=[3,1])
    ax = fig.add_subplot(gs2[0])
    ax_trend = fig.add_subplot(gs2[1], sharex=ax)
    
    # channel 1
    ax1.plot(data.times, data.ch1, c='0.7')
    ax1.plot(data.times, data.ch1_lpfilt, c='C0')
    ax1.set_title(f'ch1: {data.metadata.ch1}')
    
    # channel 2
    ax2.plot(data.times, data.ch2, c='0.7')
    ax2.plot(data.times, data.ch2_lpfilt, c='C1')
    ax2.set_title(f'ch2: {data.metadata.ch2}')
    
    # combined
    mec_start = data.metadata.mec_start_sec
    start = find_point_by_value(data.times, mec_start)
    end = find_point_by_value(data.times, mec_start + extension)
    plot_range = slice(start, end)
    short_time = data.times[plot_range] - mec_start
    
    ax.plot(short_time, (data.ch1 - data.process.get_hptrend(1))[plot_range], c='0.7')
    ax.plot(short_time, (data.ch2 - data.process.get_hptrend(2))[plot_range], c='0.7')
    ax.plot(short_time, data.ch1_lpfilt_hpfilt[plot_range])
    ax.plot(short_time, data.ch2_lpfilt_hpfilt[plot_range], alpha=0.6)
    ax.set_ylim( max(ax.get_ylim()[0], -20), min( 40, ax.get_ylim()[1]) )
    
    ax_trend.set_xlim( 0, extension)
    ax_trend.set_xlabel('time sinsce mec start (sec)')
    
    ax.set_title('Both channels after mec start')
    
    # plot baselines
    ax_trend.plot(btimes - mec_start, baselines[0])
    ax_trend.plot(btimes - mec_start, baselines[1])
    ax_trend.plot(btimes - mec_start, baselines[0], 'o', c='C0')
    ax_trend.plot(btimes - mec_start, baselines[1], 'o', c='C1')
    
    ax1.plot(btimes, baselines[0], 'k')
    ax2.plot(btimes, baselines[1], 'k')

    for axi in [ax1, ax2]:
        mec_line = axi.axvline(mec_start, ls='--', c='k', label='mec start')
        axi.set_xlabel('time (sec)')
        axi.set_xlim(data.times.min(), data.times.max())
    
    if plot_all:
        plt.savefig(save_dir + f'/{data.metadata.file.stem}.png')
        plt.close(fig)
        
        np.savez(Path(data_dir) / 'output' / 'baselines' / f'{data.metadata.file.stem}',
                 time = btimes,
                 base1 = baselines[0],
                 base2 = baselines[1],
                 mec_start = mec_start,
                 pair = [data.metadata.ch1, data.metadata.ch2]
                 )
        
#%% Process mec baselines

""" Plot all the baselines by celltype. We use two ways of "normalizing" 
(vertically displacing) the baselines: either by the value they had in the 
moment of mec application, or by the average of the value they had before mec.
The latter aims at comparing the value after mec with those before mec.

Since there are some runs that are strong outliers from the central tendency,
we filter the outliers by calculating the mode and standard deviation and 
discarding all runs that spend 50% of the time at least one std away form the 
mode. Using the mean instead of the mode is highkly skwewed because of the 
outliers and low N. We calculate this over an interval that goes up to 10 
minutes after mec, since most runs include that range. Taking longer upper b
ounds is not so good, since for long times only a few recodings remain and the 
statiscis (mode or mean) closely resemble those. This in turn makes those runs
more likely to reach the required 50%, regardless of whether that specific run
is an outlier or not.
"""


data_dir = '/media/marcos/DATA/marcos/FloClock_data/data - mecamilamina/output/baselines'

baseline_files = contenidos(data_dir)

base_n1 = {'S':[], 'L':[]}
base_n2 = {'S':[], 'L':[]}
mintimes = np.inf
maxtimes = -np.inf

for file in baseline_files:
    loaded = np.load(file)
    
    time = loaded['time']
    baseline1 = loaded['base1']
    baseline2 = loaded['base2']
    mec_start = loaded['mec_start']
    pair = loaded['pair']
    
    # normalize the data
    for baseline, celltype in zip([baseline1, baseline2], pair):
        
        if celltype=='R':
            continue
        
        # normalize baseline
        mec_point = find_point_by_value(time, mec_start)
        normalizer1 = baseline[mec_point]
        normalizer2 = baseline[:mec_point+1].mean()
        baseline_n1 = baseline - normalizer1
        baseline_n2 = baseline - normalizer2
        
        time -= mec_start
                
        interpolator1 = interpolate.interp1d(time, baseline_n1, 
                                             bounds_error=False, fill_value=np.nan)
        interpolator2 = interpolate.interp1d(time, baseline_n2, 
                                             bounds_error=False, fill_value=np.nan)
        
        # save data for smoothing
        base_n1[celltype].append(interpolator1)
        base_n2[celltype].append(interpolator2)
        
        mintimes = min(mintimes, time.min())
        maxtimes = max(maxtimes, time.max())
        
# sort and convert concatenated data
maxtimes = min(600, maxtimes)
mintimes = max(-200, mintimes)
avg_times = np.linspace(mintimes, maxtimes, 1000)
cellcolor = {'S':scolor, 'L':lcolor}

fig, (axs, axl) = plt.subplots(2,2, sharex=True, sharey=True, figsize=[14.3 ,  7.08], constrained_layout=True)

for axli, axsi, bases in zip(axl, axs, (base_n1, base_n2)):
    axes = {'S':axsi, 'L': axli}
    for celltype, ax in axes.items():
        
        
        base_std = np.nanstd( [intp(avg_times) for intp in bases[celltype]], axis=0)
        base_mode = [calc_mode([intp(t) for intp in bases[celltype] if not np.isnan(intp(t))]) for t in avg_times] 
        
        # a run is "valid" (i.e counted when computing the average) if at least
        # 50% of the run lies no more than a std away from the mode at each timepoint
        valid = [i for i, b in enumerate(bases[celltype]) if np.mean(
            np.logical_and(b(avg_times) < base_mode+base_std, 
                           b(avg_times) > base_mode-base_std)[~np.isnan(b(avg_times))] 
            ) > 0 ]
        
        base_avg = np.nanmean( [intp(avg_times) for i, intp in enumerate(bases[celltype]) if i in valid], axis=0)
        base_std = np.nanstd( [intp(avg_times) for i, intp in enumerate(bases[celltype]) if i in valid], axis=0)
        
        for i, intp in enumerate(bases[celltype]):
            color = cellcolor[celltype] if i in valid else 'k'
            ax.plot(avg_times, intp(avg_times), color, alpha=0.7)
        
        ax.plot(avg_times, base_mode, 'w', path_effects=[
            pe.Stroke(linewidth=3, foreground='0.5'), pe.Normal()], label='mode')
        ax.plot(avg_times, base_avg, 'k', lw=2, label='mean')
        ax.fill_between(avg_times, base_avg-base_std, base_avg+base_std, color='0.5', label='CI')

for ax in (*axs, *axl):
    ax.axvline(0, color='k', ls='--')
    ax.axhline(0, color='k', ls='--')
    ax.set_xlim(mintimes, maxtimes)
    ax.set_ylabel('baseline [mV]')
    ax.grid(axis='y')
    ax.legend()
    
axs[0].set_title('Normalized at mec applied')
axs[1].set_title('Normalized over average before mec')
    
axl[0].set_xlabel('time since mec applied [sec]')
axl[1].set_xlabel('time since mec applied [sec]')

#%% Calculate and save all cross correlations

# Load data
data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
savedir = Path('/media/marcos/DATA/marcos/FloClock_data/data/output/cross_correlations')

keep = 10000 # how many points of the cross correlation to keep in each direction

data_files = contenidos(data_dir, filter_ext='.abf')
# pair_guide_file = contenidos(data_dir, filter_ext='.xlsx').pop()

for i, file in enumerate(data_files):
    print(f'Running {file.stem}: {i+1}/{len(data_files)}')
    
    # Process data a bit
    data = au.load_data(file, gauss_filter=False, override_raw=False)
    data.process.lowpass_filter(filter_order=2, frequency_cutoff=10)
    data = data.process.downsample()
    data.process.highpass_filter(filter_order=2, frequency_cutoff=0.1)
    data.process.lowpass_filter(filter_order=2, frequency_cutoff=2, keep_og=True)
    
    lags, corr = data.process.cross_correlation()
    lags2, corr2 = data.process.cross_correlation('lpfilt') # lags2 is equal to lags
    
    keep_slice = slice(int(len(corr)/2)-keep, int(len(corr)/2)+keep)
    np.savez(savedir / file.stem,
             lags = lags[keep_slice],
             corr = corr[keep_slice],
             corr2 = corr2[keep_slice],
             )
    
#%% Multi cross corr

# Load data
data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
savedir = Path('/media/marcos/DATA/marcos/FloClock_data/data/output/multi_cross_correlations')

data_files = contenidos(data_dir, filter_ext='.abf')
# pair_guide_file = contenidos(data_dir, filter_ext='.xlsx').pop()

for i, file in enumerate(data_files):
    print(f'Running {file.stem}: {i+1}/{len(data_files)}')
    
    # Process data a bit
    data = au.load_data(file, gauss_filter=False, override_raw=False)
    data.process.lowpass_filter(filter_order=2, frequency_cutoff=10)
    data = data.process.downsample()
    data.process.highpass_filter(filter_order=2, frequency_cutoff=0.1)
    data.process.lowpass_filter(filter_order=2, frequency_cutoff=2, keep_og=True)
    
    lengths = [2, 10, 20, 50]
    all_times, all_lags = {}, {}
    for length in lengths:
        times, lags = data.process.multi_cross_correlation_lag(length=length)
        
        all_times[f'times_{length}'] = times
        all_lags[f'lags_{length}'] = lags
    
    np.savez(savedir / file.stem,
             lengths = lengths,
             **all_times,
             **all_lags,
             )
    
#%% Plot just one cross corr

data_dir = Path('/media/marcos/DATA/marcos/FloClock_data/data/output/cross_correlations')
pair_guide_file = data_dir.parent.parent / 'par_guide.xlsx'

file_inx = 25

correlation_files = contenidos(data_dir)
pair_guide = pd.read_excel(pair_guide_file).sort_values('name', ignore_index=True).set_index('name', drop=True)

file = correlation_files[file_inx]
loaded = np.load(file)

lags = loaded['lags']
corr = loaded['corr']

fig, ax = plt.subplots()
ax.plot(lags, corr)
ax.set_xlim(-8, 8)
ax.set_title(file.stem)
ax.set_xlabel('lag [sec]')
ax.grid()
# ax.set_xlim(-2, 2)

#%% Plot just one multi cross corr

data_dir = Path('/media/marcos/DATA/marcos/FloClock_data/data/output/multi_cross_correlations')
pair_guide_file = data_dir.parent.parent / 'par_guide.xlsx'

file_inx = 45

correlation_files = contenidos(data_dir)
pair_guide = pd.read_excel(pair_guide_file).sort_values('name', ignore_index=True).set_index('name', drop=True)

for file in correlation_files:
    file = correlation_files[file_inx] #!!! comment this to see all
    loaded = np.load(file)
    
    lengths = loaded['lengths']
    times_names = [n for n in loaded.files if 'times' in n]
    lags_names = [n for n in loaded.files if 'lags' in n]
    
    
    fig, ax = plt.subplots(figsize=(6.4, 2.8), constrained_layout=True)
    for timename, lagname in zip(times_names, lags_names):
        times = loaded[timename]
        lags = loaded[lagname]
        
        length = find_numbers(timename)[0]
        
        ax.plot(times, lags, '.', ms=np.log2(length)*3, label=length)
    
    ax.set_title(file.stem)
    ax.set_xlabel('time [sec]')
    ax.set_ylabel('lag [sec]')
    ax.grid()
    ax.set_ylim(-1, 1)
    ax.legend(title='window length', ncol=len(times_names), loc='lower left')

    break
    fig.savefig('/media/marcos/DATA/marcos/FloClock pics/Xcorr/' + file.stem)
    plt.close(fig)

#%% Analize all cross correlations
import copy
 
# load data
data_dir = Path('/media/marcos/DATA/marcos/FloClock_data/data/output/cross_correlations')
pair_guide_file = data_dir.parent.parent / 'par_guide.xlsx'

correlation_files = contenidos(data_dir)
pair_guide = pd.read_excel(pair_guide_file).sort_values('name', ignore_index=True).set_index('name', drop=True)

correlations = dict(
    LL = [],
    LS = [],
    SS = [],
    LR = [],
    SR = [],
)

correlations2 = copy.deepcopy(correlations)

lags_df = pd.DataFrame(columns=['type', 'lag', 'lag2'])

# Extract data from the saved correlations
for corr_file in correlation_files:
    loaded = np.load(corr_file)
     
    lags = loaded['lags']
    corr = loaded['corr']
    corr2 = loaded['corr2']
    
    # check if we have to invert the correlation because of switched order of channels
    line = pair_guide.loc[corr_file.stem]
    if line.ch1 + line.ch2 != line.par:
        corr = corr[::-1]
        corr = corr2[::-1]
        
    correlations[line.par].append(corr)
    correlations2[line.par].append(corr2)
    
    lag = lags[np.argmax(corr)]
    lag2 = lags[np.argmax(corr2)]
    lags_df = pd.concat(
        [lags_df, 
         pd.DataFrame([[line.par, lag, lag2]], columns=lags_df.columns)], 
        axis=0)
           
# convert correlation data to arrays
correlations = {k:np.asarray(v) for k, v in correlations.items()}
correlations2 = {k:np.asarray(v) for k, v in correlations2.items()}

corr_mean = {k:v.mean(axis=0) for k, v in correlations.items()}
corr_std = {k:v.std(axis=0) for k, v in correlations.items()}

# plot all correlations
fig, axarr = plt.subplots(2,3, constrained_layout=True, figsize=[15, 7])
for ax, (kind, corr), std in zip(axarr.flat, corr_mean.items(), corr_std.values()):
    ax.plot(lags, corr)
    ax.fill_between(lags, corr+std, corr-std, color='0.7')
    
    ax.set_title(kind)
    ax.grid(axis='x')
    ax.set_xlabel('time [sec]')

# make boxplot
ax = axarr.flat[-1]
lags_df['condensed_type'] = [(v if 'R' not in v else 'R') for v in lags_df.type]
lags_df.boxplot('lag', by='condensed_type', ax=ax, grid=False)

# add points to boxplot
for i, (kind, points) in enumerate(lags_df.groupby('condensed_type')):
    ax.plot(np.random.normal(1+i, 0.04, size=len(points)), points.lag, 'k.')
    
ax.grid(axis='y')
ax.set_ylim(-0.3, 0.3)

# a few statistical tests
print('\n Difference from zero: p_t = students-t | p_w : wilcoxon | p_s : sign')
for kind, table in lags_df.groupby('condensed_type'):
    
    points = table.lag
    res_t = stats.ttest_1samp(points, popmean=0)
    res_w = stats.wilcoxon(points)
    res_s = stats.binomtest(sum(x>0 for x in points), len(points))
    ci = res_t.confidence_interval()
    
    print(f'{kind}\tp_t:{res_t.pvalue:.2e} \tp_w:{res_w.pvalue:.2e} \tp_s:{res_s.pvalue:.2e}', end='\t| ')
    print(f'mean = ({np.mean(points):.3f} ± [{ci.low:.3f}, {ci.high:.3f}])ms', end=' ')
    print(f' (n = {len(points)})')

# pairwise statistical tests
pairs = ['LL', 'SS'], ['LL', 'LS'], ['SS', 'LS']
print('\n Pariwise tests: p_t = students-t | p_w : mann-whitney')
for pair in pairs:
    first = lags_df.lag[lags_df.condensed_type==pair[0]]
    second = lags_df.lag[lags_df.condensed_type==pair[1]]
    
    res_t = stats.ttest_ind(first, second)
    res_w = stats.mannwhitneyu(first, second)
        
    print(f'{pair[0]} vs {pair[1]} :: p_t:{res_t.pvalue:.2e} \tp_w:{res_w.pvalue:.2e}')

interesting = ['SS', 'LL', 'LS']
res_k = stats.kruskal(*[lags_df.lag[lags_df.condensed_type==what] for what in interesting])
print('\nGlobal (Kruscak-Wallis): pval =', f'{res_k.pvalue:.3e}')


#%% Align Xcorrs with tpd

import copy
 
# load data
data_dir = Path('/media/marcos/DATA/marcos/FloClock_data/data/output/multi_cross_correlations')
pair_guide_file = data_dir.parent.parent / 'par_guide.xlsx'
tpd_file = data_dir.parent.parent.parent / 'tiempos_post_diseccion' / 'info.xlsx'

LENGTH = 10 # must be one of the saved ones. That's probably [2, 10, 20, 50]

correlations_files = contenidos(data_dir)
pair_guide = (pd.read_excel(pair_guide_file)
              .sort_values('rec', ignore_index=True)
              .set_index('name', drop=True)
              )
tpd_data = pd.concat(sheet for name, sheet in pd.read_excel(tpd_file, sheet_name=None).items() 
                     if name not in ('large', 'small')).sort_values('registro', ignore_index=True)
pair_guide['tpd'] = tpd_data['tiempo_post_diseccion (mins)'].values

lag_means = dict(
    SS = [],
    LL = [],
    LS = [],
    LR = [],
    SR = [],
)

# times = copy.deepcopy(lags)

plt.rcParams["axes.prop_cycle"] = plt.cycler("color", [*plt.cm.Dark2.colors, *plt.cm.tab10.colors])

fig, axdict = plt.subplot_mosaic([[k] for k in lag_means.keys() if 'R' not in k], sharex=True, sharey=True, constrained_layout=True, figsize=(6.4, 2.8))
fig2, axdict2 = plt.subplot_mosaic([[k] for k in lag_means.keys()], sharex=True, sharey=True, constrained_layout=True)

# Extract data from the saved correlations
for corr_file in correlations_files:
    loaded = np.load(corr_file)
    if LENGTH not in loaded['lengths']:
        raise ValueError(f"LENGTH should be one of {loaded['lengths']}")
     
    times = loaded[f'times_{LENGTH}']
    lags = loaded[f'lags_{LENGTH}']
    
    # check if we have to invert the correlation because of switched order of channels
    row = pair_guide.loc[corr_file.stem]
    
    if row.ch1 + row.ch2 != row.par:
        lags *= -1
    
    # plot individual cell distributions
    file_inx = find_numbers(corr_file.stem)[0]
    kde = stats.gaussian_kde(lags)
    max_val = kde(lags).max()
    
    axdict2[row.par].plot(np.random.normal(1+file_inx, kde(lags)/max_val/8, size=len(lags)), 
                         lags, 
                         '.', alpha=0.2, rasterized=True)

    lag_means[row.par].append(lags.mean())
    
    # plot evolution over time
    if 'R' in row.par:
        continue
    
    axdict[row.par].plot(times/60 + row.tpd, lags, '.')
    
    # lags[row.par].append(lags_i)
    # times[row.par].append(cross_points/60 + row.tpd)

# format TPD plots 
axdict['SS'].set_ylim(-0.4, 0.4)
axdict['SS'].set_title(f'Cross-correlation lags as time evolves with windows length={LENGTH} sec')
axdict['LS'].set_xlabel('time [min]')

for pairtype, ax in axdict.items():
    ax.legend(handles=[], title=pairtype, loc='upper right')
    ax.grid()
    ax.set_ylabel('lag [sec]')

# format distributions plots 
axdict2['SS'].set_ylim(-0.4, 0.4)
axdict2['SS'].set_title(f'Cross-correlation lags per cell with windows length={LENGTH} sec')
axdict2['SR'].set_xlabel('cell#')

for pairtype, ax in axdict2.items():
    ax.legend(handles=[], title=pairtype, loc='upper right')
    ax.grid()
    ax.set_ylabel('lag [sec]')

# restitute default colorcycle
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)

# pairwise statistical tests
pairs = ['LL', 'SS'], ['LL', 'LS'], ['SS', 'LS'], ['LS', 'LR'], ['LS', 'SR']
print('\n Pariwise tests: p_t = students-t | p_w : mann-whitney')
for pair in pairs:
    first, second = lag_means[pair[0]], lag_means[pair[1]]
    
    res_t = stats.ttest_ind(first, second)
    res_w = stats.mannwhitneyu(first, second)
        
    print(f'{pair[0]} vs {pair[1]} :: p_t:{res_t.pvalue:.2e} \tp_w:{res_w.pvalue:.2e}')



#%% Crossings exploration

# Load data
data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
# data_dir = '/media/marcos/DATA/marcos/FloClock_data/data - mecamilamina'
file_inx = 0
max_distance = 0.3 # in units of average period

data_files = contenidos(data_dir, filter_ext='.abf')

# Process data a bit
data = au.load_data(data_files[file_inx], gauss_filter=False)
data.process.lowpass_filter(filter_order=2, frequency_cutoff=10)
data = data.process.downsample()
data.process.highpass_filter(filter_order=2, frequency_cutoff=0.1)
data.process.lowpass_filter(filter_order=2, frequency_cutoff=2)

data.process.find_peaks(period_percent=0.4, prominence=3)
rising1 = data.process.get_crossings(1, 'rising', threshold=5, peak_min_distance=0.5)
rising2 = data.process.get_crossings(2, 'rising', threshold=5, peak_min_distance=0.5)

# plot example data
plt.subplot(2,1,1)
plt.plot(data.times, data.ch1, label=data.metadata.ch1)
plt.plot(data.times[rising1], data.ch1[rising1], '.', c='C2')
# plt.plot(data.times[filtered_pairs[:, 0]], data.ch1[rising1], '.', c='C2')

plt.plot(data.times, data.ch2, label=data.metadata.ch2)
plt.plot(data.times[rising2], data.ch2[rising2], '.', c='C3')

plt.title(data_files[file_inx].stem)
plt.legend()
# plt.xlim((74.32534052419356, 99.97859576612906))
plt.gcf().set_size_inches([14.56,  4.8 ])


filtered_pairs = []
lags = []
maxd = max_distance * np.mean(data.process.get_periods(1)[1])
for crossing in rising1:
    crossing_time = data.times.values[crossing]
    closest_time = find_closest_value(data.times.values[rising2], crossing_time)
    closest = find_closest_value(rising2, crossing)
    
    if np.abs(closest_time - crossing_time) > maxd:
        print('skipping')
        continue
    
    filtered_pairs.append((crossing, closest))   
    lags.append(crossing_time - closest_time)
    
    plt.plot([data.times[crossing], data.times[closest]], [data.ch1[crossing], data.ch2[closest]], 'k')
    
filtered_pairs = np.asarray(filtered_pairs)
lags = np.asarray(lags)

# plt.plot(data.times.values[filtered_pairs[:, 1]] + lags/2, np.ones(lags.shape)*5, 'x')

plt.subplot(4,1,3)
plt.plot(data.times.values[filtered_pairs[:, 0]], lags, 'o')
ylim = np.abs(plt.ylim()).max()
plt.ylim(-ylim, ylim)
plt.grid(axis='y')

plt.subplot(4,1,4)
ptimes, periods = data.process.get_periods(1)
plt.plot(ptimes, periods, 'o')
ptimes, periods = data.process.get_periods(2)
plt.plot(ptimes, periods, 'o')

# plt.ylim(0, None)
plt.grid(axis='y')

#%% Calculate and save all crossing lags

# Load data
data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
savedir = Path('/media/marcos/DATA/marcos/FloClock_data/data/output/crossings')
# data_dir = '/media/marcos/DATA/marcos/FloClock_data/data - mecamilamina'
max_distance = 0.4 # in units of average period

data_files = contenidos(data_dir, filter_ext='.abf')

# Process data a bit
for i, file in enumerate(data_files):
    print(f'Running {file.stem}: {i+1}/{len(data_files)}')
    
    # Process data a bit
    data = au.load_data(file, gauss_filter=False)
    data.process.lowpass_filter(filter_order=2, frequency_cutoff=10)
    data = data.process.downsample()
    data.process.highpass_filter(filter_order=2, frequency_cutoff=0.1)
    data.process.lowpass_filter(filter_order=2, frequency_cutoff=2)
    
    data.process.find_peaks(period_percent=0.4, prominence=3)
    rising1 = data.process.get_crossings(1, 'rising', threshold=5, peak_min_distance=0.5)
    rising2 = data.process.get_crossings(2, 'rising', threshold=5, peak_min_distance=0.5)
        
    filtered_pairs = []
    lags = []
    maxd = max_distance * np.mean(data.process.get_periods(1)[1])
    for crossing in rising1:
        crossing_time = data.times.values[crossing]
        closest_time = find_closest_value(data.times.values[rising2], crossing_time)
        closest = find_closest_value(rising2, crossing)
        
        if np.abs(closest_time - crossing_time) > maxd:
            print('skipping')
            continue
        
        filtered_pairs.append((crossing, closest))   
        lags.append(crossing_time - closest_time)
        
    filtered_pairs = np.asarray(filtered_pairs)
    lags = np.asarray(lags)
    tperiods1, periods1 = data.process.get_periods(1)
    tperiods2, periods2 = data.process.get_periods(2)
    
    np.savez(savedir / file.stem,
             lags = lags,
             crossings = data.times.values[filtered_pairs[:, 1]] + lags/2,
             tperiods1 = tperiods1,
             periods1 = periods1,
             tperiods2 = tperiods2,
             periods2 = periods2,
             )
    
#%% Analize crossings info
import copy
 
# load data
data_dir = Path('/media/marcos/DATA/marcos/FloClock_data/data/output/crossings')
pair_guide_file = data_dir.parent.parent / 'par_guide.xlsx'

crossings_files = contenidos(data_dir)
pair_guide = pd.read_excel(pair_guide_file).sort_values('name', ignore_index=True).set_index('name', drop=True)

lags = dict(
    LL = [],
    SS = [],
    LS = [],
    
    LR = [],
    SR = [],
)

crossings = copy.deepcopy(lags)

# figure for the individual cell distributions
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", [*plt.cm.Dark2.colors, *plt.cm.tab10.colors])
figd, axdict = plt.subplot_mosaic([[k] for k in lags.keys()], constrained_layout=True, sharex=True, sharey=True)

# Extract data from the saved correlations
for cross_file in crossings_files:
    loaded = np.load(cross_file)
     
    lag_points = loaded['lags']
    cross_points = loaded['crossings']
    
    # check if we have to invert the correlation because of switched order of channels
    row = pair_guide.loc[cross_file.stem]
    if row.ch1 + row.ch2 != row.par:
        lag_points *= -1
        
    lags[row.par].append(lag_points)
    crossings[row.par].append(cross_points)
    
    # plot individual runs
    file_inx = find_numbers(cross_file.stem)[0]
    kde = stats.gaussian_kde(lag_points)
    max_val = kde(lag_points).max()
    
    axdict[row.par].plot(np.random.normal(1+file_inx, kde(lag_points)/max_val/8, size=len(lag_points)), 
                         lag_points, 
                         '.', alpha=0.2, rasterized=True)

# format dsitribution data
axdict['SS'].set_ylim(-0.4, 0.4)
axdict['SS'].set_title('Crossings lags per cell')
axdict['SR'].set_xlabel('cell#')

for pairtype, ax in axdict.items():
    ax.legend(handles=[], title=pairtype, loc='upper right')
    ax.grid()
    ax.set_ylabel('lag [sec]')    


# convert correlation data to arrays
all_lags = {k:[x for array in v for x in array] for k, v in lags.items()}
all_crossings = {k:[x for array in v for x in array] for k, v in crossings.items()}

# fig, ax = plt.subplots()
# for i, ((kind, cross), lag) in enzip(all_crossings.items(), all_lags.values()):
#     ax.plot(cross, lag, '.', label=kind, c=f'C{i}')
#     ax.axhline(np.mean(lag), color=f'C{i}')
    
# ax.legend()


fig, axarr = plt.subplots(3, 2, constrained_layout=True, width_ratios=[4,1], sharey='row')
left = axarr[:, 0]
right = axarr[:, 1]

colors_dict = {'SS': '#d06c9eff', 'LS': '#1db17eff', 'LL': '#006680ff', 'LR':'0.3', 'SR':'0.3'}

for i, (ax, (kind, cross), lag) in enzip(left, crossings.items(), lags.values()):
    if 'R' in kind:
        continue
    for this_cross, this_lag in zip(cross, lag):
        ax.plot(this_cross, this_lag, '.', label=kind, c=colors_dict[kind], rasterized=True)

    ax.grid(axis='y')
    ax.legend(handles=[], title=kind, loc='upper right')
    ax.set_xlim(0, 500)
    ax.set_ylim(-0.5, 0.5)
fig.suptitle('Crossing lag trends')

for i, (ax, (kind, lag)) in enzip(right, all_lags.items()):
    if 'R' in kind:
        continue
    ax.hist(lag, orientation='horizontal', color=colors_dict[kind], bins=25)
    ax.grid(axis='y')

# boxplots
fig2, ax = plt.subplots(figsize=[3.98, 4.8 ])
order = ['LL', 'SS', 'LS']#, 'LR', 'SR']
ax.boxplot([all_lags[kind] for kind in order], showfliers=False)
ax.set_xticklabels(order)
ax.grid(axis='y')

for i, kind in enumerate(order):
    points = all_lags[kind]
    kde = stats.gaussian_kde(points)
    max_val = kde(points).max()
    
    ax.plot(np.random.normal(1+i, kde(points)/max_val/8, size=len(points)), points, 
            '.', c=colors_dict[kind], alpha=0.2, mec='none', zorder=1, rasterized=True)

# a few statistical tests
print('\n Difference from zero: p_t = students-t | p_w : wilcoxon | p_s : sign')
for kind, points in all_lags.items():
    
    res_t = stats.ttest_1samp(points, popmean=0)
    res_w = stats.wilcoxon(points)
    res_s = stats.binomtest(sum(x>0 for x in points), len(points))
    ci = res_t.confidence_interval()
    
    print(f'{kind}\tp_t:{res_t.pvalue:.2e} \tp_w:{res_w.pvalue:.2e} \tp_s:{res_s.pvalue:.2e}', end='\t| ')
    print(f'mean = ({np.mean(points):.3f} ± [{ci.low:.3f}, {ci.high:.3f}])ms', end=' ')
    print(f' (n = {len(points)})')


# pairwise statistical tests
pairs = ['LL', 'SS'], ['LL', 'LS'], ['SS', 'LS']
print('\n Pariwise tests: p_t = students-t | p_w : mann-whitney')
for pair in pairs:
    first, second = all_lags[pair[0]], all_lags[pair[1]]
    
    res_t = stats.ttest_ind(first, second)
    res_w = stats.mannwhitneyu(first, second)
        
    print(f'{pair[0]} vs {pair[1]} :: p_t:{res_t.pvalue:.2e} \tp_w:{res_w.pvalue:.2e}')

interesting = ['SS', 'LL', 'LS']
res_k = stats.kruskal(*[all_lags[what] for what in interesting])
print('\nGlobal (Kruscak-Wallis): pval =', f'{res_k.pvalue:.3e}')

#%% Align crossings with tpd

import copy
 
# load data
data_dir = Path('/media/marcos/DATA/marcos/FloClock_data/data/output/crossings')
pair_guide_file = data_dir.parent.parent / 'par_guide.xlsx'
tpd_file = data_dir.parent.parent.parent / 'tiempos_post_diseccion' / 'info.xlsx'

crossings_files = contenidos(data_dir)
pair_guide = (pd.read_excel(pair_guide_file)
              .sort_values('rec', ignore_index=True)
              .set_index('name', drop=True)
              )
tpd_data = pd.concat(sheet for name, sheet in pd.read_excel(tpd_file, sheet_name=None).items() 
                     if name not in ('large', 'small')).sort_values('registro', ignore_index=True)
pair_guide['tpd'] = tpd_data['tiempo_post_diseccion (mins)'].values

lags = dict(
    SS = [],
    LL = [],
    LS = [],
    LR = [],
    SR = [],
)

crossings = copy.deepcopy(lags)

# Extract data from the saved correlations
for cross_file in crossings_files:
    loaded = np.load(cross_file)
     
    lag_points = loaded['lags']
    cross_points = loaded['crossings']
    
    # check if we have to invert the correlation because of switched order of channels
    line = pair_guide.loc[cross_file.stem]
    if line.ch1 + line.ch2 != line.par:
        lag_points *= -1
    
    lags[line.par].append(lag_points)
    crossings[line.par].append(cross_points/60 + line.tpd)

# concatenate all data into single arrays
all_lags = {k:[x for array in v for x in array] for k, v in lags.items()}
all_crossings = {k:[x for array in v for x in array] for k, v in crossings.items()}

plt.rcParams["axes.prop_cycle"] = plt.cycler("color", [*plt.cm.Dark2.colors, *plt.cm.tab10.colors])
fig, axarr = plt.subplots(3, 2, constrained_layout=True, figsize=[6.6, 5],
                          width_ratios=[4,1], sharey='row', sharex='col')
# fig2, axdict = plt.subplot_mosaic([[k] for k in lag_means.keys() if 'R' not in k], sharex=True, sharey=True, constrained_layout=True, figsize=(6.4, 2.8))

left = axarr[:, 0]
right = axarr[:, 1]

P = np.polynomial.Polynomial
for i, (ax, (kind, cross), lag) in enzip(left, all_crossings.items(), all_lags.values()):
    if 'R' in kind:
        continue
    
    # axdict[kind].plot(cross, lag, '.', rasterized=True)

    trend = P.fit(cross, lag, deg=1)
    slope = trend.convert().coef[1]
    
    ax.plot(cross, lag, '.', label=kind, c=f'C{i}', rasterized=True)
    ax.plot( x := np.array([min(cross), max(cross)]), trend(x), 'k')
    
    ax.grid(axis='y')
    ax.legend(handles=[], title=f'{kind}: {slope=:.1e}', loc='upper right')
    
    # ax.set_xlim(0, 500)
    ax.set_ylim(-0.5, 0.5)
    ax.set_ylabel('lag [sec]')
    
    print()
    print(kind, '| Global slope:', f'{slope:.4f}')
    for this_cross, this_lag in zip(crossings[kind], lags[kind]):
        this_trend = P.fit(this_cross, this_lag, deg=1)
        ax.plot( x := np.array([min(this_cross), max(this_cross)]), trend(x), '0.5')
        print(f'{this_trend.convert().coef[1]:.4f}')
        
fig.suptitle('Crossing lag trends')

for i, (ax, (kind, lag)) in enzip(right, all_lags.items()):
    if 'R' in kind:
        continue
    ax.hist(lag, orientation='horizontal', color=f'C{i}', bins=25)
    ax.grid(axis='y')

axarr[-1, 0].set_xlabel('time + tpd [min]')
axarr[-1, 1].set_xlabel('counts')

# plt.figure()
# concat_cross = []
# concat_lag = []
# for i, ((kind, cross), lag) in enzip(all_crossings.items(), all_lags.values()):
#     if 'R' in kind:
#         continue
    
#     plt.plot(cross, lag - np.mean(lag), '.')
#     concat_cross.extend(cross)
#     concat_lag.extend( (np.asarray(lag) - np.mean(lag)) )

# global_trend = P.fit(concat_cross, concat_lag, deg=1)
# plt.plot(x := np.array([min(concat_cross), max(concat_cross)]), global_trend(x), 
#          'k', label=f'slope={global_trend.convert().coef[1]:.1e}')
# plt.legend()

#%% Crosssings & Xcorrs with tpd

import copy
 
# load data
xcorr_data_dir = Path('/media/marcos/DATA/marcos/FloClock_data/data/output/multi_cross_correlations')
cross_data_dir = Path('/media/marcos/DATA/marcos/FloClock_data/data/output/crossings')
pair_guide_file = xcorr_data_dir.parent.parent / 'par_guide.xlsx'
tpd_file = xcorr_data_dir.parent.parent.parent / 'tiempos_post_diseccion' / 'info.xlsx'

LENGTH = 10 # must be one of the saved ones. That's probably [2, 10, 20, 50]

# load both file types
crossings_files = contenidos(cross_data_dir)
correlations_files = contenidos(xcorr_data_dir)

# load pair info
pair_guide = (pd.read_excel(pair_guide_file)
              .sort_values('rec', ignore_index=True)
              .set_index('name', drop=True)
              )

# load tpd data
tpd_data = pd.concat(sheet for name, sheet in pd.read_excel(tpd_file, sheet_name=None).items() 
                     if name not in ('large', 'small')).sort_values('registro', ignore_index=True)
pair_guide['tpd'] = tpd_data['tiempo_post_diseccion (mins)'].values

# randomly (but deterministically) shuffle the file order
pair_guide = pair_guide.sample(frac=1, random_state=2)

# build figures
order = 'SS', 'LL', 'LS', 'SR', 'LR'
# plt.rcParams["axes.prop_cycle"] = plt.cycler("color", [*plt.cm.Dark2.colors, *plt.cm.tab10.colors])

fig_C, axdictC = plt.subplot_mosaic([[k] for k in order if 'R' not in k], sharex=True, sharey=True, constrained_layout=True, figsize=(6.4, 2.8*1.7))
fig_X, axdictX = plt.subplot_mosaic([[k] for k in order if 'R' not in k], sharex=True, sharey=True, constrained_layout=True, figsize=(6.4, 2.8*1.7))

fig_Ci, axdictCi = plt.subplot_mosaic([[k] for k in order], sharex=True, sharey=True, constrained_layout=True, figsize=(6.4, 4.3*1.7))
fig_Xi, axdictXi = plt.subplot_mosaic([[k] for k in order], sharex=True, sharey=True, constrained_layout=True, figsize=(6.4, 4.3*1.7))

# make colormaps
sc = np.asarray(colors.to_rgba(scolor))
lc = np.asarray(colors.to_rgba(lcolor))
pc = np.asarray(colors.to_rgba(pcolor))

wp = 0.7 # how much white to mix into the color
bp = 0.7 # how much black to mix into the color
black = np.array((0,0,0,1))
white = np.array((1,1,1,1))

scmap = colors.LinearSegmentedColormap.from_list('small', (sc*(1-bp) + black * bp, sc, sc*(1-wp) + white * wp))
lcmap = colors.LinearSegmentedColormap.from_list('large', (lc*(1-bp) + black * bp, lc, lc*(1-wp) + white * wp))
pcmap = colors.LinearSegmentedColormap.from_list('pair' , (pc*(1-bp) + black * bp, pc, pc*(1-wp) + white * wp))
rcmap = colors.LinearSegmentedColormap.from_list('rand' , (str(1-bp), str(wp)))

srcmap = colors.LinearSegmentedColormap.from_list('large', (sc, sc))
lrcmap = colors.LinearSegmentedColormap.from_list('large', (lc, lc))

cmaps = {'SS':scmap, 'LL':lcmap, 'LS':pcmap, 'SR':srcmap, 'LR':lrcmap}

# Extract data from the saved correlations
indexes = {k:0 for k in order} # save the index of the current file of each type
count_by_type = pair_guide.par.value_counts()
for fname, row in pair_guide.iterrows():
    
    # load data
    xcorr_loaded = np.load((xcorr_data_dir / fname).with_suffix('.npz'))
    cross_loaded = np.load((cross_data_dir / fname).with_suffix('.npz'))
    
    if LENGTH not in xcorr_loaded['lengths']:
        raise ValueError(f"LENGTH should be one of {loaded['lengths']}")
     
    xtimes = xcorr_loaded[f'times_{LENGTH}']
    xlags = xcorr_loaded[f'lags_{LENGTH}']
    
    ctimes = cross_loaded['crossings']
    clags = cross_loaded['lags']    
    
    if row.ch1 + row.ch2 != row.par:
        xlags *= -1
        clags *= -1
        
    # the color
    file_inx = indexes[row.par]
    indexes[row.par] += 1
    
    color = cmaps[row.par](file_inx / (count_by_type[row.par]-1))
    
    # plot individual cell distributions
    kdeX = stats.gaussian_kde(xlags)
    kdeC = stats.gaussian_kde(clags)
    max_valX = kdeX(xlags).max()
    max_valC = kdeC(clags).max()
    
    
    axdictXi[row.par].plot(np.random.normal(1+file_inx, kdeX(xlags)/max_valX/8, size=len(xlags)), 
                         xlags, 
                         '.', alpha=0.2, rasterized=True, c=color)
    axdictCi[row.par].plot(np.random.normal(1+file_inx, kdeC(clags)/max_valC/8, size=len(clags)), 
                         clags, 
                         '.', alpha=0.2, rasterized=True, c=color)
    
    # limsC = [f'{x:.1e}' for x in axdictCi[row.par].get_xlim()]
    # limsX = [f'{x:.1e}' for x in axdictXi[row.par].get_xlim()]
    # print(row.par, file_inx, limsC, limsX)
    
    # plot evolution over time
    if 'R' in row.par:
        continue    
    
    axdictX[row.par].plot(xtimes/60 + row.tpd, xlags, '.', c=color)
    axdictC[row.par].plot(ctimes/60 + row.tpd, clags, '.', c=color)

    if fname in ('LL01', 'LS10', 'SS02'):
        print(fname, file_inx)

# Some formatting
axlim = 0.7 # vertical axis limit
titles = f'Cross-correlation lags as time evolves with windows length={LENGTH} sec', 'Crossings lag as time evolves'
for title, axdict in zip(titles, [axdictX, axdictC]):
    axdict['SS'].set_ylim(-axlim, axlim)
    axdict['SS'].set_title(title)
    axdict['LS'].set_xlabel('time [min]')    
    
    for pairtype, ax in axdict.items():
        ax.legend(handles=[], title=pairtype, loc='upper right')
        ax.grid()
        ax.set_ylabel('lag [sec]')

titles = f'Cross-correlation lags in individual recs. with windows length={LENGTH} sec', 'Crossings lag in individual recs.'
for title, axdict in zip(titles, [axdictXi, axdictCi]):
    axdict['SS'].set_ylim(-axlim, axlim)
    axdict['SS'].set_xlim(0, count_by_type.max()+1)
    axdict['SS'].set_title(title)
    axdict['LR'].set_xlabel('time [min]')    
    
    for pairtype, ax in axdict.items():
        ax.legend(handles=[], title=pairtype, loc='upper right')
        ax.grid()
        ax.set_ylabel('lag [sec]')
        
# print('saving...')

# savedir = Path('/media/marcos/DATA/marcos/FloClock pics/svgs for paper/fig S4 material V4')
# savename = savedir / 'Xcorr lags over time.svg'
# fig_X.savefig(savename, dpi=300)
# savename = savedir / 'crossing lags over time.svg'
# fig_C.savefig(savename, dpi=300)
# savename = savedir / 'Xcorr lags individuals.svg'
# fig_Xi.savefig(savename, dpi=300)
# savename = savedir / 'crossing lags individuals.svg'
# fig_Ci.savefig(savename, dpi=300)

#%% Hist and corr of the means

# load data
# xcorr_data_dir = Path('/media/marcos/DATA/marcos/FloClock_data/data/output/multi_cross_correlations')
# cross_data_dir = Path('/media/marcos/DATA/marcos/FloClock_data/data/output/crossings')

xcorr_data_dir = Path('/home/user/Documents/Doctorado/Fly clock/FlyClock_data/data/output/multi_cross_correlations')
cross_data_dir = Path('/home/user/Documents/Doctorado/Fly clock/FlyClock_data/data/output/crossings')


pair_guide_file = xcorr_data_dir.parent.parent / 'par_guide.xlsx'
tpd_file = xcorr_data_dir.parent.parent.parent / 'tiempos_post_diseccion' / 'info.xlsx'

LENGTH = 10 # must be one of the saved ones. That's probably [2, 10, 20, 50]

# load both file types
crossings_files = contenidos(cross_data_dir)
correlations_files = contenidos(xcorr_data_dir)

# load pair info
pair_guide = (pd.read_excel(pair_guide_file)
              .sort_values('rec', ignore_index=True)
              .set_index('name', drop=True)
              )

# load tpd data
tpd_data = pd.concat(sheet for name, sheet in pd.read_excel(tpd_file, sheet_name=None).items() 
                     if name not in ('large', 'small')).sort_values('registro', ignore_index=True)
pair_guide['tpd'] = tpd_data['tiempo_post_diseccion (mins)'].values

# randomly (but deterministically) shuffle the file order
pair_guide = pair_guide.sample(frac=1, random_state=2)

# build figures
order = 'SS', 'LL', 'LS'
color_dict = {'SS':scolor, 'LL':lcolor, 'LS':pcolor}
# plt.rcParams["axes.prop_cycle"] = plt.cycler("color", [*plt.cm.Dark2.colors, *plt.cm.tab10.colors])

figH, axarr = plt.subplots(3,2, sharex=True, sharey=True)

# Extract data from the saved correlations
lagsX = {k:[] for k in order}
lagsC = {k:[] for k in order}
for fname, row in pair_guide.iterrows():
    
    if 'R' in row.par:
        continue    
    
    # load data
    xcorr_loaded = np.load((xcorr_data_dir / fname).with_suffix('.npz'))
    cross_loaded = np.load((cross_data_dir / fname).with_suffix('.npz'))
    
    if LENGTH not in xcorr_loaded['lengths']:
        raise ValueError(f"LENGTH should be one of {loaded['lengths']}")
     
    xtimes = xcorr_loaded[f'times_{LENGTH}']
    xlags = xcorr_loaded[f'lags_{LENGTH}']
    
    ctimes = cross_loaded['crossings']
    clags = cross_loaded['lags']    
    
    if row.ch1 + row.ch2 != row.par:
        xlags *= -1
        clags *= -1
    
    lagsX[row.par].append(xlags.mean())
    lagsC[row.par].append(clags.mean())

# plot histograms
for (axX, axC), pair in zip(axarr, order):
    axX.hist(lagsX[pair], fc=color_dict[pair], label=pair)
    axC.hist(lagsC[pair], fc=color_dict[pair])
    
    axX.legend()

# format plot
axarr[0, 0].set_title('X-corr mean lags')
axarr[0, 1].set_title('crossings mean lags')

axarr[-1, 0].set_xlabel('lag [sec]')
axarr[-1, 1].set_xlabel('lag [sec]')

for ax in axarr[:, 0]:
    ax.set_ylabel('#counts')

fig2, ax = plt.subplots(figsize=(5,5), constrained_layout=True)

pearson_rs = {}
for pair in order:
    ax.plot(lagsC[pair], lagsX[pair], 'o', c=color_dict[pair])
    pearson_r, _ = stats.pearsonr(lagsC[pair], lagsX[pair])
    print(f'Pearson-r of {pair}: {pearson_r:.2f}')

ax.set_xlabel('Threhsold crossing lag')
ax.set_ylabel('Cross correlation lag')

ax.plot([-1, 1], [-1, 1], '--k', zorder=1)

ax.set_aspect('equal', 'box')
ax.set_xlim(-0.33, 0.05)
ax.set_ylim(-0.33, 0.05)

all_lagsX = [l for lags in lagsX.values() for l in lags]
all_lagsC = [l for lags in lagsC.values() for l in lags]
pearson_r, _ = stats.pearsonr(all_lagsC, all_lagsX)

print(f'\n Global pearson-r: {pearson_r}')

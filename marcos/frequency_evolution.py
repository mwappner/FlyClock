#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 18:05:25 2023

@author: marcos
"""

# import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm, colors
import pandas as pd
import numpy as np
from scipy import interpolate
from scipy import stats
from statsmodels.stats.weightstats import DescrStatsW

from utils import contenidos, find_point_by_value, calc_mode, sort_by, cprint, clear_frame, enzip
import analysis_utils as au

scolor = '#d06c9eff'
lcolor = '#006680ff'
#%% Visualzie raw data

import pyabf

# Load data
data_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
file_inx = 9

data_files = contenidos(data_dir, filter_ext='.abf')

file = data_files[file_inx]
abf = pyabf.ABF(file)

chcount = abf.channelCount

# create figure
fig, axarr = plt.subplots(chcount, 2, figsize=[17, chcount*2.5+1], 
                          sharex='col', width_ratios=[3,1], sharey=True)

for ch_i, ax_pair in zip(abf.channelList,  np.atleast_2d(axarr)):

    # extract data
    abf.setSweep(0, channel=ch_i)
    times = abf.sweepX
    ch = abf.sweepY
      
    ax_left, ax_right = ax_pair
    ax_left.plot(times[::100], ch[::100], lw=0.5, c='0.6')
    
    plt_slice = slice(1405000,1480000,10)
    ax_right.plot(times[plt_slice], ch[plt_slice], lw=0.5, c='0.6')
    
    ax_left.vlines([times[plt_slice.start], times[plt_slice.stop]], 
                   ch.min(), ch.max(), colors='k', linestyles='dashed')

    au.make_scalebar(ax_left)
    au.make_scalebar(ax_right)
    
fig.suptitle(file.name)

print('file:', file.stem)
print('duration:', f'{times[-1]/60:.3f}')
print('sampling rate:', abf.sampleRate)

#%% Testing step size for plotting

import pyabf

# Load data
data_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
file_inx = 9

data_files = contenidos(data_dir, filter_ext='.abf')

file = data_files[file_inx]
abf = pyabf.ABF(file)

chcount = abf.channelCount

# create figure
fig, ax = plt.subplots(figsize=[8, 4])

# extract data
abf.setSweep(0, channel=1)
times = abf.sweepX
ch = abf.sweepY

steps = 1, 5, 10, 20, 50, 100
# steps = 1, 10
for step in steps:
    plt_slice = slice(1500000,1570000,step)
    # plt_slice = slice(None, None, step)
    ax.plot(times[plt_slice], ch[plt_slice], lw=0.4, label=step)

plt.legend()
fig.suptitle(file.name)

#%% Time dependent period in one file

# Load data
# data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
data_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
file_inx = 9
outlier_mode_proportion = 1.8 #1.8 for normal runs

data_files = contenidos(data_dir, filter_ext='.abf')

# data containing relevant intervals
tpd_file = Path(data_dir) / 'info.xlsx'
tpd_data = pd.concat(sheet for name, sheet in pd.read_excel(tpd_file, sheet_name=None).items() 
                     if name in ('large', 'small'))
tpd_data['registro'] = tpd_data.registro.astype(str)
tpd_data = tpd_data.set_index('registro')

rec_nr = data_files[file_inx].stem
interval = tpd_data.loc[rec_nr]['rango_de_minutos'] if rec_nr in tpd_data.index else None

# Process data a bit
data = au.load_any_data(data_files[file_inx], gauss_filter=False)
data.process.lowpass_filter(frequency_cutoff=10, keep_og=True)
data = data.process.downsample()
data.process.highpass_filter(frequency_cutoff=0.1, keep_og=True, channels='lpfilt')
data.process.lowpass_filter(frequency_cutoff=2, keep_og=True, channels='lpfilt_hpfilt')

data.process.find_peaks(channels='lpfilt_hpfilt_lpfilt', period_percent=0.4, prominence=3)

# Plot data and peaks
fig, (ax1, ax3, ax2, ax4) = plt.subplots(4, 1, figsize=(12, 8), constrained_layout=True, sharex=True, height_ratios=[2,1,2,1])

modes = []
trends = []
P = np.polynomial.Polynomial
for ax, ax_p, ch in zip((ax1, ax2), (ax3, ax4), (1, 2)):
    # ax.plot(data.times, data[f'ch{ch}'] - data.process.get_trend(ch), color='0.6')
    # ax.plot(data.times, data[f'ch{ch}_gfilt'])
    # ax.plot(data.times, data[f'ch{ch}_gfilt_gfilt2'])
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
    period_mode = calc_mode(periods)   
    # ax_p.plot(period_times, periods, 'o', color='C04')
    # ax_p.plot(period_times[periods > outlier_mode_proportion*period_mode], periods[periods > outlier_mode_proportion*period_mode], 'ro')
    
    # plot mode, mean and trend line
    valid_period_inxs = au.find_valid_periods(period_times, periods, outlier_mode_proportion, passes=2)
    valid_periods = periods[valid_period_inxs]
    period_mean = np.mean(valid_periods)
    modeline = ax_p.axhline(period_mode, color='0.3', linestyle='--', zorder=1)
    meanline = ax_p.axhline(period_mean, color='0.3', linestyle=':', zorder=1)
    
    trend_poly = P.fit(period_times[valid_period_inxs], valid_periods, 1)
    # trendline, = ax_p.plot(data.times, trend_poly(data.times), 'C00', zorder=0.9)
    # trends.append(trend_poly.convert().coef[1])
    
    # plot edge crossing periods
    # data.process.find_peaks(channels='lpfilt_hpfilt', period_percent=0.4, prominence=3)
    rising, multi_rising = data.process.get_multi_crossings(ch, 'rising', 
                                                            threshold=5, threshold_var=5, 
                                                            peak_min_distance=0.4)
    
    rr = multi_rising.flat[~np.isnan(multi_rising.flat)].astype(int)
    ax.plot(data.times.values[rr], data[f'ch{ch}_lpfilt_hpfilt_lpfilt'][rr], '.', c='k', ms=3)
     
    # "mrising" for multi-rising edge crossing
    mrising_out = data.process.get_multi_edge_periods(ch, 'rising',
                                                    threshold=5, threshold_var=5,
                                                    peak_min_distance=0.4)
    mrising_times, mrising_periods, mrising_errors, mrising_ptp = mrising_out
    ax_p.errorbar(mrising_times, mrising_periods, mrising_ptp, fmt='.', color='C0')
    
    # plot mode, mean and trend line for edge periods
    valid_period_inxs = au.find_valid_periods(mrising_times, mrising_periods, outlier_mode_proportion, passes=2)
    valid_periods = mrising_periods[valid_period_inxs]
    period_mean = np.mean(mrising_periods)
    period_mode = calc_mode(mrising_periods) 
    modeline = ax_p.axhline(period_mode, color='0.3', linestyle='--', zorder=1)
    meanline = ax_p.axhline(period_mean, color='0.3', linestyle=':', zorder=1)
    
    # plot average periods on the data
    mrising_indexes = np.asarray([find_point_by_value(data.times.values, tt) for tt in mrising_times-mrising_periods/2])
    ax.plot(data.times.values[mrising_indexes], data[f'ch{ch}_lpfilt_hpfilt_lpfilt'][mrising_indexes], '.', c='r')
    
    trend_poly = P.fit(mrising_times[valid_period_inxs], valid_periods, 1)
    trendline, = ax_p.plot(data.times, trend_poly(data.times), 'k', zorder=0.9)
    ax_p.plot(mrising_times[~valid_period_inxs], mrising_periods[~valid_period_inxs], 'xr')
    
    trends.append(trend_poly.convert().coef[1])
    modes.append(period_mode)

    if not data.metadata.twochannel:
        trends.append(trend_poly.convert().coef[1])
        modes.append(period_mode)
        break

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

#%% Paper: Time dependent period in one file for paper figure

# Load data
# data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
data_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
file_inx = 9
downsampling_rate = 100 # a factor of 10 is already applied, so this number is divided further down in the code
outlier_mode_proportion = 1.8 #1.8 for normal runs

data_files = contenidos(data_dir, filter_ext='.abf')

# data containing relevant intervals
tpd_file = Path(data_dir) / 'info.xlsx'
tpd_data = pd.concat(sheet for name, sheet in pd.read_excel(tpd_file, sheet_name=None).items() 
                     if name in ('large', 'small'))
tpd_data['registro'] = tpd_data.registro.astype(str)
tpd_data = tpd_data.set_index('registro')

rec_nr = data_files[file_inx].stem
interval = tpd_data.loc[rec_nr]['rango_de_minutos'] if rec_nr in tpd_data.index else None

# Process data a bit
data = au.load_any_data(data_files[file_inx], gauss_filter=False)
data.process.lowpass_filter(frequency_cutoff=10, keep_og=True)
data = data.process.downsample()
data.process.highpass_filter(frequency_cutoff=0.1, keep_og=True, channels='lpfilt')
data.process.lowpass_filter(frequency_cutoff=2, keep_og=True, channels='lpfilt_hpfilt')

data.process.find_peaks(channels='lpfilt_hpfilt_lpfilt', period_percent=0.4, prominence=3)

# Plot data and peaks
fig, (ax1, ax3, ax2, ax4) = plt.subplots(4, 1, figsize=(12, 8), constrained_layout=True, sharex=True, height_ratios=[2,1,2,1])

modes = []
trends = []
P = np.polynomial.Polynomial
step = slice(None, None, downsampling_rate//10)
for ax, ax_p, ch in zip((ax1, ax2), (ax3, ax4), (1, 2)):

    # plot timeseries        
    ax.plot(data.times, data[f'ch{ch}'] - data.process.get_hptrend(ch), color='0.6')
    ax.plot(data.times[step], data[f'ch{ch}_lpfilt_hpfilt'][step])
    ax.plot(data.times[step], data[f'ch{ch}_lpfilt_hpfilt_lpfilt'][step])

    ax.set_xlim(data.times.values[0], data.times.values[-1])

    # plot periods
    period_times, periods = data.process.get_periods(ch)
    period_mode = calc_mode(periods)   
    
    # plot mode, mean and trend line
    valid_period_inxs = au.find_valid_periods(period_times, periods, outlier_mode_proportion, passes=2)
    valid_periods = periods[valid_period_inxs]
    period_mean = np.mean(valid_periods)
    modeline = ax_p.axhline(period_mode, color='0.3', linestyle='--', zorder=1)
    meanline = ax_p.axhline(period_mean, color='0.3', linestyle=':', zorder=1)
    
    trend_poly = P.fit(period_times[valid_period_inxs], valid_periods, 1)
    
    # plot edge crossing periods
    rising, multi_rising = data.process.get_multi_crossings(ch, 'rising', 
                                                            threshold=5, threshold_var=5, 
                                                            peak_min_distance=0.4)
    
    
    # "mrising" for multi-rising edge crossing
    mrising_out = data.process.get_multi_edge_periods(ch, 'rising',
                                                    threshold=5, threshold_var=5,
                                                    peak_min_distance=0.4)
    mrising_times, mrising_periods, mrising_errors, mrising_ptp = mrising_out
    ax_p.errorbar(mrising_times, mrising_periods, mrising_ptp, fmt='.', color='C0')
    
    # plot mode, mean and trend line for edge periods
    valid_period_inxs = au.find_valid_periods(mrising_times, mrising_periods, outlier_mode_proportion, passes=2)
    valid_periods = mrising_periods[valid_period_inxs]
    period_mean = np.mean(mrising_periods)
    period_mode = calc_mode(mrising_periods) 
    modeline = ax_p.axhline(period_mode, color='0.3', linestyle='--', zorder=1)
    meanline = ax_p.axhline(period_mean, color='0.3', linestyle=':', zorder=1)
    
    # plot average periods on the data
    
    trend_poly = P.fit(mrising_times[valid_period_inxs], valid_periods, 1)
    trendline, = ax_p.plot(data.times, trend_poly(data.times), 'k', zorder=0.9)
    ax_p.plot(mrising_times[~valid_period_inxs], mrising_periods[~valid_period_inxs], 'xr')
    
    trends.append(trend_poly.convert().coef[1])
    modes.append(period_mode)

    if not data.metadata.twochannel:
        trends.append(trend_poly.convert().coef[1])
        modes.append(period_mode)
        break

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



#%% Baselines

# Load data
# data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
data_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
file_inx = 1
outlier_mode_proportion = 1.8 #1.8 for normal runs

data_files = contenidos(data_dir, filter_ext='.abf')

# data containing relevant intervals
tpd_file = Path(data_dir) / 'info.xlsx'
tpd_data = pd.concat(sheet for name, sheet in pd.read_excel(tpd_file, sheet_name=None).items() 
                     if name in ('large', 'small'))
tpd_data['registro'] = tpd_data.registro.astype(str)
tpd_data = tpd_data.set_index('registro')

rec_nr = data_files[file_inx].stem
interval = tpd_data.loc[rec_nr]['rango_de_minutos'] if rec_nr in tpd_data.index else None

# Process data a bit
data = au.load_any_data(data_files[file_inx], gauss_filter=False)
data.process.lowpass_filter(frequency_cutoff=10, keep_og=True)
data = data.process.downsample()
data.process.lowpass_filter(frequency_cutoff=2, keep_og=True, channels='lpfilt')

btimes, *baselines = data.process.multi_baseline('lpfilt', drop_quantile=0.3, length=20)

# Plot data and peaks
fig, (ax1, ax3, ax2, ax4) = plt.subplots(4, 1, figsize=(12, 8), constrained_layout=True, sharex=True, height_ratios=[2,1,2,1])

P = np.polynomial.Polynomial
for ax, ax_p, ch in zip((ax1, ax2), (ax3, ax4), (1, 2)):
    
    # plot timeseries    
    ax.plot(data.times, data[f'ch{ch}'], color='0.6')
    ax.plot(data.times, data[f'ch{ch}_lpfilt'])
    ax.plot(data.times, data[f'ch{ch}_lpfilt_lpfilt'])
    
    # plot baseline
    mininx, minima, fmininx, fminima = data.process.baseline_in_one_channel(data[f'ch{ch}_lpfilt'], drop_quantile=0.3)
    # ax.plot(data.times.values[mininx], minima, 'k.')
    # ax.plot(data.times.values[fmininx], fminima, 'r.')
    
    ax.plot(btimes, baselines[ch-1], 'o')
    ax.plot(btimes, baselines[ch-1])
    
    # set limits
    ax.set_xlim(data.times.values[0], data.times.values[-1])

    if not data.metadata.twochannel:
        break

fig.suptitle(data.metadata.file.stem)
ax4.set_xlabel('time (s)')

ax1.set_ylabel('mV')
ax2.set_ylabel('mV')
ax3.set_ylabel('period (s)')
ax4.set_ylabel('period (s)')

ax1.set_title('Channel 1')
ax2.set_title('Channel 2')
au.make_scalebar(ax2)

    
print('Running', data.metadata.file.stem)

#%% Extract time dependent periods

data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
tpd_file = Path(data_dir) / 'tabla_referencia_tiempo_post_diseccion.xlsx' #tpd = tiempo post disección
pair_guide_file = Path(data_dir) / 'par_guide.xlsx' 

file_inx = 13
outlier_mode_proportion = 1.6 #1.8 for normal runs

# Load data
data_files = contenidos(data_dir, filter_ext='.abf')
tpd_data = pd.read_excel(tpd_file).set_index('# referencia')
pair_guide = pd.read_excel(pair_guide_file).set_index('name')

print('Running', data_files[file_inx].stem)

# get tpd
rec_nr = pair_guide.loc[data_files[file_inx].stem].rec
tpd = tpd_data.loc[rec_nr]['TPD inicio'] # in minutes

# Process data a bit
data = au.load_data(data_files[file_inx], gauss_filter=True, override_raw=True)
data = data.process.downsample()
data.process.poly_detrend(degree=5)
data.process.gaussian_filter(sigma_ms=100)
data.process.find_peaks(period_percent=0.4, prominence=3)

# plot data
fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)

modes = []
trends = []
P = np.polynomial.Polynomial
for ch in (1, 2):
    
    # plot timeseries    
    peak_pos = data.process.get_peak_pos(ch)
    peak_val = data.process.get_peak_values(ch)
 
    # plot periods
    period_times, periods = data.process.get_periods(ch)
    period_mode = calc_mode(periods)
    ax.plot(period_times, periods, 'o', color=f'C0{ch}')
    ax.plot(period_times[periods > outlier_mode_proportion*period_mode], periods[periods > outlier_mode_proportion*period_mode], 'ro')
    
    # plot mode, mean and trend line
    valid_period_inxs = periods < outlier_mode_proportion*period_mode
    valid_periods = periods[valid_period_inxs]
    period_mean = np.mean(valid_periods)
    modeline = ax.axhline(period_mode, color='0.3', linestyle='--', zorder=1)
    meanline = ax.axhline(period_mean, color='0.3', linestyle=':', zorder=1)
    
    trend_poly = P.fit(period_times[valid_period_inxs], valid_periods, 1)
    trendline, = ax.plot(data.times, trend_poly(data.times), 'C00', zorder=0.9)
    trends.append(trend_poly.convert().coef[1])
    
    modes.append(period_mode)

if hasattr(data.metadata, 'mec_start_sec'):
    for ax in (ax1, ax3, ax2, ax4):
        mec_line = ax.axvline(data.metadata.mec_start_sec, ls='--', c='k', label='mec start')
    
    fig.legend(handles=[mec_line], ncol=6, loc='center right', bbox_to_anchor=(1, 0.97))    

fig.suptitle(data.metadata.file.stem)
ax.set_xlabel('time (s)')
ax.set_ylabel('period (s)')

# ax3.set_title(f'Channel 1 period | mode = {modes[0]:.2f} sec')
# ax4.set_title(f'Channel 2 period | mode = {modes[1]:.2f} sec')

# ax3.legend(handles=[modeline, meanline, trendline], loc='upper left',
#            labels=['mode', 'mean (no outliers)', f'slope={trends[0]*1000:.1f}e3'])
# ax4.legend(handles=[trendline], loc='upper left', labels=[f'slope={trends[1]*1000:.1f}e3'])


#%% Plot all period types (all runs)

# Load data
# data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
data_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
file_inx = 80
outlier_mode_proportion = 1.6 #1.8 for normal runs

data_files = contenidos(data_dir, filter_ext='.abf')
file = data_files[file_inx]
print('Running', file.stem)

# data containing relevant intervals
tpd_file = Path(data_dir) / 'info.xlsx'
tpd_data = pd.concat(sheet for name, sheet in pd.read_excel(tpd_file, sheet_name=None).items() 
                     if name in ('large', 'small'))
tpd_data['registro'] = tpd_data.registro.astype(str)
tpd_data = tpd_data.set_index('registro')

rec_nr = file.stem
interval = tpd_data.loc[rec_nr]['rango_de_minutos'] if rec_nr in tpd_data.index else None

# Process data a bit
data = au.load_any_data(file, gauss_filter=False, interval=interval)
data.process.lowpass_filter(frequency_cutoff=10, keep_og=False)
data = data.process.downsample()
data.process.highpass_filter(frequency_cutoff=0.1, keep_og=False)
data.process.lowpass_filter(frequency_cutoff=2, keep_og=False)

data.process.find_peaks( period_percent=0.4, prominence=3)

# Plot data and peaks

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True,)

modes = []
trends = []
P = np.polynomial.Polynomial
for ax, ch in zip((ax1, ax2), (1, 2)):
    period_times, periods = data.process.get_periods(ch)
    
    # plot periods
    valid_period_inxs = au.find_valid_periods(period_times, periods, passes=2)
    ax.plot(period_times, periods, 'o', color='C02')
    ax.plot(period_times[~valid_period_inxs], periods[~valid_period_inxs], 'ro')
    
    # plot mode, mean and trend line
    period_mean = np.mean(periods[valid_period_inxs])
    period_mode = calc_mode(period_times)
    modeline = ax_p.axhline(period_mode, color='0.3', linestyle='--', zorder=1)
    meanline = ax_p.axhline(period_mean, color='0.3', linestyle=':', zorder=1)
    
    trend_poly = P.fit(period_times[valid_period_inxs], periods[valid_period_inxs], 1)
    trendline, = ax_p.plot(data.times, trend_poly(data.times), 'C00', zorder=0.9)
    trends.append(trend_poly.convert().coef[1])
    
    modes.append(period_mode)
    
    # add period calculated through threshold crossings
    rising_times, rising_periods = data.process.get_edge_periods(ch, 'rising')
    falling_times, falling_periods = data.process.get_edge_periods(ch, 'falling')
    mrising_times, mrising_periods, mrising_errors = data.process.get_multi_edge_periods(1, 'rising', threshold_var=5)
    
    # plot edge crossing periods
    ax.plot(rising_times, rising_periods, 'x', c='C3')
    ax.plot(falling_times, falling_periods, '*', c='C4')
    ax.errorbar(mrising_times, mrising_periods, mrising_errors, fmt='.', color='C0')

    if not data.metadata.twochannel:    
        modes.append(period_mode)
        break

fig.suptitle(data.metadata.file.stem)
ax2.set_xlabel('time (s)')

ax1.set_ylabel('period (s)')
ax2.set_ylabel('period (s)')

ax1.set_title(f'Channel 1: {data.metadata.ch1} | mode = {modes[0]:.2f} sec')
ax2.set_title(f'Channel 2: {data.metadata.ch2} | mode = {modes[1]:.2f} sec')


#%% Get and save all period types (all runs)

# Load data
# data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
data_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
output_dir = Path(data_dir) / 'output' / 'frequency_time_dependency'
tpd_file = Path(data_dir) / 'info.xlsx'
pair_guide_file = Path(data_dir) / 'par_guide.xlsx' 

# load relevant info
data_files = contenidos(data_dir, filter_ext='.abf')
tpd_data = pd.concat(sheet for name, sheet in pd.read_excel(tpd_file, sheet_name=None).items())
                      # if name not in ('large', 'small'))
# tpd_data = pd.concat(pd.read_excel(tpd_file, sheet_name=None).values())
pair_guide = pd.read_excel(pair_guide_file)

# set rec names to string
tpd_data['registro'] = tpd_data.registro.astype(str)
tpd_data = tpd_data.set_index('registro')

pair_guide['rec'] = pair_guide.rec.astype(str)
pair_guide = pair_guide.set_index('rec')

output_info = {'name':[], 'type':[], 'ch':[], 'rec':[], 'og_name':[], 
               'tpd':[], 'duration':[], 'time_hours':[]}

data_files = data_files[40:]
# data_files = [f for f in data_files if f.stem in ['23602001', '23615000', '23615003']]
for i, file in enumerate(data_files):
    print('Running', f'{i+1}/{len(data_files)}:', file.stem, end='. ')
    
    # Get tpd
    rec_nr = file.stem

    if rec_nr not in tpd_data.index:# or pair_guide.loc[file.stem].par =='single':
        print('No tpd info available. Skipping...')
        continue
    else:
        print()
    tpd = tpd_data.loc[rec_nr]['tiempo_post_diseccion (mins)'] # in minutes
        
    # data containing relevant intervals
    rec_nr = file.stem
    interval = tpd_data.loc[rec_nr]['rango_de_minutos'] if rec_nr in tpd_data.index else None
    
    # Process data a bit
    data = au.load_any_data(file, gauss_filter=False, interval=interval)
    data.process.lowpass_filter(frequency_cutoff=10, keep_og=False)
    data = data.process.downsample()
    data.process.highpass_filter(frequency_cutoff=0.1, keep_og=False)
    data.process.lowpass_filter(frequency_cutoff=2, keep_og=False)
    
    data.process.find_peaks( period_percent=0.4, prominence=3)
    
    # Plot data and peaks
    for ch in (1, 2):
                
        cell_type = getattr(data.metadata, f'ch{ch}')
        if cell_type == 'R':
            continue
        
        period_times, periods = data.process.get_periods(ch)
            
        # add period calculated through threshold crossings
        falling_times, falling_periods = data.process.get_edge_periods(ch, 'falling')
        mrising_out = data.process.get_multi_edge_periods(ch, 'rising', threshold=5, threshold_var=5)
        mrising_times, mrising_periods, mrising_errors, mrising_ptp = mrising_out
        
        new_name = data.metadata.file.stem + f'_ch{ch}'
        np.savez(output_dir / new_name,
                  times_peaks = period_times,
                  periods_peaks = periods,
                  falling_times = falling_times,
                  falling_periods = falling_periods,
                  rising_times  = mrising_times,
                  rising_periods = mrising_periods, 
                  rising_std = mrising_errors,
                  rising_ptp = mrising_ptp,
                  
                  # set mrising periods as the default one to use (mor disck space, but it's still never a lot)
                  times = mrising_times,
                  periods = mrising_periods,
                  
                  )
        
        output_info['name'].append(new_name)
        output_info['type'].append(cell_type)
        output_info['ch'].append(ch)
        output_info['rec'].append(rec_nr)
        output_info['og_name'].append(data.metadata.file.stem)
        output_info['tpd'].append(tpd)
        output_info['duration'].append(data.metadata.duration_min)
        output_info['time_hours'].append(data.metadata.rec_datetime.hour + data.metadata.rec_datetime.minute/60)
        
        if not data.metadata.twochannel:    
            break

output_info_df = pd.DataFrame(output_info)
output_info_df.to_csv(output_dir / 'periods_info4.csv', index=False)


#%% Get and save baselines (all runs)

# Load data
# data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
data_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
output_dir = Path(data_dir) / 'output' / 'baseline_time_dependency'
tpd_file = Path(data_dir) / 'info.xlsx'
pair_guide_file = Path(data_dir) / 'par_guide.xlsx' 

# load relevant info
data_files = contenidos(data_dir, filter_ext='.abf')
tpd_data = pd.concat(sheet for name, sheet in pd.read_excel(tpd_file, sheet_name=None).items())
                      # if name not in ('large', 'small'))
# tpd_data = pd.concat(pd.read_excel(tpd_file, sheet_name=None).values())
pair_guide = pd.read_excel(pair_guide_file)

# set rec names to string
tpd_data['registro'] = tpd_data.registro.astype(str)
tpd_data = tpd_data.set_index('registro')

pair_guide['rec'] = pair_guide.rec.astype(str)
pair_guide = pair_guide.set_index('rec')

data_files = data_files[80:]
for i, file in enumerate(data_files):
    print('Running', f'{i+1}/{len(data_files)}:', file.stem, end='. ')
    
    # Get tpd
    rec_nr = file.stem

    if rec_nr not in tpd_data.index:# or pair_guide.loc[file.stem].par =='single':
        print('No tpd info available. Skipping...')
        continue
    else:
        print()
    
    # data containing relevant intervals
    rec_nr = file.stem
    interval = tpd_data.loc[rec_nr]['rango_de_minutos'] if rec_nr in tpd_data.index else None
    
    # Process data a bit
    data = au.load_any_data(file, gauss_filter=False, interval=interval)
    data.process.lowpass_filter(frequency_cutoff=10, keep_og=False)
    data = data.process.downsample()
        
    btimes, *baselines = data.process.multi_baseline(drop_quantile=0.3, length=20)

    # Save the relevant data (one or two channels)
    for ch in (1, 2):
                
        cell_type = getattr(data.metadata, f'ch{ch}')
        if cell_type == 'R':
            continue
        
        new_name = data.metadata.file.stem + f'_ch{ch}'
        np.savez(output_dir / new_name,
                  btimes = btimes,
                  baselines = baselines[ch-1]
                  )
    
        if not data.metadata.twochannel:    
            break

# output_info_df = pd.DataFrame(output_info)
# output_info_df.to_csv(output_dir / 'periods_info1.csv', index=False)


#%% Get and save time dependent periods (dual channel runs)

data_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
output_dir = Path(data_dir) / 'output' / 'frequency_time_dependency'
tpd_file = Path(data_dir) / 'info.xlsx'
pair_guide_file = Path(data_dir) / 'par_guide.xlsx' 

file_inx = 13

# Load data
data_files = contenidos(data_dir, filter_ext='.abf')
tpd_data = pd.concat(sheet for name, sheet in pd.read_excel(tpd_file, sheet_name=None).items() 
                     if name not in ('large', 'small'))
pair_guide = pd.read_excel(pair_guide_file)

# set rec names to string
tpd_data['registro'] = tpd_data.registro.astype(str)
tpd_data = tpd_data.set_index('registro')

pair_guide['rec'] = pair_guide.rec.astype(str)
pair_guide = pair_guide.set_index('rec')

output_info = {'name':[], 'type':[], 'ch':[], 'rec':[], 'og_name':[], 'tpd':[], 'time_hours':[]}

for file in data_files:
    print('Running', file.stem, end='. ')
    
    # Get tpd
    # rec_nr = pair_guide.loc[file.stem].rec
    rec_nr = file.stem
    
    if rec_nr not in tpd_data.index:# or pair_guide.loc[file.stem].par =='single':
        print('No tpd info available. Skipping...')
        continue
    elif pair_guide.loc[file.stem].par =='single':
        print('Skippping single channel file', file.stem)
        continue
    else:
        print()
    tpd = tpd_data.loc[rec_nr]['tiempo_post_diseccion (mins)'] # in minutes
        
    # Process data a bit
    data = au.load_data(file, gauss_filter=True, override_raw=True)
    data = data.process.downsample()
    data.process.poly_detrend(degree=5)
    data.process.gaussian_filter(sigma_ms=100)
    data.process.find_peaks(period_percent=0.4, prominence=3)
    
    # Extract periods
    for ch in (1, 2):
        
        cell_type = getattr(data.metadata, f'ch{ch}')
        if cell_type == 'R':
            continue
        
        # plot timeseries    
        peak_pos = data.process.get_peak_pos(ch)
        peak_val = data.process.get_peak_values(ch)
     
        # plot periods
        period_times, periods = data.process.get_periods(ch)
        
        new_name = data.metadata.file.stem + f'_ch{ch}'
        np.savez(output_dir / new_name,
                  times = period_times,
                  periods = periods
                  )
        
        output_info['name'].append(new_name)
        output_info['type'].append(cell_type)
        output_info['ch'].append(ch)
        output_info['rec'].append(rec_nr)
        output_info['og_name'].append(data.metadata.file.stem)
        output_info['tpd'].append(tpd)
        output_info['time_hours'].append(data.metadata.rec_datetime.hour + data.metadata.rec_datetime.minute/60)
        
    # break
output_info_df = pd.DataFrame(output_info)
output_info_df.to_csv(output_dir / 'periods_info.csv', index=False)


#%% Get and save time dependent periods (single channel runs)

data_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
output_dir = Path(data_dir) / 'output' / 'frequency_time_dependency'
tpd_file = Path(data_dir) / 'info.xlsx'
pair_guide_file = Path(data_dir) / 'par_guide.xlsx' 

file_inx = 13

# Load data
data_files = contenidos(data_dir, filter_ext='.abf')
tpd_data = pd.concat(sheet for name, sheet in pd.read_excel(tpd_file, sheet_name=None).items() 
                     if name in ('large', 'small'))
pair_guide = pd.read_excel(pair_guide_file)

# set rec names to string
tpd_data['registro'] = tpd_data.registro.astype(str)
tpd_data = tpd_data.set_index('registro')

pair_guide['rec'] = pair_guide.rec.astype(str)
pair_guide = pair_guide.set_index('rec')

output_info = {'name':[], 'type':[], 'ch':[], 'rec':[], 'og_name':[], 'tpd':[], 'time_hours':[]}

for file in data_files[90:]:
    print('Running', file.stem, end='. ')
    
    # Get tpd
    # rec_nr = pair_guide.loc[file.stem].rec
    rec_nr = file.stem
    
    if rec_nr not in tpd_data.index:# or pair_guide.loc[file.stem].par =='single':
        print('No tpd info available. Skipping...')
        continue
    elif pair_guide.loc[file.stem].par !='single':
        print('Skippping multi-channel file', file.stem)
        continue
    else:
        print()
    tpd = tpd_data.loc[rec_nr]['tiempo_post_diseccion (mins)'] # in minutes
        
    # Process data a bit
    data = au.load_single_channel(file, interval=tpd_data.loc[rec_nr]['rango_de_minutos'], 
                                  gauss_filter=True, override_raw=True)
    data = data.process.downsample()
    data.process.poly_detrend(degree=5)
    data.process.gaussian_filter(sigma_ms=100)
    data.process.find_peaks(period_percent=0.4, prominence=3)
    
    # Extract periods
    for ch in (1, ):
        
        cell_type = getattr(data.metadata, f'ch{ch}')
        if cell_type == 'R':
            continue
        
        # plot timeseries    
        peak_pos = data.process.get_peak_pos(ch)
        peak_val = data.process.get_peak_values(ch)
     
        # plot periods
        period_times, periods = data.process.get_periods(ch)
        
        new_name = data.metadata.file.stem + f'_ch{ch}'
        np.savez(output_dir / new_name,
                  times = period_times,
                  periods = periods
                  )
        
        output_info['name'].append(new_name)
        output_info['type'].append(cell_type)
        output_info['ch'].append(ch)
        output_info['rec'].append(rec_nr)
        output_info['og_name'].append(data.metadata.file.stem)
        output_info['tpd'].append(tpd)
        output_info['time_hours'].append(data.metadata.rec_datetime.hour + data.metadata.rec_datetime.minute/60)
        
output_info_df = pd.DataFrame(output_info)
output_info_df.to_csv(output_dir / 'periods_info_single2.csv', index=False)

#%% Get and save periods for all recordigns (1D incldued)

import pyabf
import re

data_dir = Path('/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion')
info_file = data_dir / 'info.xlsx'
pair_guide_file = data_dir / '..' / 'data' / 'par_guide.xlsx'

# find data and relevant info files
info = pd.read_excel(info_file, sheet_name=None)
pair_guide = pd.read_excel(pair_guide_file).set_index('rec')
files = contenidos(data_dir, filter_ext='.abf')


# process info file:
def process_line(line):
    if line == 'todo' or (not isinstance(line, str) and np.isnan(line)):
        return 0, np.nan
    else:
        return map(float, line.split('-'))
    
for name, sheet in info.items():
    # start, end = [], []
    # for line in sheet.rango_de_minutos:
    #     s, e = process_line(line)
    #     start.append(s)
    #     end.append(e)
    # sheet['start_minutes'] = start
    # sheet['end_minutes'] = end
    
    sheet['tpd'] = sheet['tiempo_post_diseccion (mins)']*60
    sheet.registro = sheet.registro.astype(str)
    sheet['celltype'] = np.full(sheet.registro.shape, name)
    
    
info = pd.concat([sheet.set_index('registro') for sheet in info.values()])    

# process data
i = 3
for file in files[i:i+1]:
    abf = pyabf.ABF(file)
    
    # check if we have info available on that run
    if file.stem not in info.index:
        print('Skipping', file.name)
        continue
    
    
    this_info = info.loc[file.stem]
    # get the tpd in seconds
    rango = this_info.rango_de_minutos
    if rango=='todo' or (not isinstance(rango, str) and np.isnan(rango)):
        start = 0
        end = -1
    else:
        start_min, end_min = map(float, rango.split('-'))
        start = int(start_min * 60 * abf.sampleRate)
        end = int(end_min * 60 * abf.sampleRate)
      
    interval = slice(start, end)
    
    # load data
    abf.setSweep(0)
    times = abf.sweepX[interval]
    data = [ abf.sweepY[interval] ]
    celltype = [this_info.cell_type]

    # load second channel if needed
    if abf.channelCount > 1:
        
        # grab data from pair info
        pair_info = pair_guide.loc[file.stem]
        
        # cell_type = ['small' if pair_info]
        
        # check the type
        
            
        abf.setSweep(0, channel=1)
        data.append(abf.sweepY[interval])
        

  

# %% Plot periods

# data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
data_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
data_dir = Path(data_dir) / 'output' / 'frequency_time_dependency'

outlier_mode_proportion = 1.8 #1.8 for normal runs

info_dir = data_dir / 'periods_info.csv'
data_files = contenidos(data_dir, filter_ext='.npz')

# load info
info = pd.read_csv(info_dir).sort_values('name').set_index('name')

# add tpd in seconds
info['tpd_sec'] = info.tpd * 60

# load and plot data
# fig, (top, mid, bot) = plt.subplots(3, 2, constrained_layout=True, 
fig, (top, mid) = plt.subplots(2, 2, constrained_layout=True, 
                               sharex='col', sharey='row', figsize=(12, 8))
ax1, ax2 = top
ax3, ax4 = mid
# ax5, ax6 = bot

norm = colors.Normalize(info.time_hours.min(), info.time_hours.max())
color_list = cm.viridis([norm(info.loc[file.stem].time_hours) for file in data_files])

# color_list = cm.jet(np.linspace(0, 1, len(data_files)+1))[1:]
P = np.polynomial.Polynomial
times = []
times_tpd = []
periods = []
times_valid = []
times_tpd_valid = []
periods_valid = []
tpds = []
durations = []
for file, c in zip(data_files, color_list):
    # load data
    data = np.load(file)
    t = data['times'] / 60
    p = data['periods']
    tpd = info.loc[file.stem].tpd
    
    durations.append(t[-1])
    small = info.loc[file.stem]['type'] == 'S'
    if not small:
        continue
    
    # plot points
    ax1.plot(t, p, '.', c=c)
    ax2.plot(t+tpd, p, '.', c=c)
    
    # plot trend lines
    # period_mode = calc_mode(p)
    # valid_indexes = p <= (period_mode * outlier_mode_proportion)
    valid_indexes = au.find_valid_periods(t, p, outlier_mode_proportion, passes=2)
    trend_poly = P.fit(t[valid_indexes], p[valid_indexes], 1)
    
    # # c = 'r' if trend_poly.convert().coef[1]>0 else 'b'
    # if trend_poly.convert().coef[1]<0: 
    #     print(file.stem, 'has slope', f'{trend_poly.convert().coef[1]:.1e}')
    
    ax3.plot(t, trend_poly(t), c=c)
    ax4.plot(t+tpd, trend_poly(t), c=c)
    
    # append data to lists
    periods.extend(p)
    times.extend(t)
    times_tpd.extend(t+tpd)
    tpds.append(tpd)
    
    times_valid.extend(t[valid_indexes])
    periods_valid.extend(p[valid_indexes])
    times_tpd_valid.extend(t[valid_indexes]+tpd)
    
    
durations_df = pd.DataFrame({'duration':durations, 
                             'name':list(data_files.stems())}).sort_values('name').set_index('name')

info['duration'] = durations_df.duration


# Add local variability
def calc_variability(t, x, step=20):
    """ Calculte the local variability of data in windows of size "step", in 
    units of t."""
    
    # sort the data by time
    x = sort_by(x, t)
    t = sorted(t)
    
    #calculate the variability
    steps_count = int(max(t)//step)
    steps = [find_point_by_value(t, step*i) for i in range(steps_count)]
    variability = [np.std(x[start:end]) for start, end in zip(steps, steps[1:])]
    times_variability = [np.mean(t[start:end]) for start, end in zip(steps, steps[1:])]
    
    return times_variability, variability

def calc_variability2(t, x, count=30):
    """ Calculate the local variability fo data by clumping in gropus of "count"
    datapoints."""
    
    # sort the data by time
    x = sort_by(x, t)
    t = sorted(t)
    
    steps = [slice(i*count, (i+1)*count) for i in range(len(t)//count)]
    variability = [np.std(x[interval]) for interval in steps]
    times_variability = [np.mean(t[interval]) for interval in steps]
    
    return times_variability, variability

# times_variability, variability = calc_variability(times, periods)
# ax5.plot(times_variability, variability, '.')

# times_variability, variability = calc_variability(times_tpd, periods)
# ax6.plot(times_variability, variability, '.')

# Add linear fit for all datapoints at once

def mse(y, y_fit):
    """ Mean square error: average of the square of the residuals of a fit"""
    return np.mean( (y-y_fit)**2 )

# plot fit for all data
times = np.asarray(times)
times_tpd = np.asarray(times_tpd)
periods = np.asarray(periods)

trend_poly = P.fit(times, periods, 1)
label = f'mse = {mse(periods, trend_poly(times)):.2f}'
ax1.plot(times, trend_poly(times), c='k', label=label)

trend_poly = P.fit(times_tpd, periods, 1)
label = f'mse = {mse(periods, trend_poly(times_tpd)):.2f}'
ax2.plot(times_tpd, trend_poly(times_tpd), c='k', label=label)

times_all = times_tpd
periods_all = periods

# make fit for "valid" data (under a threshold)
times = np.asarray(times_valid)
times_tpd = np.asarray(times_tpd_valid)
periods = np.asarray(periods_valid)

trend_poly = P.fit(times, periods, 1)
label = f'mse = {mse(periods, trend_poly(times)):.2f}'
ax1.plot(times, trend_poly(times), '--', c='k', label=label)

trend_poly = P.fit(times_tpd, periods, 1)
label = f'mse = {mse(periods, trend_poly(times_tpd)):.2f}'
ax2.plot(times_tpd, trend_poly(times_tpd), '--', c='k', label=label)

pearson_tsr, _ = stats.pearsonr(times, periods)
pearson_tpd, _ = stats.pearsonr(times_tpd, periods)


# ax1.plot(times_valid, periods_valid, 'kx')
# ax2.plot(times_tpd_valid, periods_valid, 'kx')

# Format plots

# limits
# ax3.set_ylim(ax1.get_ylim())
# ax4.set_ylim(ax2.get_ylim())

# ax6.set_ylim(ax5.get_ylim())

# style
# ax5.set_xlabel('time since recording started [sec]')
# ax6.set_xlabel('time since disection [sec]')
ax3.set_xlabel('time since recording started [min]')
ax4.set_xlabel('time since disection [min]')

ax1.set_title('Data aligned at the start of recording')
ax2.set_title('Data aligned at the disection time')

ax1.set_ylabel('period [sec]')
ax3.set_ylabel('period (slope only) [sec]')

ax1.legend(title=f'$r_{{perason}}$ = {pearson_tsr:.2f}')
ax2.legend(title=f'$r_{{perason}}$ = {pearson_tpd:.2f}')

#%% Plot only the valid points 

""" This cell is just a cleanup of the cell above. You need to run that one 
right before this one. It plots only the valid periods for the run aligned at
the right tpd and fits a line."""

# fig_main = plt.figure(constrained_layout=True, figsize=[9.8, 5.6])
# fig, fig2 = fig_main.subfigures(2,1, height_ratios=[4,1])
# axarr = fig.subplots(2, 1, sharex=True, sharey=True)
# (axl, axlh) = fig2.subplots(1,2, sharey=True, width_ratios=[5,1])
                          
# (ax, axh) = axarr
VALID = True
upper_value = 7

if VALID:
    this_periods, this_times = periods, times_tpd
else:
    this_periods, this_times = periods_all.copy(), times_all.copy()
    this_times = this_times[this_periods<=upper_value]
    this_periods = this_periods[this_periods<=upper_value]
    
fig, axarr = plt.subplots(3,2, constrained_layout=True, figsize=[9.8, 4.8],
                   sharex='col', sharey='row', width_ratios=[5,1])
(ax, axh, axl) = axarr[:, 0]
axbar = axarr[0:2, 1]
axlh = axarr[2, 1]

# scatter plot
ax.plot(this_times, this_periods, '.', rasterized=True, c=[191/255,  62/255, 128/255])
trend_poly = P.fit(this_times, this_periods, 1)
ax.plot(this_times, trend_poly(this_times), '--k')

# 2D histogram
colors_keypoints = [
    (0,   [191,  62, 128]),
    # (0.4, [205, 102, 102]),
    (1,   [241, 199,  43]),
    ]

cmap = colors.LinearSegmentedColormap.from_list('custom_colormap', [(i, [x/255 for x in c]) for i, c in colors_keypoints])

# cmap = cm.viridis.with_extremes(under='w')
*_, hi = axh.hist2d(this_times, this_periods, bins=[200, 50], cmin=0.5, 
                    rasterized=True, cmap=cmap)
# axh.plot(this_times, trend_poly(this_times), '--r')

plt.colorbar(hi, ax=axbar, orientation='vertical', label='point count')

for axi in axbar:
    clear_frame(axi)

# plot durations
info.sort_values('tpd', inplace=True)
valid = info.type=='S'
for i, (tpd, dur) in enzip(info[valid].tpd, info[valid].duration):
# for i, (tpd, dur) in enzip(info[:].tpd, info[:].duration):
    # axl.plot([tpd, tpd+dur], [dur, dur], '0.3')
    axl.plot([tpd, tpd+dur], [i, i], '0.3')

axlh.hist(info.duration, bins='auto', orientation='vertical')

# Format and stuff
pearson_tpd, _ = stats.pearsonr(this_times, this_periods)
mse_val = mse(this_periods, trend_poly(this_times))
slope = trend_poly.convert().coef[1]
_, cov = np.polyfit(this_times, this_periods, deg=1, cov='unscaled')
slope_err = np.sqrt(cov[0, 0])

axl.set_xlabel('time relative to tpd [min]')
axh.set_ylabel('period [s]')
ax.set_ylabel('period [s]')
axl.set_ylabel('duration [min]')

ax.legend(handles=[], title = f'$r_{{pearson}}$ = {pearson_tpd:.2f}\nmse={mse_val:.2f}\nslope={slope:.3f}±{slope_err:.3f}')

# axlh.set_xlim(0, 40)
axh.set_ylim(0, upper_value)
# ax.set_ylim(0, 7)
ax.set_ylim(axh.get_ylim())



#%% Correlate small and large CDs

""" This cell depends on the cell above being run at least once, becauses that
one defines some data structures we use here."""


# data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
data_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
data_dir = Path(data_dir) / 'output' / 'frequency_time_dependency'

outlier_mode_proportion = 1.8 #1.8 for normal runs
time_bins = 30, # use an iterable. If just value, usea a tuple like so: 10,

info_dir = data_dir / 'periods_info.csv'
data_files = contenidos(data_dir, filter_ext='.npz')

# load info
info = pd.read_csv(info_dir).sort_values('name').set_index('name')

# add tpd in seconds
info['tpd_sec'] = info.tpd * 60

# load and plot data

norm = colors.Normalize(info.time_hours.min(), info.time_hours.max())
color_list = cm.viridis([norm(info.loc[file.stem].time_hours) for file in data_files])

# color_list = cm.jet(np.linspace(0, 1, len(data_files)+1))[1:]
times = {'L' : [], 'S': []}
periods = {'L' : [], 'S': []}
for file, c in zip(data_files, color_list):
    # load data
    data = np.load(file)
    t = data['times'] / 60
    p = data['periods']
    tpd = info.loc[file.stem].tpd
    celltype = info.loc[file.stem]['type']
    
    # plot trend lines
    valid_indexes = au.find_valid_periods(t, p, outlier_mode_proportion, passes=2)
        
    # append data to lists
    periods[celltype].extend(p[valid_indexes])
    times[celltype].extend(t[valid_indexes]+tpd)

fig1 = plt.figure()
colors_dict = {'L':lcolor, 'S':scolor}
for celltype in 'LS':
    plt.plot(times[celltype], periods[celltype], '.', ms=3,
             label=celltype, c=colors_dict[celltype], rasterized=True )
plt.legend()
plt.xlabel('TPD')
plt.ylabel('CD')
plt.ylim(0.5, 7.5)


fig, ax = plt.subplots(figsize=(5, 5))

for time_bins in time_bins:
    # calculate time step size
    time_min = min(min(t) for t in times.values())
    time_max = max(max(t) for t in times.values())
    time_step = (time_max - time_min) / time_bins

    mean_CDs = {}
    std_CDs = {}
    for (celltype, time), period in zip(times.items(), periods.values()):    
        time = np.asarray(time)
        period = np.asarray(period)
    
        mean_CD = []
        std_CD = []
        for i in range(time_bins):
            lower_time = time_min + i*time_step
            upper_time = time_min + (i+1)*time_step
            
            indexes_in_bin = np.logical_and(time>=lower_time, time<upper_time)
            
            periods_inbin = period[indexes_in_bin]
            mean_CD.append(periods_inbin.mean() if sum(indexes_in_bin)>0 else np.nan)
            std_CD.append(periods_inbin.std()/np.sqrt(len(periods_inbin)) if sum(indexes_in_bin)>0 else np.nan)
        
        mean_CDs[celltype] = np.asarray(mean_CD)
        std_CDs[celltype] = np.asarray(std_CD)
            
    time_binned = [time_min + (i+1)*time_step/2 for i in range(time_bins)]
    
    ax.errorbar(mean_CDs['L'], mean_CDs['S'], xerr=std_CDs['L'], yerr=std_CDs['S'], 
                fmt='k.', label=time_bins, ms=(50-time_bins)/3)
    #ax.plot(mean_CDs['L'], mean_CDs['S'], '.', label=time_bins, ms=(50-time_bins)/3)
    
    # correlation values
    data = np.vstack( tuple(mean_CDs.values()) ).T[~np.isnan(mean_CDs['S'])]
    weights = (std_CDs['L']**2 +  std_CDs['S']**2)[~np.isnan(mean_CDs['S'])]
    data = DescrStatsW(data, weights)
    print(f'With {time_bins} bins: r={data.corrcoef[0,1]}')

ax.plot([0,6], [0,6], '--', c='0.7')
ax.legend(title='time bins')

ax.set_xlim(1.25, 5.35)
ax.set_ylim(1.25, 5.35)

ax.set_xlabel('CD lLNvs')
ax.set_ylabel('CD sLNvs')

ax.set_aspect('equal', 'box')


#%% Plot recording duration

""" This cell depends on the cell above being run at least once, becauses that
one defines some data structures we use here."""

VALID = True

# data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
data_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
data_dir = Path(data_dir) / 'output' / 'frequency_time_dependency'

info_dir = data_dir / 'periods_info.csv'
data_files = contenidos(data_dir, filter_ext='.npz')

# load info
info = pd.read_csv(info_dir).sort_values('name').set_index('name')

durations = []
for file in data_files:
    # load data
    data = np.load(file)
    t = data['times'] / 60
    
    durations.append(t[-1])
    
durations_df = pd.DataFrame({'duration':durations, 
                             'name':list(data_files.stems())}
                            ).sort_values('name').set_index('name')

# add durations and sort
info['duration'] = durations_df.duration
info = info.sort_values('tpd')

# plot durations

fig, axarr = plt.subplots(2, 1, figsize=[6.75, 3.56],
                          sharex=True, constrained_layout=True)

for ax, celltype in zip(axarr, ('L', 'S')):
    valid = info.type==celltype
    for i, (tpd, dur) in enzip(info[valid].tpd, info[valid].duration):
        ax.plot([tpd, tpd+dur], [i, i], '0.3')
    
    ax.set_ylabel('recording (ordered)')

# Format and stuff
ax.set_xlabel('time relative to tpd [min]')

#%% Save all individual runs

""" Save the period distribution of all individual runs to see which one (if 
any) are contriibuting the most noise."""

# data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
data_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
data_dir = Path(data_dir) / 'output' / 'frequency_time_dependency'

outlier_mode_proportion = 1.6 #1.8 for normal runs

info_dir = data_dir / 'periods_info.csv'
data_files = contenidos(data_dir, filter_ext='.npz')

# load info
info = pd.read_csv(info_dir).set_index('name')

# add tpd in seconds
info['tpd_sec'] = info.tpd * 60

# load and plot data
fig, ax2 = plt.subplots(constrained_layout=True, figsize=(6, 4))

norm = colors.Normalize(info.time_hours.min(), info.time_hours.max())
color_list = cm.viridis([norm(info.loc[file.stem].time_hours) for file in data_files])

P = np.polynomial.Polynomial
times_tpd = []
periods = []
period_errs = []
times_valid = []
times_tpd_valid = []
periods_valid = []
period_errs_valid = []
tpds = []
for file, c in zip(data_files, color_list):
    # load data
    data = np.load(file)
    t = data['times'] / 60
    p = data['periods']
    dp = data['rising_std']
    tpd = info.loc[file.stem].tpd
    
    # plot points
    ax2.plot(t+tpd, p, '.', c=c)
    
    # plot trend lines
    period_mode = calc_mode(p)
    valid_indexes = p <= (period_mode * outlier_mode_proportion)
    trend_poly = P.fit(t[valid_indexes], p[valid_indexes], 1)
    
    # append data to lists
    periods.extend(p)
    period_errs.extend(dp)
    times_tpd.extend(t+tpd)
    tpds.append(tpd)
    
    times_valid.extend(t[valid_indexes])
    periods_valid.extend(p[valid_indexes])
    period_errs_valid.extend(dp[valid_indexes])
    times_tpd_valid.extend(t[valid_indexes]+tpd)
    
def mse(y, y_fit):
    """ Mean square error: average of the square of the residuals of a fit"""
    return np.mean( (y-y_fit)**2 )

# plot fit for all data
times_tpd = np.asarray(times_tpd)
periods = np.asarray(periods)
period_errs = np.asarray(period_errs)

trend_poly = P.fit(times_tpd, periods, deg=1, w=1/period_errs**2)
label = f'mse = {mse(periods, trend_poly(times_tpd)):.2f}'
ax2.plot(times_tpd, trend_poly(times_tpd), c='k', label=label)

# make fit for "valid" data (under a threshold)
times_tpd = np.asarray(times_tpd_valid)
periods = np.asarray(periods_valid)
period_errs = np.asarray(period_errs_valid)

trend_poly = P.fit(times_tpd, periods, deg=1, w=1/period_errs**2)
label = f'mse = {mse(periods, trend_poly(times_tpd)):.2f}'
ax2.plot(times_tpd, trend_poly(times_tpd), '--', c='k', label=label)

pearson_tsr, _ = stats.pearsonr(times, periods)
pearson_tpd, _ = stats.pearsonr(times_tpd, periods)

# Format plots

ax2.set_title('Data aligned at the disection time')
ax2.set_ylabel('period [sec]')
ax2.legend(title=f'$r_{{perason}}$ = {pearson_tpd:.2f}')
ax2.set_xlabel('time since disection [min]')

# Draw the individual datasets

# we use _i for "individual"
for file, c in zip(data_files, color_list):
    
    fig_i, ax_i = plt.subplots(constrained_layout=True, figsize=[6, 4])
    
    # load data
    data = np.load(file)
    t = data['times'] / 60
    p = data['periods']
    tpd = info.loc[file.stem].tpd
    
    # find valid_periods
    period_mode = calc_mode(p)
    valid_indexes = p <= (period_mode * outlier_mode_proportion)
    
    # plot data
    ax_i.plot(t+tpd, p, '.')
    ax_i.plot(t[~valid_indexes]+tpd, p[~valid_indexes], '.', c='r')

    # plot trend
    trend_poly_i = P.fit(t[valid_indexes]+tpd, p[valid_indexes], 1)
    slope = trend_poly_i.convert().coef[1]
    ax_i.plot(t+tpd, trend_poly_i(t+tpd), c='0.6', label=f'slope: {slope:.1e}')

    xlim = np.asarray(ax_i.get_xlim())
    global_slope = trend_poly.convert().coef[1]
    ax_i.plot(xlim, trend_poly(xlim), '--', c='k', zorder=1, label=f'global slope: {global_slope:.1e}')
    ax_i.set_xlim(xlim)
    
    # calc mse against global trend
    mse_i = mse(p, trend_poly(t+tpd)) 
    mse_i_v = mse(p[valid_indexes], trend_poly(t[valid_indexes]+tpd))

    ax_i.set_ylim(ax2.get_ylim())    
    ax_i.set_title(f'{file.stem} | mse = {mse_i:.2f} | mse$_{{valid}}$ = {mse_i_v:.2f} | duration = {round(t[-1])}mins')
    ax_i.set_ylabel('period [sec]')
    ax_i.set_xlabel('time since disection [min]')
    ax_i.legend(loc='upper left')
    
    savename = '/media/marcos/DATA/marcos/FloClock pics/Period variability/all period evolutions/'
    fig_i.savefig(savename + file.stem)
    plt.close(fig_i)

#%% Plot initial period as a function of tpd

# data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
data_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
data_dir = Path(data_dir) / 'output' / 'frequency_time_dependency'

outlier_mode_proportion = 1.6 #1.8 for normal runs

info_dir = data_dir / 'periods_info.csv'
data_files = contenidos(data_dir, filter_ext='.npz')

# load info
info = pd.read_csv(info_dir).set_index('name')
norm = colors.Normalize(info.time_hours.min(), info.time_hours.max())
color_list = cm.viridis([norm(info.loc[file.stem].time_hours) for file in data_files])

# add tpd in seconds
info['tpd_sec'] = info.tpd * 60
# color_list = cm.jet(np.linspace(0, 1, len(data_files)+1))[1:]

initial_periods = []
initial_periods_error = []
tpds = []
daytime = []
issmall = []
for file, c in zip(data_files, color_list):
    # load data
    data = np.load(file)
    t = data['times']
    p = data['periods']
    tpd = info.loc[file.stem].tpd
    time_hours = info.loc[file.stem].time_hours
    small = info.loc[file.stem]['type'] == 'S'
    
    ip = np.mean(p[:40])
    if ip > 5:
        continue

    initial_periods.append(ip)
    initial_periods_error.append(np.std(p[:40]))
    tpds.append(tpd)
    daytime.append(time_hours)
    issmall.append(small)

tpds = np.asarray(tpds)
initial_periods = np.asarray(initial_periods)
initial_periods_error = np.asarray(initial_periods_error)
issmall = np.asarray(issmall)

# plot    
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 3), constrained_layout=True)

# ax1
sc = ax1.scatter(tpds, initial_periods, c=daytime)
ax1.set_xlabel('time since disection [sec]')
ax1.set_ylabel('initial period [sec]')
cbar = plt.colorbar(sc, label='time of day [hours]')

# trend line
trend = np.polynomial.Polynomial.fit(tpds, initial_periods, deg=1, w=1/initial_periods_error**2)
pearson, _ = stats.pearsonr(tpds, initial_periods)
slope = trend.convert().coef[1]
_, cov = np.polyfit(tpds, initial_periods, deg=1, cov='unscaled')
slope_err = np.sqrt(cov[0, 0])
fit_error = au.linear_fit_error(tpds, initial_periods_error.mean())

# small trend
small_trend = np.polynomial.Polynomial.fit(tpds[issmall], initial_periods[issmall], deg=1, w=1/initial_periods_error[issmall]**2)
small_slope = small_trend.convert().coef[1]
_, cov = np.polyfit(tpds[issmall], initial_periods[issmall], deg=1, w=1/initial_periods_error[issmall]**2, cov='unscaled')
small_slope_err = np.sqrt(cov[0, 0])

# large trend
large_trend = np.polynomial.Polynomial.fit(tpds[~issmall], initial_periods[~issmall], deg=1, w=1/initial_periods_error[~issmall]**2)
large_slope = large_trend.convert().coef[1]
_, cov = np.polyfit(tpds[~issmall], initial_periods[~issmall], w=1/initial_periods_error[~issmall]**2, deg=1, cov='unscaled')
large_slope_err = np.sqrt(cov[0, 0])


ax1.plot(tpds, trend(tpds), 'k', label=f'slope = {slope:.3f}±{slope_err:.3f}')
ax1.legend(loc='lower right')

# ax2
# ax2.scatter(tpds, initial_periods, c=issmall, cmap='cool')
ax2.errorbar(tpds[issmall], initial_periods[issmall], initial_periods_error[issmall], 
             fmt='o', label='small')
ax2.errorbar(tpds[~issmall], initial_periods[~issmall], initial_periods_error[~issmall], 
             fmt='o', label='large')
ax2.set_xlabel('time since disection [sec]')
ax2.set_ylabel('initial period [sec]')

# trend line
# line_time = np.linspace(min(tpds), max(tpds), 200)
line_time = np.linspace(min(tpds), max(tpds), 200)
ax2.plot(line_time, trend(line_time), 'k', label=f'$r_{{pearson}}$ = {pearson:.2f}')
ax2.plot(line_time, trend(line_time) + fit_error(line_time), '--k', lw=0.5)
ax2.plot(line_time, trend(line_time) - fit_error(line_time), '--k', lw=0.5)

ax2.plot(line_time, small_trend(line_time), 'C0')
ax2.plot(line_time, large_trend(line_time), 'C1')

ax2.legend(loc='lower right')
ax2.set_ylim(ax1.get_ylim())

# print pearson-r
pearson_lLNv, _ = stats.pearsonr(tpds[~issmall], initial_periods[~issmall])
pearson_sLNv, _ = stats.pearsonr(tpds[issmall], initial_periods[issmall])

cprint('&ly \nPearson-r values:')
print(f'\tlLNv : {pearson_lLNv:.1e} (n={sum(~issmall)})')
print(f'\tsLNv : {pearson_sLNv:.1e} (n={sum(issmall)})')



#%% Plot initial period as a function of TOD

"""NOTE: this requires normalizing the period to the value extrapolated from 
its TPD, so you need to run the previous cell before this one"""

detrended_periods = np.asarray(initial_periods)# - trend(np.asarray(tpds))
daytime = np.asarray(daytime)

plt.figure()
plt.errorbar(daytime[issmall], detrended_periods[issmall], initial_periods_error[issmall], fmt='.', label='small')
plt.errorbar(daytime[~issmall], detrended_periods[~issmall], initial_periods_error[~issmall], fmt='.', label='large')
plt.xlabel('time of day [hours]')
plt.ylabel('period (normalized) [sec]')
plt.legend()

# print pearson-r
pearson_lLNv, _ = stats.pearsonr(daytime[~issmall], detrended_periods[~issmall])
pearson_sLNv, _ = stats.pearsonr(daytime[issmall], detrended_periods[issmall])

cprint('&ly \nPearson-r values:')
print(f'\tlLNv : {pearson_lLNv:.1e} (n={sum(~issmall)})')
print(f'\tsLNv : {pearson_sLNv:.1e} (n={sum(issmall)})')


#%% Create 2d plot with TOD and TPD

# data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
data_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
data_dir = Path(data_dir) / 'output' / 'frequency_time_dependency'

outlier_mode_proportion = 1.8 #1.8 for normal runs
DRAW_TEXT = True
EXCLUDE_OUTLIERS = True
bin_count = 5 # for tod binning
bin_count_tpd = 12 # fot tpd binning

info_dir = data_dir / 'periods_info.csv'
data_files = contenidos(data_dir, filter_ext='.npz')

# load info
info = pd.read_csv(info_dir).set_index('name')

# add tpd in seconds
info['tpd_sec'] = info.tpd * 60
# color_list = cm.jet(np.linspace(0, 1, len(data_files)+1))[1:]

periods = []
tpds = []
tods = []

for file in data_files:
    # load data
    data = np.load(file)
    t = data['times'] # in seconds
    p = data['periods']
    tpd = info.loc[file.stem].tpd # in minutes
    tod = info.loc[file.stem].time_hours # in hours
    
    # handle separately the case where outlieras are included or excluded
    
    if EXCLUDE_OUTLIERS:
        period_mode = calc_mode(p)
        valid_indexes = p <= (period_mode * outlier_mode_proportion)
        
        periods.extend(p[valid_indexes])
        tpds.extend( t[valid_indexes]/60 + tpd )
        tods.extend( t[valid_indexes]/(60*60) + tod )

    else:
        periods.extend(p)
        tpds.extend( t/60 + tpd )
        tods.extend( t/(60*60) + tod )
    
# convert data to numpy and dataframe
periods = np.asarray(periods)
tpds = np.asarray(tpds)
tods = np.asarray(tods)

data = pd.DataFrame({'period':periods, 'tpd':tpds, 'tod':tods}).sort_values(['tpd', 'tod'])

# # plot data
# fig, ax = plt.subplots()
# sc = ax.scatter(data.tpd, data.tod, c=data.period, cmap='plasma')
# ax.set_xlabel('time post disection [min]')
# ax.set_ylabel('time of day [min]')
# plt.colorbar(sc)

# bin data
tpd_range = data.tpd.min(), data.tpd.max()
tod_range = data.tod.min(), data.tod.max()

tpd_step = (tpd_range[1] - tpd_range[0]) / bin_count_tpd
tod_step = (tod_range[1] - tod_range[0]) / bin_count

# fig_line, ax4 = plt.subplots()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True, constrained_layout=True, 
                                    figsize=[8.5, 3.9] if DRAW_TEXT else [6.44, 3.13])


periods_matrix = np.full((bin_count, bin_count_tpd), np.nan)
periods_variability = np.full((bin_count, bin_count_tpd), np.nan)
periods_counts = np.full((bin_count, bin_count_tpd), np.nan)
for i in range(bin_count_tpd):
    valid_tpds = np.logical_and(tpds >= tpd_range[0] + i*tpd_step, tpds < tpd_range[0] + (i+1)*tpd_step )
    for j in range(bin_count):
        valid_tods = np.logical_and(tods >= tod_range[0] + j*tod_step, tods < tod_range[0] + (j+1)*tod_step )    
        
        # get index ranges in the interval
        valid_indexes = np.logical_and(valid_tpds, valid_tods)
        
        if np.any(valid_indexes):
            
            periods_counts[j, i] = np.sum(valid_indexes)
            periods_matrix[j, i] = np.mean(periods[valid_indexes])
            periods_variability[j, i] = np.std(periods[valid_indexes])
            # plt.plot(tpds[valid_indexes], tods[valid_indexes], 'o')
            
            
            # Add numbers to plots
            if DRAW_TEXT:
                
                ax1.text(tpd_range[0] + (i+0.5)*tpd_step, tod_range[0] + (j+0.5)*tod_step,
                          f'{periods_matrix[j, i]: .2f}', 
                          horizontalalignment='center', verticalalignment='center',
                          color='w', fontsize=8)
                
                ax2.text(tpd_range[0] + (i+0.5)*tpd_step, tod_range[0] + (j+0.5)*tod_step,
                          f'{periods_variability[j, i]: .2f}', 
                          horizontalalignment='center', verticalalignment='center',
                          color='w', fontsize=8)
                
                ax3.text(tpd_range[0] + (i+0.5)*tpd_step, tod_range[0] + (j+0.5)*tod_step,
                          f'{periods_counts[j, i]: .0f}', 
                          horizontalalignment='center', verticalalignment='center',
                          color='w', fontsize=8)

    # plot trendlines
    # ax4.errorbar(x = np.arange(bin_count)+0.03*i, y = periods_matrix[:, i], yerr = periods_variability[:, i],
    #              label=f'{tpd_range[0] + (i+0.5)*tpd_step:.0f}')
    

# Plot    

i1 = ax1.imshow(periods_matrix, origin='lower', extent=[*tpd_range, *tod_range], aspect='auto', vmax=None)
i2 = ax2.imshow(periods_variability, origin='lower', extent=[*tpd_range, *tod_range], aspect='auto')
i3 = ax3.imshow(periods_counts, origin='lower', extent=[*tpd_range, *tod_range], aspect='auto')

plt.colorbar(i1, orientation='horizontal', label='period [sec]')
plt.colorbar(i2, orientation='horizontal', label='std [sec]')
plt.colorbar(i3, orientation='horizontal', label='# counts')

# style plots
ax1.set_title('Average period')
ax2.set_title('Period standard dev.')
ax3.set_title('Point counts')

ax1.set_xlabel('time post disection [min]')
ax2.set_xlabel('time post disection [min]')
ax3.set_xlabel('time post disection [min]')

ax1.set_ylabel('time of day [hours]')

# ax4.set_xticks(np.arange(bin_count), [round(tod_range[0] + (j+0.5)*tod_step) for j in range(bin_count)])
# ax4.set_xlabel('time of day [hours]')
# ax4.set_ylabel('period [sec]')
# ax4.legend(ncol=5, title='tpd [min]')

#%% Distributions of binned data in TPD vs TOD

# data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
data_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
data_dir = Path(data_dir) / 'output' / 'frequency_time_dependency'

outlier_mode_proportion = 1.8 #1.8 for normal runs
DRAW_TEXT = True
EXCLUDE_OUTLIERS = True
bin_count = 5 # for tod binning
bin_count_tpd = 12 # fot tpd binning

info_dir = data_dir / 'periods_info.csv'
data_files = contenidos(data_dir, filter_ext='.npz')

# load info
info = pd.read_csv(info_dir).set_index('name')

# add tpd in seconds
info['tpd_sec'] = info.tpd * 60
# color_list = cm.jet(np.linspace(0, 1, len(data_files)+1))[1:]

periods = []
tpds = []
tods = []

tpd_vals = []
for file in data_files:
    # load data
    data = np.load(file)
    t = data['times']
    p = data['periods']
    tpd = info.loc[file.stem].tpd
    tod = info.loc[file.stem].time_hours
    
    tpd_vals.append(tpd)
    
    # handle separately the case where outlieras are included or excluded
    
    if EXCLUDE_OUTLIERS:
        period_mode = calc_mode(p)
        valid_indexes = p <= (period_mode * outlier_mode_proportion)
        
        periods.extend(p[valid_indexes])
        tpds.extend( t[valid_indexes]/60 + tpd )
        tods.extend( t[valid_indexes]/(60*60) + tod )

    else:
        periods.extend(p)
        tpds.extend( t/60 + tpd )
        tods.extend( t/(60*60) + tod )
    
# convert data to numpy and dataframe
periods = np.asarray(periods)
tpds = np.asarray(tpds)
tods = np.asarray(tods)

data = pd.DataFrame({'period':periods, 'tpd':tpds, 'tod':tods}).sort_values(['tpd', 'tod'])

# # plot data
# fig, ax = plt.subplots()
# sc = ax.scatter(data.tpd, data.tod, c=data.period, cmap='plasma')
# ax.set_xlabel('time post disection [min]')
# ax.set_ylabel('time of day [min]')
# plt.colorbar(sc)

# bin data
tpd_range = data.tpd.min(), data.tpd.max()
tod_range = data.tod.min(), data.tod.max()

tpd_step = (tpd_range[1] - tpd_range[0]) / bin_count_tpd
tod_step = (tod_range[1] - tod_range[0]) / bin_count

fig, axarr = plt.subplots(bin_count, bin_count_tpd, sharex=True, sharey=True, constrained_layout=True, 
                                    figsize=[8.5, 6.9] if DRAW_TEXT else [6, 5.7])
fig.suptitle(f'Distribution of periods inside each bin with{"out" if EXCLUDE_OUTLIERS else ""} outliers', fontsize=15)

ncol = int(np.ceil(np.sqrt(bin_count_tpd)))
nrow = int(round(np.sqrt(bin_count_tpd)))
fig_p, axarr_p = plt.subplots(nrow, ncol, constrained_layout=True, sharex=True, sharey=True, figsize=[6.56, 5.53])
fig_p.suptitle(f'Median test p-value results for fixed tpd values with{"out" if EXCLUDE_OUTLIERS else ""} outliers', fontsize=15)

fig_extra, axarr_extra = plt.subplots(1, bin_count_tpd, figsize=[18.46,  3.18], constrained_layout=True)

all_period_distributions = []
for i in range(bin_count_tpd):
    valid_tpds = np.logical_and(tpds >= tpd_range[0] + i*tpd_step, tpds < tpd_range[0] + (i+1)*tpd_step )
    
    # create list to store the period distrs.
    period_distributions = []
    
    # iterate over TOD
    for j in range(bin_count):
        valid_tods = np.logical_and(tods >= tod_range[0] + j*tod_step, tods < tod_range[0] + (j+1)*tod_step )    
        
        # get index ranges in the interval
        valid_indexes = np.logical_and(valid_tpds, valid_tods)
        
        # count the first coord of te axis from back to front to
        ax = axarr[-j-1 , i]
        
        if np.any(valid_indexes):
            
            c = 'C01' if np.sum(valid_indexes)>800 else 'C00'
            ax.hist(periods[valid_indexes], density=True, bins='auto', color=c)
            period_distributions.append(periods[valid_indexes])
            
            axarr_extra[i].hist(periods[valid_indexes], density=True, bins=20, histtype='step', linewidth=2)
            
            if DRAW_TEXT:
                
                # ax.set_title(f'N = {np.sum(valid_indexes)}')
                
                # tpd_str = f'tpd = [{tpd_range[0] + i*tpd_step :.0f}, {tpd_range[0] + (i+1)*tpd_step :.0f}]'
                # tod_str = f'tod = [{tod_range[0] + j*tod_step :.0f}, {tod_range[0] + (j+1)*tod_step :.0f}]'
                
                # ax.legend(handles=[], title=f'{tpd_str}\n{tod_str}')
                
                ax.legend(handles=[], title=f'N = {np.sum(valid_indexes)}')
                
                tpd_str = f'tpd = [{tpd_range[0] + i*tpd_step :.0f}, {tpd_range[0] + (i+1)*tpd_step :.0f}]'
                tod_str = f'tod = [{tod_range[0] + j*tod_step :.0f}, {tod_range[0] + (j+1)*tod_step :.0f}]'
                
                if j == 0:
                    ax.set_xlabel(tpd_str)
                    axarr_extra[i].set_title(tpd_str)
                if i == 0:
                    ax.set_ylabel(tod_str)
                    
            else:
                clear_frame(ax, hidespine=False)
            
        else:
            clear_frame(ax)
    
    # if not all bins had periods in them, there's probably not enough data to 
    # say anything relevant anyway
    if len(period_distributions) != bin_count:
        continue
    # create array to fill up with pvalues
    p_values = np.full((bin_count, bin_count), np.nan)
    
    cmap = cm.winter.with_extremes(over='r')
    ax_p = axarr_p.flat[i]
    
    for m in range(bin_count):
        this_period_distr = period_distributions[m]
        for n in range(m, bin_count):
            other_period_distr = period_distributions[n]
            
            ks_result = stats.ks_2samp(this_period_distr, other_period_distr)
            p_values[n, m] = ks_result.pvalue
    
            mood_result = stats.median_test(this_period_distr, other_period_distr)
            p_values[n, m] = mood_result[1]
            
            # add numbers to plots
            # pval_order = np.round(np.log10(p_values[n, m])).astype(int)
            # pval_str = f'$10^{{{pval_order}}}$'
            # pval_str = f'{p_values[n, m]:.1g}'
            pval_str = f'{np.log10(p_values[n, m]):.0f}'
            
            ax_p.text(m, n, f'{pval_str}', 
                      horizontalalignment='center', verticalalignment='center',
                      color='w', fontsize=8)
    
    anova_pval = stats.f_oneway(*period_distributions).pvalue
    mood_pval = stats.median_test(*period_distributions)[1]
    
    im_pval = ax_p.imshow(p_values, cmap=cmap, norm=colors.LogNorm(vmax=0.05))
    ax_p.set_title(f'tpd $\\in$ [{tpd_range[0] + i*tpd_step :.0f}, {tpd_range[0] + (i+1)*tpd_step :.0f}]\np-value$_{{mood}}$={mood_pval:.2g}')
    
    all_period_distributions.append(period_distributions)
    
ax.set_xlim(0, 10)
ax.set_ylim(0, 2)

cbar = plt.colorbar(im_pval, ax=axarr_p.flatten(), extend='max')
cbar.ax.set_title('p-value')

# set some axis labels
axarr[int(bin_count/2), 0].set_ylabel('prob. density')
axarr[-1, int(bin_count_tpd/2)].set_xlabel('period [sec]')

# tick labels
xtick_labels = []
for j in range(bin_count):
    label = f'[{tod_range[0] + j*tod_step :.0f}, {tod_range[0] + (j+1)*tod_step :.0f}]'
    xtick_labels.append(label)
    
for ax_p in axarr_p[-1, :]:
    ax_p.set_xticks(range(bin_count), xtick_labels, rotation=90)
    ax_p.set_xlabel('tod')

for ax_p in axarr_p[:, 0]:
    ax_p.set_yticks(range(bin_count), xtick_labels)
    ax_p.set_ylabel('tod')
    
#%% HLE bootstrap interval

period_distributions = all_period_distributions[1]

for m in range(bin_count):
    this_period_distr = period_distributions[m]
    
    print(f'P{m}')
    for n in range(m+1, bin_count):
        other_period_distr = period_distributions[n]
        
        hle = au.hodges_lehmann_distance_estimator(this_period_distr, other_period_distr)
        CI = au.hodges_lehmann_distance_confidence_interval(this_period_distr, other_period_distr)
        not_significant = CI.low <= 0 <= CI.high
        cprint(f'{hle:.3f} in [{CI.low:.3f}, {CI.high:.3f}]: {"&r NOT &s" if not_significant else ""} significant')

#%% Try outlier filtering with initial trend

""" Set an approximate trend with the first few points. Extrapolate that trend 
along the whole run and use it to detrend the data. From the detrended data 
extract the modal period and use that value to filter outliers (around the 
trend)"""

data_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
data_dir = Path(data_dir) / 'output' / 'frequency_time_dependency'

outlier_mode_proportion = 1.6 #1.8 for normal runs
file_inx = 98

data_files = contenidos(data_dir, filter_ext='.npz')

file = data_files[file_inx]
P = np.polynomial.Polynomial

data = np.load(file)
t = data['times']
p = data['periods']

fig, ax = plt.subplots(constrained_layout=True, figsize=(8,4))

ax.plot(t, p, '.')
ax.set_xlabel('Time [sec]')
ax.set_ylabel('Period')
ax.set_title(file.stem)

count = 20
trend_poly = P.fit(t, p, 1)
# ax.plot(t, trend_poly(t), c='C1')

# detrended_mode = calc_mode(p - trend_poly(t))
ax.plot(t, p - trend_poly(t) + trend_poly.convert().coef[0], '.')


# period_mode = calc_mode(p)
# valid_indexes = p <= (period_mode * outlier_mode_proportion)


#%% Try two-step outlier filtering

""" Find outlier periods by exluding periods some multiple away from the mode 
of the periods and then fitting a line to the remaining ones. Then using that 
line to do a second evaluation"""


data_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
data_dir = Path(data_dir) / 'output' / 'frequency_time_dependency'
info_dir = data_dir / 'periods_info.csv'

outlier_mode_proportion = 1.8 #1.8 for normal runs
file_inx = 103

data_files = contenidos(data_dir, filter_ext='.npz')
info = pd.read_csv(info_dir).set_index('name')

# add tpd in seconds
info['tpd_sec'] = info.tpd * 60


file = data_files[file_inx]
P = np.polynomial.Polynomial

data = np.load(file)
t = data['times']
p = data['periods']

period_mode = calc_mode(p)
valid_indexes = p <= period_mode * outlier_mode_proportion

fig, ax = plt.subplots(2, 1, constrained_layout=True, figsize=(8,4))

ax, ax2 = ax

ax.plot(t, p, '.')
ax.plot(t[~valid_indexes], p[~valid_indexes], '.')

#mode visualized
ax.axhline(period_mode, color='k')
ax.axhline(period_mode*outlier_mode_proportion, ls='--', color='k')

trend = P.fit(t[valid_indexes], p[valid_indexes], deg=1)
ax.plot(t, trend(t))

ax.set_title(file.stem)

dp = p - trend(t) + trend(0)

period_mode = calc_mode(dp)
valid_indexes2 = dp <= period_mode * outlier_mode_proportion

ax2.plot(t, dp, '.')
ax2.plot(t[~valid_indexes2], dp[~valid_indexes2], '.')
ax.plot(t[valid_indexes2], p[valid_indexes2], 'x', c='C2')

#mode visualized
ax.axhline(period_mode, color='k')
ax.axhline(period_mode*outlier_mode_proportion, ls='--', color='k')

#%% Quantify period variability as a function of outlier threshold

""" We want to know what's the optimal threshold for the period outlier 
discrimination. To that end, we calculate the period variability for multiple 
values of the threshold with one and tow pass outlier detection."""

# data_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
data_dir = '/home/user/Documents/Doctorado/Fly clock/FlyClock_data/tiempos_post_diseccion'
data_dir = Path(data_dir) / 'output' / 'frequency_time_dependency'
info_dir = data_dir / 'periods_info.csv'

outlier_mode_proportion = 1.8 #1.8 for normal runs
file_inx = 8

data_files = contenidos(data_dir, filter_ext='.npz')
info = pd.read_csv(info_dir).set_index('name')

# add tpd in seconds
info['tpd_sec'] = info.tpd * 60


file = data_files[file_inx]
P = np.polynomial.Polynomial

data = np.load(file)
t = data['times']
p = data['periods']

fig, axarr = plt.subplots(2, 1, constrained_layout=True, figsize=(8,4), sharex=True)

thresholds = np.linspace(1, 2, 20)

for thr in thresholds:
    
    valid = au.find_valid_periods(t, p, threshold=thr, passes=1)
    valid2 = au.find_valid_periods(t, p, threshold=thr, passes=2)
    
    for ax, v in zip(axarr, (valid, valid2)):
        valid_proportion = sum(valid) / len(valid)
        ax.plot(thr, valid_proportion, 'o', c='C0')
        
axarr[0].set_title(file.stem)
axarr[0].set_ylabel('valid proportion\none pass')
axarr[1].set_ylabel('valid proportion\ntwo passes')
axarr[1].set_xlabel('threshold')


#%% (All) Quantify period variability as a function of outlier threshold

""" We want to know what's the optimal threshold for the period outlier 
discrimination. To that end, we calculate the period variability for multiple 
values of the threshold with one and tow pass outlier detection."""

data_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
# data_dir = '/home/user/Documents/Doctorado/Fly clock/FlyClock_data/tiempos_post_diseccion'
data_dir = Path(data_dir) / 'output' / 'frequency_time_dependency'
info_dir = data_dir / 'periods_info.csv'

outlier_mode_proportion = 1.8 #1.8 for normal runs

data_files = contenidos(data_dir, filter_ext='.npz')
info = pd.read_csv(info_dir).set_index('name')

# add tpd in seconds
info['tpd_sec'] = info.tpd * 60
P = np.polynomial.Polynomial

fig, axarr = plt.subplots(2, 1, constrained_layout=True, figsize=(8,4), sharex=True)

thresholds = np.linspace(1, 2, 20)
all_props1 = []
all_props2 = []
for file in data_files:
    props1 = []
    props2 = []
    
    data = np.load(file)
    t = data['times']
    p = data['periods']
    
    for thr in thresholds:
        
        valid = au.find_valid_periods(t, p, threshold=thr, passes=1)
        valid2 = au.find_valid_periods(t, p, threshold=thr, passes=2)
        
        for v, props in zip((valid, valid2), (props1, props2)):
            valid_proportion = sum(valid) / len(valid)
            props.append(valid_proportion)
     
    axarr[0].plot(thresholds, props1)
    axarr[1].plot(thresholds, props2)
    
    all_props1.append(props1)
    all_props2.append(props2)
    
all_props1 = np.asarray(all_props1).mean(axis=0)
all_props2 = np.asarray(all_props2).mean(axis=0)

axarr[0].plot(thresholds, all_props1, 'k', lw=3)
axarr[1].plot(thresholds, all_props1, 'k', lw=3)

axarr[0].set_title(file.stem)
axarr[0].set_ylabel('valid proportion\none pass')
axarr[1].set_ylabel('valid proportion\ntwo passes')
axarr[1].set_xlabel('threshold')


#%% Set outlier_mode_proportion threshold

""" Use this to set the threshold that selects outliners when calculating
period trends. See for a few runs what values look right."""

data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
# data_dir = '/home/user/Documents/Doctorado/Fly clock/FlyClock_data/data'

data_dir = Path(data_dir) / 'output' / 'frequency_time_dependency'

outlier_mode_proportion = 1.6 #1.8 for normal runs

info_dir = data_dir / 'periods_info.csv'
data_files = contenidos(data_dir, filter_ext='.npz')

# load info
info = pd.read_csv(info_dir).set_index('name')

# add tpd in seconds
info['tpd_sec'] = info.tpd * 60

count_per_plot = 8
for i in range( np.ceil(len(data_files) / count_per_plot).astype(int)):
# for i in range(1):
    # load and plot data
    fig, axarr = plt.subplots(4, 2, constrained_layout=True, figsize=(16, 6))
    
    color_list = cm.jet(np.linspace(0, 1, len(data_files)+1))[1:]
    for file, ax in zip(data_files[i:(i+1)*count_per_plot], axarr.flat):
        # load data
        data = np.load(file)
        t = data['times']
        p = data['periods']
        tpd = info.loc[file.stem].tpd_sec
        
        period_mode = calc_mode(p)
        valid = p <= (period_mode * 1.4)
        invalid1 = (p>= (period_mode * 1.4)) & (p<= (period_mode * 1.6))
        invalid2 = (p>= (period_mode * 1.6)) & (p<= (period_mode * 1.8))
        invalid3 = p>= (period_mode * 1.8)
        
        ax.plot(t[valid], p[valid], '.', c='C02')
        ax.plot(t[invalid1], p[invalid1], '.', c='C01')
        ax.plot(t[invalid2], p[invalid2], '.', c='C03')
        ax.plot(t[invalid3], p[invalid3], '.', c='C04')
        ax.axhline(period_mode, color='0.3', linestyle='--', zorder=1)
              
        ax.fill_between(t, period_mode * 1.8, color='0.9', zorder=0.8)
        ax.fill_between(t, period_mode * 1.6, color='0.8', zorder=0.8)
        ax.fill_between(t, period_mode * 1.4, color='0.6', zorder=0.8)
        
    # savename = '/media/marcos/DATA/marcos/FloClock pics/Period variability/'
    # fig.savefig(savename + f'setting_outlier_example_{i}.png')

#%% Check the distribution of daytimes we have

""" Extract date time info from all the runs """

import pyabf
import pandas as pd

BASE_DIR = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'

files = contenidos(BASE_DIR, filter_ext='.abf')

dates_list = []
for file in files:
    abf = pyabf.ABF(file)
    dates_list.append(abf.abfDateTime)
    
dates = pd.DataFrame({'dates':dates_list})
dates['day'] = dates.dates.dt.day
dates['hour'] = dates.dates.dt.hour
dates['minute'] = dates.dates.dt.minute

plt.figure()
plt.hist(dates.hour)
plt.ylabel('countes')
plt.xlabel('hour of day')

# dates.groupby(dates["dates"].dt.hour).count()['hour' ].plot(kind='bar', rot=0)
# plt.ylabel('countes')
# plt.xlabel('hour of day')

#%% Do the same but form the saved data


# data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
data_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
data_dir = Path(data_dir) / 'output' / 'frequency_time_dependency'

outlier_mode_proportion = 1.6 #1.8 for normal runs

info_dir = data_dir / 'periods_info.csv'
data_files = contenidos(data_dir, filter_ext='.npz')


#%% Study distribution of tpds

data_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
data_dir = Path(data_dir) / 'output' / 'frequency_time_dependency'

info_dir = data_dir / 'periods_info.csv'
data_files = contenidos(data_dir, filter_ext='.npz')

# load info
info = pd.read_csv(info_dir).set_index('name')

fig, ax = plt.subplots(figsize=[4,4])

ax.hist(info.tpd, bins='auto')
ax.set_xlabel('tpd [min]')
ax.set_ylabel('counts')
ax.set_title('Distribution of TPDs')


#%% Study slope distribution vs duration and vs tpd

# data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
data_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
data_dir = Path(data_dir) / 'output' / 'frequency_time_dependency'

outlier_mode_proportion = 1.8 #1.8 for normal runs

info_dir = data_dir / 'periods_info.csv'
data_files = contenidos(data_dir, filter_ext='.npz')

# load info
info = pd.read_csv(info_dir).set_index('name')
info = info.drop(columns='duration')

# load and plot data
P = np.polynomial.Polynomial
slopes = []
slopes_valid = []
slopes_valid_err = []
durations = []
for file in data_files:
    # load data
    data = np.load(file)
    t = data['times']
    p = data['periods']
    tpd = info.loc[file.stem].tpd 
    t /= 60
    
    # fit trendlines
    valid_indexes = au.find_valid_periods(t, p, passes=2)
    trend_poly = P.fit(t, p, 1)
    trend_poly_valid = P.fit(t[valid_indexes], p[valid_indexes], 1)

    _, cov = np.polyfit(t[valid_indexes], p[valid_indexes], deg=1, cov='unscaled')
    slope_err = np.sqrt(cov[0, 0])

    # append data to lists
    slopes.append(trend_poly.convert().coef[1])
    slopes_valid.append(trend_poly_valid.convert().coef[1])
    slopes_valid_err.append(slope_err)
    durations.append(t[-1])
    
# put data into dataframe
slope_df = pd.DataFrame({'slope':slopes, 'slope_valid':slopes_valid, 'slope_err':slopes_valid_err, 'duration':durations},
                        index=pd.Index(data_files.stems(), name='name'))
info = pd.concat([info, slope_df], axis='columns').sort_values('slope_valid')

# plot slopes distributions
fig, (top, bot) = plt.subplots(2, 3, constrained_layout=True, 
                               sharex='col', sharey=True, figsize=(8, 5.5),
                               width_ratios=[1,2,2])
ax1, ax3, ax5 = top
ax2, ax4, ax6 = bot
left = ax1, ax2

# find single channel recordings
single_channel_recs = np.asarray([len(str(name))>2 for name in info.og_name.values])
sLNv_recs = np.asarray([ 'S' == info.loc[file].type for file in data_files.stems() ])

# slope histograms
binning = np.linspace(-0.4, 0.4, 15)
ax1.hist(slopes, bins=binning, orientation='horizontal', fc='0.6')
ax2.hist(slopes_valid, bins=binning, orientation='horizontal', fc='0.6')

ax2.hist(info.slope_valid[~sLNv_recs], bins=binning, orientation='horizontal', 
         fc=lcolor, alpha=0.5)
ax2.hist(info.slope_valid[sLNv_recs], bins=binning, orientation='horizontal', 
         fc=scolor, alpha=0.5)


# slope vs duration
ax3.plot(info.duration[~sLNv_recs], info.slope[~sLNv_recs], '.', c=lcolor)
ax4.plot(info.duration[~sLNv_recs], info.slope_valid[~sLNv_recs], '.', label='lLNvs', c=lcolor)

# color small LNvs differently
ax3.plot(info.duration[sLNv_recs], info.slope[sLNv_recs], '.', c=scolor)
ax4.plot(info.duration[sLNv_recs], info.slope_valid[sLNv_recs], '.', label='sLNvs', c=scolor)

# slope vs tpd
ax5.plot(info.tpd[~sLNv_recs], info.slope[~sLNv_recs], '.', c=lcolor)
ax6.errorbar(info.tpd[~sLNv_recs], info.slope_valid[~sLNv_recs], info.slope_err[~sLNv_recs], fmt='.', c=lcolor)
avg = np.average(info.slope_valid[~sLNv_recs], weights=1/info.slope_err[~sLNv_recs]**2)
avg_err = np.sqrt(1 / np.sum( 1/info.slope_err[~sLNv_recs]**2 ))
ax6.axhline(avg, c=lcolor, label=f'{avg:.3f}±{avg_err:.3f}')

# color the sigle channel recordings differently
ax5.plot(info.tpd[sLNv_recs], info.slope[sLNv_recs], '.', c=scolor)
ax6.errorbar(info.tpd[sLNv_recs], info.slope_valid[sLNv_recs], info.slope_err[sLNv_recs], fmt='.', c=scolor)
avg = np.average(info.slope_valid[sLNv_recs], weights=1/info.slope_err[sLNv_recs]**2)
ax6.axhline(avg, c=scolor, label=f'{avg:.3f}±{avg_err:.3f}')

for ax in left:
    ax.set_ylabel('slope [sec/min]')

ax2.set_xlabel('counts')
ax4.set_xlabel('duration of rec. [min]')
ax6.set_xlabel('tpd [min]')

ax3.set_title('All periods')
ax4.set_title('"Valid" periods only')
ax4.legend() 
ax6.legend(title='Weighted avg.\nslope [sec/min]')

for ax in (*top, *bot):
    ax.grid(axis='y')
    
# some of the runs with highest and lowest slopes
count = 5
rec_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
rec_files = list(contenidos(rec_dir, filter_ext='.abf').stems())

cprint('&ly LOWEST SLOPES')
for name, line in info.iloc[:count].iterrows():
    index = rec_files.index(line.rec)
    cprint(f'\t&lc {index}&s : {name} ({line.slope:.3f})')

cprint('&ly \nHIGHEST SLOPES')
for name, line in info.iloc[-count:].iterrows():
    index = rec_files.index(line.rec)
    cprint(f'\t&lc {index}&s : {name} ({line.slope:.3f})')
    
cprint('&ly \nCLOSEST SLOPES TO ZERO')
info['dist_to_zero'] = np.abs(info.slope)
info = info.sort_values('dist_to_zero')
for name, line in info.iloc[:count].iterrows():
    index = rec_files.index(line.rec)
    cprint(f'\t&lc {index}&s : {name} ({line.slope:.5f})')
    
ax6.set_ylim(-0.3, 0.3)

# a few statistical tests
cprint('\n &ly Difference from zero: &s p_t = students-t | p_w : wilcoxon | p_s : sign')
for name, indexes in zip(('lLNv', 'sLNv'), (~sLNv_recs, sLNv_recs)):
    
    points = info.slope_valid[indexes]
    res_t = stats.ttest_1samp(points, popmean=0)
    res_w = stats.wilcoxon(points)
    res_s = stats.binomtest(sum(x>0 for x in points), len(points))
    ci = res_t.confidence_interval()
    
    print(f'{name}\tp_t:{res_t.pvalue:.2e} \tp_w:{res_w.pvalue:.2e} \tp_s:{res_s.pvalue:.2e} \t| mean = ({np.mean(points):.3f} ± [{ci.low:.3f}, {ci.high:.3f}])ms')

# print pearson-r
pearson_lLNv, _ = stats.pearsonr(info.tpd[~sLNv_recs], info.slope_valid[~sLNv_recs])
pearson_sLNv, _ = stats.pearsonr(info.tpd[sLNv_recs], info.slope_valid[sLNv_recs])

cprint('&ly \nPearson-r values:')
print(f'\tlLNv : {pearson_lLNv:.1e} (n={sum(~sLNv_recs)})')
print(f'\tsLNv : {pearson_sLNv:.1e} (n={sum(sLNv_recs)})')

# print weighted person-r
pearson_lLNv, _ = stats.pearsonr(info.tpd[~sLNv_recs], info.slope_valid[~sLNv_recs])
pearson_sLNv, _ = stats.pearsonr(info.tpd[sLNv_recs], info.slope_valid[sLNv_recs])

data_lLNv = DescrStatsW(info[['tpd', 'slope_valid']][~sLNv_recs], info.slope_err[~sLNv_recs])
data_sLNv = DescrStatsW(info[['tpd', 'slope_valid']][ sLNv_recs], info.slope_err[ sLNv_recs])

cprint('&ly \nWeighted correlation values:')
print(f'\tlLNv : {data_lLNv.corrcoef[0,1]:.1e} (n={sum(~sLNv_recs)})')
print(f'\tsLNv : {data_sLNv.corrcoef[0,1]:.1e} (n={sum(sLNv_recs)})')

#%% Study variability around the global slope

""" Plot the standard deviation of the points in a run (as a measure of noise),
which will hold as long as the increase in period over time is not greater than
the variability.
Also plot the mean square error (mse) of each run as measured against the global
trend line. Plot both quantities as a function of duration of the recording.
"""

# data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
data_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
data_dir = Path(data_dir) / 'output' / 'frequency_time_dependency'

outlier_mode_proportion = 1.6 #1.8 for normal runs

info_dir = data_dir / 'periods_info.csv'
data_files = contenidos(data_dir, filter_ext='.npz')

# load info
info = pd.read_csv(info_dir).set_index('name')

# add tpd in seconds
info['tpd_sec'] = info.tpd * 60

# load and plot data
P = np.polynomial.Polynomial
var = []
var_valid = []
mse = []
mse_valid = []
durations = []
for file in data_files:
    # load data
    data = np.load(file)
    t = data['times']
    p = data['periods']
    tpd = info.loc[file.stem].tpd_sec
    
    # fit trendlines
    period_mode = calc_mode(p)
    valid_indexes = p <= (period_mode * outlier_mode_proportion)
    
    trend_poly = P.fit(t, p, 1)
    trend_poly_valid = P.fit(t[valid_indexes], p[valid_indexes], 1)
    
    # append data to lists
    
    durations.append(t[-1]/60)
    
    var.append(np.std(p))
    var_valid.append(np.std(p[valid_indexes]))
    
    mse.append( np.mean( (trend_poly(t) - p)**2 ) )
    mse_valid.append( np.mean( (trend_poly(t[valid_indexes]) - p[valid_indexes])**2 ) )
    
# plot mse and variability

fig, (top, mid) = plt.subplots(2, 2, constrained_layout=True, 
                               sharex=True, sharey='row', figsize=(6, 4))
ax1, ax2 = top
ax3, ax4 = mid

ax1.plot(durations, var, '.')
ax2.plot(durations, var_valid, '.')

ax3.plot(durations, mse, '.')
ax4.plot(durations, mse_valid, '.')

ax1.set_title('All periods')
ax2.set_title('Valid periods')

ax1.set_ylabel('variability')
ax3.set_ylabel('mse')

ax3.set_xlabel('duration [min]')
ax4.set_xlabel('duration [min]')

# see highest mse runs

ordered_files = sort_by(data_files.stems(), mse)[::-1]
ordered_indexes = sort_by(range(len(mse)), mse)[::-1]
ordered_mse = sorted(mse)[::-1]

rec_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
rec_files = list(contenidos(rec_dir, filter_ext='.abf').stems())

count = 5
cprint('&ly HIGHEST MSE RUNS')
for f, _, m in zip(ordered_files[:count], ordered_indexes[:count], ordered_mse[:count]):
    i = rec_files.index(f.split('_')[0])
    cprint(f'\t&lc {i}&s : {f} ({m:.1f})')

# see highest variability runs

ordered_files = sort_by(data_files.stems(), var)[::-1]
ordered_indexes = sort_by(range(len(mse)), var)[::-1]
ordered_var = sorted(var)[::-1]

rec_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
rec_files = list(contenidos(rec_dir, filter_ext='.abf').stems())

count = 5
cprint('&ly HIGHEST VARIABILITY RUNS')
for f, _, m in zip(ordered_files[:count], ordered_indexes[:count], ordered_var[:count]):
    i = rec_files.index(f.split('_')[0])
    cprint(f'\t&lc {i}&s : {f} ({m:.1f})')
    
#%% Quantify outliers as a function of tpd and threshold

# data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
data_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
# data_dir = '/home/user/Documents/Doctorado/Fly clock/FlyClock_data/tiempos_post_diseccion'

data_dir = Path(data_dir) / 'output' / 'frequency_time_dependency'

info_dir = data_dir / 'periods_info.csv'
data_files = contenidos(data_dir, filter_ext='.npz')

# load info
info = pd.read_csv(info_dir).set_index('name')

outlier_mode_proportions = 1.8,

# load and plot data
P = np.polynomial.Polynomial
valid_proportion = [[] for _ in outlier_mode_proportions]
valid_counts = [[] for _ in outlier_mode_proportions]
durations = []
tpds = []
for file in data_files:
    # load data
    data = np.load(file)
    t = data['times']
    p = data['periods']
    tpd = info.loc[file.stem].tpd
    
    # fit trendlines
    period_mode = calc_mode(p)
    
    for i, outlier_mode_proportion in enumerate(outlier_mode_proportions):
        valid_indexes = p <= (period_mode * outlier_mode_proportion)
        
        valid_counts[i].append( sum(valid_indexes) )
        valid_proportion[i].append( sum(valid_indexes) / p.size )
        
    durations.append(t[-1]/60)
    tpds.append(tpd)
    
    
valid_counts = np.asarray(valid_counts).T
valid_proportion = np.asarray(valid_proportion).T
    
fig, (top, bot) = plt.subplots(2,2, constrained_layout=True, figsize=[8,6])

ax1, ax3 = top
ax2, ax4 = bot

lines = ax1.plot(durations, valid_counts, '.')
ax2.plot(durations, valid_proportion, '.')

ax3.plot(tpds, valid_counts, '.')
ax4.plot(tpds, valid_proportion, '.')

# format axes
ax1.set_ylabel('valid counts')
ax2.set_ylabel('valid proportion')

ax2.set_xlabel('duration [min]')
ax4.set_xlabel('tpd [min]')

ax1.set_title('Valid periods vs duration')
ax3.set_title('Valid periods vs tpd')

fig.suptitle('\n\n')
fig.legend(lines, outlier_mode_proportions, ncols=len(outlier_mode_proportions),
           title='mode proportion for outliers',
           loc='upper center', bbox_to_anchor=(0.5, 1))

#%% Quantify outliers as as the run advances

# data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
data_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
# data_dir = '/home/user/Documents/Doctorado/Fly clock/FlyClock_data/tiempos_post_diseccion'
data_dir = Path(data_dir) / 'output' / 'frequency_time_dependency'

outlier_mode_proportion = 1.8 #1.8 for normal runs
KIND = 'S'  # 'L', 'S', or None

COLOR_INVALID = '#db4a4a'

info_dir = data_dir / 'periods_info.csv'
data_files = contenidos(data_dir, filter_ext='.npz')

# load info
info = pd.read_csv(info_dir).set_index('name')

# add tpd in seconds
info['tpd_sec'] = info.tpd * 60

# load and plot data

fig, (top, mid, bot) = plt.subplots(3, 2, constrained_layout=True, 
                               sharex='col', sharey='row', figsize=(9.7, 6.7))
ax1, ax2 = top
ax3, ax4 = mid
ax5, ax6 = bot

times = []
valids = []
periods = []
times_norm = []
durations = []
for file in data_files:
    
    if KIND is not None and info.loc[file.stem].type != KIND:
        continue
    
    # load data
    data = np.load(file)
    t = data['times'] / 60
    p = data['periods']
    
    # plot trend lines
    valid = au.find_valid_periods(t, p, passes=2)
    # period_mode = calc_mode(p)
    # valid = p <= (period_mode * outlier_mode_proportion)
    trend_poly = P.fit(t[valid], p[valid], 1)
    
    t_norm = t / t[-1] # this (correctly) assumes t[0] = 0
    # ax3.axvline(t[-1], color='k', alpha=0.3)
    
    # append data to lists
    periods.extend(p)
    times.extend(t)
    times_norm.extend(t_norm)
    valids.extend(valid)
    durations.append(t[-1])

p = np.asarray(periods)
t = np.asarray(times)
t_norm = np.asarray(times_norm)
valid = np.asarray(valids)

### Plot and process data

color = scolor if KIND=='S' else lcolor
color = color if KIND is not None else 'C2'

# Plot data
ax1.plot(t[valid], p[valid], '.', c=color, rasterized=True, ms=3)
ax1.plot(t[~valid], p[~valid], '.', c=COLOR_INVALID, rasterized=True, ms=3)
# ax1.plot(t[valid], p[valid], '.', c='C0', alpha=0.02)

# Plot data on normalized time [0, 1]
ax2.plot(t_norm[valid], p[valid], '.', c=color, rasterized=True, ms=3)
ax2.plot(t_norm[~valid], p[~valid], '.', c=COLOR_INVALID, rasterized=True, ms=3)

# Average data in small windows
steps = 20
time_bins = np.linspace(t.min(), t.max(), steps)

for t0, t1 in zip(time_bins, time_bins[1:]):

    # For normal scale time
    where = np.logical_and(t>=t0, t<=t1)
    
    total_count = sum(where)
    valid_count = sum(np.logical_and(where, valid))
    tt = (t0+t1)/2

    ax3.plot(tt, valid_count/total_count, 'o', c=color)
    ax5.plot(tt, total_count, 'x', c='C3')
    
    # For normalized time
    t0 /= t.max()
    t1 /= t.max()
    
    where = np.logical_and(t_norm>=t0, t_norm<=t1)
    
    total_count = sum(where)
    valid_count = sum(np.logical_and(where, valid))
    tt = (t0+t1)/2

    ax4.plot(tt, valid_count/total_count, 'o', c=color)
    ax6.plot(tt, total_count, 'x', c='C3')
    
ylims3 = ax3.get_ylim()
ax3.hist(durations, bins=time_bins, bottom=ylims3[0 ], density=True, color='0.7')

### Format plots
ax5.set_yscale('log')

ax5.set_xlabel('Time [sec]')
ax6.set_xlabel('Normalzied time')

ax1.set_ylabel('Period [sec]')
ax3.set_ylabel('Valid period proportion')
ax5.set_ylabel('Period counts per bin')

ax1.set_title('Runs aligned at the beginning of the recording')
ax2.set_title('Time of each run normalized to [0,1]')

suptitle = 'Proportion of valid periods as time advances' + ('' if KIND is None else f' ({KIND} only)')
fig.suptitle(suptitle, fontsize=14)

ax1.set_ylim(0, 20)
ax3.set_ylim(0, 1.05)


#%% Quantify baselines

# data_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
data_dir = '/home/user/Documents/Doctorado/Fly clock/FlyClock_data/tiempos_post_diseccion'
data_dir = Path(data_dir) / 'output' / 'baseline_time_dependency'

info_dir = data_dir / 'baselines_info.csv'
data_files = contenidos(data_dir, filter_ext='.npz')

# load info
info = pd.read_csv(info_dir).sort_values('name').set_index('name')

# add tpd in seconds
info['tpd_sec'] = info.tpd * 60


fig, (ax1, ax2, axh) = plt.subplots(1, 3, constrained_layout=True, 
                                    figsize=[12, 3.2],
                                    width_ratios=[1, 1.6, 0.4], sharey=True)
mean_baseline = {}
stimes, sbaselines = [], []
ltimes, lbaselines = [], []
for file in data_files:
    loaded = np.load(file)
    
    celltype = info.loc[file.stem].type
    tpd = info.loc[file.stem].tpd
    
    mean_baseline[file.stem] = loaded['baselines'].mean()
    times = loaded['btimes']/60 + tpd
    baselines = loaded['baselines']
    
    times = times[baselines < -20]
    baselines = baselines[baselines < -20]
    
    if celltype =='S':
        stimes.extend(times)
        sbaselines.extend(baselines)
    else:
        ltimes.extend(times)
        lbaselines.extend(baselines)
    
    thecolor = scolor if celltype=='S' else lcolor
    thezorder = 2 if celltype=='S' else 1.9
    ax2.plot(times, baselines, c=thecolor, zorder=thezorder)
    
    if any(baselines<-80):
        print(file.name, celltype)
    
mean_baseline = pd.DataFrame.from_dict(mean_baseline, orient='index', columns=['baseline'])
info = info.join(mean_baseline)

colors_list = [scolor if row.type=='S' else lcolor for _, row in info.iterrows()]
ax1.scatter(info.tpd + info.duration/2, info.baseline, c=colors_list)

# ax2.plot(ltimes, lbaselines,  c=lcolor, label='L', rasterized=True)
# ax2.plot(stimes, sbaselines,  c=scolor, label='S', rasterized=True)

axh.hist([*sbaselines, *lbaselines], bins='auto', fc='0.6', orientation='horizontal')    
axh.hist(lbaselines, bins='auto', fc=lcolor, orientation='horizontal', alpha=0.4)
axh.hist(sbaselines, bins='auto', fc=scolor, orientation='horizontal', alpha=0.4)


# format plots
ax1.set_xlabel('tpd [min]')
ax1.set_title('Avearge baseline')
ax1.grid()

ax2.set_xlabel('tpd [min]')
ax2.set_title('Local baseline over time')
ax2.grid()

axh.set_xlabel('# counts')
axh.set_title('Distribution of baselines\n(data over time)')
axh.grid()
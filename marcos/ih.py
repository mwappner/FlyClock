#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:51:42 2023

@author: marcos
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import interpolate, stats

from utils import contenidos, enzip, find_point_by_value, find_closest_value, find_numbers, calc_mode
import analysis_utils as au


scolor = '#d06c9eff'
lcolor = '#006680ff'
#%% Visualzie raw data

import pyabf

# Load data
data_dir = '/media/marcos/DATA/marcos/FloClock_data/Ih'
# data_dir = '/media/marcos/DATA//marcos/FloClock_data/data - mecamilamina'
file_inx = 2

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

plt_slice = slice(120000,170000)
ax3.plot(times[plt_slice], ch1[plt_slice])
ax4.plot(times[plt_slice], ch2[plt_slice])

fig.suptitle(file.name)

print('file:', file.stem)
print('duration (min):', f'{times[-1]/60:.3f}')
print('sampling rate:', abf.sampleRate)

#%% One run

# Load data
# data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
data_dir = '/media/marcos/DATA/marcos/FloClock_data/Ih'
file_inx = 0
outlier_mode_proportion = 1.8 #1.8 for normal runs
max_distance = 0.3 # in units of average period

data_files = contenidos(data_dir, filter_ext='.abf')

# # data containing relevant intervals
# tpd_file = Path(data_dir) / 'info.xlsx'
# tpd_data = pd.concat(sheet for name, sheet in pd.read_excel(tpd_file, sheet_name=None).items() 
#                      if name in ('large', 'small'))
# tpd_data['registro'] = tpd_data.registro.astype(str)
# tpd_data = tpd_data.set_index('registro')

# rec_nr = data_files[file_inx].stem
# interval = tpd_data.loc[rec_nr]['rango_de_minutos'] if rec_nr in tpd_data.index else None

# Process data a bit
data = au.load_any_data(data_files[file_inx], gauss_filter=False)
data.process.lowpass_filter(frequency_cutoff=10, keep_og=True)
data = data.process.downsample()
data.process.highpass_filter(frequency_cutoff=0.1, keep_og=True, channels='lpfilt')
data.process.lowpass_filter(frequency_cutoff=2, keep_og=True, channels='lpfilt_hpfilt')

data.process.find_peaks(channels='lpfilt_hpfilt_lpfilt', period_percent=0.4, prominence=3)

# Plot data and peaks
fig, (ax1, ax2, ax_p, ax_l) = plt.subplots(4, 1, figsize=(12, 8), 
                                     constrained_layout=True, sharex=True, 
                                     height_ratios=[2,2,1,1])

mrising_outputs = []
P = np.polynomial.Polynomial
for ax, ch in zip((ax1, ax2), (1, 2)):
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
    
    # plot mode, mean and trend line
    valid_period_inxs = au.find_valid_periods(period_times, periods, outlier_mode_proportion, passes=2)
    valid_periods = periods[valid_period_inxs]
    period_mean = np.mean(valid_periods)
    modeline = ax_p.axhline(period_mode, color='0.3', linestyle='--', zorder=1)
    meanline = ax_p.axhline(period_mean, color='0.3', linestyle=':', zorder=1)
    
    trend_poly = P.fit(period_times[valid_period_inxs], valid_periods, 1)
    
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
    ax_p.errorbar(mrising_times, mrising_periods, mrising_ptp, fmt='.', color=f'C{ch}')
    
    # plot mode, mean and trend line for edge periods
    valid_period_inxs = au.find_valid_periods(mrising_times, mrising_periods, outlier_mode_proportion, passes=2)
    valid_periods = mrising_periods[valid_period_inxs]
    period_mean = np.mean(mrising_periods)
    period_mode = calc_mode(mrising_periods) 
    # modeline = ax_p.axhline(period_mode, color='0.3', linestyle='--', zorder=1)
    # meanline = ax_p.axhline(period_mean, color='0.3', linestyle=':', zorder=1)
    
    # plot average periods on the data
    mrising_indexes = np.asarray([find_point_by_value(data.times.values, tt) for tt in mrising_times-mrising_periods/2])
    ax.plot(data.times.values[mrising_indexes], data[f'ch{ch}_lpfilt_hpfilt_lpfilt'][mrising_indexes], '.', c='r')
    
    # trend_poly = P.fit(mrising_times[valid_period_inxs], valid_periods, 1)
    # trendline, = ax_p.plot(data.times, trend_poly(data.times), 'k', zorder=0.9)
    ax_p.plot(mrising_times[~valid_period_inxs], mrising_periods[~valid_period_inxs], 'xr')
    
    mrising_outputs.append(mrising_out)

    # handle single channel recodings
    if not data.metadata.twochannel:
        mrising_outputs.append(mrising_out)
        break


# calculate the lags
times1 = mrising_outputs[0][0]
times2 = mrising_outputs[1][0]

lag_times = []
lags = []
maxd = max_distance * np.mean(data.process.get_periods(1)[1])
for crossing_time in times1:
    
    closest_time = find_closest_value(times2, crossing_time)
    
    if np.abs(closest_time - crossing_time) > maxd:
        print('skipping')
        continue
    
    lag = crossing_time - closest_time
    lags.append(lag)
    lag_times.append( crossing_time - lag/2)
    
lag_times= np.asarray(lag_times)
lags = np.asarray(lags)

ax_l.plot(lag_times, lags, '.')

# format plots
fig.suptitle(data.metadata.file.stem)
ax_l.set_xlabel('time (s)')

ax1.set_ylabel('mV')
ax2.set_ylabel('mV')
ax_p.set_ylabel('period (s)')
ax_p.set_ylabel('lag (s)')

ax1.set_title('Channel 1')
ax2.set_title('Channel 2')

ax_l.grid()

print('Running', data.metadata.file.stem)



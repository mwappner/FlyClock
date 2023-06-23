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

from utils import contenidos, find_point_by_value, calc_mode, sort_by, cprint, clear_frame
import analysis_utils as au

#%% Time dependent period in one file

# Load data
# data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
data_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
file_inx = 80
outlier_mode_proportion = 1.6 #1.8 for normal runs

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
data = au.load_any_data(data_files[file_inx], interval=interval, gauss_filter=True, override_raw=False)
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

outlier_mode_proportion = 1.6 #1.8 for normal runs

info_dir = data_dir / 'periods_info.csv'
data_files = contenidos(data_dir, filter_ext='.npz')

# load info
info = pd.read_csv(info_dir).set_index('name')

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
for file, c in zip(data_files, color_list):
    # load data
    data = np.load(file)
    t = data['times'] / 60
    p = data['periods']
    tpd = info.loc[file.stem].tpd
    
    # plot points
    ax1.plot(t, p, '.', c=c)
    ax2.plot(t+tpd, p, '.', c=c)
    
    # plot trend lines
    period_mode = calc_mode(p)
    valid_indexes = p <= (period_mode * outlier_mode_proportion)
    trend_poly = P.fit(t[valid_indexes], p[valid_indexes], 1)
    
    # c = 'r' if trend_poly.convert().coef[1]>0 else 'b'
    if trend_poly.convert().coef[1]<0: 
        print(file.stem, 'has slope', f'{trend_poly.convert().coef[1]:.1e}')
    
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

fig, ax = plt.subplots(constrained_layout=True, figsize=[6.4, 3.3])

ax.plot(times_tpd, periods, '.', rasterized=True)
trend_poly = P.fit(times_tpd, periods, 1)
ax.plot(times_tpd, trend_poly(times_tpd), '--k')

ax.set_xlabel('time relative to tpd [min]')
ax.set_ylabel('period [s]')

pearson_tpd, _ = stats.pearsonr(times_tpd, periods)
mse_val = mse(periods, trend_poly(times_tpd))

ax.legend(handles=[], title = f'$r_{{pearson}}$ = {pearson_tpd:.2f}\nmse={mse_val:.2f}')

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
times_valid = []
times_tpd_valid = []
periods_valid = []
tpds = []
for file, c in zip(data_files, color_list):
    # load data
    data = np.load(file)
    t = data['times'] / 60
    p = data['periods']
    tpd = info.loc[file.stem].tpd
    
    # plot points
    ax2.plot(t+tpd, p, '.', c=c)
    
    # plot trend lines
    period_mode = calc_mode(p)
    valid_indexes = p <= (period_mode * outlier_mode_proportion)
    trend_poly = P.fit(t[valid_indexes], p[valid_indexes], 1)
    
    # append data to lists
    periods.extend(p)
    times_tpd.extend(t+tpd)
    tpds.append(tpd)
    
    times_valid.extend(t[valid_indexes])
    periods_valid.extend(p[valid_indexes])
    times_tpd_valid.extend(t[valid_indexes]+tpd)
    
def mse(y, y_fit):
    """ Mean square error: average of the square of the residuals of a fit"""
    return np.mean( (y-y_fit)**2 )

# plot fit for all data
times_tpd = np.asarray(times_tpd)
periods = np.asarray(periods)

trend_poly = P.fit(times_tpd, periods, 1)
label = f'mse = {mse(periods, trend_poly(times_tpd)):.2f}'
ax2.plot(times_tpd, trend_poly(times_tpd), c='k', label=label)

# make fit for "valid" data (under a threshold)
times_tpd = np.asarray(times_tpd_valid)
periods = np.asarray(periods_valid)

trend_poly = P.fit(times_tpd, periods, 1)
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
tpds = []
daytime = []
celltype = []
for file, c in zip(data_files, color_list):
    # load data
    data = np.load(file)
    t = data['times']
    p = data['periods']
    tpd = info.loc[file.stem].tpd
    time_hours = info.loc[file.stem].time_hours
    ctype = info.loc[file.stem]['type'] == 'S'
    
    initial_periods.append(np.mean(p[:40]))
    tpds.append(tpd)
    daytime.append(time_hours)
    celltype.append(ctype)

tpds = np.asarray(tpds)
initial_periods = np.asarray(initial_periods)
celltype = np.asarray(celltype)

# plot    
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 5), constrained_layout=True)

# ax1
sc = ax1.scatter(tpds, initial_periods, c=daytime)
ax1.set_xlabel('time since disection [sec]')
ax1.set_ylabel('initial period [sec]')
cbar = plt.colorbar(sc, label='time of day [hours]')

# trend line
trend = np.polynomial.Polynomial.fit(tpds, initial_periods, 1)
pearson, _ = stats.pearsonr(tpds, initial_periods)

ax1.plot(tpds, trend(tpds), 'k', label=f'$r_{{pearson}}$ = {pearson:.2f}')
ax1.legend(loc='lower right')

# ax2
# ax2.scatter(tpds, initial_periods, c=celltype, cmap='cool')
ax2.plot(tpds[celltype], initial_periods[celltype], 'o', label='small')
ax2.plot(tpds[~celltype], initial_periods[~celltype], 'o', label='large')
ax2.set_xlabel('time since disection [sec]')
ax2.set_ylabel('initial period [sec]')

# trend line
trend = np.polynomial.Polynomial.fit(tpds, initial_periods, 1)
pearson, _ = stats.pearsonr(tpds, initial_periods)

ax2.plot(tpds, trend(tpds), 'k', label=f'$r_{{pearson}}$ = {pearson:.2f}')
ax2.legend(loc='lower right')


#%% Plot initial period as a function of TOD

"""NOTE: this requires normalizing the period to the value extrapolated from 
its TPD, so you need to run the previous cell before this one"""

detrended_periods = np.asarray(initial_periods)# - trend(np.asarray(tpds))

plt.figure()
plt.plot(daytime, detrended_periods,'.')
plt.xlabel('time of day [hours]')
plt.ylabel('period (normalized) [sec]')

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

data_dir = '/media/marcos/DATA/marcos/FloClock_data/tiempos_post_diseccion'
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
    
            
axarr[0].set_title(file.stem)
axarr[0].set_ylabel('valid proportion\none pass')
axarr[1].set_ylabel('valid proportion\ntwo passes')
axarr[1].set_xlabel('threshold')


#%% Set outlier_mode_proportion threshold

""" Use this to set the threshold that selects outliners when calculating
period trends. See for a few runs what values look right."""

data_dir = '/media/marcos/DATA/marcos/FloClock_data/data'
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

# load and plot data
P = np.polynomial.Polynomial
slopes = []
slopes_valid = []
durations = []
for file in data_files:
    # load data
    data = np.load(file)
    t = data['times']
    p = data['periods']
    tpd = info.loc[file.stem].tpd
    
    # fit trendlines
    period_mode = calc_mode(p)
    valid_indexes = p <= (period_mode * outlier_mode_proportion)
    
    trend_poly = P.fit(t, p, 1)
    trend_poly_valid = P.fit(t[valid_indexes], p[valid_indexes], 1)
    
    # append data to lists
    
    slopes.append(trend_poly.convert().coef[1])    
    slopes_valid.append(trend_poly_valid.convert().coef[1])
    durations.append(t[-1]/60)
    
# put data into dataframe
slope_df = pd.DataFrame({'slope':slopes, 'slope_valid':slopes_valid, 'duration':durations},
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

# slope histograms
ax1.hist(slopes, bins='auto', orientation='horizontal', fc='0.6')
ax2.hist(slopes_valid, bins='auto', orientation='horizontal', fc='0.6')

# slope vs duration
ax3.plot(info.duration[~single_channel_recs], info.slope[~single_channel_recs], '.')
ax4.plot(info.duration[~single_channel_recs], info.slope_valid[~single_channel_recs], '.', label='dual channel')

# color the sigle channel recordings differently
ax3.plot(info.duration[single_channel_recs], info.slope[single_channel_recs], '.')
ax4.plot(info.duration[single_channel_recs], info.slope_valid[single_channel_recs], '.', label='single channel')

# slope vs tpd
ax5.plot(info.tpd[~single_channel_recs], info.slope[~single_channel_recs], '.')
ax6.plot(info.tpd[~single_channel_recs], info.slope_valid[~single_channel_recs], '.')

# color the sigle channel recordings differently
ax5.plot(info.tpd[single_channel_recs], info.slope[single_channel_recs], '.')
ax6.plot(info.tpd[single_channel_recs], info.slope_valid[single_channel_recs], '.')


for ax in left:
    ax.set_ylabel('slope')

ax2.set_xlabel('counts')
ax4.set_xlabel('duration of rec. [min]')
ax6.set_xlabel('tpd [min]')

ax3.set_title('All periods')
ax4.set_title('"Valid" periods only')
ax4.legend() 

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
data_dir = Path(data_dir) / 'output' / 'frequency_time_dependency'

outlier_mode_proportion = 1.6 #1.8 for normal runs

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
    # load data
    data = np.load(file)
    t = data['times'] / 60
    p = data['periods']
    
    # plot trend lines
    period_mode = calc_mode(p)
    valid = p <= (period_mode * outlier_mode_proportion)
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

# Plot data
ax1.plot(t[valid], p[valid], '.', c='C0', rasterized=True)
ax1.plot(t[~valid], p[~valid], '.', c='C1', rasterized=True)
# ax1.plot(t[valid], p[valid], '.', c='C0', alpha=0.02)

# Plot data on normalized time [0, 1]
ax2.plot(t_norm[valid], p[valid], '.', c='C0', rasterized=True)
ax2.plot(t_norm[~valid], p[~valid], '.', c='C1', rasterized=True)

# Average data in small windows
steps = 20
time_bins = np.linspace(t.min(), t.max(), steps)

for t0, t1 in zip(time_bins, time_bins[1:]):

    # For normal scale time
    where = np.logical_and(t>=t0, t<=t1)
    
    total_count = sum(where)
    valid_count = sum(np.logical_and(where, valid))
    tt = (t0+t1)/2

    ax3.plot(tt, valid_count/total_count, 'o', c='C2')
    ax5.plot(tt, total_count, 'x', c='C3')
    
    # For normalized time
    t0 /= t.max()
    t1 /= t.max()
    
    where = np.logical_and(t_norm>=t0, t_norm<=t1)
    
    total_count = sum(where)
    valid_count = sum(np.logical_and(where, valid))
    tt = (t0+t1)/2

    ax4.plot(tt, valid_count/total_count, 'o', c='C2')
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
fig.suptitle('Proportion of valid periods as time advances, by binning the time', fontsize=14)

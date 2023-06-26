#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:01:58 2023

@author: marcos
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np
from scipy import interpolate, signal

from utils import contenidos, find_point_by_value, calc_mode, sort_by, enzip
import analysis_utils as au

import pyabf

#%% Check out one file

BASE_DIR = '/media/marcos/DATA/marcos/FloClock_data/phase_response_curve'
files = contenidos(BASE_DIR, filter_ext='.abf')

file = files[-2]

# Load data
abf = pyabf.ABF(file)

fig, axarr = plt.subplots(abf.sweepCount, sharex=True)

# Set sweep and channel, extract tmes and data

for i, ax in enumerate(axarr):
    abf.setSweep(sweepNumber=i)
    times = abf.sweepX
    data = abf.sweepY
    perturbation = abf.sweepC
    
    ax.plot(times, data)
    ax.plot(times, perturbation+data.mean(), c='C1')
    ax.grid()
    
    for p1 in abf.sweepEpochs.p1s:
        ax.axvline(times[p1], color='k', ls='--', alpha=.5)

ax.set_xlabel('time [sec]')
ax.set_xlim(times.min(), times.max())

#%% Cut sweeps at stimuli to line them all up (check if we have sufficiently scanned everything)


BASE_DIR = '/media/marcos/DATA/marcos/FloClock_data/phase_response_curve'
files = contenidos(BASE_DIR, filter_ext='.abf')

file = files[-5]

# Load data
abf = pyabf.ABF(file)

fig, ax = plt.subplots()

# Set sweep and channel, extract tmes and data

for i in abf.sweepList:
    # load sweep and unpack data
    abf.setSweep(sweepNumber=i)
    times = abf.sweepX
    data = abf.sweepY
    perturbation = abf.sweepC
    
    # find points where perturbations happened and rewind tree datapoints
    if perturbation.mean() > 0:
        perturbation_indexes = np.nonzero(np.diff(perturbation) > 0)[0] - 3
    else:
        perturbation_indexes = np.nonzero(np.diff(perturbation) < 0)[0] - 3
    perturbation_indexes = np.append(perturbation_indexes, -1) # add the last datapoint too
    
    for start, end in zip(perturbation_indexes, perturbation_indexes[1:]):
        interval = slice(start, end, 10)
        ax.plot(data[interval], c='C0')
    

# ax.set_xlabel('time [sec]')
# ax.set_xlim(times.min(), times.max())

#%% Cut sweeps at oscillations and line them up

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

def polydetrend(times, data, degree=5):
    
    P = np.polynomial.Polynomial
    
    # some filter methods leave behind nans in the data, that raises a LinAlgError
    nan_locs = np.isnan(data)
    
    # fit the data
    trend_poly = P.fit(times[~nan_locs], data[~nan_locs], degree)
    
    # remove the trend from the data, this reintroduces the nans
    detrended = data - trend_poly(times)
    
    # save processed data
    return detrended, trend_poly
    
def abf_gauss_filt(abf, sigma=20, sweep=None):
    
    pyabf.filter.gaussian(abf, sigma, channel=0)
    
    if sweep is None:
        data = []
        for sweep in abf.sweepList:
            abf.setSweep(sweepNumber=sweep, channel=0)        
            filtered_data = abf.sweepY
            data.append(filtered_data)
        
        return data
    else:
        abf.setSweep(sweepNumber=sweep, channel=0)        
        filtered_data = abf.sweepY
        
        return filtered_data    
    

def gauss_filt(times, data, sigma_ms=100):
    
    # some filter methods leave behind nans in the data, that raises a LinAlgError
    nan_locs = np.isnan(data)
    
    # calculate the sigma in untis of datapoints
    sampling_rate = 1/(times[1]-times[0])
    sigma_points = (sigma_ms / 1000) * sampling_rate
    
    filtered = gaussian_filter1d(data[~nan_locs], sigma_points)
    
    nan_locs = np.isnan(data)

    # if np.sum(~nan_locs) != data.size:
    #     raise ValueError("The size of the input data couldn't be matched to the non nan values in the original data")

    ch1_nans = np.full(data.shape, np.nan)
    ch1_nans[~nan_locs] = filtered
    filtered = ch1_nans
    
    return filtered


def my_find_peaks(times, data, period_percent=0.4, prominence=3):
    
    ## first pass, with threshold 0mv
    p_inx, _ = find_peaks(data, height=0)
    
    ## second pass, with threshold given by otsu
    # threshold = Processors.get_threshold(peaks)
    # p_inx, _ = signal.find_peaks(ch, height=threshold)
    # peaks = ch.values[p_inx]
    t_peaks = times[p_inx]

    ## third pass, with minimum distance between peaks
    counts, bins = np.histogram(np.diff(t_peaks))
    bin_centers = bins[:-1] + np.diff(bins) / 2
    period_mode = bin_centers[ np.argmax(counts) ]
    distance_points = int(period_mode * period_percent / (times[1] - times[0]))
    # p_inx, _ = signal.find_peaks(ch, height=threshold, distance=distance_points)
    p_inx, _ = find_peaks(data, distance=distance_points, prominence=prominence)
    
    return p_inx
    


BASE_DIR = '/media/marcos/DATA/marcos/FloClock_data/phase_response_curve'
BASE_DIR = '/home/user/Documents/Doctorado/Fly clock/FlyClock_data/phase_response_curve'

files = contenidos(BASE_DIR, filter_ext='.abf')

file = files[-2]

# Load data
abf = pyabf.ABF(file)
fig, ax = plt.subplots()
fig, ax2 = plt.subplots()
    
for sweep in abf.sweepList[:2]:
    print('Processing sweep', sweep)
    
    # raw data
    abf.setSweep(sweep)
    times = abf.sweepX
    data = abf.sweepY
    ax.plot(times, data)
    
    # first gaussian filter
    data = abf_gauss_filt(abf, sweep=sweep)
    ax.plot(times, data)
    
    # downsample
    times_d = times[::10]
    data = data[::10]
    
    # polynomial detrend
    data, trend = polydetrend(times_d, data)
    ax.plot(times_d, data)
    ax.plot(times, trend(times), c='C2')
    
    # second (harsher) gaussian filter
    data = gauss_filt(times_d, data)
    ax.plot(times_d, data)
    
    # find maxima
    peaks = my_find_peaks(times_d, data)
    ax.plot(times_d[peaks], data[peaks], 'o')
    
    mean_period_in_points = round(np.diff(peaks).mean())
    cut_points = peaks #- int(mean_period_in_points/2)
    cut_points = np.append(cut_points, -1)
    ax.vlines(times_d[cut_points], -10, 20, colors='0.5', linestyles='--')
    
    # add perturbation
    perturbation = abf.sweepC
    ax.plot(times, perturbation/3 -5, 'C0')
    
    
    for start, end in zip(cut_points, cut_points[1:]):
        interval = slice(start, end)
        ax2.plot( data[interval])
        
        ax2.plot( perturbation[::10][interval], c='k')

#%% Plot periods for the whole run (concatenate sweeps)

""" Run previous cell for function deffinitions """


BASE_DIR = '/media/marcos/DATA/marcos/FloClock_data/phase_response_curve'
BASE_DIR = '/home/user/Documents/Doctorado/Fly clock/FlyClock_data/phase_response_curve'
files = contenidos(BASE_DIR, filter_ext='.abf')

file = files[19]
threshold = 10

# Load data
abf = pyabf.ABF(file)
fig, (ax, ax2, axp) = plt.subplots(3, sharex=True, figsize=[17.69,  7.29], constrained_layout=True)
# fig, ax2 = plt.subplots()

# to prime it for the first sweep
times = [0]

for sweep in abf.sweepList:
    print('Processing sweep', sweep)
    
    # raw data
    abf.setSweep(sweep)
    times = abf.sweepX + times[-1]
    data = abf.sweepY
    raw_a, = ax.plot(times, data, 'C0', label='raw data')


# reload the data with a small gaussian filtering
times = [0]
pyabf.filter.gaussian(abf, sigmaMs=20)

all_time = []
all_data = []
all_pert = []
for sweep in abf.sweepList:
    print('Processing sweep', sweep)

    # first gaussian filter
    abf.setSweep(sweep)
    times = abf.sweepX + times[-1]
    data = abf.sweepY
    filt_a, = ax.plot(times, data, 'C1', label='small gaussian filter')

    ax.axvline(times[-1], linestyle='--', color='0.7')
    
    all_time.append(times)
    all_data.append(data)
    all_pert.append(abf.sweepC)

time = np.concatenate(all_time)
data = np.concatenate(all_data)
perturbation = np.concatenate(all_pert)

# add perturbation
ax.plot(time, np.nanmean(data)+perturbation/3 -5, 'k', zorder=2.1)
ax2.plot(time, perturbation/3 -5, 'k', zorder=2.1)

# downsample
time = time[::10]
data = data[::10]

# polynomial detrend
data, trend = polydetrend(time, data)
ax2.plot(time, data, label='detrended')
trend_a, = ax.plot(time, trend(time), c='C2', label='trendline')

# second (harsher) gaussian filter
data = gauss_filt(time, data)
ax2.plot(time, data, label='aggressive filtering')

# find maxima
peaks = my_find_peaks(time, data)
ax2.plot(time[peaks], data[peaks], 'o')

mean_period_in_points = round(np.diff(peaks).mean())
# cut_points = peaks - int(mean_period_in_points/2)
# cut_points = np.append(cut_points, -1)
# ax.vlines(time[cut_points], -10, 20, colors='0.5', linestyles='--')

# find crossover points
rising_edge = []
falling_edge = []
filtered_peaks = []
prev_peak = -np.inf
for peak in peaks:
    
    # skip maxima that are too low
    if data[peak] < threshold:
        continue
    
    # skip maxima that are too close together
    if peak - prev_peak < mean_period_in_points / 2:
        continue
    
    # find rising edge point (before peak)
    interval = data[:peak]
    raising = np.nonzero(interval < threshold)[0][-1]
    
    # find falling edge point (after peak)    
    starting_point = min(peak + int(mean_period_in_points / 2), len(data))
    interval = data[:starting_point]
    falling = np.nonzero(interval > threshold)[0][-1]
    
    rising_edge.append(raising)
    falling_edge.append(falling)
    filtered_peaks.append(peak)
    prev_peak = peak

rising_edge = np.asarray(rising_edge)
falling_edge = np.asarray(falling_edge)
peaks = np.asarray(filtered_peaks)
ax2.plot(time[rising_edge], data[rising_edge], 'o')
ax2.plot(time[falling_edge], data[falling_edge], 'o')

# calculate periods for the three different datapoints

# rising edge
rperiods = np.diff(time[rising_edge])
axp.plot(time[rising_edge][:-1], rperiods, 'x', c='C3', label='rising edge')

# fallign edge
fperiods = np.diff(time[falling_edge])
axp.plot(time[falling_edge][:-1], fperiods, 'x', c='C4', label='falling edge')

# peaks
pperiods = np.diff(time[peaks])
axp.plot(time[peaks][:-1], pperiods, '*', c='C2', label='maximum')
axp.axhline(mean_period_in_points*(time[1]-time[0]), color='k', label='average')

axp.set_xlabel('time [sec]')

ax.grid(axis='x')
ax2.grid(axis='x')
axp.grid(axis='x')

ax.legend(handles=[raw_a, filt_a, trend_a], loc='lower left')
ax2.legend(loc='lower left')
axp.legend(loc='upper left', ncol=4)
axp.set_ylabel('period [sec]')


#%% Try to get PRC

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


BASE_DIR = '/media/marcos/DATA/marcos/FloClock_data/phase_response_curve'
BASE_DIR = '/home/user/Documents/Doctorado/Fly clock/FlyClock_data/phase_response_curve'
files = contenidos(BASE_DIR, filter_ext='.abf')

file = files[17]
threshold = 8

# Load data
abf = pyabf.ABF(file)

# to prime it for the first sweep
times = [0]
all_time = []
all_data = []
all_pert = []

for sweep in abf.sweepList:
    print('Processing sweep', sweep)

    # first gaussian filter
    abf.setSweep(sweep)
    times = abf.sweepX + times[-1]
    data = abf.sweepY
    
    all_time.append(times)
    all_data.append(data)
    all_pert.append(abf.sweepC)

time = np.concatenate(all_time)
data = np.concatenate(all_data)
perturbation = np.concatenate(all_pert)

# filter data a bit
data_lp = lowpass_filter(time, data, filter_order=2, frequency_cutoff=10)
data_lp = data_lp[::10]
time = time[::10]
data = data[::10]
perturbation = perturbation[::10]
data_hp = highpass_filter(time, data_lp, filter_order=2, frequency_cutoff=0.1)
data_hplp = lowpass_filter(time, data_lp, filter_order=2, frequency_cutoff=2)


# plot
fig, (ax, ax2, axp) = plt.subplots(3, sharex=True, figsize=[17.69,  7.29], constrained_layout=True)

ax.plot(time, data)
ax.plot(time, perturbation + data.mean())
ax.plot(time, data_lp)

ax2.plot(time, data_hp)
ax2.plot(time, perturbation + data_hp.mean())

data = data_hp
# find peaks and edge crossings

peaks = my_find_peaks(time, data)
ax2.plot(time[peaks], data[peaks], 'o')

mean_period_in_points = round(np.diff(peaks).mean())

# find crossover points
rising_edge = []
falling_edge = []
filtered_peaks = []
prev_peak = -np.inf
for peak in peaks:
    
    # skip maxima that are too low
    if data[peak] < threshold:
        continue
    
    # skip maxima that are too close together
    if peak - prev_peak < mean_period_in_points / 2:
        continue
    
    # find rising edge point (before peak)
    interval = data[:peak]
    raising = np.nonzero(interval < threshold)[0][-1]
    
    # find falling edge point (after peak)    
    starting_point = min(peak + int(mean_period_in_points / 2), len(data))
    interval = data[:starting_point]
    falling = np.nonzero(interval > threshold)[0][-1]
    
    rising_edge.append(raising)
    falling_edge.append(falling)
    filtered_peaks.append(peak)
    prev_peak = peak

rising_edge = np.asarray(rising_edge)
falling_edge = np.asarray(falling_edge)
peaks = np.asarray(filtered_peaks)

# plot peaks and threshold crossings
ax2.plot(time[peaks], data[peaks], 'ko')
ax2.plot(time[rising_edge], data[rising_edge], 'o')

# find cycles with a perturbation
pert_inxs,  = np.where(np.diff(perturbation) < -1)
pert_moments = time[pert_inxs]
ax2.plot(pert_moments, perturbation[pert_inxs], 'kx')

cross_moments = time[rising_edge]

perturbed_cylces = []
perturb_phase = []
try:
    for i, (re1, re2) in enzip(rising_edge, rising_edge[1:]):
        # find the first perturbation after the current crossing
        for pert in pert_inxs:
            if pert > re1:
                break
        else:
            raise StopIteration()
        
        # if te perturbation happened before the next peak, store it
        if pert <= re2:
            perturbed_cylces.append(i)
            
            # get phase of perturbation
            phase = (pert-re1) / (re2-re1)
            perturb_phase.append(phase)
            
except StopIteration:
    # this happens if there are no more perturbations to find
    pass

# mark the cycles that have a perturbation
for inx in perturbed_cylces:
    start = rising_edge[inx]
    end = rising_edge[inx+1]
    ax2.plot(time[start:end], data_hp[start:end], 'r')
    
for pi, ph in zip(pert_inxs, perturb_phase):
    ax2.text(time[pi], perturbation[pi]+0.5, f'{ph:.2f}')

# calculate and store the (normalzied) Δipi
pert_dipi = []
prev_dipi = []
for inx in perturbed_cylces:

    re_2 = rising_edge[inx-2]
    re_1 = rising_edge[inx-1]
    re0 = rising_edge[inx]
    re1 = rising_edge[inx+1]
    
    pert_ipi = time[re1] - time[re0]
    base_ipi = time[re0] - time[re_1]
    prev_ipi = time[re_1] - time[re_2]

    pert_dipi.append( (pert_ipi - base_ipi) / base_ipi )
    prev_dipi.append( (prev_ipi - base_ipi) / base_ipi )
    
pert_dipi = np.asarray(pert_dipi)
prev_dipi = np.asarray(prev_dipi)
dipi_times = time[rising_edge[perturbed_cylces]]

# plot them
axp.plot(dipi_times, pert_dipi, 'x')
axp.plot(dipi_times, prev_dipi, '+')

# plot PRC
fig2, ax_prc = plt.subplots()

ax_prc.plot(perturb_phase, pert_dipi, 'x')
ax_prc.plot(perturb_phase, prev_dipi, '+')

#%%

file = '/media/marcos/DATA/marcos/FloClock_data/Raw data preps con ojos/14320023.abf'
# file = '/home/user/Documents/Doctorado/Fly clock/FlyClock_data/phase_response_curve'

abf = pyabf.ABF(file)
abf.setSweep(0)

times = abf.sweepX[::10]
data = abf.sweepY[::10]
plt.plot(times, data)

pyabf.filter.gaussian(abf, sigmaMs=20)

times = abf.sweepX[::10]
data = abf.sweepY[::10]
plt.plot(times, data)
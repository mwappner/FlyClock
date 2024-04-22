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
from scipy import interpolate, signal, optimize

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from utils import contenidos, find_point_by_value, calc_mode, sort_by, enzip, smooth, kde_scatter
import analysis_utils as au

import pyabf

#%% Check out one file

# BASE_DIR = '/media/marcos/DATA/marcos/FloClock_data/phase_response_curve'
BASE_DIR = '/home/user/Documents/Doctorado/Fly clock/FlyClock_data/phase_response_curve'

files = contenidos(BASE_DIR, filter_ext='.abf')

file = files[-2]
print('Running', file.stem)

# Load data
abf = pyabf.ABF(file)

fig, axarr = plt.subplots(abf.sweepCount, sharex=True)
axarr = np.atleast_1d(axarr)

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


# BASE_DIR = '/media/marcos/DATA/marcos/FloClock_data/phase_response_curve'
BASE_DIR = '/home/user/Documents/Doctorado/Fly clock/FlyClock_data/phase_response_curve'

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



#%% Estimate the relaxation time


# BASE_DIR = '/media/marcos/DATA/marcos/FloClock_data/phase_response_curve'
BASE_DIR = '/home/user/Documents/Doctorado/Fly clock/FlyClock_data/phase_response_curve'

files = contenidos(BASE_DIR, filter_ext='.abf')

file = files[-5]

# Load data
abf = pyabf.ABF(file)

# fig, ax = plt.subplots()

def exp(t, A, t0, l, c):
    return A * np.exp(-(t-t0)/l) + c

# Set sweep and channel, extract tmes and data
relaxation = []
relaxation_error = []
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
    
    fig, axarr = plt.subplots(3,3, constrained_layout=True)
    # plot the perturbations
    for start, ax in zip(perturbation_indexes, axarr.flat):
        # find the interesting bit
        end = start+5000
        interval = slice(start, end, 10)
        bit = data[interval]
        tbit = times[interval] - times[start]
        
        # find the peak
        peak = signal.find_peaks(-bit, height=75, prominence=5)[0][0]
        
        # fit a curve
        p0 = [bit[peak]-bit[peak:].max(), tbit[peak], 0.1, bit[peak:].max()]
        popt, pcov = optimize.curve_fit(exp, tbit[peak:], bit[peak:], p0)
        
        relax = popt[2]
        relax_err = np.sqrt(np.diag(pcov))[2]
        
        relaxation.append(relax)
        relaxation_error.append(relax_err)
        
        # plot
        ax.plot(tbit, bit)
        ax.plot(tbit[peak:], exp(tbit[peak:], *popt), 'r--')
        ax.plot(tbit[peak:], exp(tbit[peak:], *p0), 'k--')
        ax.plot(tbit[peak], bit[peak], 'ok')
        
        ax.set_title(f'{relax:3f}')

relaxation = np.asarray(relaxation)
relaxation_error = np.asarray(relaxation_error)

mean_relax = np.average(relaxation, weights = 1/relaxation_error**2)

print(f'Relaxation time = {mean_relax:.4f}sec')

#%% Cut sweeps at oscillations and line them up

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

file = files[40]
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


#%% Get PRC

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

# BASE_DIR = Path('/media/marcos/DATA/marcos/FloClock_data/phase_response_curve')
BASE_DIR = '/home/user/Documents/Doctorado/Fly clock/FlyClock_data/phase_response_curve'
files = contenidos(BASE_DIR, filter_ext='.abf')

file = files[26] # using 40, 39, 38, 36, 35, 34, 33, 32, 30, 29, 26
threshold = 5
METHOD = 'rising' # one of 'rising', 'falling' or 'peaks'
SMOOTH = True # decides if we should do a harsh lowpass filter on the signal

# Load data
abf = pyabf.ABF(file)
print('Running', file.stem)

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
if SMOOTH:
    data_hp = lowpass_filter(time, data_hp, filter_order=2, frequency_cutoff=2)

# plot
fig_main = plt.figure(figsize=[17.69,  7.29], constrained_layout=True)
fig, fig2 = fig_main.subfigures(1,2, width_ratios=[3,1])
ax, ax2, axp = fig.subplots(3, sharex=True)
ax_prc = fig2.subplots()

ax.plot(time, data, label='raw')
ax.plot(time, perturbation + data.mean(), label='perturb.')
ax.plot(time, data_lp, label='lowpass $f_c$=10')
ax.plot(time, lowpass_filter(time, data_lp, filter_order=2, frequency_cutoff=2), label='lowpass $f_c$=2')

ax2.plot(time, data_hp, label='detrended')
ax2.plot(time, perturbation + data_hp.mean(), label='perturb.')

data = data_hp
# find peaks and edge crossings

peaks = my_find_peaks(time, data)
ax2.plot(time[peaks], data[peaks], 'o', label='peaks')

mean_period_in_points = round(np.diff(peaks).mean())

# find crossing points
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

### Choose which event to use as reference
if METHOD == 'rising':
    event_index_array = rising_edge # one of rising_edge, falling_edge, peaks
elif METHOD == 'falling':
    event_index_array = falling_edge # one of rising_edge, falling_edge, peaks
elif METHOD == 'peaks':
    event_index_array = peaks # one of rising_edge, falling_edge, peaks
else:
    raise ValueError('"METHOD" has to be one of "rising", "falling" or "peaks"')

# plot peaks and threshold crossings
ax2.plot(time[peaks], data[peaks], 'k.', label='filtered_peaks')
ax2.plot(time[rising_edge], data[rising_edge], 'o', label='edge crossings')

# find cycles with a perturbation
pert_inxs,  = np.where(np.diff(perturbation) < -1)
pert_moments = time[pert_inxs]
ax2.plot(pert_moments, perturbation[pert_inxs], 'kx')

cross_moments = time[event_index_array]

# filter out cases where we can't detect the cycle because the perturbation fell right on it
cross_period = np.diff(cross_moments)
period_times = cross_moments[:-1] + cross_period/2
valid = au.find_valid_periods(period_times, cross_period, passes=2, threshold=1.6)

perturbed_cylces = []
perturb_phase = []
invalid_perturbed_cylces = []
try:
    for i, (e1, e2, v) in enzip(event_index_array, event_index_array[1:], valid):
        # find the first perturbation after the current crossing
        for pert in pert_inxs:
            if pert > e1:
                break
        else:
            raise StopIteration()
        
        # if te perturbation happened before the next peak, store it
        if pert <= e2:
            
            # keep it only if the period in question was valid
            if v:
                perturbed_cylces.append(i)
                
                # get phase of perturbation
                phase = (pert-e1) / (e2-e1) # here dividing indexes or actual time values is identical
                perturb_phase.append(phase)
            else:
                invalid_perturbed_cylces.append(i)
            
except StopIteration:
    # this happens if there are no more perturbations to find
    pass

# mark the cycles that have a perturbation
for inx in perturbed_cylces:
    start = event_index_array[inx]
    end = event_index_array[inx+1]
    ax2.plot(time[start:end], data_hp[start:end], 'r')

for inx in invalid_perturbed_cylces:
    start = event_index_array[inx]
    end = event_index_array[inx+1]
    # ax2.plot(time[start:end], data_hp[start:end], 'r')
    ax2.fill_betweenx([data_hp.min(), data_hp.max()], time[start], time[end], color='0.7', zorder=1)

# plot one extra patch to label it
ax2.fill_betweenx([data_hp.min(), data_hp.max()], time[start], time[end], color='0.7', zorder=1, label='invalid')

for pi, ph in zip(pert_inxs, perturb_phase):
    ax2.text(time[pi], perturbation[pi]+0.5, f'{ph:.2f}')

# calculate and store the (normalzied) ΔCD
pert_dcd = []
prev_dcd = []
for inx in perturbed_cylces:

    e_1 = event_index_array[inx-2]
    e0 = event_index_array[inx-1]
    e1 = event_index_array[inx]
    e2 = event_index_array[inx+1]
    
    pert_cd = time[e2] - time[e1]
    base_cd = time[e1] - time[e0]
    prev_cd = time[e0] - time[e_1]

    pert_dcd.append( (pert_cd - base_cd) / base_cd )
    prev_dcd.append( (prev_cd - base_cd) / base_cd )
    
    # prev_dcd.append( (base_cd - prev_cd) / prev_cd )
    
    
# pert_dcd = np.asarray(pert_dcd)
# prev_dcd = np.asarray(prev_dcd)
# dcd_times = time[event_index_array[perturbed_cylces]]

# convert lists to sorted arrays
pert_dcd = np.asarray( sort_by(pert_dcd, perturb_phase))
prev_dcd = np.asarray( sort_by(prev_dcd, perturb_phase))
dcd_times = np.asarray( sort_by(time[event_index_array[perturbed_cylces]],perturb_phase))
perturb_phase = np.asarray(sorted(perturb_phase))


# plot them
axp.plot(dcd_times, pert_dcd, 'x', label=r'$T_0$')
# axp.plot(dcd_times, prev_dcd, '+', label=r'$T_{-1}$', c='k')
axp.plot(dcd_times, prev_dcd, '+', label=r'$T_{-2}$', c='k')

# plot PRC

ax_prc.plot(perturb_phase, pert_dcd, 'o', c='C0')
ax_prc.plot(perturb_phase, smooth(pert_dcd, size=9), c='C0', label='smoothed PRC')
ax_prc.plot(perturb_phase, smooth(prev_dcd, size=9), c='k', label='smoothed reference')
ax_prc.plot(perturb_phase, prev_dcd, '+', c='k')
ax_prc.fill_between([0, 1],-prev_dcd.std(), prev_dcd.std(), color='0.6', label='algo. detection limit')
# av = np.abs(prev_dcd).mean()
# ax_prc.fill_between([0, 1],-av, av, color='C1', alpha=0.5)

# Format plots
ax_prc.grid()
ax_prc.set_xlabel('Phase [normalized]')
ax_prc.set_ylabel(r'ΔCD = $\frac{T_0-T_{-1}}{T_{-1}}$')
ax_prc.set_title(f'PRC from {METHOD}')

ax.set_title('Time series and analysis')
axp.set_xlabel('Time [sec]')
ax.set_ylabel('mV')
ax2.set_ylabel('detrended signal [mV]')
axp.set_ylabel('periods [s]')

ax.legend()
ax2.legend()
axp.legend()
axp.grid()
ax_prc.legend()

#%% Save all methods for one file

"""Run previous cell first"""

for METHOD in ('rising', 'falling', 'peaks'):
    ### Choose which event to use as reference
    if METHOD == 'rising':
        event_index_array = rising_edge # one of rising_edge, falling_edge, peaks
    elif METHOD == 'falling':
        event_index_array = falling_edge # one of rising_edge, falling_edge, peaks
    elif METHOD == 'peaks':
        event_index_array = peaks # one of rising_edge, falling_edge, peaks
    else:
        raise ValueError('"METHOD" has to be one of "rising", "falling" or "peaks"')
    
    # find cycles with a perturbation
    pert_inxs,  = np.where(np.diff(perturbation) < -1)
    pert_moments = time[pert_inxs]
    
    cross_moments = time[event_index_array]
    
    # filter out cases where we can't detect the cycle because the perturbation fell right on it
    cross_period = np.diff(cross_moments)
    period_times = cross_moments[:-1] + cross_period/2
    valid = au.find_valid_periods(period_times, cross_period, passes=2, threshold=1.6)
    
    perturbed_cylces = []
    perturb_phase = []
    try:
        for i, (re1, re2, v) in enzip(event_index_array, event_index_array[1:], valid):
            # find the first perturbation after the current crossing
            for pert in pert_inxs:
                if pert > re1:
                    break
            else:
                raise StopIteration()
            
            # keep it only if the period in question was valid
            if not v:
                continue
            
            # if te perturbation happened before the next peak, store it
            if pert <= re2:
                perturbed_cylces.append(i)
                
                # get phase of perturbation
                phase = (pert-re1) / (re2-re1) # here dividing indexes or actual time values is identical
                perturb_phase.append(phase)
                
    except StopIteration:
        # this happens if there are no more perturbations to find
        pass
    
    # calculate and store the (normalzied) ΔCD
    pert_dcd = []
    prev_dcd = []
    for inx in perturbed_cylces:
    
        # locations of the events (previous cycle, perturbed cycle, next cycle)
        e_1 = event_index_array[inx-2]
        e0 = event_index_array[inx-1]
        e1 = event_index_array[inx]
        e2 = event_index_array[inx+1]
        
        pert_cd = time[e2] - time[e1]
        base_cd = time[e1] - time[e0]
        prev_cd = time[e0] - time[e_1]
    
        pert_dcd.append( (pert_cd - base_cd) / base_cd )
        prev_dcd.append( (prev_cd - base_cd) / base_cd )
        
    # convert lists to sorted arrays
    pert_dcd = np.asarray( sort_by(pert_dcd, perturb_phase))
    prev_dcd = np.asarray( sort_by(prev_dcd, perturb_phase))
    perturb_phase = np.asarray(sorted(perturb_phase))

    np.savez(BASE_DIR / 'output' / 'PRC' / METHOD / file.stem,
            pert_dcd = pert_dcd,
            prev_dcd = prev_dcd,
            perturb_phase = perturb_phase,
            
            pert_cd = pert_cd,
            base_cd = base_cd,
            prev_cd = prev_cd,
            
            time = time,
            event_index_array = event_index_array
            )


#%% Analize all clustered PRC data

BASE_DIR = Path('/media/marcos/DATA/marcos/FloClock_data/phase_response_curve/output/PRC')
methods_dirs = contenidos(BASE_DIR)


fig, axarr = plt.subplots(3, 1, sharey=True, sharex=True, constrained_layout=True)

for ax, method_dir in zip(axarr, methods_dirs):
    data_files = contenidos(method_dir)
    method = method_dir.name

    phases = []
    perturbed = []
    reference = []
    for file in data_files:
        loaded = np.load(file)
        
        pert_dcd = loaded['pert_dcd']
        prev_dcd = loaded['prev_dcd']
        perturb_phase = loaded['perturb_phase']
        
        phases.extend(perturb_phase)
        perturbed.extend(pert_dcd)
        reference.extend(prev_dcd)
    
    perturbed = np.asarray( sort_by(perturbed, phases))
    reference = np.asarray( sort_by(reference, phases))
    phases = np.asarray(sorted(phases))
    
    ax.plot(phases, perturbed, 'o', c='C0')
    ax.plot(phases, reference, 'kx')
    
    repeated = np.concatenate((perturbed, perturbed, perturbed)) 
    repeated_phase = np.concatenate((phases-1, phases, phases+1))
    ax.plot(repeated_phase, smooth(repeated, size=13), c='C03')
    
    step = 11
    binned_phase = [np.mean(repeated_phase[i:i+1]) for i in range(0, len(repeated), step)]
    binned_PRC = [np.mean(repeated[i:i+step]) for i in range(0, len(repeated), step)]
    binned_PRC_err = [np.std(repeated[i:i+step]) for i in range(0, len(repeated), step)]
    ax.errorbar(binned_phase, binned_PRC, binned_PRC_err, fmt='o', color='C03')
    
    reference_std = np.std(reference[np.abs(reference) < 1])
    ax.fill_between([0, 1], -reference_std, reference_std, color='0.6')

    ax.set_ylabel(f'PRC ({method})')
    
ax.set_ylim(-1, 1)
ax.set_xlim(0, 1)
ax.set_xlabel('phase')


#%% Estimate the relaxation time only in perturbations away from pulses

""" This is hacked together from the "Get PRC" and "Estimate relaxatoin times"
sections, so it likely has a bunch of extra unneeded code."""


# BASE_DIR = Path('/media/marcos/DATA/marcos/FloClock_data/phase_response_curve')
BASE_DIR = '/home/user/Documents/Doctorado/Fly clock/FlyClock_data/phase_response_curve'
files = contenidos(BASE_DIR, filter_ext='.abf')

file = files[36] # using 40, 39, 38, 36, 35, 34, 33, 32, 30, 29, 26
threshold = 5
METHOD = 'rising' # one of 'rising', 'falling' or 'peaks'
SMOOTH = True # decides if we should do a harsh lowpass filter on the signal


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


def exp(t, A, t0, l, c):
    return A * np.exp(-(t-t0)/l) + c

# Load data
abf = pyabf.ABF(file)
print('Running', file.stem)

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
if SMOOTH:
    data_hp = lowpass_filter(time, data_hp, filter_order=2, frequency_cutoff=2)

# plot
fig_main = plt.figure(figsize=[17.69,  7.29], constrained_layout=True)
fig, fig2 = fig_main.subfigures(1,2, width_ratios=[3,1])
ax, ax2, axp = fig.subplots(3, sharex=True)
ax_prc = fig2.subplots()

ax.plot(time, data, label='raw')
ax.plot(time, perturbation + data.mean(), label='perturb.')
ax.plot(time, data_lp, label='lowpass $f_c$=10')
ax.plot(time, lowpass_filter(time, data_lp, filter_order=2, frequency_cutoff=2), label='lowpass $f_c$=2')

ax2.plot(time, data_hp, label='detrended')
ax2.plot(time, perturbation + data_hp.mean(), label='perturb.')

raw_data = data
data = data_hp
# find peaks and edge crossings

peaks = my_find_peaks(time, data)
ax2.plot(time[peaks], data[peaks], 'o', label='peaks')

mean_period_in_points = round(np.diff(peaks).mean())

# find crossing points
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

### Choose which event to use as reference
if METHOD == 'rising':
    event_index_array = rising_edge # one of rising_edge, falling_edge, peaks
elif METHOD == 'falling':
    event_index_array = falling_edge # one of rising_edge, falling_edge, peaks
elif METHOD == 'peaks':
    event_index_array = peaks # one of rising_edge, falling_edge, peaks
else:
    raise ValueError('"METHOD" has to be one of "rising", "falling" or "peaks"')

# plot peaks and threshold crossings
ax2.plot(time[peaks], data[peaks], 'k.', label='filtered_peaks')
ax2.plot(time[rising_edge], data[rising_edge], 'o', label='edge crossings')

# find cycles with a perturbation
pert_inxs,  = np.where(np.diff(perturbation) < -1)
pert_moments = time[pert_inxs]
ax2.plot(pert_moments, perturbation[pert_inxs], 'kx')

cross_moments = time[event_index_array]

# filter out cases where we can't detect the cycle because the perturbation fell right on it
cross_period = np.diff(cross_moments)
period_times = cross_moments[:-1] + cross_period/2
valid = au.find_valid_periods(period_times, cross_period, passes=2, threshold=1.6)

perturbed_cylces = []
perturb_phase = []
perturb_indexes = []
invalid_perturbed_cylces = []
try:
    for i, (e1, e2, v) in enzip(event_index_array, event_index_array[1:], valid):
        # find the first perturbation after the current crossing
        for pert in pert_inxs:
            if pert > e1:
                break
        else:
            raise StopIteration()
        
        # if te perturbation happened before the next peak, store it
        if pert <= e2:
            
            # keep it only if the period in question was valid
            if v:
                perturbed_cylces.append(i)
                
                # get phase of perturbation
                phase = (pert-e1) / (e2-e1) # here dividing indexes or actual time values is identical
                perturb_phase.append(phase)
                perturb_indexes.append(pert)
            else:
                invalid_perturbed_cylces.append(i)
            
except StopIteration:
    # this happens if there are no more perturbations to find
    pass

# mark the cycles that have a perturbation
for inx in perturbed_cylces:
    start = event_index_array[inx]
    end = event_index_array[inx+1]
    ax2.plot(time[start:end], data_hp[start:end], 'r')

for inx in invalid_perturbed_cylces:
    start = event_index_array[inx]
    end = event_index_array[inx+1]
    # ax2.plot(time[start:end], data_hp[start:end], 'r')
    ax2.fill_betweenx([data_hp.min(), data_hp.max()], time[start], time[end], color='0.7', zorder=1)

# plot one extra patch to label it
ax2.fill_betweenx([data_hp.min(), data_hp.max()], time[start], time[end], color='0.7', zorder=1, label='invalid')

for pi, ph in zip(pert_inxs, perturb_phase):
    ax2.text(time[pi], perturbation[pi]+0.5, f'{ph:.2f}')

# plot PRC

ax.set_title('Time series and analysis')
axp.set_xlabel('Time [sec]')
ax.set_ylabel('mV')
ax2.set_ylabel('detrended signal [mV]')
axp.set_ylabel('periods [s]')

ax.legend()
ax2.legend()
axp.legend()
axp.grid()
ax_prc.legend()

# plot valid cycles aligned at start of the cycle
keep_perturbation = []
for i, (inx, pert_inx, pert_phase) in enzip(perturbed_cylces, perturb_indexes, perturb_phase):
    start = event_index_array[inx]
    end = event_index_array[inx+1]
    
    phase = (time[start:end] - time[start]) / (time[end] - time[start])
    
    # ax_prc.plot(time[start:end] - time[start], raw_data[start:end])
    ax_prc.plot(phase, raw_data[start:end])
    ax_prc.plot(pert_phase, raw_data[pert_inx], 'ko', zorder=3)

    if 0.3 < pert_phase < 0.8:
        keep_perturbation.append(i)

fig, axarr = plt.subplots(4,3, figsize=[16.03,  7.6 ])

relaxation = []
relaxation_error = []
cycle_duration = []
for j, i in enumerate(keep_perturbation):
    ax = axarr.flat[j % axarr.size]
    
    inx = perturbed_cylces[i]
    pert_inx = perturb_indexes[i]
    
    # define some things about the cycle
    start = event_index_array[inx]
    end = event_index_array[inx+1]
    pert_moment = time[pert_inx] - time[start]
    
    # grab only the bit we are interested in
    bit = raw_data[pert_inx: pert_inx+400]
    tbit = time[pert_inx: pert_inx+400] - time[pert_inx]

    # find the peak
    peak = signal.find_peaks(-bit, prominence=5)[0][0]

    # fit a curve
    p0 = [bit[peak]-bit[peak:].max(), tbit[peak], 0.1, bit[peak:].max()]
    popt, pcov = optimize.curve_fit(exp, tbit[peak:], bit[peak:], p0)

    relax = popt[2]
    relax_err = np.sqrt(np.diag(pcov))[2]

    relaxation.append(relax)
    relaxation_error.append(relax_err)
    cycle_duration.append(time[end] - time[start])

    # plot
    offset =  time[pert_inx] - time[start]
    ax.plot(time[start:end] - time[start], raw_data[start:end])
    ax.plot(tbit + offset, bit)
    ax.plot(pert_moment, raw_data[pert_inx], 'ro')

    ax.plot(tbit[peak:] + offset, exp(tbit[peak:], *p0), '--', c='0.7')
    ax.plot(tbit[peak:] + offset, exp(tbit[peak:], *popt), 'k--')    
    ax.plot(tbit[peak] + offset, bit[peak], 'ok')

    ax.set_title(f'{relax:3f}')


output_dir = file.parent / 'output' / 'relaxation_times'
savefile = output_dir / file.stem
np.savez(savefile, 
         relaxation = relaxation,
         relaxation_error = relaxation_error,
         cycle_duration = cycle_duration
         )

#%% Analize relaxation time data


# BASE_DIR = Path('/media/marcos/DATA/marcos/FloClock_data/phase_response_curve')
BASE_DIR = Path('/home/user/Documents/Doctorado/Fly clock/FlyClock_data/phase_response_curve')
data_dir = BASE_DIR / 'output' / 'relaxation_times'
files = contenidos(data_dir, filter_ext='.npz')

max_relax = 0.4

def wheighted_mean(num, err):
    if num.shape != err.shape:
        return num.mean()
    else:
        return np.average(num, weights = 1/err**2)

layout = [['a', 'b', 'e'], ['a', 'c', 'f']]
fig, axdict  = plt.subplot_mosaic(layout, constrained_layout=True)
ax1, ax2, ax3, axs1, axs2 = [axdict[l] for l in'abcef']

relaxation_times = []
relaxation_times_normalized = []
for i, file in enumerate(files):
    loaded = np.load(file)
    relax = loaded['relaxation']
    dur = loaded['cycle_duration']
    err = loaded['relaxation_error']
    
    # filter outliers
    dur = dur[relax<0.4]
    relax = relax[relax<0.4]
    
    # call out outlier runs
    if any(r>0.2 for r in relax):
        print('Run with high relaxation times:', file.stem)
    
    # scatter plot
    mean_relax = wheighted_mean(relax, err)
    ax1.plot(dur, relax, '.', c=f'C{i}')
    ax1.plot(dur.mean(), mean_relax, 'o', c=f'C{i}')
    ax1.set_xlabel('cycle duration')
    ax1.set_ylabel('relaxaion time')
    
    # histograms
    ax2.hist(relax, alpha=0.3, fc=f'C{i}')
    ax2.plot(mean_relax, 10, 'v', c=f'C{i}')
    ax2.set_xlabel('relaxaion time')
    
    # cloud
    kde_scatter(i+1, relax, horizontal_scale=0.1, ax=axs1, alpha=1, 
                mec=None, c=f'C{i}')    
    
    # normalized histograms
    mean_relax_norm =  wheighted_mean(relax/dur, err)
    ax3.hist(relax/dur, alpha=0.3, fc=f'C{i}')
    ax3.plot(mean_relax_norm, 10, 'v', c=f'C{i}')
    ax3.set_xlabel('normalized relaxaion time')
    
    # normalized cloud
    kde_scatter(i+1, relax/dur, horizontal_scale=0.1, ax=axs2, alpha=1, 
                mec=None, c=f'C{i}')
    
    # save data for a boxplot
    relaxation_times.append(relax)
    relaxation_times_normalized.append(relax/dur)

axs1.boxplot(relaxation_times, showfliers=False, medianprops={'color':'k'})
axs2.boxplot(relaxation_times_normalized, showfliers=False, medianprops={'color':'k'})

axs1.set_xlabel('run number')
axs1.set_ylabel('relaxation time')

axs2.set_xlabel('run number')
axs2.set_ylabel('normalized relaxation time')
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
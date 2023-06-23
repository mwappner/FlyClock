#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 16:53:04 2023

@author: marcos
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import signal
from utils import find_point_by_value


def peak_periods(t, x):
    
    peak_inx, _ = signal.find_peaks(x)
    peak_times = t[peak_inx]
    
    periods = np.diff(peak_times)
    period_times = peak_times[:-1] + periods/2
    
    return period_times, periods

# def get_crossings(t, x, thresh):
    
#     rising1 = 
    
#     for crossing in rising1:
#         crossing_time = data.times.values[crossing]
#         closest_time = find_closest_value(data.times.values[rising2], crossing_time)
#         closest = find_closest_value(rising2, crossing)
        
#         if np.abs(closest_time - crossing_time) > maxd:
#             print('skipping')
#             continue
        
#         filtered_pairs.append((crossing, closest))   
#         lags.append(crossing_time - closest_time)
        

def peak_lags(t, x1, x2):
    
    peak_inx1, _ = signal.find_peaks(x1)
    peak_times1 = t[peak_inx1]
    
    peak_inx2, _ = signal.find_peaks(x2)
    peak_times2 = t[peak_inx2]
    
    lags = []
    times = []
    for peak in peak_times1:
        closest_inx = find_point_by_value(peak_times2, peak)
        
        lag = peak - peak_times2[closest_inx]
        moment = (peak + peak_times2[closest_inx])/2
       
        lags.append(lag)
        times.append(moment)
       
    return np.array(times), np.asarray(lags)


#%% Test all steps (oscillation, phase, lag)
a = 0.002
b = 1
phi = np.pi*0.99

period = lambda t: a*t+b

t = np.linspace(0, 1000, 50000)
# phase = 2*np.pi / period(t) * t
phase = 200* np.log(t+5)

# x = lambda phi: np.sin(2*np.pi / (period(t))**0.75 * t + phi)
x = lambda phi: np.sin(phase + phi)
x1 = x(0)
x2 = x(phi)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True, sharex=True)

ax1.plot(t, x1)
ax1.plot(t, x2)
ax1.set_title('data')

ax2.plot(*peak_periods(t, x1))
ax2.plot(*peak_periods(t, x2))

trend = np.polynomial.Polynomial.fit(*peak_periods(t, x1), deg=1)
ax2.plot(t, trend(t), '--k')
ax2.plot(t, period(t), ':k')
ax2.set_title('periods')

ax3.plot(*peak_lags(t, x1, x2), '.')
ax3.set_title('lag')
lag_trend = np.polynomial.Polynomial.fit(*peak_lags(t, x1, x2), deg=1)

print('Period trend:', trend.convert().coef[1])
print('Lag trend:', lag_trend.convert().coef[1])

#%% Lag evolution in time for multiple initial offsets (lags)

a = 0.002
b = 1
phis = np.linspace(1e-3, np.pi, endpoint=False)

period = lambda t: a*t+b

t = np.linspace(0, 1000, 50000)
# phase = 2*np.pi / period(t) * t
phase = 500 * np.log(t+5)

# x = lambda phi: np.sin(2*np.pi / (period(t))**0.75 * t + phi)
x = lambda phi: np.sin(phase + phi)
x1 = x(0)

fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)
colors_list = cm.viridis(np.linspace(0, 1, phis.size))
for c, phi in zip(colors_list, phis):
    x2 = x(phi)
    
    trend = np.polynomial.Polynomial.fit(*peak_periods(t, x2), deg=1)
    lag_trend = np.polynomial.Polynomial.fit(*peak_lags(t, x1, x2), deg=1)
    ax1.plot(*peak_lags(t, x1, x2), '.', c=c)

    period_slope = trend.convert().coef[1]
    # lag_slope = (c := lag_trend.convert().coef)[1] if len(c)==2 else 0
    lag_slope = lag_trend.convert().coef[1]
    
    phi /= np.pi
    t_slope, = ax2.plot(phi, a, '.', c='0.5', label='theory slope')
    p_slope, = ax2.plot(phi, period_slope, '.', c='k', label='period slope')
    ax2.plot(phi, lag_slope, '.', c=c)

ax1.set_xlabel('time')
ax1.set_ylabel('lag')
ax1.set_ylim(-0.05, None)

ax2.set_xlabel('initial phase [π rad]')
ax2.set_ylabel('lag slope')
ax2.grid()

ax2.legend(handles = [t_slope, p_slope])

#%% Diff equation governing the phase

from scipy.integrate import odeint
from scipy.optimize import curve_fit

w = 200

def func(x, t):
    return w/(t+5)
    

x0 = [0]
times = np.linspace(0, 100, 20000)

xs = odeint(func, x0, times)

plt.subplot(2,1,1)
plt.plot(times, xs)

def log(t, w, t0):
    
    c = w * np.log(-t0)
    return w * np.log(t-t0) - c

plt.plot(times, log(times, 200, -5), '--')
plt.title('Phase')

plt.subplot(2,1,2)
plt.plot(times, np.sin(xs))
plt.title('sin(phase)')

#%%

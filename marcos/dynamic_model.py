#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:38:22 2024

@author: marcos
"""

#%% Try out the model

import numpy as np
import matplotlib.pyplot as plt
from jitcode import jitcode, y
from symengine import sin, cos

# parameters
Tf = 2          # forcing period (s)
Ω  = (2*np.pi)/Tf     # forcing frequency (Hz)

rho0 = 11.4 # large

μ  = -3.46
a  =  5.53 / rho0**2
Δ  =  -0.6 * Ω
b  =  0.0
f0 =  9.14 * rho0

dy = -4
length = 8.4

ω = Ω - Δ

# variables and models
ρ = y(0)
θ = y(1)
φ = y(2)
f = [ ρ * (μ - a*ρ**2) + f0 * cos(φ - θ) , 
      ω - b*ρ**2 + f0/ρ * sin(φ-θ) ,
      Ω
	]

initial_state = np.array([1, 1, 0])

# run integrator
ODE = jitcode(f)
ODE.set_integrator("dopri5")
ODE.set_initial_value(initial_state, 0)

# integrate a bit
times = np.linspace(0, 20, 1000)
data = np.vstack([ODE.integrate(time) for time in times])

# extract variables
rho = data[:, 0]
theta = data[:, 1]
phi = data[:, 2]

# plot
fig = plt.figure(figsize=[8.58, 7.2 ])#constrained_layout=True)
ax1 = plt.subplot(211)
ax2 = plt.subplot(223, projection='polar')
ax3 = plt.subplot(224, projection='polar')

driver_scale = np.ones_like(phi)*rho[-1]*1.2
ax1.plot(times, driver_scale*np.sin(phi), label='driver')
ax1.plot(times, rho * np.sin(theta))

ax2.plot(phi, driver_scale)
ax2.plot(theta, rho)

ax3.plot(phi-theta, rho)


# format plots
ax1.legend()
ax1.set_title('ρ sin(θ)')
# ax2.set_title('Polar representation')
# ax3.set_title('Rotating frame of reference')

#%% Add one perturbation in y

import numpy as np
import matplotlib.pyplot as plt
from jitcode import jitcode, y
from symengine import sin, cos

from utils import find_point_by_value

# parameters
Tf = 2.2              # forcing period (s)
Ω  = (2*np.pi)/Tf     # forcing frequency (Hz)

rho0 = 11.4 # large

μ  = -3.46
a  =  5.53 / rho0**2
Δ  =  -0.6 * Ω
b  =  0.0
f0 =  9.14 * rho0

dy = -10
length = 6.83

ω = Ω - Δ

# variables and models
ρ = y(0)
θ = y(1)
φ = y(2)
f = [ ρ * (μ - a*ρ**2) + f0 * cos(φ - θ) , 
      ω - b*ρ**2 + f0/ρ * sin(φ-θ) ,
      Ω
	]

initial_state = np.array([1, 1, 0])

# run integrator
ODE = jitcode(f)
ODE.set_integrator("dopri5")
ODE.set_initial_value(initial_state, 0)

# integrate a bit
times = np.linspace(2, length, 1000)
data = np.vstack([ODE.integrate(time) for time in times])

# perturbate
rho0 = data[-1, 0]
theta0 = data[-1, 1]
rho_pert = np.sqrt( rho0**2 + dy**2 +  2*rho0 * dy * np.sin(theta0))
theta_pert = np.arctan2(rho0 * np.sin(theta0) + dy, rho0 * np.cos(theta0))

# integrate perturbation
perturbed_state = np.array([rho_pert, theta_pert, data[-1, 2]])
ODE.set_initial_value(perturbed_state, ODE.t)
more_times = np.linspace(0, 3, 1000) + ODE.t
pert_data = np.vstack([ODE.integrate(time) for time in more_times])

# integrate unperturbed, for comparison
ODE.set_initial_value(data[-1], times[-1])
more_data = np.vstack([ODE.integrate(time) for time in more_times])

# concatenate the arrays
times = np.concatenate((times, more_times))
pert_data = np.concatenate((data, pert_data))
data = np.concatenate((data, more_data))

# extract variables
rho = data[:, 0]
theta = data[:, 1]
phi = data[:, 2]

pert_rho = pert_data[:, 0]
pert_theta = pert_data[:, 1]

# plot
fig = plt.figure(figsize=[8.58, 7.2 ])#constrained_layout=True)
ax1 = plt.subplot(211)
ax2 = plt.subplot(223, projection='polar')
ax3 = plt.subplot(224, projection='polar')

ax1.plot(times, rho * np.sin(phi) ,label='driver')
ax1.plot(times, rho * np.sin(theta), label='unperturbed')
ax1.plot(times, pert_rho * np.sin(pert_theta), label='perturbed')

ax2.plot(phi, rho)
ax2.plot(theta, rho)
ax2.plot(pert_theta, pert_rho)

ax3.plot(phi-theta, rho)
ax3.plot(phi-pert_theta, pert_rho)

delta_theta = theta_pert%(2*np.pi) - theta0%(2*np.pi)
ax2.plot([theta0, theta0+delta_theta], [rho0, rho_pert])

# plot a point
point = 600 * np.pi / 180
point_index = find_point_by_value(theta[:theta.size//3], point)
instant = times[point_index]
size = rho[point_index]
driver_point = phi[point_index]

ax1.plot(instant, size * np.sin(point), 'ko')
ax1.plot(instant, size * np.sin(driver_point), 'ro')

ax2.plot(point, size, 'ko')
ax2.plot(driver_point, size, 'ro')

# format plots
ax1.legend()
ax1.set_title('ρ cos(θ)')
# ax2.set_rlim(0, 12)
# ax3.set_rlim(0, 12)

#%% Add one perturbation in x

import numpy as np
import matplotlib.pyplot as plt
from jitcode import jitcode, y
from symengine import sin, cos

# parameters
b = 0
a = 2
μ = -0.6
ω = 2.8

Ω = 3
f0 = 0.5

dx = -0.3
length = 10

# variables and models
ρ = y(0)
θ = y(1)
φ = y(2)
f = [ ρ * (μ - a*ρ**2) + f0 * cos(φ - θ) , 
      ω - b*ρ**2 + f0/ρ * sin(φ-θ) ,
      Ω
	]

initial_state = np.array([1, 1, 0])

# run integrator
ODE = jitcode(f)
ODE.set_integrator("dopri5")
ODE.set_initial_value(initial_state, 0)

# integrate a bit
times = np.linspace(0, length, 1000)
data = np.vstack([ODE.integrate(time) for time in times])

# perturbate
rho0 = data[-1, 0]
theta0 = data[-1, 1]
rho_pert = np.sqrt( rho0**2 + dx**2 +  2*rho0 * dx * np.cos(theta0))
# theta_pert = np.arctan( 1 / ( 1/np.tan(theta0) + dx / (rho0 * np.sin(theta0)) ) )
theta_pert = np.arctan2(rho0 * np.sin(theta0), rho0 * np.cos(theta0) + dx)

# integrate perturbation
perturbed_state = np.array([rho_pert, theta_pert, data[-1, 2]])
ODE.set_initial_value(perturbed_state, ODE.t)
more_times = np.linspace(0, length, 1000) + ODE.t
pert_data = np.vstack([ODE.integrate(time) for time in more_times])

# integrate unperturbed, for comparison
ODE.set_initial_value(data[-1], times[-1])
more_data = np.vstack([ODE.integrate(time) for time in more_times])

# concatenate the arrays
times = np.concatenate((times, more_times))
pert_data = np.concatenate((data, pert_data))
data = np.concatenate((data, more_data))

# extract variables
rho = data[:, 0]
theta = data[:, 1]
phi = data[:, 2]

pert_rho = pert_data[:, 0]
pert_theta = pert_data[:, 1]

# plot
fig = plt.figure(figsize=[8.58, 7.2 ])#constrained_layout=True)
ax1 = plt.subplot(211)
ax2 = plt.subplot(223, projection='polar')
ax3 = plt.subplot(224, projection='polar')

ax1.plot(times, f0*np.cos(phi) ,label='driver')
ax1.plot(times, rho * np.cos(theta), label='unperturbed')
ax1.plot(times, pert_rho * np.cos(pert_theta), label='perturbed')

ax2.plot(phi, np.ones_like(phi)*f0)
ax2.plot(theta, rho)
ax2.plot(pert_theta, pert_rho)

ax3.plot(phi-theta, rho)

delta_theta = theta_pert%(2*np.pi) - theta0%(2*np.pi)
ax2.plot([theta0, theta0+delta_theta], [rho0, rho_pert])

# format plots
ax1.legend()
ax1.set_title('ρ cos(θ)')

#%% Add multiple perturbations

import numpy as np
import matplotlib.pyplot as plt
from jitcode import jitcode, y
from symengine import sin, cos

# parameters
Tf = 2          # forcing period (s)
Ω  = (2*np.pi)/Tf     # forcing frequency (Hz)

rho0 = 11.4 # large

μ  = -3.46
a  =  5.53 / rho0**2
Δ  =  -0.6 * Ω
b  =  0.0
f0 =  9.14 * rho0

dy = -10
pert_every = 4.3
tf = 20

ω = Ω - Δ

# variables and models
ρ = y(0)
θ = y(1)
φ = y(2)
f = [ ρ * (μ - a*ρ**2) + f0 * cos(φ - θ) , 
      ω - b*ρ**2 + f0/ρ * sin(φ-θ) ,
      Ω
	]

initial_state = np.array([1, 1, 0])

# run integrator
ODE = jitcode(f)
ODE.set_integrator("dopri5")
ODE.set_initial_value(initial_state, 0)

# integrate a bit
times = np.arange(0, pert_every, 1e-2)
data = np.vstack([ODE.integrate(time) for time in times])

pert_data = data
pert_count = int(tf // pert_every) - 1
for _ in range(pert_count):
    
    # perturbate
    rho0 = pert_data[-1, 0]
    theta0 = pert_data[-1, 1]
    rho_pert = np.sqrt( rho0**2 + dy**2 +  2*rho0 * dy * np.sin(theta0))
    theta_pert = np.arctan2(rho0 * np.sin(theta0) + dy, rho0 * np.cos(theta0))
    
    # integrate perturbation
    perturbed_state = np.array([rho_pert, theta_pert, pert_data[-1, 2]])
    ODE.set_initial_value(perturbed_state, ODE.t)
    pert_times = np.linspace(0, pert_every, 1000) + ODE.t
    pert_data_bit = np.vstack([ODE.integrate(time) for time in pert_times])
    
    pert_data = np.concatenate((pert_data, pert_data_bit))
    times = np.concatenate((times, pert_times))

# integrate unperturbed, for comparison
ODE.set_initial_value(initial_state, 0)
data = np.vstack([ODE.integrate(time) for time in times])

# # concatenate the arrays
# times = np.concatenate((times, more_times))
# pert_data = np.concatenate((data, pert_data))
# data = np.concatenate((data, more_data))

# extract variables
rho = data[:, 0]
theta = data[:, 1]
phi = data[:, 2]

pert_rho = pert_data[:, 0]
pert_theta = pert_data[:, 1]

# plot
fig = plt.figure(figsize=[8.58, 7.2 ])#constrained_layout=True)
ax1 = plt.subplot(211)
ax2 = plt.subplot(223, projection='polar')
ax3 = plt.subplot(224, projection='polar')

ax1.plot(times, rho * np.sin(phi) ,label='driver')
ax1.plot(times, rho * np.sin(theta), label='unperturbed')
ax1.plot(times, pert_rho * np.sin(pert_theta), label='perturbed')

ax2.plot(phi, rho)
ax2.plot(theta, rho)
ax2.plot(pert_theta, pert_rho)

ax3.plot(phi-theta, rho)

# format plots
ax1.legend()
ax1.set_title('ρ cos(θ)')

#%% Numeric PRC testing

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from jitcode import jitcode, y
from symengine import sin, cos

def limits_fun(pert_data, pert_isntants, times, *others):
    
    # # extract variables
    # rho = data[:, 0]
    # theta = data[:, 1]
    # phi = data[:, 2]
    # x = rho * np.sin(theta)
    
    pert_rho = pert_data[:, 0]
    pert_theta = pert_data[:, 1]
    pert_x = pert_rho * np.sin(pert_theta)
    
    # ref_peaks, _ = signal.find_peaks(x)
    # maximum = x[ref_peaks[2:]].mean()
    # peaks, _ = signal.find_peaks(pert_x, height=maximum*0.8)

    # peaks as maxima
    peaks, _ = signal.find_peaks(pert_x, height=0)

    # peaks from the phase itself
    # cycle_offset = np.pi*3/2 # adjust what part of the cycle we count as "beginning"
    # peaks, _ = signal.find_peaks( (pert_theta+cycle_offset) % (2*np.pi))

    # peaks as 0-crossings
    # peaks, _ = signal.find_peaks(-np.abs(pert_x))
    # peaks = [p for p in peaks if pert_x[p]<pert_x[p+1]]
    
    # filter some peaks
    # peaks = np.asarray([p for p in peaks if times[p] not in pert_instants]) # ignore peaks due to the perturbation    
    peaks = np.asarray([p for p in peaks if np.abs(times[p]-pert_instants).min()>0.05]) # ignore peaks due to the perturbation    
    
    return peaks

# parameters
Tf = 2.2          # forcing period (s)
Ω  = (2*np.pi)/Tf     # forcing frequency (Hz)


rho0 = 11.4 # large

μ  = -3.46
a  =  5.53 / rho0**2
Δ  =  -0.6 * Ω
b  =  0.0
f0 =  9.14 * rho0 /10

dy = -4.5
pert_every = 7.2
tf = 20

ω = Ω - Δ

# variables and models
ρ = y(0)
θ = y(1)
φ = y(2)
f = [ ρ * (μ - a*ρ**2) + f0 * cos(φ - θ) , 
      ω - b*ρ**2 + f0/ρ * sin(φ-θ) ,
      Ω
	]

initial_state = np.array([.1, 0, 0])

# run integrator
ODE = jitcode(f)
ODE.set_integrator("dopri5")
ODE.set_initial_value(initial_state, 0)

# integrate a bit
times = np.arange(0, pert_every, 1e-2)
data = np.vstack([ODE.integrate(time) for time in times])

pert_data = data
pert_count = int(tf // pert_every) - 1
pert_instants = []
pert_phase = []
theo_PRC = []
for _ in range(pert_count):
    
    # perturbate
    rho0 = pert_data[-1, 0]
    theta0 = pert_data[-1, 1]
    rho_pert = np.sqrt( rho0**2 + dy**2 +  2*rho0 * dy * np.sin(theta0))
    theta_pert = np.arctan2(rho0 * np.sin(theta0) + dy, rho0 * np.cos(theta0))
        
    pert_instants.append(ODE.t)
    pert_phase.append(theta0)
    theo_PRC.append(theta_pert%(2*np.pi) - theta0%(2*np.pi))
    
    # integrate perturbation
    perturbed_state = np.array([rho_pert, theta_pert, pert_data[-1, 2]])
    ODE.set_initial_value(perturbed_state, ODE.t)
    pert_times = np.linspace(0, pert_every, 1000) + ODE.t
    pert_data_bit = np.vstack([ODE.integrate(time) for time in pert_times])
    
    pert_data = np.concatenate((pert_data, pert_data_bit))
    times = np.concatenate((times, pert_times))

pert_instants = np.asarray(pert_instants)
pert_phase = (np.asarray(pert_phase) % (2*np.pi)) / (2*np.pi)
theo_PRC = np.asarray(theo_PRC)

# integrate unperturbed, for comparison
ODE.set_initial_value(initial_state, 0)
data = np.vstack([ODE.integrate(time) for time in times])

# extract variables
rho = data[:, 0]
theta = data[:, 1]
phi = data[:, 2]
x = rho * np.sin(theta)

pert_rho = pert_data[:, 0]
pert_theta = pert_data[:, 1]
pert_x = pert_rho * np.sin(pert_theta)

# find maxima
peaks = limits_fun(pert_data, pert_instants, times)

# calculate PRC
next_peak_indexes = []
prev_peak_indexes = []
linear_phase = []
real_phase = []
PRC = []
for pertt, theta0 in zip(pert_instants, pert_phase):
    
    # skip perturbations that land way too close to the start of the oscillation
    # if np.isclose(pertt, times[ref_peaks], rtol=0, atol=0.02).any():
    #     continue

    # find the end of the perturbed cycle
    next_peak_index = next( i for i, p in zip(range(peaks.size), peaks) if times[p]>pertt )
    # pert_index = np.where(pertt==times)[0][0]
    # if np.all(pert_x[pert_index+1:peaks[next_peak_index]] < pert_x[pert_index]):
    #     print('deleting', next_peak_index)
    #     peaks = np.delete(peaks, next_peak_index)
    # next_peak_index = next( i for i, p in zip(range(peaks.size), peaks) if times[p]>pertt )
    
    next_peak = peaks[next_peak_index]
    prev_peak = peaks[next_peak_index-1]
    prev_prev_peak = peaks[next_peak_index-2]
    
    # calculate the perturbed and unperturbed cycle durations
    p0 = times[prev_peak] - times[prev_prev_peak]
    p1 = times[next_peak] - times[prev_peak]
    
    # PRC
    dp = (p1-p0)/p0
    
    # pahse of the perturbation
    phase = (pertt - times[prev_peak]) / p1
    theta0 = (pertt - times[prev_peak]) / p0
    # if (abs(phase - theta0) / theta0) > 0.25:
    #     continue
    
    # save stuff
    linear_phase.append(phase)
    real_phase.append(theta0)
    PRC.append(dp)
    
    # I just need this for plotting
    next_peak_indexes.append(next_peak_index)

# convert to array
linear_phase = np.asarray(linear_phase)
real_phase = np.asarray(real_phase)
PRC = np.asarray(PRC)
next_peak_indexes = np.asarray(next_peak_indexes)

# plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[17, 4 ], constrained_layout=True, 
                               gridspec_kw={'width_ratios':[3,1]})

# plot lines
# ax1.plot(times, rho.max() * np.sin(phi) ,label='driver')
ax1.plot(times, x, label='unperturbed')
ax1.plot(times, pert_x, label='perturbed')

# plot peaks
triangle_height = pert_x.min()*1.1

# ax1.plot(times[ref_peaks], x[ref_peaks], 'k.')
ax1.plot(times[peaks], pert_x[peaks], 'o')
ax1.plot(pert_instants, triangle_height*np.ones(pert_instants.shape), 'k^')

# plot perturbed periods
for peak_inx, lphase, rphase in zip(next_peak_indexes, linear_phase, real_phase):
    
    # get peaks
    next_peak = peaks[peak_inx]
    prev_peak = peaks[peak_inx-1]
    
    # get times
    peak_time = times[[prev_peak, next_peak]]
    val = triangle_height, triangle_height
    
    # get phase
    pertt = next(pt for pt in pert_instants if peak_time[0] < pt < peak_time[1])
    
    # plot perturbed period
    ax1.plot(peak_time, val, 'b^')
    pert_period = slice(prev_peak, next_peak)
    ax1.plot(times[pert_period], pert_x[pert_period], 'r')
    
    txt = f'{lphase:.2f}\n{rphase:.2f}'
    ax1.text(pertt, triangle_height*1.1, txt, horizontalalignment='center', verticalalignment='top')
    
# plot PRC
ax2.plot(linear_phase, PRC, '.')
ax2.plot(real_phase, PRC, '.')
ax2.set_xlabel('phase of perturbation / 2π')
ax2.set_ylabel('PRC')

#%% Full PRC


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from jitcode import jitcode, y
from symengine import sin, cos

SAVE = True
savedir = r'/media/marcos/DATA/marcos/FloClock_data/dynamic_model/PRC/paper potential/new parameters/'

def limits_fun(pert_data, pert_isntants, times, *others):
    
    # # extract variables
    # rho = data[:, 0]
    # theta = data[:, 1]
    # phi = data[:, 2]
    # x = rho * np.sin(theta)
    
    pert_rho = pert_data[:, 0]
    pert_theta = pert_data[:, 1]
    pert_x = pert_rho * np.sin(pert_theta)
    
    # ref_peaks, _ = signal.find_peaks(x)
    # maximum = x[ref_peaks[2:]].mean()
    # peaks, _ = signal.find_peaks(pert_x, height=maximum*0.8)

    # peaks as maxima
    peaks, _ = signal.find_peaks(pert_x)
    peaks, _ = signal.find_peaks(pert_x, height=0)#, distance=0.5*np.diff(peaks).mean())

    # peaks from the phase itself
    # cycle_offset = np.pi*3/2 # adjust what part of the cycle we count as "beginning"
    # peaks, _ = signal.find_peaks( (pert_theta+cycle_offset) % (2*np.pi))

    # peaks as 0-crossings
    # peaks, _ = signal.find_peaks(-np.abs(pert_x))
    # peaks = [p for p in peaks if pert_x[p]<pert_x[p+1]]
    
    # filter some peaks
    peaks = np.asarray([p for p in peaks if times[p] not in pert_instants]) # ignore peaks due to the perturbation    
    # peaks = np.asarray([p for p in peaks if np.abs(times[p]-pert_instants).min()>0.05]) # ignore peaks due to the perturbation    
    
    return peaks

# # parameters
# Tf = 2.2          # forcing period (s)
# Ω  = (2*np.pi)/Tf     # forcing frequency (Hz)

# rho0 = 11.4 # large

# μ  = -3.46
# a  =  5.53 / rho0**2
# Δ  =  -0.6 * Ω
# b  =  0.0
# f0 =  9.14 * rho0/10

# dy = -4.5
# pert_every = 9.1

# ω = Ω - Δ

# tf = 4000
# dt = 1e-3

# # variables and models
# ρ = y(0)
# θ = y(1)
# φ = y(2)
# f = [ ρ * (μ - a*ρ**2) + f0 * cos(φ - θ) , 
#       ω - b*ρ**2 + f0/ρ * sin(φ-θ) ,
#       Ω
#  	]

# initial_state = np.array([1, 1, 0])

# # run integrator
# ODE = jitcode(f)
# ODE.set_integrator("dopri5")
# ODE.set_initial_value(initial_state, 0)

# # integrate a bit
# times = np.arange(0, pert_every, dt)
# data = np.vstack([ODE.integrate(time) for time in times])

# # build perturbation intervals
# pert_count = int(tf // pert_every) - 1
# pert_intervals = np.random.randn(pert_count*2)*2 + pert_every

# # iterate and integrate
# pert_data = data
# pert_instants = []
# pert_phase = []
# theo_PRC = []
# for pert_interval in pert_intervals:
    
#     # perturbate
#     rho0 = pert_data[-1, 0]
#     theta0 = pert_data[-1, 1]
#     rho_pert = np.sqrt( rho0**2 + dy**2 +  2*rho0 * dy * np.sin(theta0))
#     theta_pert = np.arctan2(rho0 * np.sin(theta0) + dy, rho0 * np.cos(theta0))
        
#     pert_instants.append(ODE.t)
#     pert_phase.append(theta0)
#     theo_PRC.append(theta_pert%(2*np.pi) - theta0%(2*np.pi))
    
#     # integrate perturbation
#     perturbed_state = np.array([rho_pert, theta_pert, pert_data[-1, 2]])
#     ODE.set_initial_value(perturbed_state, ODE.t)
#     pert_times = np.arange(0, pert_interval, dt) + ODE.t
#     pert_data_bit = np.vstack([ODE.integrate(time) for time in pert_times])
    
#     pert_data = np.concatenate((pert_data, pert_data_bit))
#     times = np.concatenate((times, pert_times))
    
#     # end early if needed
#     if ODE.t > tf:
#         break

# pert_instants = np.asarray(pert_instants)
# pert_phase = (np.asarray(pert_phase) % (2*np.pi)) / (2*np.pi)
# theo_PRC = np.asarray(theo_PRC)

# # # integrate unperturbed, for comparison
# # ODE.set_initial_value(initial_state, 0)
# # data = np.vstack([ODE.integrate(time) for time in times])

# # # extract variables
# # rho = data[:, 0]
# # theta = data[:, 1]
# # phi = data[:, 2]
# # x = rho * np.sin(theta)

# pert_rho = pert_data[:, 0]
# pert_theta = pert_data[:, 1]
# pert_x = pert_rho * np.sin(pert_theta)

# find maxima
peaks = limits_fun(pert_data, pert_instants, times)

# filter some peaks
# peaks = np.asarray([p for p in peaks if times[p] not in pert_instants]) # ignore peaks due to the perturbation

# calculate PRC
next_peak_indexes = []
linear_phase = []
real_phase = []
PRC = []
for pertt, theta0 in zip(pert_instants, pert_phase):
    
    # skip perturbations that land way too close to the start of the oscillation
    # if np.isclose(pertt, times[ref_peaks], rtol=0, atol=0.1).any():
    #     continue

    # find the end of the perturbed cycle
    next_peak_index = next( i for i, p in zip(range(peaks.size), peaks) if times[p]>pertt )
    
    # pert_index = np.where(pertt==times)[0][0]
    # if np.all(pert_x[pert_index+1:peaks[next_peak_index]] < pert_x[pert_index]):
    #     peaks = np.delete(peaks, next_peak_index)
    # next_peak_index = next( i for i, p in zip(range(peaks.size), peaks) if times[p]>pertt )
    
    next_peak = peaks[next_peak_index]
    prev_peak = peaks[next_peak_index-1]
    prev_prev_peak = peaks[next_peak_index-2]
    
    # calculate the perturbed and unperturbed cycle durations
    p0 = times[prev_peak] - times[prev_prev_peak]
    p1 = times[next_peak] - times[prev_peak]
    
    # PRC
    dp = (p1-p0)/p0
    
    # pahse of the perturbation
    phase = (pertt - times[prev_peak]) / p1
    theta0 = (pertt - times[prev_peak]) / p0
    # if (abs(phase - theta0) / theta0) > 0.25:
    #     continue
    
    # save stuff
    linear_phase.append(phase)
    real_phase.append(theta0)
    PRC.append(dp)
    
    # I just need this for plotting
    next_peak_indexes.append(next_peak_index)

# convert to array
linear_phase = np.asarray(linear_phase)
real_phase = np.asarray(real_phase)
PRC = np.asarray(PRC)
next_peak_indexes = np.asarray(next_peak_indexes)

# save data
fname = f'{b=}, {a=}, mu={μ}, omega={ω:.2f}, Omega={Ω:.2f}, {f0=}, {dy=}'
savefile = savedir + fname
if SAVE:
    np.savez(savefile,
        b = b,
        a = a,
        mu = μ,
        omega=ω,
        Omega = Ω,
        f0 = f0,
        
        dy = dy,
        pert_every = pert_every,
        tf = tf,
        dt = dt,
        
        linear_phase = linear_phase,
        real_phase = real_phase,
        PRC = PRC,    
              )

# plot
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[17, 4 ], constrained_layout=True, 
#                                gridspec_kw={'width_ratios':[3,1]})

fig, ax = plt.subplots(figsize=(8,8)) 

# plot PRC
# PRC[np.abs(PRC)> 1] = np.nan
ax.plot(linear_phase, PRC, '.')
ax.plot(real_phase, PRC, '.')
# ax.plot(-(pert_phase-0.25)%1, ((theo_PRC+2)%(2*np.pi)-2)/(2*np.pi), '.')
ax.set_title(fname)

#%% Plot PRC as a function of parameters

import functools

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils import contenidos

def big_or(column, values):
    return functools.reduce(lambda x, y: x|y, (column==v for v in values))

def filter_data(df, **filters):
    """ Takes in varargs with keys being the column name to check and 
    values being the values to match for that column, either floats or tuples,
    as well as the dataframe to filter. Returns the filtered dataframe."""
    
    #return the original if tehre's nothing to filter
    if len(filters)==0:
        return df

    # make floats into iterables
    filters = {k: (v if hasattr(v, '__iter__') else [v]) for k, v in filters.items()}
    
    # create bool masks
    bools = [big_or(df[k], v) for k, v in filters.items()]
    
    # make a big and
    where = functools.reduce(lambda x, y: x&y, bools)
    
    return df[where]

PHASE_VAR = 'linear_phase' # either 'linear_phase' or 'real_phase'

datadir = r'/media/marcos/DATA/marcos/FloClock_data/dynamic_model/PRC/old parameters'
files = contenidos(datadir, filter_ext='.npz')

example = np.load(files[0])
fields = [f for f in example.files if example[f].size == 1]

# load the data parameters
data = {f:[] for f in fields}
data['path'] = files
data['file'] = list(files.stems())
for file in files:
    loaded = np.load(file)
    
    for field in fields:
        data[field].append(loaded[field].tolist())
        
data = pd.DataFrame(data)

# find parameter values
param_names = fields[:6]

# first element of each list is the "default" value
param_values = {p:data[p].value_counts().index.tolist() for p in param_names}
base_params = {p:v[0] for p, v in param_values.items()}

# plot
fig, axarr = plt.subplots(2, 3, constrained_layout=True, figsize=(11,6))

param_name_dict = {'mu':'μ', 'omega':'ω', 'Omega':'Ω'}

# load the base case
base_line = filter_data(data, **base_params)
base_run = np.load(base_line.path.values[0])
for pname, ax in zip(param_names, axarr.flat):
    
    runs = filter_data(data, **{pname:param_values[pname][1:]})
    
    for i, line in runs.iterrows():
        loaded = np.load(line.path)
        
        phase = loaded[PHASE_VAR]
        PRC = loaded['PRC']
        
        ax.plot(phase, PRC, '.', label=line[pname])
    
    ax.plot(base_run[PHASE_VAR], base_run['PRC'], 'k.', label=base_run[pname])
    ax.set_title(param_name_dict.get(pname, pname))
    ax.set_xlabel('phase of perturbation [2π]')
    ax.set_ylabel('PRC')
    ax.legend()

axlims = [[-0.039, 0.092],
     [-0.058, 0.14],
     [-0.036, 0.134],
     [-0.192, 0.3],
     [-0.086, 0.378],
     [-0.178, 0.317]]

for lims, ax in zip(axlims, axarr.flat):
    ax.set_ylim(lims)


#%% Plot PRC as a function f0

import functools

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils import contenidos, sort_by

def big_or(column, values):
    return functools.reduce(lambda x, y: x|y, (column==v for v in values))

def filter_data(df, **filters):
    """ Takes in varargs with keys being the column name to check and 
    values being the values to match for that column, either floats or tuples,
    as well as the dataframe to filter. Returns the filtered dataframe."""
    
    #return the original if tehre's nothing to filter
    if len(filters)==0:
        return df

    # make floats into iterables
    filters = {k: (v if hasattr(v, '__iter__') else [v]) for k, v in filters.items()}
    
    # create bool masks
    bools = [big_or(df[k], v) for k, v in filters.items()]
    
    # make a big and
    where = functools.reduce(lambda x, y: x&y, bools)
    
    return df[where]

PHASE_VAR = 'linear_phase' # either 'linear_phase' or 'real_phase'

datadir = r'/media/marcos/DATA/marcos/FloClock_data/dynamic_model/PRC/f0'
files = contenidos(datadir, filter_ext='.npz')

example = np.load(files[0])
fields = [f for f in example.files if example[f].size == 1]

# load the data parameters
data = {f:[] for f in fields}
data['path'] = files
data['file'] = list(files.stems())
for file in files:
    loaded = np.load(file)
    
    for field in fields:
        data[field].append(loaded[field].tolist())
        
data = pd.DataFrame(data)

# find parameter values
param_names = fields[:6]

# plot
fig, ax = plt.subplots(constrained_layout=True, figsize=(11,6))

param_name_dict = {'mu':'μ', 'omega':'ω', 'Omega':'Ω'}

for i, line in data.iterrows():
    if i == 0:
        continue
    loaded = np.load(line.path)
    
    phase = loaded[PHASE_VAR]
    PRC = sort_by(loaded['PRC'], phase)
    phase = sorted(phase)
    
    ax.plot(phase, PRC, label=line.f0)

ax.set_xlabel('phase of perturbation [2π]')
ax.set_ylabel('PRC')
ax.legend()

## Sensitivity band

rising_PRC_dir = '/media/marcos/DATA/marcos/FloClock_data/phase_response_curve/output/PRC/rising'
data_files = contenidos(rising_PRC_dir)

reference = []
for file in data_files:
    loaded = np.load(file)
    
    prev_dcd = loaded['prev_dcd']
    reference.extend(prev_dcd)

reference = np.asarray(reference)
reference_std = np.std(reference[np.abs(reference) < 1])
ax.fill_between([0, 1], -reference_std, reference_std, color='0.6')

ax.set_xlim(0, 1)
ax.set_ylim(-1, 1)



#%% Plot PRC as a function dy

import functools

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils import contenidos, sort_by

def big_or(column, values):
    return functools.reduce(lambda x, y: x|y, (column==v for v in values))

def filter_data(df, **filters):
    """ Takes in varargs with keys being the column name to check and 
    values being the values to match for that column, either floats or tuples,
    as well as the dataframe to filter. Returns the filtered dataframe."""
    
    #return the original if tehre's nothing to filter
    if len(filters)==0:
        return df

    # make floats into iterables
    filters = {k: (v if hasattr(v, '__iter__') else [v]) for k, v in filters.items()}
    
    # create bool masks
    bools = [big_or(df[k], v) for k, v in filters.items()]
    
    # make a big and
    where = functools.reduce(lambda x, y: x&y, bools)
    
    return df[where]

PHASE_VAR = 'linear_phase' # either 'linear_phase' or 'real_phase'

datadir = r'/media/marcos/DATA/marcos/FloClock_data/dynamic_model/PRC/positive mu'
files = contenidos(datadir, filter_ext='.npz')

example = np.load(files[0])
fields = [f for f in example.files if example[f].size == 1]

# load the data parameters
data = {f:[] for f in fields}
data['path'] = files
data['file'] = list(files.stems())
for file in files:
    loaded = np.load(file)
    
    for field in fields:
        data[field].append(loaded[field].tolist())
        
data = pd.DataFrame(data)

# find parameter values
param_names = fields[:6]

# plot
fig, ax = plt.subplots(constrained_layout=True, figsize=(11,6))

param_name_dict = {'mu':'μ', 'omega':'ω', 'Omega':'Ω'}

for i, line in data.iterrows():
    # if i == 0:
    #     continue
    if line.mu == 7.8:
        continue

    loaded = np.load(line.path)
    
    phase = loaded[PHASE_VAR]
    PRC = sort_by(loaded['PRC'], phase)
    phase = sorted(phase)
    
    ax.plot(phase, PRC, label=f'dy={line.dy}, mu={line.mu}')

ax.set_xlabel('phase of perturbation [2π]')
ax.set_ylabel('PRC')
ax.legend()

## Sensitivity band

rising_PRC_dir = '/media/marcos/DATA/marcos/FloClock_data/phase_response_curve/output/PRC/rising'
data_files = contenidos(rising_PRC_dir)

reference = []
for file in data_files:
    loaded = np.load(file)
    
    prev_dcd = loaded['prev_dcd']
    reference.extend(prev_dcd)

reference = np.asarray(reference)
reference_std = np.std(reference[np.abs(reference) < 1])
ax.fill_between([0, 1], -reference_std, reference_std, color='0.6')

# ax.set_xlim(0, 1)
ax.set_ylim(-1, 1)



#%% Plot PRC for paper

import functools

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils import contenidos, sort_by

def big_or(column, values):
    return functools.reduce(lambda x, y: x|y, (column==v for v in values))

def filter_data(df, **filters):
    """ Takes in varargs with keys being the column name to check and 
    values being the values to match for that column, either floats or tuples,
    as well as the dataframe to filter. Returns the filtered dataframe."""
    
    #return the original if tehre's nothing to filter
    if len(filters)==0:
        return df

    # make floats into iterables
    filters = {k: (v if hasattr(v, '__iter__') else [v]) for k, v in filters.items()}
    
    # create bool masks
    bools = [big_or(df[k], v) for k, v in filters.items()]
    
    # make a big and
    where = functools.reduce(lambda x, y: x&y, bools)
    
    return df[where]

PHASE_VAR = 'real_phase' # either 'linear_phase' or 'real_phase'

datadir = r'/media/marcos/DATA/marcos/FloClock_data/dynamic_model/PRC/paper potential/new parameters/'
files = contenidos(datadir, filter_ext='.npz')

example = np.load(files[0])
fields = [f for f in example.files if example[f].size == 1]

# load the data parameters
data = {f:[] for f in fields}
data['path'] = files
data['file'] = list(files.stems())
for file in files:
    loaded = np.load(file)
    
    for field in fields:
        data[field].append(loaded[field].tolist())
        
data = pd.DataFrame(data)

# find parameter values
param_names = fields[:6]

# plot
fig, ax = plt.subplots(constrained_layout=True, figsize=(11,6))

param_name_dict = {'mu':'μ', 'omega':'ω', 'Omega':'Ω'}

for i, line in data.iterrows():
    # if i == 0:
    #     continue

    loaded = np.load(line.path)
    
    phase = loaded[PHASE_VAR]
    PRC = np.asarray(sort_by(loaded['PRC'], phase))
    phase = np.asarray(sorted(phase))

    if i > 2:
        phase = phase[PRC>-0.5]
        PRC = PRC[PRC>-0.5]
    
    label = f'$f_0$={line.f0}, μ={line.mu}'
    ax.plot(phase, PRC, label=label, lw=2)

ax.set_xlabel('phase of perturbation [2π]')
ax.set_ylabel('PRC')
ax.legend()

## Sensitivity band

rising_PRC_dir = '/media/marcos/DATA/marcos/FloClock_data/phase_response_curve/output/PRC/rising'
data_files = contenidos(rising_PRC_dir)

reference = []
for file in data_files:
    loaded = np.load(file)
    
    prev_dcd = loaded['prev_dcd']
    reference.extend(prev_dcd)

reference = np.asarray(reference)
reference_std = np.std(reference[np.abs(reference) < 1])
ax.fill_between([0, 1], -reference_std, reference_std, color='0.6')

ax.set_xlim(0, 1)
ax.set_ylim(-1, 1)



#%% Estimated vs actual phase of perturbation

import functools

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils import contenidos

def big_or(column, values):
    return functools.reduce(lambda x, y: x|y, (column==v for v in values))

def filter_data(df, **filters):
    """ Takes in varargs with keys being the column name to check and 
    values being the values to match for that column, either floats or tuples,
    as well as the dataframe to filter. Returns the filtered dataframe."""
    
    #return the original if tehre's nothing to filter
    if len(filters)==0:
        return df

    # make floats into iterables
    filters = {k: (v if hasattr(v, '__iter__') else [v]) for k, v in filters.items()}
    
    # create bool masks
    bools = [big_or(df[k], v) for k, v in filters.items()]
    
    # make a big and
    where = functools.reduce(lambda x, y: x&y, bools)
    
    return df[where]

datadir = r'/media/marcos/DATA/marcos/FloClock_data/dynamic_model/PRC/old parameters'
files = contenidos(datadir)

example = np.load(files[0])
fields = [f for f in example.files if example[f].size == 1]

# load the data parameters
data = {f:[] for f in fields}
data['path'] = files
data['file'] = list(files.stems())
for file in files:
    loaded = np.load(file)
    
    for field in fields:
        data[field].append(loaded[field].tolist())
        
data = pd.DataFrame(data)

# find parameter values
param_names = fields[:6]

# first element of each list is the "default" value
param_values = {p:data[p].value_counts().index.tolist() for p in param_names}
base_params = {p:v[0] for p, v in param_values.items()}

# plot
fig, axarr = plt.subplots(2, 3, constrained_layout=True, figsize=(11,6))

param_name_dict = {'mu':'μ', 'omega':'ω', 'Omega':'Ω'}

# load the base case
base_line = filter_data(data, **base_params)
base_run = np.load(base_line.path.values[0])
for pname, ax in zip(param_names, axarr.flat):
    
    runs = filter_data(data, **{pname:param_values[pname][1:]})
    
    for i, line in runs.iterrows():
        loaded = np.load(line.path)
        
        PRC = loaded['PRC']
        linear_phase = loaded['linear_phase']
        real_phase = loaded['real_phase']
                
        # ax.plot(real_phase, linear_phase, '.', label=line[pname])
        ax.plot(PRC, real_phase-linear_phase, '.', label=line[pname])
        # ax.plot(real_phase, (linear_phase - real_phase) / real_phase, '.', label=line[pname])
    
    ax.plot(base_run['real_phase'], base_run['linear_phase'], 'k.', label=base_run[pname])
    # rp = base_run['real_phase']
    # ax.plot(rp, (base_run['linear_phase'] - rp)/rp, 'k.', label=base_run[pname])
    
    ax.set_title(param_name_dict.get(pname, pname))
    ax.set_xlabel('real phase [2π]')
    ax.set_ylabel('linear phase [2π]')
    ax.legend()
    
    # ax.plot([0, 1], [0, 1], '--', c='0.7')

# axlims = [[-0.039, 0.092],
#      [-0.058, 0.14],
#      [-0.036, 0.134],
#      [-0.192, 0.3],
#      [-0.086, 0.378],
#      [-0.178, 0.317]]

# for lims, ax in zip(axlims, axarr.flat):
#     ax.set_ylim(lims)


#%% Theoretical PRC

import numpy as np
import matplotlib.pyplot as plt

# parameters
μ  =  1
a  =  0.122

rho0 = np.sqrt(μ/a)
# dyorho0 = 0.3 # perturbación como fracción de rho0


plt.figure()

for dyorho0 in (-0.1, -0.3, ):#-0.5, -1, -1.2):
    dy = dyorho0 * rho0
    
    theta0 = np.linspace(0, 2*np.pi, 1000)
    rho_pert = np.sqrt( rho0**2 + dy**2 +  2*rho0 * dy * np.sin(theta0))
    theta_pert = np.arctan2(rho0 * np.sin(theta0) + dy, rho0 * np.cos(theta0))
    
    theta_plot = -(theta0/(2*np.pi)-0.25)%1
    PRC = ((theta_pert+2)%(2*np.pi)-2)/(2*np.pi)
    
    # plt.plot(theta0, theta_pert)
    plt.plot(theta_plot, PRC, '.', label=dyorho0)
plt.legend()

#%% Examples


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from jitcode import jitcode, y
from symengine import sin, cos

from utils import find_point_by_value

def limits_fun(pert_data, *others):
    
    # # extract variables
    # rho = data[:, 0]
    # theta = data[:, 1]
    # phi = data[:, 2]
    # x = rho * np.sin(theta)
    
    pert_rho = pert_data[:, 0]
    pert_theta = pert_data[:, 1]
    pert_x = pert_rho * np.sin(pert_theta)
    
    # ref_peaks, _ = signal.find_peaks(x)
    # maximum = x[ref_peaks[2:]].mean()
    # peaks, _ = signal.find_peaks(pert_x, height=maximum*0.8)

    # peaks as maxima
    # peaks, _ = signal.find_peaks(pert_x)

    # peaks from the phase itself
    cycle_offset = np.pi*3/2 # adjust what part of the cycle we count as "beginning"
    peaks, _ = signal.find_peaks( (pert_theta+cycle_offset) % (2*np.pi))

    # peaks as 0-crossings
    # peaks, _ = signal.find_peaks(-np.abs(pert_x))
    # peaks = [p for p in peaks if pert_x[p]<pert_x[p+1]]
    
    # filter some peaks
    # peaks = np.asarray([p for p in peaks if times[p] not in pert_instants]) # ignore peaks due to the perturbation    
    # peaks = np.asarray([p for p in peaks if np.abs(times[p]-pert_instants).min()>0.05]) # ignore peaks due to the perturbation    
    
    return peaks

# parameters
Tf = 2          # forcing period (s)
Ω  = (2*np.pi)/Tf     # forcing frequency (Hz)

μ  = -0.98
a  =  0.10
Δ  = -3.14
b  =  0.00
f0 = .65

dy = -.25
pert_count = 11

dt = 0.00987

ω = Ω - Δ

# variables and models
ρ = y(0)
θ = y(1)
φ = y(2)
f = [ ρ * (μ - a*ρ**2) + f0 * cos(φ - θ) , 
      ω - b*ρ**2 + f0/ρ * sin(φ-θ) ,
      Ω
	]

initial_rho = calc_rho0(μ, a, Δ, b, f0)
initial_delta = calc_delta(μ, a, Δ, b, f0)
initial_state = np.array([initial_rho, 0, initial_delta])
# initial_state = [ 0.1975, 59.041 , 57.7739]

# run integrator
ODE = jitcode(f)
ODE.set_integrator("dopri5")

# integrate unperturbed, to get the total integration time
ODE.set_initial_value(initial_state, 0)
minima = []
data = initial_state
times = np.array([0])
while len(minima) < 4:
    int_time = np.arange(dt, 0.2, dt) + ODE.t
    bit = np.vstack([ODE.integrate(time) for time in int_time])

    times = np.hstack((times, int_time))
    data = np.vstack((data, bit))
    x = data[:, 0] * np.sin(data[:, 1])

    minima, _ = signal.find_peaks(-x)
    
    if ODE.t > 20:
        break

# we will end the itnegration after the fourth minimum
tf = times[minima[-1]]

times = times[:minima[-1]+1]
x = x[:minima[-1]+1]

# find the interval of time corresponding to the second oscillation
maxima, _ = signal.find_peaks(x)
pert_between = maxima[1:3]*dt

# offset the limits a bit from the maxima
pert_between[0] += 10*dt
pert_between[1] -= 3*dt

# decide when to perturbate
pert_instants = np.linspace(*pert_between, pert_count)

fig, axarr = plt.subplots(pert_count, constrained_layout=True, 
                          sharex=True, sharey=True, figsize=[6.4, 9.2])
# integrate and perturbate the model
for pinst, ax in zip(pert_instants, axarr.flat):
    
    # find perturbation time
    closest_time_index = find_point_by_value(times, pinst)
    closest_time = times[closest_time_index]
    closest_state = data[closest_time_index]
    
    # perturbate
    rho0 = closest_state[0]
    theta0 = closest_state[1]
    rho_pert = np.sqrt( rho0**2 + dy**2 +  2*rho0 * dy * np.sin(theta0))
    theta_pert = np.arctan2(rho0 * np.sin(theta0) + dy, rho0 * np.cos(theta0))
        
    # integrate perturbation
    perturbed_state = np.array([rho_pert, theta_pert, closest_state[2]])
    ODE.set_initial_value(perturbed_state, closest_time)
    
    int_time = np.arange(closest_time+dt, tf, dt)
    pert_data_bit = np.vstack([ODE.integrate(time) for time in int_time])
    
    # pert_data = np.vstack((data[closest_time_index-1], pert_data_bit))
    # pert_data[closest_time_index] = perturbed_state
    
    int_time = np.hstack((int_time[0], int_time))
    pert_data_bit = np.vstack((closest_state, pert_data_bit))
    
    # extract data and plot
    pert_rho = pert_data_bit[:, 0]
    pert_theta = pert_data_bit[:, 1]
    pert_x = pert_rho * np.sin(pert_theta)

    ax.plot(times, x)
    ax.plot(int_time, pert_x)
    
    ax.plot(times[maxima[1:3]], x[maxima[1:3]], 'o')
    
    pert_max = limits_fun(pert_data_bit)
    for imax in pert_max[:2]:
        ax.plot(int_time[imax], pert_x[imax], 'o')

fname = f'{b=}, {a=}, mu={μ}, omega={ω:.2f}, Omega={Ω:.2f}, {f0=}, {dy=}' + '\n cycle finding thorugh maxima'
fig.suptitle(fname)

#%%

pert_instants = np.asarray(pert_instants)
pert_phase = (np.asarray(pert_phase) % (2*np.pi)) / (2*np.pi)
theo_PRC = np.asarray(theo_PRC)

# integrate unperturbed, for comparison
ODE.set_initial_value(initial_state, 0)
data = np.vstack([ODE.integrate(time) for time in times])

# extract variables
rho = data[:, 0]
theta = data[:, 1]
phi = data[:, 2]
x = rho * np.sin(theta)

pert_rho = pert_data[:, 0]
pert_theta = pert_data[:, 1]
pert_x = pert_rho * np.sin(pert_theta)

# find maxima
peaks = limits_fun(pert_data, pert_instants, times)

# calculate PRC
next_peak_indexes = []
prev_peak_indexes = []
linear_phase = []
real_phase = []
PRC = []
for pertt, theta0 in zip(pert_instants, pert_phase):
    
    # skip perturbations that land way too close to the start of the oscillation
    # if np.isclose(pertt, times[ref_peaks], rtol=0, atol=0.02).any():
    #     continue

    # find the end of the perturbed cycle
    next_peak_index = next( i for i, p in zip(range(peaks.size), peaks) if times[p]>pertt )
    # pert_index = np.where(pertt==times)[0][0]
    # if np.all(pert_x[pert_index+1:peaks[next_peak_index]] < pert_x[pert_index]):
    #     print('deleting', next_peak_index)
    #     peaks = np.delete(peaks, next_peak_index)
    # next_peak_index = next( i for i, p in zip(range(peaks.size), peaks) if times[p]>pertt )
    
    next_peak = peaks[next_peak_index]
    prev_peak = peaks[next_peak_index-1]
    prev_prev_peak = peaks[next_peak_index-2]
    
    # calculate the perturbed and unperturbed cycle durations
    p0 = times[prev_peak] - times[prev_prev_peak]
    p1 = times[next_peak] - times[prev_peak]
    
    # PRC
    dp = (p1-p0)/p0
    
    # pahse of the perturbation
    phase = (pertt - times[prev_peak]) / p1
    theta0 = (pertt - times[prev_peak]) / p0
    # if (abs(phase - theta0) / theta0) > 0.25:
    #     continue
    
    # save stuff
    linear_phase.append(phase)
    real_phase.append(theta0)
    PRC.append(dp)
    
    # I just need this for plotting
    next_peak_indexes.append(next_peak_index)

# convert to array
linear_phase = np.asarray(linear_phase)
real_phase = np.asarray(real_phase)
PRC = np.asarray(PRC)
next_peak_indexes = np.asarray(next_peak_indexes)

# plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[17, 4 ], constrained_layout=True, 
                               gridspec_kw={'width_ratios':[3,1]})

# plot lines
# ax1.plot(times, rho.max() * np.sin(phi) ,label='driver')
ax1.plot(times, x, label='unperturbed')
ax1.plot(times, pert_x, label='perturbed')

# plot peaks
triangle_height = pert_x.min()*1.1

# ax1.plot(times[ref_peaks], x[ref_peaks], 'k.')
ax1.plot(times[peaks], pert_x[peaks], 'o')
ax1.plot(pert_instants, triangle_height*np.ones(pert_instants.shape), 'k^')

# plot perturbed periods
for peak_inx, lphase, rphase in zip(next_peak_indexes, linear_phase, real_phase):
    
    # get peaks
    next_peak = peaks[peak_inx]
    prev_peak = peaks[peak_inx-1]
    
    # get times
    peak_time = times[[prev_peak, next_peak]]
    val = triangle_height, triangle_height
    
    # get phase
    pertt = next(pt for pt in pert_instants if peak_time[0] < pt < peak_time[1])
    
    # plot perturbed period
    ax1.plot(peak_time, val, 'b^')
    pert_period = slice(prev_peak, next_peak)
    ax1.plot(times[pert_period], pert_x[pert_period], 'r')
    
    txt = f'{lphase:.2f}\n{rphase:.2f}'
    ax1.text(pertt, triangle_height*1.1, txt, horizontalalignment='center', verticalalignment='top')
    
# plot PRC
ax2.plot(linear_phase, PRC, '.')
ax2.plot(real_phase, PRC, '.')
ax2.set_xlabel('phase of perturbation / 2π')
ax2.set_ylabel('PRC')
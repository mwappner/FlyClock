#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:46:54 2024

@author: marcos
"""

#%% Solve cubic for ρ0
import numpy as np
import matplotlib.pyplot as plt

# parameters of the model
μ  =  -1.0
a  =  0.10
Δ  =  3 # Ω-ω
b  =  0.7
f0 =  5

# coefficients of the cubic equation over ρ0**2:=r
a3 = a**2 + b**2    # a
a2 = 2*(b*Δ - μ*a)  # b
a1 = μ**2 + Δ**2    # c
a0 = -f0**2         # d

h = lambda r: a3 * r**3 + a2 * r**2 + a1 * r + a0

# let's calcualte the depressed cubic parameters
# https://en.wikipedia.org/wiki/Cubic_equation#Depressed_cubic
# we will solve t³ + pt + q = 0 where 

t = lambda r: r + a2 / (3*a3)
p = (3*a3*a1 - a2**2) / (3*a3**2)
q = (2*a2**3 - 9*a3*a2*a1 + 27*a3**2*a0) / (27*a3**3)

# the inverse coordinate change
r = lambda t: t - a2 / (3*a3)

# by Cardanos formula, the real root when (p/3)³ + (q/2)² > 0 is
# https://en.wikipedia.org/wiki/Cubic_equation#Cardano's_formula

D = (p/3)**3 + (q/2)**2
u1 = -q/2 + np.sqrt(D)
u2 = -q/2 - np.sqrt(D)
t0 = np.cbrt(u1) + np.cbrt(u2)

# ts = np.linspace(-1, 1, 200)
# j = lambda t: t**3 + p*t + q
# plt.plot(ts, j(ts))
# plt.plot(t0, j(t0), 'o')

# we go back to r-space and plot to check the answer

r0 = r(t0)
rs = np.linspace(-10, 5, 200)
plt.plot(rs, h(rs))
plt.plot(r0, h(r0), 'o')

j = lambda t: t**3 + p*t + q
plt.plot(t(rs), j(t(rs)))
plt.plot(t0, j(t0), 'o')

#%% Get ρ0, δ and λ
import numpy as np

def calc_rho0(μ, a, Δ, b, f0):
    
    # coefficients of the cubic equation over ρ0**2:=r
    a3 = a**2 + b**2    # a
    a2 = 2*(b*Δ - μ*a)  # b
    a1 = μ**2 + Δ**2    # c
    a0 = -f0**2         # d
    
    # let's calcualte the depressed cubic parameters
    # https://en.wikipedia.org/wiki/Cubic_equation#Depressed_cubic
    # we will solve t³ + pt + q = 0 where 
    
    p = (3*a3*a1 - a2**2) / (3*a3**2)
    q = (2*a2**3 - 9*a3*a2*a1 + 27*a3**2*a0) / (27*a3**3)
    
    # the inverse coordinate change
    r = lambda t: t - a2 / (3*a3)
    
    # by Cardanos formula, the real root when (p/3)³ + (q/2)² > 0 is
    # https://en.wikipedia.org/wiki/Cubic_equation#Cardano's_formula
    
    D = (p/3)**3 + (q/2)**2
    u1 = -q/2 + np.sqrt(D)
    u2 = -q/2 - np.sqrt(D)
    t0 = np.cbrt(u1) + np.cbrt(u2)
    
    # we go back to r-space and plot to check the answer
    r0 = r(t0)
    
    # finally, ρ0 = √r
    return np.sqrt(r0)

def calc_delta(μ, a, Δ, b, f0):
    
    ρ0 = calc_rho0(μ, a, Δ, b, f0)
    # return np.arctan2(-Δ + b * ρ0**2, μ - a * ρ0**2)
    return np.arctan(-(Δ+b*ρ0**2) / (μ - a * ρ0**2))    

def calc_lambda_plus(μ, a, Δ, b, f0):
    
    ρ0 = calc_rho0(μ, a, Δ, b, f0)
    r0 = ρ0**2
    return μ - 2*a*r0 + np.emath.sqrt( a**2 * r0**2 - 3*b**2 * r0**2 - 4*b*Δ*r0-Δ**2)

def calc_lambda_minus(μ, a, Δ, b, f0):
    
    ρ0 = calc_rho0(μ, a, Δ, b, f0)
    r0 = ρ0**2
    return μ - 2*a*r0 - np.emath.sqrt( a**2 * r0**2 - 3*b**2 * r0**2 - 4*b*Δ*r0-Δ**2)

def calc_trace(μ, a, Δ, b, f0):
    
    ρ0 = calc_rho0(μ, a, Δ, b, f0)
    return 2*μ - 4*a * ρ0**2
    
def calc_determinant(μ, a, Δ, b, f0):
    
    ρ0 = calc_rho0(μ, a, Δ, b, f0)
    r0 = ρ0**2
    return 3*(a**2 + b**2) * r0**2 + 4*(b*Δ - μ*a)*r0 + μ**2 + Δ**2
    

Tf = 2.2          # forcing period (s)
Ω  = (2*np.pi)/Tf     # forcing frequency (Hz)

μ  = -3.46
a  =  5.53/11.4**2
Δ  =  -0.6 * Ω
b  =  0.0
f0 =  9.14*11.4

print(f'ρ0 = {calc_rho0(μ, a, Δ, b, f0):.2f}\nδ = {calc_delta(μ, a, Δ, b, f0):.2f}')
print(f'λ+ = {calc_lambda_plus(μ, a, Δ, b, f0):.2f}')
print(f'λ- = {calc_lambda_minus(μ, a, Δ, b, f0):.2f}')


# check lambda calculation
tr = calc_trace(μ, a, Δ, b, f0)
det = calc_determinant(μ, a, Δ, b, f0)
olambdap = 1/2 * (tr + np.emath.sqrt(tr**2 - 4*det))
olambdam = 1/2 * (tr - np.emath.sqrt(tr**2 - 4*det))
print(f'{olambdap:.2f}')
print(f'{olambdam:.2f}')
#%% Plot analytical quantities as function of params

import numpy as np
import matplotlib.pyplot as plt
from utils import significant_digits

fig, axarr = plt.subplots(5, 5, sharey='row', sharex='col', figsize=[13, 9])


# base parameters
Tf = 2.2          # forcing period (s)
Ω  = (2*np.pi)/Tf     # forcing frequency (Hz)

μ  = -3.46
a  =  5.53/11.4**2
Δ  =  -0.6 * Ω
b  =  0.0
f0 =  9.14*11.4

# parameter ranges
mu_range = -4, -1e-3
a_range = 1e-3, 1
nu_range = -4.5*Ω, 1*Ω
b_range = 0, 0.3
f0_range = 1e-3, 120

# useful iterables
param_names = r'$\mu$ [$Hz$]', r'a [$\frac{Hz}{mV^2}$]', r'$\Delta$ [$Hz$]', r'b [$\frac{Hz}{mV^2}$]', r'$f_0$  [mV Hz]'
param_values = {'mu':μ, 'a':a, 'nu':Δ, 'b':b, 'f0':f0}
param_ranges = {'mu':mu_range, 'a':a_range, 'nu':nu_range, 'b':b_range, 'f0':f0_range}

calc_tr = lambda *params: -1 / (calc_lambda_plus(*params).real * Tf) 
calc_tau = lambda *params: calc_delta(*params) / Ω
real_lambdas = lambda *params: calc_lambda_plus(*params).real, lambda *params: calc_lambda_minus(*params).real

for (param_name, param_range), ax_col in zip(param_ranges.items(), axarr.T):
    
    # build parameter dictionary
    param_linspace = np.linspace(*param_range, 2000)
    params = param_values.copy()
    params[param_name] = param_linspace
    
    params_list = list(params.values())
    base_params = list(param_values.values())
    
    base_param_val = param_values[param_name]
    
    quantities = calc_rho0, calc_delta, calc_tau, *real_lambdas, calc_tr
    for i, (calc_q, ax) in enumerate(zip(quantities, [*ax_col[:4], *ax_col[3:]])):
        # calculate quantity
        value = calc_q(*params_list)
        base_val= calc_q(*base_params)
        
        # plot
        color = plt.cm.Dark2(i)
        
        ax.plot(param_linspace, value, c=color)
        ax.plot(base_param_val, base_val, 'o', c=color)
        
# label x axes
for param_name, ax in zip(param_names, axarr[-1]):
    ax.set_xlabel(param_name, fontsize=15)

# label y axes
var_names = r'$\rho_0$ [mV]', 'δ [rad]', 'τ [sec]', r'$\lambda_-$, $\lambda_+$ [Hz]', '$t_{r-}$\n[norm. to CD]'
for var_name, ax in zip(var_names, axarr[:, 0]):
    ax.set_ylabel(var_name, fontsize=15)
    
title = ', '.join(f'{n}={significant_digits(v, 3)}' for n, v in zip(param_names, param_values.values()))
axarr[0, 2].set_title(title, fontsize=15)
        

#%% Plot analytical quantities over multiple params

import matplotlib.pyplot as plt
from utils import adjust_color

fig, axarr = plt.subplots(4, 5, sharey='row', sharex='col', figsize=[13, 9])

# experimental
rho0_exp = 11.4 # half the average amplitude in mV
t_relax_exp = 0.023 # as a fraction of CD
CD = 2.20 # in seconds, measured 25min TSD

# base parameters
# μ  = -1#-4, -0.98, 0.01
# a  =  0.05#0.01, 0.05, 0.10, 1
# Δ  = -5, -3, -1
# b  =  0 #0, 0.1, 0.2
# f0 =  60 #15, 30, 60

Tf = 2.2          # forcing period (s)
Ω  = (2*np.pi)/Tf     # forcing frequency (Hz)

μ  = -3.46
a  =  5.53/11.4**2
Δ  =  -0.6 * Ω
b  =  0.0, 0.1, 0.2
f0 =  9.14*11.4

# parameter ranges
mu_range = -4, -1e-5
a_range = 1e-6, 0.1
nu_range = -7, 7
b_range = 0, 0.3
f0_range = 1e-5, 120

# useful iterables
param_names = r'$\mu$ [$Hz$]', r'a [$\frac{Hz}{mV^2}$]', r'$\nu$ [$Hz$]', r'b [$\frac{Hz}{mV^2}$]', r'$f_0$  [mV Hz]'
param_values = {'mu':μ, 'a':a, 'nu':Δ, 'b':b, 'f0':f0}
param_ranges = {'mu':mu_range, 'a':a_range, 'nu':nu_range, 'b':b_range, 'f0':f0_range}

# multi-valued parameter
multivalued_param = [k for k, v in param_values.items() if hasattr(v, '__iter__')].pop()
param_values = {k:(np.array([v]) if k!=multivalued_param else np.array(v)) for k, v in param_values.items()}
# param_values['nu'] *= -1

# relaxation time lambda
calc_tr = lambda *params: -1 / (calc_lambda_plus(*params).real * CD) 
real_lambdas = lambda *params: calc_lambda_plus(*params).real, lambda *params: calc_lambda_minus(*params).real
# four_det = lambda *params: calc_determinant(*params)

for (param_name, param_range), ax_col in zip(param_ranges.items(), axarr.T):
    
    # build parameter dictionary
    param_linspace = np.linspace(*param_range, 2000)
        
    quantities = calc_rho0, calc_delta, *real_lambdas, calc_tr
    for i, (calc_q, ax) in enumerate(zip(quantities, [*ax_col[:3], *ax_col[2:]])):
    # quantities = calc_rho0, calc_delta, *real_lambdas, calc_trace, four_det, calc_tr
    # for i, (calc_q, ax) in enumerate(zip(quantities, [*ax_col[:3], *ax_col[2:4], *ax_col[3:]])):
        
        for j, mvp in enumerate(param_values[multivalued_param]):
    
            params = param_values.copy()
            params[multivalued_param] = mvp
            params[param_name] = param_linspace
            params_list = list(params.values())
            
            # calculate quantity
            value = calc_q(*params_list)
            
            # plot
            color = plt.cm.Dark2(i)
            color = adjust_color(color, 0.5*(j+1))
            lcolor = 'k' if param_name == multivalued_param else color
            
            ax.plot(param_linspace, value, c=lcolor)
            
            # plot points
            base_params = param_values.copy()
            base_params[multivalued_param] = mvp
            
            base_params_list = list(base_params.values())
            base_param_val = base_params[param_name]
            
            base_val = calc_q(*base_params_list)
            ax.plot(base_param_val, base_val, 'o', c=color, zorder=2.1)

# measured values
for ax in axarr[0]:
    ax.axhline(rho0_exp, color='k', zorder=1, linestyle='--')

for ax in axarr[3]:
    ax.axhline(t_relax_exp, color='k', zorder=1, linestyle='--')

# x limits
for xlims, ax in zip(param_ranges.values(), axarr[-1]):
    ax.set_xlim(xlims)

# y limits
ylim = axarr[-1, -1].get_ylim()
ymax = min(ylim[1], 0.05)
ymin = max(ylim[0], 0)
axarr[-1, -1].set_ylim(ymin, ymax)

# label x axes
for param_name, ax in zip(param_names, axarr[-1]):
    ax.set_xlabel(param_name, fontsize=15)

# label y axes
var_names = r'$\rho_0$ [mV]', 'δ [rad]', r'$\lambda_-$, $\lambda_+$ [Hz]', r'$t_{r-}$ [norm. to CD]'
for var_name, ax in zip(var_names, axarr[:, 0]):
    ax.set_ylabel(var_name, fontsize=15)

# title
title = ', '.join(f'{n}={v}' for n, v in zip(param_names, param_values.values()))
fig.suptitle(title, fontsize=15)


#%% Fit the data by inverting relations

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils import enzip, adjust_color

# experimental
rho0_exp = 11 # half the average amplitude in mV
t_relax_exp = 0.023 # as a fraction of CD
CD_exp = 2.20 # in seconds, measured 25min TSD
lag_exp = 0.13 # lag between small an large in seconds

surface_ratio = 5**2/10**2 # small cell surface over large cell surface

# values for the δ and Δ lines
delta_line_value = -0.05,
Delta_line_value = -0.6, -1, -1.4

max_val = 80 # maximum plotted value

# some constants calcualted from the experimental values above
tr = t_relax_exp * CD_exp# * surface_ratio
λp = -1 / tr
ρ0 = rho0_exp
Ω = 2*np.pi / CD_exp
dδ = lag_exp * Ω

# we choose δ and Δ as controls variables and pot a and μ
μa = lambda δ, Δ: -( (Δ/np.tan(δ) + λp)**2 + Δ**2) / (2*(Δ/np.tan(δ) + λp))
μ = lambda δ, Δ: μa(δ, Δ) - Δ/np.tan(δ)
μf = lambda δ, Δ: Δ / np.sin(δ)

# make linspaces
delta_range = -0.08 * Ω, 0.08 * Ω
# delta_range = -0.3 , 0.3 
Delta_range = Ω - 2*np.pi / 1, Ω - 2*np.pi / 6

delta_lsp = np.linspace(*delta_range, 1001)
Delta_lsp = np.linspace(*Delta_range, 1001)

# make sure values where the singularities happen are included
delta_single = np.arctan(-np.asarray(Delta_line_value) / λp)
Delta_single = - np.tan(delta_line_value) * λp

where_insert = np.searchsorted(delta_lsp, delta_single)
delta_lsp = np.insert(delta_lsp, where_insert, delta_single)

where_insert = np.searchsorted(Delta_lsp, Delta_single)
Delta_lsp = np.insert(Delta_lsp, where_insert, Delta_single)

# make grids
delta_grid, Delta_grid = np.meshgrid(delta_lsp, Delta_lsp)

mua_grid = μa(delta_grid, Delta_grid)
mu_grid = μ(delta_grid, Delta_grid)
muf_grid = μf(delta_grid, Delta_grid)

# hide values (set to NaN) where μ>=0
mu_mask = mu_grid>=0
a_mask = mua_grid<=0
f_mask = muf_grid<=0

muf_grid[mu_mask | a_mask | f_mask] = np.nan
mua_grid[mu_mask | a_mask | f_mask] = np.nan
mu_grid[mu_mask | a_mask | f_mask] = np.nan

# plot
fig, axarr = plt.subplots(3, 3, constrained_layout=True, figsize=[10, 8])
extent = *delta_range, *Delta_range
ax1, ax2, ax3 = axarr[:, 0]

# maps
def make_custom_map(cmap_name):
    cmap = plt.cm.get_cmap(cmap_name)
    
    colors1 = cmap(np.linspace(0., .9, 128))
    colors2 = plt.cm.Greys_r(np.linspace(0.2, 1, 128))

    # combine them and build a new colormap
    colors = np.vstack((colors2, colors1))
    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    
    return mymap

def make_mono_map(color):
    if isinstance(color, str):
        color = mcolors.to_rgba(color)
        
    color = adjust_color(color, 0.5)
    
    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', [(0,0,0,0), color])
    mymap.set_bad(alpha=0)
    return mymap

cmaps = 'Oranges, Greens, Blues'
for ax, grid, cmap_name in zip(axarr[:, 0], (mua_grid, mu_grid, muf_grid), cmaps.split(', ')):
    
    # cap max value
    vmax = min(np.nanmax(grid), max_val)
    vmin = max(np.nanmin(grid), -max_val)
    
    # colormap
    cmap = make_custom_map(cmap_name)
    
    # norm
    if vmax > 0 and vmin <0:
        norm = mcolors.TwoSlopeNorm(vcenter=0, vmax=vmax, vmin=vmin)
    else:
        cmap = plt.cm.get_cmap(cmap_name)
        norm = mcolors.Normalize(vmax=vmax, vmin=vmin)
    cmap.set_bad(alpha=0)
    
    # plot masks
    ax.imshow(a_mask, extent=extent, aspect='auto', origin='lower',
                    cmap=make_mono_map('xkcd:burgundy'), interpolation='none')
    ax.imshow(mu_mask, extent=extent, aspect='auto', origin='lower',
                    cmap=make_mono_map('xkcd:dark green'), interpolation='none')
    ax.imshow(f_mask, extent=extent, aspect='auto', origin='lower',
                    cmap=make_mono_map('xkcd:dark blue'), interpolation='none')    
    # plot image
    im1 = ax.imshow(grid, extent=extent, aspect='auto', origin='lower',
                    cmap=cmap, norm=norm )
    
    # im1 = ax.contourf(delta_grid, Delta_grid, grid, levels=10,
    #                 vmax=vmax, vmin=vmin, cmap=cmap)
    print(vmin, vmax)
    ax.set_xlabel('δ = ψ-θ')
    ax.set_ylabel('Δ = Ω-ω')
    
    # colorbar
    fig.colorbar(im1, ax=ax)
    
    # reference lines for the curves
    ax.vlines(delta_line_value, *Delta_range, color='r', ls='--')
    ax.hlines(Delta_line_value, *delta_range, color='r', ls='--')
    
ax1.set_title(r'$\mu_a = a{\rho_0}^2$', fontsize=15)
ax2.set_title(r'$\mu$', fontsize=15)
ax3.set_title(r'$\mu_f = f_0 / \rho_0$', fontsize=15)
    
# δ and Δ curves
axlabels = 'δ = ψ-θ', 'Δ = Ω-ω'
for i, (delta_vals, Delta_vals, axlabel) in enzip((delta_lsp, delta_line_value),
                                                  (Delta_line_value, Delta_lsp),
                                                  axlabels):

    # calculate the curves
    deltas_temp_grid = np.meshgrid(delta_vals, Delta_vals)
    curves = [p(*deltas_temp_grid) for p in (μa, μ, μf)]
    
    # get mask
    mu_line = curves[1]
    where = mu_line<0
    
    if where.shape[0] > where.shape[1]:
        where = where.T
    
    # get the variable to plot against
    lsp_var = delta_vals if isinstance(delta_vals, np.ndarray) else Delta_vals
    const_var = delta_vals if not isinstance(delta_vals, np.ndarray) else Delta_vals
    
    # plot
    axcol = axarr[:, i+1]
    ax1, ax2, ax3 = axcol
    for ax, curve, cmap_name in zip(axcol, curves, cmaps.split(', ')):
        
        # get cmap
        cmap = plt.cm.get_cmap(cmap_name)
        
        # check if we need to transpose curves array to iterate over it
        if curve.shape[0] > curve.shape[1]:
            curve = curve.T
        
        # plot curves
        for j, (one_curve, one_mask, cvar) in enzip(curve, where, const_var):
            color = cmap(0.5 + 0.25*(j-len(const_var)//2))
            
            ax.plot(lsp_var[one_mask], one_curve[one_mask], c=color, label=cvar)
            ax.plot(lsp_var[~one_mask], one_curve[~one_mask], '--', c=color)
    
        # set plotting limits
        ylims = ax.get_ylim()
        
        ymax = min(ylims[1], max_val)
        ymin = max(ylims[0], -max_val)
        
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(lsp_var.min(), lsp_var.max())
        ax.set_xlabel(axlabel)
        ax.legend()
    
    ax1.set_ylabel(r'$\mu_a = a{\rho_0}^2$')
    ax2.set_ylabel(r'$\mu$')
    ax3.set_ylabel(r'$\mu_f = f_0 / \rho_0$')

# plot points on curves
for axrow, param_calc in zip(axarr, (μa, μ, μf)):
    ax1, ax2, ax3 = axrow

    # calculate paramater values    
    for dval in delta_line_value:
        for Dval in Delta_line_value:
            pval = param_calc(dval, Dval)
            
            ax2.plot(dval, pval, 'k.')
            ax3.plot(Dval, pval, 'k.')

# hackfix values of axis
for axrow in axarr:
    ax1, ax2, ax3 = axrow
    ax2.set_ylim(ax3.get_ylim())


#%% Fit fit separate values for large and small

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import enzip

scolor = '#d06c9eff'
lcolor = '#006680ff'

### Experimental values
# large
rho0_exp_large = 11.4 # half the average amplitude in mV
t_relax_exp_large = 0.023 # as a fraction of CD

# small
rho0_exp_small = 8.85 # half the average amplitude in mV
t_relax_exp_small = t_relax_exp_large # as a fraction of CD

#general
lag_exp = 0.13 # lag between small an large in seconds
CD_exp = 2.20 # in seconds, measured 25min TSD

# turn general measurements into radians and whatnot
π2 = 2*np.pi
Ω = π2/CD_exp
dδ = lag_exp * Ω

# a parameter choice for δl, Δl and Δs
Delta_large = -0.6 * Ω
delta_large = -0.03 * π2

Delta_small = 0.6 * Ω
delta_small = delta_large + dδ

# Delta_large = -1.1 * Ω
# delta_large = -0.065 * π2

# Delta_small = -0.18 * Ω
# delta_small = delta_large + dδ


# make linspaces
delta_range = -0.05 * π2, 0.05 * π2
# delta_range = -0.3 , 0.3 
Delta_range = Ω - Ω*2.5, Ω # - 2*np.pi / 6
# Delta_range = Ω - 2*np.pi / 0.6, 0 # - 2*np.pi / 6

# make figure
fig, axarr = plt.subplots(3,3, constrained_layout=True, sharex='col', figsize=[8.4, 4.8], sharey='row')

rhos = rho0_exp_large, rho0_exp_small
trelaxs = t_relax_exp_large, t_relax_exp_small
deltas = delta_large, delta_small
Deltas = Delta_large, Delta_small
colors = lcolor, scolor
parameters = []

def plot_masked(ax, independent, dependent, mask, color):
    
    # make valid array
    valid = dependent.copy()
    valid[~mask] = np.nan
    
    # make invalid array
    invalid = dependent.copy()
    invalid[mask] = np.nan
    
    ax.plot(independent, valid, c=color)
    ax.plot(independent, invalid, '--', c='0.7')

def add_singularity(lsp, singularity, count):
    
    single_lsp = singularity + np.linspace(-0.02, 0.02, 10001)    
    where_insert = np.searchsorted(lsp, single_lsp)
    lsp = np.insert(lsp, where_insert, single_lsp)

    return lsp

for rho0_exp, t_relax_exp, Delta, delta, color in zip(rhos, trelaxs, Deltas, deltas, colors):

    # some constants calcualted from the experimental values above
    tr = t_relax_exp * CD_exp
    λp = -1 / tr
    ρ0 = rho0_exp
        
    # generate parameter masks
    μa = lambda δ, Δ: -( (Δ/np.tan(δ) + λp)**2 + Δ**2) / (2*(Δ/np.tan(δ) + λp))
    μ = lambda δ, Δ: μa(δ, Δ) - Δ/np.tan(δ)
    μf = lambda δ, Δ: Δ / np.sin(δ)

    # make linspaces
    delta_lsp = np.linspace(*delta_range, 2001)
    Delta_lsp = np.linspace(*Delta_range, 2001)

    # add detail around the non zero singularity
    delta_single = np.arctan(-np.asarray(Delta) / λp)
    Delta_single = - np.tan(delta) * λp
    
    delta_lsp = add_singularity(delta_lsp, delta_single, 10001)
    Delta_lsp = add_singularity(Delta_lsp, Delta_single, 10001)    

    # make mask
    delta_mask = μ(delta_lsp, Delta) < 0
    Delta_mask = μ(delta, Delta_lsp) < 0
    params = []
    for what, ax in zip((μa, μ, μf), axarr):
        
        # δ axis
        plot_masked(ax[0], delta_lsp/π2, what(delta_lsp, Delta), delta_mask, color)
        ax[0].plot(delta/π2, what(delta, Delta), 'o', c=color)
        
        # Δ axis
        plot_masked(ax[1], Delta_lsp/Ω, what(delta, Delta_lsp), Delta_mask, color)
        ax[1].plot(Delta/Ω, what(delta, Delta), 'o', c=color)
        
        # ω axis
        plot_masked(ax[2], (Ω-Delta_lsp)/Ω, what(delta, Delta_lsp), Delta_mask, color)
        ax[2].plot((Ω-Delta)/Ω, what(delta, Delta), 'o', c=color)
        
        # save fitter param
        params.append(what(delta, Delta))
        
        # ax[0].axvline(delta_single, color='k')
        # ax[1].axvline(Delta_single, color='k')
        # ax[2].axvline(1-Delta_single, color='k')
        
    # save parameter values for large and small
    params.append(Delta/Ω)
    params.append((Ω-Delta)/Ω)
    params.append(delta/π2)
    params.append(rho0_exp)
    params.append(t_relax_exp)
    parameters.append(params)
    
# μa limits and label
for ax in axarr[0]:
    ax.set_ylim(0, 10)
    ax.set_ylabel(r'$\mu_a = a{\rho_0}^2$')
    
# μ limits and label
for ax in axarr[1]:
    ax.set_ylim(-18, 5)
    ax.set_ylabel(r'$\mu$')
    
# μf limits and label
for ax in axarr[2]:
    ax.set_ylim(0, 18)
    ax.set_ylabel(r'$\mu_f = f_0 / \rho_0$')
    
# set xlabels
axarr[-1, 0].set_xlabel('δ/2π = (ψ-θ)/2π')
axarr[-1, 1].set_xlabel('Δ/Ω = (Ω-ω)/Ω')
axarr[-1, 2].set_xlabel('ω/Ω')

# set x limits
axarr[-1, 0].set_xlim(*np.asarray(delta_range)/π2)
axarr[-1, 1].set_xlim(*np.asarray(Delta_range)/Ω)
axarr[-1, 2].set_xlim(*(Ω-np.asarray(Delta_range))[::-1]/Ω)

# print parameters
index = 'μa', 'μ', 'μf', 'Δ/Ω', 'ω/Ω', 'δ/2π', 'ρ0', 'tr'
df = pd.DataFrame(np.array(parameters).T, columns=['large', 'small'], index=index)

print(df)

# set as title
large_params = 'LARGE:   ' + '    '.join(f'{n}:{v:.2f}' for n, v in zip(index, df['large']))
small_params = 'SMALL:   ' + '    '.join(f'{n}:{v:.2f}' for n, v in zip(index, df['small']))

fig.suptitle('\n'.join((large_params, small_params)))


#%% Plot analytical quantities for small and large

import numpy as np
import matplotlib.pyplot as plt
from utils import significant_digits

fig, axarr = plt.subplots(5, 5, sharey='row', sharex='col', figsize=[13, 9])

scolor = '#d06c9eff'
lcolor = '#006680ff'

# base parameters
Tf = 2.2          # forcing period (s)
Ω  = (2*np.pi)/Tf     # forcing frequency (Hz)

large_params = dict(
    μ  = -3.46,
    a  =  5.53/11.4**2,
    Δ  =  -0.6 * Ω,
    b  =  0.0,
    f0 =  9.14*11.4,
    )

small_params = dict(
    μ  = -3.88,
    a  =  5.37/8.85**2,
    Δ  =  -0.6 * Ω,
    b  =  0.0,
    f0 =  9.427*8.85,
    )

# parameter ranges
mu_range = -8, -1e-3
a_range = 1e-3, 0.1
Delta_range = -4.5*Ω, 1*Ω
b_range = 0, 0.2
f0_range = 1e-3, 130

# useful iterables
param_names = r'$\mu$ [$Hz$]', r'a [$\frac{Hz}{mV^2}$]', r'$\Delta$ [$Hz$]', r'b [$\frac{Hz}{mV^2}$]', r'$f_0$  [mV Hz]'

param_ranges = {'mu':mu_range, 'a':a_range, 'Delta':Delta_range, 'b':b_range, 'f0':f0_range}

calc_tr = lambda *params: -1 / (calc_lambda_plus(*params).real * Tf) 
calc_tau = lambda *params: calc_delta(*params) / Ω
real_lambdas = lambda *params: calc_lambda_plus(*params).real, lambda *params: calc_lambda_minus(*params).real

for color, (μ, a, Δ, b, f0) in zip((lcolor, scolor), (large_params.values(), small_params.values())):

    param_values = {'mu':μ, 'a':a, 'Delta':Δ, 'b':b, 'f0':f0}
    for (param_name, param_range), ax_col in zip(param_ranges.items(), axarr.T):
        
        # build parameter dictionary
        param_linspace = np.linspace(*param_range, 2000)
        params = param_values.copy()
        params[param_name] = param_linspace
        
        params_list = list(params.values())
        base_params = list(param_values.values())
        
        base_param_val = param_values[param_name]
        
        quantities = calc_rho0, calc_delta, calc_tau, *real_lambdas, calc_tr
        for calc_q, ax in zip(quantities, [*ax_col[:4], *ax_col[3:]]):
            # calculate quantity
            value = calc_q(*params_list)
            base_val= calc_q(*base_params)
            
            ax.plot(param_linspace, value, c=color)
            ax.plot(base_param_val, base_val, 'o', c=color)

# set x limits
for ax, lims in zip(axarr[0], param_ranges.values()):
    ax.set_xlim(lims)

axarr[-1, -1].set_ylim(0, 0.1)
        
# label x axes
for param_name, ax in zip(param_names, axarr[-1]):
    ax.set_xlabel(param_name, fontsize=15)

# label y axes
var_names = r'$\rho_0$ [mV]', 'δ [rad]', 'τ [sec]', r'$\lambda_-$, $\lambda_+$ [Hz]', '$t_{r-}$\n[norm. to CD]'
for var_name, ax in zip(var_names, axarr[:, 0]):
    ax.set_ylabel(var_name, fontsize=15)
    
plarge = '   '.join(f'{n}={significant_digits(v, 3)}' for n, v in zip(param_names, large_params.values()))
psmall = '   '.join(f'{n}={significant_digits(v, 3)}' for n, v in zip(param_names, small_params.values()))
title = 'LARGE:   ' + plarge + '\nSMALL:   ' + psmall
axarr[0, 2].set_title(title, fontsize=15)

#%% Plot analytical quantities for small and large with rescaled axes

import numpy as np
import matplotlib.pyplot as plt
from utils import significant_digits

fig, axarr = plt.subplots(5, 5, sharey='row', sharex='col', figsize=[13, 9])

scolor = '#d06c9eff'
lcolor = '#006680ff'

# base parameters
Tf = 2.2          # forcing period (s)
Ω  = (2*np.pi)/Tf     # forcing frequency (Hz)

rho0_large = 11.4
rho0_small = 8.85

# symmetric parameter set
large_params = dict(
    μ  = -3.46,
    a  =  5.53 / rho0_large**2,
    Δ  =  -0.6 * Ω,
    b  =  0.0,
    f0 =  9.14 * rho0_large,
    ω  =  1.6 * Ω, 
    )

small_params = dict(
    μ  = -3.88,
    a  =  5.37 / rho0_small**2,
    Δ  =  0.6 * Ω,
    b  =  0.0,
    f0 =  9.427 * rho0_small,
    ω  =  0.4 * Ω, 
    )

# # very different parameter set
# large_params = dict(
#     μ  = -0.61,
#     a  =  6.65 / rho0_large**2,
#     Δ  =  -1.1 * Ω,
#     b  =  0.0,
#     f0 =  7.91 * rho0_large,
#     )

# small_params = dict(
#     μ  = -10.86,
#     a  =  2.98 / rho0_small**2,
#     Δ  =  -0.18 * Ω,
#     b  =  0.0,
#     f0 =  13.85 * rho0_small,
#     )

# parameter ranges
mu_range = -15, -1e-3
a_range = 1e-3, 1
Delta_range = -4.5*Ω, 1*Ω
b_range = 0, 0.3
f0_range = 1e-3, 160

# parameter transformers
param_transformers = {
    'mu'   : lambda mu, r0: mu,
    'a'    : lambda a, r0: a * r0**2,
    'Delta': lambda Delta, r0: Delta / Ω,
    'b'    : lambda b, r0: b * r0**2,
    'f0'   : lambda f0, r0: f0 / r0,
    }

# useful iterables
param_names = (r'$\mu$ [$Hz$]', r'$a\rho_0^2$ [$Hz$]', r'$\Delta/\Omega$',
               r'$b\rho_0^2$ [$Hz$]', r'$f_0/\rho_0$  [Hz]')
param_ranges = {'mu':mu_range, 'a':a_range, 'Delta':Delta_range, 'b':b_range, 'f0':f0_range}

calc_tr = lambda *params: -1 / (calc_lambda_plus(*params).real * Tf) 
calc_tau = lambda *params: calc_delta(*params) / Ω
real_lambdas = lambda *params: calc_lambda_plus(*params).real, lambda *params: calc_lambda_minus(*params).real

for color, rho0, (μ, a, Δ, b, f0, ω) in zip((lcolor, scolor), (rho0_large, rho0_small),
                                         (large_params.values(), small_params.values())):

    param_values = {'mu':μ, 'a':a, 'Delta':Δ, 'b':b, 'f0':f0}
    for (param_name, param_range), ax_col in zip(param_ranges.items(), axarr.T):
        
        # build parameter dictionary
        param_linspace = np.linspace(*param_range, 2000)
        params = param_values.copy()
        params[param_name] = param_linspace
        
        params_list = list(params.values())
        base_params = list(param_values.values())
        
        # rescale parameters
        param_linspace = param_transformers[param_name](param_linspace, rho0)
        base_param_val = param_transformers[param_name](param_values[param_name], rho0)
        
        quantities = calc_rho0, calc_delta, calc_tau, *real_lambdas, calc_tr
        for calc_q, ax in zip(quantities, [*ax_col[:4], *ax_col[3:]]):
            # calculate quantity
            value = calc_q(*params_list)
            base_val= calc_q(*base_params)
            
            ax.plot(param_linspace, value, c=color)
            ax.plot(base_param_val, base_val, 'o', c=color)

# set xlims
for ax in axarr[-1]:
    ax.set_xlim(0, 12)
    
axarr[0,0].set_xlim(-10, 0) # mu
axarr[0,2].set_xlim(-1.5, 1) # delta

axarr[ 2, -1].set_ylim(-0.3, 0.3) # tau
axarr[-2, -1].set_ylim(-30, 0) # lambdas
axarr[-1, -1].set_ylim(0, 0.1) # tr
        
# label x axes
for param_name, ax in zip(param_names, axarr[-1]):
    ax.set_xlabel(param_name, fontsize=15)

# label y axes
var_names = r'$\rho_0$ [mV]', 'δ [rad]', 'τ [sec]', r'$\lambda_-$, $\lambda_+$ [Hz]', '$t_{r-}$\n[norm. to CD]'
for var_name, ax in zip(var_names, axarr[:, 0]):
    ax.set_ylabel(var_name, fontsize=15)
    
plarge = '   '.join(f'{n}={significant_digits(t(v, rho0_large), 3)}' for n, v, t in zip(param_names, large_params.values(), param_transformers.values(), ))
psmall = '   '.join(f'{n}={significant_digits(t(v, rho0_small), 3)}' for n, v, t in zip(param_names, small_params.values(), param_transformers.values(), ))
title = 'LARGE:   ' + plarge + '\nSMALL:   ' + psmall
axarr[0, 2].set_title(title, fontsize=15)

#%% Perturbation
"""Calculate how the perturbation evolves using the linearized system"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

from utils import find_point_by_value

# Parameters
Tf = 2.2              # forcing period (s)
Ω  = (2*np.pi)/Tf     # forcing frequency (Hz)

μ  = -3.46
a  =  5.53/11.4**2
Δ  =  -0.6 * Ω
b  =  0.0
f0 =  9.14*11.4

δ = calc_delta(μ, a, Δ, b, f0)
λp = calc_lambda_plus(μ, a, Δ, b, f0)
λm = calc_lambda_minus(μ, a, Δ, b, f0)

# perturbation time and magnitude
t0 = 2.4
dy = -1

# Just before perturbation
rho0 = calc_rho0(μ, a, Δ, b, f0)
theta0 = lambda t0: Ω*t0 - δ 
tr = lambda theta: ((theta + np.pi) % (2*np.pi)) - np.pi # a transformation to fold theta into [-π, π]

# Perturbed state
rho_p = lambda t0: np.sqrt( rho0**2 + dy**2 +  2*rho0 * dy * np.sin(theta0(t0)) )
theta_p = lambda t0: np.arctan2(rho0 * np.sin(theta0(t0)) + dy, rho0 * np.cos(theta0(t0)) )
# theta_p = lambda t0: np.arctan(np.tan(theta0(t0)) + dy/ (rho0 * np.cos(theta0(t0))) )

# Projections into the eigenspace
α = μ - 3*a * rho0**2
β = -f0 * np.sin(δ)
A = lambda t0: (tr(theta0(t0)) - theta_p(t0) + (rho0-rho_p(t0)) * (α - λm)/β) / ((λm - λp)/β)
B = lambda t0: rho_p(t0) - rho0 - A(t0)

# eigenvector elements
vp1 = 1
vp2 = (α-λp)/β
vm1 = 1
vm2 = (α-λm)/β

### make figure
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Dark2.colors)

fig = plt.figure(constrained_layout=True, figsize=(11, 5.4))
subfig1, subfig2 = fig.subfigures(1, 2, wspace=0.07)

ax1, ax2, ax3 = subfig1.subplots(3, 1)
ax4, ax5 = subfig2.subplots(2, 1)

### Plot A and B
t0s = np.linspace(0, 6, 2000)
ax1.plot(t0s, A(t0s), label=f'A={A(t0):.2f}', c='C0')
ax1.plot(t0s, B(t0s), label=f'B={B(t0):.2f}', c='C1')

ax1.plot(t0, A(t0), 'o', c='C0')
ax1.plot(t0, B(t0), 'o', c='C1')

# ax0.plot(t0s, tr(theta0(t0s)), label = 'θ0', c='C04')
# ax0.plot(t0s, theta0(t0s), label = 'θ0', c='C06')
# ax0.plot(t0s, theta_p(t0s), label='θp', c='C05')

ax1.legend()
ax1.set_xlabel('t0')
ax1.set_ylabel('A(t0), B(t0)')
ax1.set_xlim(t0s.min(), t0s.max())

### Plot oscillation
# calculate it
time = np.linspace(0, 5, 10000)
psi = Ω*time 
theta_stationary = Ω*time - δ
y_stationary = rho0 * np.sin(theta_stationary)
forcing = f0/rho0 * np.sin(psi)

# plot it
ax2.plot(time, forcing, c='C3', label='forcing')
ax2.plot(time, y_stationary, c='C2', label='cell')

# format
ax2.legend()
ax2.set_xlabel('time [sec]')
ax2.set_ylabel('amplitude [mV]')
ax2.set_xlim(time[0], time[-1])

### Plot perturbation
# calculate it
exp_p = np.exp(λp * (time-t0))
exp_m = np.exp(λm * (time-t0)) 

rho_transitory = rho0 + A(t0) * vp1 * exp_p + B(t0) * vm1 * exp_m
phi_transitory = δ    + A(t0) * vp2 * exp_p + B(t0) * vm2 * exp_m

theta_transitory  = Ω*time - phi_transitory 
y_transitory = rho_transitory * np.sin(theta_transitory)

# find perturbation instant and relpace data before perturbation
pert_index = find_point_by_value(time, t0)
y_transitory[:pert_index] = y_stationary[:pert_index]

# plot it
ax3.plot(time, y_stationary, c='C2', label='unperturbed')
ax3.plot(time, y_transitory, c='C4', label='linearized')

# format
ax3.set_xlabel('time [sec]')
ax3.set_ylabel('amplitude [mV]')
ax3.set_xlim(time[0], time[-1])
ax3.set_ylim(y_stationary.min() + dy, ax3.get_ylim()[1])

### Add exact solution

import numpy as np
import matplotlib.pyplot as plt
from jitcode import jitcode, y
from symengine import sin, cos

from utils import find_point_by_value

# variables and models
ρ = y(0)
θ = y(1)
φ = y(2)
f = [ ρ * (μ - a*ρ**2) + f0 * cos(φ - θ) , 
      ω - b*ρ**2 + f0/ρ * sin(φ-θ) ,
      Ω
	]

# run integrator
ODE = jitcode(f)
ODE.set_integrator("dopri5")

rho_pert = rho_p(t0)
theta_pert = theta_p(t0)

# integrate perturbation
perturbed_state = np.array([rho_pert, theta_pert, Ω*t0])
ODE.set_initial_value(perturbed_state, t0)
int_times = time[pert_index+1:]
pert_data = np.vstack([ODE.integrate(time) for time in int_times])

# extract variables
rho_int = pert_data[:, 0]
theta_int = pert_data[:, 1]
y_int = rho_int * np.sin(theta_int)

# plot
ax3.plot(int_times, y_int, c='C5', label='exact')
ax3.legend(loc='lower left')

fig.suptitle(f'Perturbation at {t0=} sec (θo = {tr(theta0(t0)):.1f} rad)')

### Add perturbation plots

tp = -1 / λp
tm = -1 / λm
ts = time[pert_index:] - t0

# Whole signal
y_linear = np.abs( (y_transitory[pert_index:] - y_stationary[pert_index:]) / dy)
y_exact = np.abs( (y_int - y_stationary[pert_index+1:]) / dy)

rho_linear = np.abs(rho_transitory[pert_index:] - rho0)
rho_exact = np.abs(rho_int - rho0)

for ax, exact, linearized in zip((ax4, ax5), (y_exact, rho_exact), (y_linear, rho_linear)):

    exact = np.log10(exact)
    linearized = np.log10(linearized)    

    # plot
    ax.plot(ts, linearized, c='C4', label='linearized', lw=2)
    ax.plot(int_times-t0, exact, c='C5', label='exact', lw=2)
    
    # plot lines
    ax.axvline(tp, color='k', linestyle='--', lw=0.5)
    ax.axvline(tm, color='k', linestyle='--', lw=0.5)
    ax.text(tp, linearized[0], r' $t_{r+}$', va='top')
    ax.text(tm, linearized[0], r' $t_{r-}$', va='top')

    # fit
    max_fit_time = 1.5*tp
    max_fit_time_index = find_point_by_value(ts, max_fit_time)
    
    time_bit = ts[:max_fit_time_index]
    data_bit = linearized[:max_fit_time_index]
    
    lin = lambda x, a, b: a*x+b
    popt, pcov = optimize.curve_fit(lin, time_bit, data_bit)
    ax.plot(time_bit, lin(time_bit, *popt), c='C3', label=f'slope={popt[0]:.1f}')

    # format
    # ax.set_yscale('log')
    
    ax.set_xlim(0, 4*tp)
    ax.set_ylim(-2, ax.get_ylim()[1])
    ax.set_xlabel('Time since perturbation [sec]')
    ax.legend(loc='lower right')
    
ax4.set_ylabel(r'$log(|y(t) - y_{stationary}|)$')
ax5.set_ylabel(r'$log(|\rho(t) - \rho_0|)$')

# fig.savefig(savedir / f'perturbation - {t0=}.svg')
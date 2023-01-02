#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 16:46:26 2022

This is a module containing function for the code that analyzes sync of two fly neurons
@author: luis
"""

import numpy as np
from numpy.polynomial import polynomial as poly
import matplotlib.pyplot as plt
from scipy import signal
# from scipy import stats
import pyabf
import pandas as pd





#%%
""" this function
filters the input data and returns result """
def filter(x, y, dt):
    print("Plot whole timeseries split in panels.")

    """
    filters high frequency noise """
    # print("I filter data to remove high frequencies using savitky-golay.")
    # xf = signal.savgol_filter(x, 1001, 1, mode='mirror')
    # yf = signal.savgol_filter(y, 1001, 1, mode='mirror')
    
    print("I filter data to remove high frequencies using a butterworth filter.")
    filter_order = 2
    filter_critical_frequencies = 10
    filter_type = 'lp'    
    sos = signal.butter(filter_order, filter_critical_frequencies, filter_type, fs=1/dt , output='sos')
    xf = signal.sosfilt(sos, x)
    yf = signal.sosfilt(sos, y)        
    return xf, yf



#%%
""" this function
finds the peaks of cycles for later backtracking upstrokes """
def peakfind(t, xfdd, yfdd):
    print("Find the peaks of cycles for later backtracking upstrokes.")

    # find the traces peaks : sliced traces
    xfdd_relmax, = signal.argrelmax (xfdd, axis=0, order=10, mode='clip')
    yfdd_relmax, = signal.argrelmax (yfdd, axis=0, order=10, mode='clip')
    # find the traces troughs : sliced traces
    # relativemin_xfs, = signal.argrelmin (xfs, axis=0, order=10000, mode='clip')
    # relativemin_yfs, = signal.argrelmin (yfs, axis=0, order=10000, mode='clip')
    
    xfdd_relmaxcln = xfdd_relmax
    yfdd_relmaxcln = yfdd_relmax
        
    # removes from the list the maxima below zero threshold
    peak_threshold = 5 # this is the threshold value in mV for accepting a peak
    
    i = len(xfdd_relmaxcln)-1
    while i >= 0:
        if xfdd[xfdd_relmaxcln[i]] < peak_threshold:
            xfdd_relmaxcln = np.delete(xfdd_relmaxcln, i)
        i=i-1
    
    i = len(yfdd_relmaxcln)-1
    while i >= 0:
        if yfdd[yfdd_relmaxcln[i]] < peak_threshold:
            yfdd_relmaxcln = np.delete(yfdd_relmaxcln, i)
        i=i-1

    # removes from the list the spurious maxima from spikes
    allowedtimedifference = 1.2
    sweeps = 1
    
    for j in range(sweeps): # do several swaeeps to remove iteratively
        # print(f"length of cleaned vector is {len(xfdd_relmaxcln)}")
        i = 0
        while i <= len(xfdd_relmaxcln)-2:
            if t[xfdd_relmaxcln[i+1]] - t[xfdd_relmaxcln[i]] < allowedtimedifference:
                if xfdd[xfdd_relmaxcln[i]] <= xfdd[xfdd_relmaxcln[i+1]]:    
                    xfdd_relmaxcln = np.delete(xfdd_relmaxcln, i)
                    i=i-1
                else:
                    xfdd_relmaxcln = np.delete(xfdd_relmaxcln, i+1)
                    i=i-1                    
            i=i+1
    
    for j in range(sweeps):
        # print(f"length of cleaned vector is {len(xfdd_relmaxcln)}")            
        i = 0
        while i <= len(yfdd_relmaxcln)-2:
            if t[yfdd_relmaxcln[i+1]] - t[yfdd_relmaxcln[i]] < allowedtimedifference:
                if yfdd[yfdd_relmaxcln[i]] <= yfdd[yfdd_relmaxcln[i+1]]:    
                    yfdd_relmaxcln = np.delete(yfdd_relmaxcln, i)
                    i=i-1
                else:
                    yfdd_relmaxcln = np.delete(yfdd_relmaxcln, i+1)
                    i=i-1                    
            i=i+1


    print("Next, find the upstrokes backtracking from maxima.")
    # searches backwards from the maxima for the threshold crossing
    xfdd_cross = np.array([], dtype=int)
    for maxindex in xfdd_relmaxcln:    
        j = 1
        while xfdd[maxindex-j] > 0:    
            j=j+1
        xfdd_cross = np.append(xfdd_cross, maxindex-j)
    
    yfdd_cross = np.array([], dtype=int)
    for maxindex in yfdd_relmaxcln:    
        j = 1
        while yfdd[maxindex-j] > 0:    
            j=j+1
        yfdd_cross = np.append(yfdd_cross, maxindex-j)

    return xfdd_relmax, yfdd_relmax, xfdd_relmaxcln, yfdd_relmaxcln, xfdd_cross, yfdd_cross
    



#%%
""" this function
computes the upstroke delays """
def upstrokedelay(t, xfdd_cross, yfdd_cross):

    upstroke_delay_gap_threshold = 0.8
    
    print("Compute the upstroke delays.")
    print("We define the delays as channel 1 minus channel 2 upstroke timings.\n")
    
    print(f"To filter out cycles that do not have a corresponding match in the other channel,\n\
    we neglect the meassurement when the difference exceeds a threshold of {upstroke_delay_gap_threshold} seconds.\n")
    
    upstroke_delays   = np.array([], dtype=float)
    upstroke_delays_t = np.array([], dtype=float)
    
    for crossindex in yfdd_cross:  
        delay = t[ xfdd_cross[ np.argmin( np.abs( xfdd_cross-crossindex ) ) ] ] - t[crossindex]
        if np.abs(delay) < upstroke_delay_gap_threshold:
            upstroke_delays   = np.append(upstroke_delays,   delay)
            upstroke_delays_t = np.append(upstroke_delays_t, t[crossindex])
            print(f"delay is {upstroke_delays[-1]:.4f} seconds at time {t[crossindex]:.1f}.")
        else:
            print(f"delay of {delay:.4f} seconds is above the {upstroke_delay_gap_threshold:.4f} threshold at time {t[crossindex]:.1f}, discard.")

    return upstroke_delays, upstroke_delays_t


#%%
""" this function
performs the cross correlation analysis """
def crosscorrelate(t, xf, yf, dt, approximate_cycle_duration):

    print("Next I run a cross correlation analysis for alternative estimation of sync delays.")

    # print("Here we fix the number of windows per slice.")    
    # per  = cycleduration         # period of one cycle in seconds - estimate
    # nwin = 5                     # number of slices for cross-correlation estimate
    # win  = (t[-1]-t[0]) / nwin   # approximate window duration
    # win  = int(win)
    # nper = win / per             # approximate number of cycles per window
    # print(f"Slicing the data in {nwin} windows of ~{win} seconds, with ~{nper} cycles.")
        
    print("Here we fix the number of cycles per slice to ensure similar statistics for all recordinngs.")
    nper = 30                                           # approximate number of cycles per window
    per  = approximate_cycle_duration                   # period of one cycle in seconds - estimate / from GLOBAL value
    nwin = int( np.round( (t[-1]-t[0]) / (nper*per) ) ) # number of slices for cross-correlation estimate
    win  = (t[-1]-t[0]) / nwin                          # approximate window duration
    win  = int(win)
    print(f"Slicing the data in {nwin} windows of ~{win} seconds, with ~{nper} cycles.")
        
    # variables for recording results
    delt = np.array([], dtype=float)
    dela = np.array([], dtype=float)
    valu = np.array([], dtype=float)
    
    for iwin in range(nwin):
        """ slice data with a non overlapping sliding window """
        print(f"Slicing the data iwin = {iwin} of {nwin-1}")
        t1 = iwin * win + t[0]
        t2 = (iwin+1) * win + t[0]
        if t1 > t[-1] or t1 < t[0]: t1 = t[0] 
        if t2 > t[-1]: t2 = t[-1]     
        print(f"t1 = {t1}")
        print(f"t2 = {t2}")
        tt1 = np.where(t==t1)[0][0]
        tt2 = np.where(t==t2)[0][0]
        
        """ slice filtered data """
        ts = t  [tt1:tt2]
        xs = xf [tt1:tt2]
        ys = yf [tt1:tt2]
    
        """ compute cross correlations """
        # cross correlation between small and large neuron
        corr = signal.correlate(xs, ys, mode='full', method='fft')
        
        lags = signal.correlation_lags(len(xs), len(ys), mode='full')
        lags = lags * dt
        zerolagindex = np.where(lags==0)[0][0]
        # this comprehension normalizes each value of corr according to the number of terms in its sum
        xcor = [ corr[i] / (1+i) if i < zerolagindex else corr[i] / ( len(corr) - i ) for i in range(0, len(corr)) ]
        
        # subtract baseline and normalize to variance :: connected auto correlation
        # xcor = ( xcor - np.mean(xs)**2 ) / np.var(xs) 
        # xcor = ( xcor - np.mean(ys[0])**2 ) / np.var(ys[0]) 
        xcor = ( xcor - np.mean(xs) * np.mean(ys) ) / np.cov(((xs,ys)))[0][1] 
            
        # find the crosscorrelation peaks
        relativemax, = signal.argrelmax (xcor, axis=0, order=10, mode='clip')
        # find the minimum time distance between zerolag and a correlation peak
        # delay = np.amin(np.abs(lags[relativemax]-lags[zerolagindex]))
        delay = lags[relativemax[np.argmin(np.abs(lags[relativemax]-lags[zerolagindex]))]]
        # identify the index of this peak
        delayindex = np.where(lags==delay)[0][0]
        print(f"Channel 2 is {delay:.4f} seconds advanced with correlation {xcor[delayindex]}.")
        # record the values in an array
        delt = np.append(delt, t1+0.5*(t2-t1))
        dela = np.append(dela, delay)
        valu = np.append(valu, xcor[delayindex])             # cross correlation value at first peak

    return delt, dela, valu






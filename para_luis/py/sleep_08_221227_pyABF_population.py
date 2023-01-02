# -*- coding: utf-8 -*-
"""

This is a temporary script file.

Parameters of the analysis are:
polydeg:    the degree of the polynomial used for direct fit detrending
upstroke_delay_gap_threshold 



"""

import numpy as np
from numpy.polynomial import polynomial as poly
import matplotlib.pyplot as plt
from scipy import signal
# from scipy import stats
import pyabf
import pandas as pd
import sleep_module_plot as plo
import sleep_module_analysis as ana



"""
GLOBAL paramters
"""

approximate_cycle_duration = 3



"""
MAIN code
"""

#%% List files
print("Define the path and data file names for import and analysis.")
path = '//home/user/Documents/Doctorado/Fly clock/para_luis/data/'
par_list = ['LL', 'SS', 'LS', 'LR', 'SR']
par_id_LL = ['01', '02', '03']
par_id_SS = ['01', '02', '03', '04']
par_id_LS = ['01', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '02']
par_id_LR = ['01', '02', '03', '04', '05', '06']
par_id_SR = ['01', '02', '03', '04', '05']

# choose the par type from par_list and par ids for analysis
# par = par_list[0] 
par = 'SR'
par_id = par_id_SR
par_id = ['04', '05']

print("Set the path for output files:")
pathout = '/Users/luis/Dropbox/Desktop Dropbox/coding/python/sleep/output/'
print(pathout)

#%% See if files exist
print(f'\nTest if files exist for selected {par} recordings.')
for i in range(len(par_id)):
    filename = par+par_id[i]+'.abf'
    print(path+filename)
    abf = pyabf.ABF(path+filename)
    # print(abf)
    del abf
print(f'All {len(par_id)} listed files exist, ready to go.')


#%% Display pair guide
print('\nOpen and display par guide to asign channel identities.')
par_guide = pd.read_excel(path+'par_guide.xlsx')
# print(par_guide)

print(f'List channels for selected {par} recordings.')
for i in range(len(par_id)):    
    recname = par+par_id[i]
    recid   = np.where(par_guide.name==recname)[0][0]
    channel1 = par_guide.ch1[recid]
    channel2 = par_guide.ch2[recid]    
    print(f'Recording {recname} contains {channel1} on channel 1 and {channel2} on channel 2.')


#%% Run analysis 
print('\nRun the full analysis for all selected recordings.')

for i in range(len(par_id)):    

    recname  = par+par_id[i]
    recid    = np.where(par_guide.name==recname)[0][0]
    channel1 = par_guide.ch1[recid]
    channel2 = par_guide.ch2[recid]    

    print(f"\nLoading the data for recording {recname}.")
    filename = recname+'.abf'
    abf = pyabf.ABF(path+filename)
    # print(abf)    
    
    
    #%% Load data and define parameters
    print(f'Define arrays with the data from {recname} recording.')
    """ 
    there is one sweep with two channels - see print(abf) """
    t = abf.sweepX
    abf.setSweep(sweepNumber=0, channel=0)
    x = abf.sweepY
    abf.setSweep(sweepNumber=0, channel=1)
    y = abf.sweepY
    """ 
    remove abf file once data was loaded into arrays """
    del abf

    
    """ 
    set t=0 at the begining of the recording """ 
    t = t - t[0]  

    """ 
    set parameters: total recording duration and sampling rate """
    N  = len(t)
    dt = t[1]-t[0]
    T  = N * dt
    print(f"Recording duration is {T:.4f} seconds with sampling interval of {dt:.5f} seconds, for a total of {N} datapoints.")
    

    """
    defines the time window for plotting timeseries data """
    cycleduration = approximate_cycle_duration # assumes 3 seconds per cycles
    cyclesperpanel = 10 # set the number of cycles per panel
    timewindow = cyclesperpanel * cycleduration
    numpanels = np.ceil(T / timewindow).astype(int)
    print(f"Assuming cycle duration of {cycleduration:.3f} seconds, split into {numpanels} panels of {timewindow:.3f} seconds, holding about {cyclesperpanel} cycles per panel.")    

    
    #%% ---
    """
    plot raw data """
    # plo.plotraw(t, x, y, numpanels, timewindow, pathout, recname)

    #%% Remove high frequencies (bursts)
    print("I filter data to remove high frequencies using a butterworth filter.")
    filter_order = 2
    filter_critical_frequencies = 10
    filter_type = 'lp'    
    sos = signal.butter(filter_order, filter_critical_frequencies, filter_type, fs=1/dt , output='sos')
    xf = signal.sosfilt(sos, x)
    yf = signal.sosfilt(sos, y)        

    # xf, yf = ana.filter(x, y, dt)

    # print("I filter data to remove high frequencies using savitky-golay.")
    # xf = signal.savgol_filter(x, 1001, 1, mode='mirror')
    # yf = signal.savgol_filter(y, 1001, 1, mode='mirror')
    
    
    #%% plot filtered data
    """
    plot filtered data """
    # plo.plotfil(t, x, y, xf, yf, numpanels, timewindow, pathout, recname)


    #%% Run cross-corr
    """
    run a cross correlation analysis for alternative estimation of sync delays """
    delt, dela, valu = ana.crosscorrelate(t, xf, yf, dt, approximate_cycle_duration)

    # plo.plotcroscorr (dela, delt, valu, t, pathout, recname)

    
    #%% Downsample data
    """ 
    slice downsample traces 
    take recordings from 0 to end and skip every downsampling_step """
    downsampling_step = 10
    tdw = t [0::downsampling_step]
    xdw = x [0::downsampling_step]
    ydw = y [0::downsampling_step]
    
    # plo.plotdow(t, x, y, tdw, xdw, ydw, numpanels, timewindow, pathout, recname)


    #%% Detrend signal
    polydeg = 5
    print(f"Fit a polynomial of degree {polydeg} to all data.")
    xf_coeff = poly.polyfit(t, xf, polydeg)
    yf_coeff = poly.polyfit(t, yf, polydeg)    
    print(f"Detrend data with direct polynomial fit of degree {polydeg} to all data.")
    xfdd = xf - poly.polyval(t, xf_coeff)
    yfdd = yf - poly.polyval(t, yf_coeff)
    
    # plo.plotdet(t, xfdd, yfdd, numpanels, timewindow, pathout, recname)

    #%% Delete unused variables
    """
    delete iddle variables """
    del x, y, xf, yf

    #%% Find peaks and upstokes
    """    
    find the peaks of cycles, 
    then find the upstrokes backtracking from maxima """

    xfdd_relmax, yfdd_relmax, xfdd_relmaxcln, yfdd_relmaxcln, xfdd_cross, yfdd_cross = ana.peakfind(t, xfdd, yfdd)    

    plo.plotupstrokex(t, xfdd, yfdd, xfdd_relmax, yfdd_relmax, xfdd_relmaxcln, yfdd_relmaxcln, xfdd_cross, yfdd_cross, numpanels, timewindow, pathout, recname)

    #%% Compute upstrok delays
    """    
    compute the upstroke delays """

    upstroke_delays, upstroke_delays_t = ana.upstrokedelay(t, xfdd_cross, yfdd_cross)

    plo.plotupstrokedeltime (t, xfdd, yfdd, xfdd_relmaxcln, yfdd_relmaxcln, xfdd_relmax, yfdd_relmax, 
                             xfdd_cross, yfdd_cross, upstroke_delays_t, upstroke_delays, numpanels, timewindow, pathout, recname)


    #%% Calculate histogram of upstroke delays
    print("Compute the histogram of upstroke delays.")
    upstroke_delays_hist, upstroke_delays_edges = np.histogram(upstroke_delays, bins=31, range=(-1.5,1.5), 
                                                               normed=None, weights=None, density=None) 

    plo.plothis (upstroke_delays_hist, upstroke_delays_edges, pathout, recname)
    
    print("Report of upstroke delays statistics.")
    print(f"Number of events compared is {len(upstroke_delays)}.")
    print(f"Mean of upstroke delays is {np.mean(upstroke_delays):.4f} seconds.")
    print(f"Standard deviation is {np.std(upstroke_delays):.4f} seconds.") 

    #%% Save upstroke data to disc
    '''
    saves upstroke_delays data for later processing and pooling histogram plotting '''
    datafilename = 'upstroke_delays_'+recname+'.dat'
    datfile = open(pathout+datafilename, "w+")
    np.savetxt(datfile, upstroke_delays, fmt='% .4f') # use [array] to save a horizontal text line
    datfile.close()

    '''
    saves stats of upstroke_delays in a common file for comparison '''
    datafilename = 'upstroke_delay_stats.dat'
    datfile  = open(pathout+datafilename, "a+")
    # creates a numpy array line that mixes data types
    dataline = np.zeros(1, dtype=[('var1', 'U6'), ('var2', 'U6'), ('var3', 'U6'), ('var4', float), ('var5', float), ('var6', int), ('var7', float)])
    dataline['var1'] = recname
    dataline['var2'] = channel1
    dataline['var3'] = channel2
    dataline['var4'] = np.mean(upstroke_delays)
    dataline['var5'] = np.std(upstroke_delays)
    dataline['var6'] = len(upstroke_delays)
    dataline['var7'] = T
    np.savetxt(datfile, dataline, fmt="%4s %1s %1s %.4f %.4f %d %.2f")
    datfile.close()



#%% Prints

    print("Mixed stats of delays from upstroke and crosscorrelations:\n")
    print(f"Mean of upstroke delays is {np.mean(upstroke_delays):.4f} seconds.")
    print(f"Standard deviation is {np.std(upstroke_delays):.4f} seconds.\n")
    print(f"Mean delays from crosscorr is {np.mean(dela):.4f} seconds.")
    print(f"Standard deviation is {np.std(dela):.4f} seconds.")
    
    




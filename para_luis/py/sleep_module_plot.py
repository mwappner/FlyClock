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
makes plots of raw timeseries data """
def plotraw(t, x, y, numpanels, timewindow, pathout, recname):
    print("Plot whole timeseries split in panels.")
    fig1 = plt.figure(1, figsize=(12,2.5*numpanels))
    plt.clf()    
    # onecol = 2.3    # science one column size
    # fig1 = plt.figure(num=55, figsize=(2*onecol, 5)) # , dpi=80, facecolor='w', edgecolor='k')
    plt.title('raw timeseries for both channels')
    
    for i in range(numpanels):
    
        ax1 = plt.subplot(2*numpanels,1,1+2*i)
        # ax1.set_aspect(0.05, 'box')
        ax1.plot(t,x,  lw=1, label="Channel 1")
        plt.xlim(i*timewindow,(i+1)*timewindow)    
        # Specify tick label size and set both ticks to be inside
        # ax1.tick_params(axis = 'both', which = 'major', labelsize = 12)
        # ax1.tick_params(axis = 'both', which = 'minor', labelsize = 0)            
        # ax1.tick_params(which = 'both', direction = 'in')
        plt.grid()    
        # plt.ylim(-1.1,-1.1)
        # ax1.set_title('1 neuron')
        ax1.set_xlabel('time (s)')
        
        ax2 = plt.subplot(2*numpanels,1,2+2*i)
        ax2.plot(t,y,  lw=1, label="Channel 2")
        plt.xlim(i*timewindow,(i+1)*timewindow)
        plt.grid()
        # plt.ylim(-1.1,-1.1)
        # ax2.set_title('2 neuron')
        ax2.set_xlabel('time (s)')
        # plt.legend(bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0.)
        # plt.legend(bbox_to_anchor=(1.002, 1), loc=2, borderaxespad=0.)
        # plt.legend()
        ax1.margins(0, 0.1)
        ax2.margins(0, 0.1)
        fig1.tight_layout()

    # plt.savefig(folder+'fhn_07_defect_detection_diagnosis.pdf',format='pdf')        
    outputfile = 'raw_'+recname+'.pdf'
    plt.savefig(pathout+outputfile)
    plt.close(1)
    del ax1, ax2
    # plt.show()
    return ()



#%%
""" this function
makes plots of filtered timeseries data """
def plotfil(t, x, y, xf, yf, numpanels, timewindow, pathout, recname):
    print("Plot filtered timeseries together with raw data.")
    fig3 = plt.figure(3, figsize=(12,2.5*numpanels))
    plt.clf()    
    plt.title('filtered timeseries for both channels')
    
    for i in range(numpanels):
    
        ax1 = plt.subplot(2*numpanels,1,1+2*i)
        # ax1.set_aspect(0.05, 'box')
        ax1.plot(t,x,   '-', c='peru',   lw=3, label="Channel 1")
        ax1.plot(t,xf,  '-', c='sienna', lw=1, label="Channel 1 filter")
        plt.xlim(i*timewindow,(i+1)*timewindow)
        # Specify tick label size and set both ticks to be inside
        # ax1.tick_params(axis = 'both', which = 'major', labelsize = 12)
        # ax1.tick_params(axis = 'both', which = 'minor', labelsize = 0)            
        # ax1.tick_params(which = 'both', direction = 'in')    
        plt.grid()    
        #plt.ylim(-1.1,-1.1)
        #ax1.set_title('1 neuron')
        ax1.set_xlabel('time (s)')
        
        ax2 = plt.subplot(2*numpanels,1,2+2*i)
        # ax2.set_aspect(0.05, 'box')
        ax2.plot(t,y,   '-', c='peru',   lw=3, label="Channel 2")
        ax2.plot(t,yf,  '-', c='sienna', lw=1, label="Channel 2 filter")
        plt.xlim(i*timewindow,(i+1)*timewindow)
        plt.grid()
        # plt.ylim(-1.1,-1.1)
        # ax2.set_title('2 neuron')
        ax2.set_xlabel('time (s)')
        
        # plt.legend(bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0.)
        # plt.legend(bbox_to_anchor=(1.002, 1), loc=2, borderaxespad=0.)
        # plt.legend()

        ax1.margins(0, 0.1)
        ax2.margins(0, 0.1)
        fig3.tight_layout()
        
    # plt.savefig(folder+'fhn_07_defect_detection_diagnosis.pdf',format='pdf')        
    outputfile = 'filtered_'+recname+'.pdf'
    plt.savefig(pathout+outputfile)

    plt.close(3)
    del ax1, ax2
    # plt.show()

#%%
""" this function
makes plots of downsampled timeseries data """
def plotdow(t, x, y, t1, x1, y1, numpanels, timewindow, pathout, recname):
    print("Plot downsampled timeseries together with raw data.")
    fig3 = plt.figure(3, figsize=(12,2.5*numpanels))
    plt.clf()    
    plt.title('filtered timeseries for both channels')
    
    for i in range(numpanels):
    
        ax1 = plt.subplot(2*numpanels,1,1+2*i)
        # ax1.set_aspect(0.05, 'box')
        ax1.plot(t,x,   '-', c='peru',   lw=3, label="Channel 1")
        ax1.plot(t1,x1,  '-', c='sienna', lw=1, label="Channel 1 filter")
        plt.xlim(i*timewindow,(i+1)*timewindow)
        # Specify tick label size and set both ticks to be inside
        # ax1.tick_params(axis = 'both', which = 'major', labelsize = 12)
        # ax1.tick_params(axis = 'both', which = 'minor', labelsize = 0)            
        # ax1.tick_params(which = 'both', direction = 'in')    
        plt.grid()    
        #plt.ylim(-1.1,-1.1)
        #ax1.set_title('1 neuron')
        ax1.set_xlabel('time (s)')
        
        ax2 = plt.subplot(2*numpanels,1,2+2*i)
        # ax2.set_aspect(0.05, 'box')
        ax2.plot(t,y,   '-', c='peru',   lw=3, label="Channel 2")
        ax2.plot(t1,y1,  '-', c='sienna', lw=1, label="Channel 2 filter")
        plt.xlim(i*timewindow,(i+1)*timewindow)
        plt.grid()
        # plt.ylim(-1.1,-1.1)
        # ax2.set_title('2 neuron')
        ax2.set_xlabel('time (s)')
        
        # plt.legend(bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0.)
        # plt.legend(bbox_to_anchor=(1.002, 1), loc=2, borderaxespad=0.)
        # plt.legend()

        ax1.margins(0, 0.1)
        ax2.margins(0, 0.1)
        fig3.tight_layout()
        
    # plt.savefig(folder+'fhn_07_defect_detection_diagnosis.pdf',format='pdf')        
    outputfile = 'downsampled_'+recname+'.pdf'
    plt.savefig(pathout+outputfile)
    plt.close(3)
    del ax1, ax2
    # plt.show()



#%%
""" this function
makes plots of detrended timeseries data 
both channels on same panel for comparison"""
def plotdet(t, xfdd, yfdd, numpanels, timewindow, pathout, recname):
    print("Plot detrended data.")
    fig7 = plt.figure(7, figsize=(12,2.5*numpanels))
    plt.clf()    
    plt.title('detrended timeseries for both channels')
    
    for i in range(numpanels):
    
        ax1 = plt.subplot(2*numpanels,1,1+2*i)
        # ax1.set_aspect(0.05, 'box')
        ax1.plot(t,xfdd,  '-', c='sienna',    lw=1, label="Channel 1 detrend")    
        ax1.plot(t,yfdd,  '-', c='steelblue', lw=1, label="Channel 2 detrend")
        plt.xlim(i*timewindow,(i+1)*timewindow)
        # Specify tick label size and set both ticks to be inside
        # ax1.tick_params(axis = 'both', which = 'major', labelsize = 12)
        # ax1.tick_params(axis = 'both', which = 'minor', labelsize = 0)            
        # ax1.tick_params(which = 'both', direction = 'in')    
        plt.grid(which='both')    
        #plt.ylim(-1.1,-1.1)
        #ax1.set_title('1 neuron')
        ax1.set_xlabel('time (s)')
        # plt.legend(bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0.)
        # plt.legend(bbox_to_anchor=(1.002, 1), loc=2, borderaxespad=0.)
        # plt.legend()
        ax1.margins(0, 0.1)
        # ax2.margins(0, 0.1)
        fig7.tight_layout()
        
    # plt.savefig(folder+'fhn_07_defect_detection_diagnosis.pdf',format='pdf')        
    outputfile = 'dirdetrended_'+recname+'.pdf'
    plt.savefig(pathout+outputfile)
    plt.close(7)
    del ax1
    # plt.show()


#%%

def plotupstrokex(t, xfdd, yfdd, 
                  xfdd_relmax, yfdd_relmax, xfdd_relmaxcln, yfdd_relmaxcln, xfdd_cross, yfdd_cross,
                  numpanels, timewindow, pathout, recname):

    print("Plot detrended data together with local maxima and upstroke crossings.")
    fig9 = plt.figure(9, figsize=(12,2.5*numpanels))
    plt.clf()    
    plt.title('maxima and upstroke crossings for both channels')
    
    for i in range(numpanels):
    
        ax1 = plt.subplot(2*numpanels,1,1+2*i)
        # ax1.set_aspect(0.05, 'box')
        ax1.plot(t,xfdd,  '-', c='sienna', lw=1, label="Channel 1 filter")
        # ax1.plot(t[xfdd_relmax],    xfdd[xfdd_relmax],    'o', c='peru',  markersize=6)
        ax1.plot(t[xfdd_relmaxcln], xfdd[xfdd_relmaxcln], 'o', c='black', markersize=3)
        ax1.plot(t[xfdd_cross], xfdd[xfdd_cross], 'o', c='red', markersize=3)
        plt.xticks(t[yfdd_cross])
        plt.xlim(i*timewindow,(i+1)*timewindow)
        # Specify tick label size and set both ticks to be inside
        # ax1.tick_params(axis = 'both', which = 'major', labelsize = 12)
        # ax1.tick_params(axis = 'both', which = 'minor', labelsize = 0)            
        # ax1.tick_params(which = 'both', direction = 'in')    
        plt.grid()    
        #plt.ylim(-1.1,-1.1)
        #ax1.set_title('1 neuron')
        ax1.set_xlabel('time (s)')
        
        ax2 = plt.subplot(2*numpanels,1,2+2*i)
        # ax2.set_aspect(0.05, 'box')
        ax2.plot(t,yfdd,  '-', c='sienna', lw=1, label="Channel 2 filter")
        # ax2.plot(t[yfdd_relmax],    yfdd[yfdd_relmax],    'o', c='peru',  markersize=6)
        ax2.plot(t[yfdd_relmaxcln], yfdd[yfdd_relmaxcln], 'o', c='black', markersize=3)
        ax2.plot(t[yfdd_cross], yfdd[yfdd_cross], 'o', c='red', markersize=3)
        plt.xticks(t[yfdd_cross])
        plt.xlim(i*timewindow,(i+1)*timewindow)
        plt.grid()
        # plt.ylim(-1.1,-1.1)
        # ax2.set_title('2 neuron')
        ax2.set_xlabel('time (s)')
        
        # plt.legend(bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0.)
        # plt.legend(bbox_to_anchor=(1.002, 1), loc=2, borderaxespad=0.)
        # plt.legend()

        ax1.margins(0, 0.1)
        ax2.margins(0, 0.1)
        fig9.tight_layout()
        
    # plt.savefig(folder+'fhn_07_defect_detection_diagnosis.pdf',format='pdf')        
    outputfile = 'crossings_'+recname+'.pdf'
    plt.savefig(pathout+outputfile)
    plt.close(9)
    del ax1, ax2    
    # plt.show()
    
    
#%%
def plotupstrokedeltime (t, xfdd, yfdd, xfdd_relmaxcln, yfdd_relmaxcln, xfdd_relmax, yfdd_relmax, 
                             xfdd_cross, yfdd_cross, upstroke_delays_t, upstroke_delays, numpanels, timewindow, pathout, recname):

    print("Plot upstroke delays as a function of time.")
    fig10 = plt.figure(10, figsize=(12,2.5*numpanels))
    plt.clf()    
    plt.title('upstroke delays as a function of time together with crossings for channel 1')
    
    for i in range(numpanels):
    
        ax1 = plt.subplot(2*numpanels,1,1+2*i)
        # ax1.set_aspect(0.05, 'box')
        ax1.plot(t,xfdd,  '-', c='sienna', lw=1, label="Channel 1 filter")
        ax1.plot(t[xfdd_relmaxcln], xfdd[xfdd_relmaxcln], 'o', c='black', markersize=3)
        ax1.plot(t[xfdd_cross], xfdd[xfdd_cross], 'o', c='red', markersize=3)
        plt.xticks(t[yfdd_cross])
        plt.xlim(i*timewindow,(i+1)*timewindow)
        # Specify tick label size and set both ticks to be inside
        # ax1.tick_params(axis = 'both', which = 'major', labelsize = 12)
        # ax1.tick_params(axis = 'both', which = 'minor', labelsize = 0)            
        # ax1.tick_params(which = 'both', direction = 'in')    
        plt.grid()    
        #plt.ylim(-1.1,-1.1)
        #ax1.set_title('1 neuron')
        ax1.set_xlabel('time (s)')
        
        ax2 = plt.subplot(2*numpanels,1,2+2*i)
        # ax2.set_aspect(0.05, 'box')
        ax2.plot(upstroke_delays_t, upstroke_delays, 'o', c='peru', markersize=5)    
        plt.xticks(upstroke_delays_t)
        plt.xlim(i*timewindow,(i+1)*timewindow)
        plt.ylim(-0.5,0.5)
        plt.grid()
        # plt.ylim(-1.1,-1.1)
        # ax2.set_title('2 neuron')
        ax2.set_xlabel('time (s)')
        # plt.legend(bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0.)
        # plt.legend(bbox_to_anchor=(1.002, 1), loc=2, borderaxespad=0.)
        # plt.legend()

        ax1.margins(0, 0.1)
        ax2.margins(0, 0.1)
        fig10.tight_layout()
        
    # plt.savefig(folder+'fhn_07_defect_detection_diagnosis.pdf',format='pdf')        
    outputfile = 'delays_time_'+recname+'.pdf'
    plt.savefig(pathout+outputfile)

    plt.close(10)
    del ax1, ax2
    # plt.show()


#%%
def plothis (upstroke_delays_hist, upstroke_delays_edges, pathout, recname):

    # obtain the coordinates of the histogram bar centers for the bar plot
    upstroke_delays_coord = (upstroke_delays_edges[1:] + upstroke_delays_edges[:-1]) / 2

    print("Plot histogram of upstroke delays.")
    fig11 = plt.figure(11, figsize=(12,6))
    plt.clf()    
    plt.title('histogram of upstroke delays')
    ax1 = plt.subplot(1,1,1)
    
    ax1.bar(upstroke_delays_coord, upstroke_delays_hist, width=0.9*np.diff(upstroke_delays_edges), edgecolor="none", align="center")

    # ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
    #        ylim=(0, 8), yticks=np.arange(1, 8))
    # plt.xticks(upstroke_delays_hist[1])
    # plt.xlim(upstroke_delays_hist[1][0],upstroke_delays_hist[1][-1])
    plt.xlim(-1.25,1.25)
    ax1.set_xlabel('time (s)')
        
    ax1.margins(0, 0.1)
    fig11.tight_layout()

    outputfile = 'delays_histogram_'+recname+'.pdf'
    plt.savefig(pathout+outputfile)
    plt.close(11)
    del ax1
    # plt.show()


#%%
def plotcroscorr (dela, delt, valu, t, pathout, recname):

    """ make plot of sliding cross correlation lags """
    print("Plot cross correlation lags ad values.")
    fig12 = plt.figure(12, figsize=(12,4))
    plt.clf()    
    plt.title('cross correlation lags ad values')
    
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(delt, 1000*dela, 'o')
    plt.xlim(t[0],t[-1])
    plt.ylim(-500,500)
    #plt.ylim(-0.5,0.5)
    plt.grid()
    ax1.set_title('lags')
    ax1.set_xlabel('time (s)')
    ax1.margins(0, 0.1)
    
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(delt, valu, 'o')
    plt.xlim(t[0],t[-1])
    plt.ylim(-1.2,1.2)
    plt.grid()
    ax2.set_title('values')
    ax2.set_xlabel('time (s)')
    ax2.margins(0, 0.1)
    
    ax1.margins(0, 0.1)
    ax2.margins(0, 0.1)
    fig12.tight_layout()
    
    outputfile = 'delays_xcor_'+recname+'.pdf'
    plt.savefig(pathout+outputfile)

    plt.close(12)
    # plt.show()





    
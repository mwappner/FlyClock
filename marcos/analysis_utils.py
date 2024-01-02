#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 15:35:04 2023

@author: marcos
"""

# from pathlib import Path
import re
import itertools, numbers

import numpy as np
from scipy import signal, interpolate, stats
from scipy.ndimage import gaussian_filter1d
from skimage import filters
import pandas as pd

import pyabf
import pyabf.filter

from utils import sort_by, enzip, calc_mode

#%% Classes

@pd.api.extensions.register_dataframe_accessor("metadata")
class RunInfo:
    """ This should add a few properties to the pandas.DataFrame """

    def __init__(self, pandas_obj):
        self.file = None
        self._guide = None
            
    @property
    def guide(self):
        return self._guide 
    @guide.setter
    def guide(self, guide_info):
        self._guide = guide_info
        self._ch1 = guide_info['ch1']
        self._ch2 = guide_info['ch2']
        self._pair = guide_info['par']
        self._raw_sampling_rate = guide_info['samplerate']
        self._duration_min = guide_info['duration(min)'] # this is the duration in minutes, encoded like this because namedtuples
        self._twochannel = True
        self.interval = None
        
        if 'mec_start' in guide_info:
            self._mec_start_sec = guide_info['mec_start(sec)']
        if 'comment' in guide_info:
            self._comment = guide_info['comment']
        
    @property
    def ch1(self):
        return self._ch1
    @property
    def ch2(self):
        return self._ch2
    @property
    def pair(self):
        return self._pair
    @property
    def duration_min(self):
        return self._duration_min
    @property
    def raw_sampling_rate(self):
        return self._raw_sampling_rate
    @property
    def mec_start_sec(self):
        return self._mec_start_sec
    @property
    def comment(self):
        return self._comment
    @property
    def twochannel(self):
        return self._twochannel


@pd.api.extensions.register_dataframe_accessor("process")
class Processors:
    """ Makes the process functions methods of the dataframe. All processors 
    will add an entry into self.process.info stating what they did and 
    what parameters were used. 
    All processors have an optional argument to keep the original values or
    override them. If kept, a new column will be added, appending the processing
    type to the column name.
    All processors have an optional argument to be applied to a given column, 
    rather than the default ch1 and ch2. To select a different column, insert
    into 'columns' the name after ch1_ or ch2_. For example, 'det' will target
    channels 'ch1_det' and 'ch2_det'.
    """
    
    _all_info_attributes = '_info', '_trends', '_peaks'
    
    def __init__(self, pandas_obj):
        # save the dataframe as an object to later access
        self._df = pandas_obj
        
        self._init_info_attributes()
    
    def _init_info_attributes(self):
        """Initialize the values of the info attributes as None"""
        for attr in self._all_info_attributes:
            setattr(self, attr, None)
            
    @property
    def info(self):
        return self._info
    @info.setter
    def info(self, value):
        if self.info is None:
            self._info = (value, )
        else:
            self._info = (*self._info, value)
    @property
    def steps(self):
        return tuple( step['step'] for step in self.info )
    
    @property
    def trends(self):
        return self._trends
    @trends.setter
    def trends(self, trend_pair):
        if self.trends is not None:
            print('A detrending job was already done. Overriding the previous one.')
        self._trends = trend_pair
    def get_trend_poly(self, inx):
        assert inx in (1,2), 'inx must be either 1 or 2'
        assert 'pdetrend' in self.steps, 'You must run a detrending job first'
        return self.trends[inx-1]
    def get_trend(self, inx):
        assert inx in (1,2), 'inx must be either 1 or 2'
        assert 'pdetrend' in self.steps, 'You must run a detrending job first'
        return self.trends[inx-1](self._df.times)
    
    def get_hptrend(self, inx):
        assert inx in (1,2), 'inx must be either 1 or 2'
        assert 'hpfilt' in self.steps, 'You must run a detrending job first'
        
        # check if data was kept after the filter
        step_info = self.get_step_info('hpfilt')
        if not step_info['keep_og']:
            raise ValueError("The data was not kept after filtering, so we can't calculate the trends")
        
        # calculate the trend as the difference between before and after applying the filter
        applied_on = step_info['channels']
        prefilt = self._get_channels(applied_on)[inx-1] # channel data on which the filter was applied
        posfilt = self._get_channels(applied_on+'_hpfilt')[inx-1] # channel data of the applied filter
        return prefilt-posfilt
    
    def get_phase_difference(self, channels=None):
        """ Get the mean phase difference for the given channel, if it has been
        calculated. If channels=None, get the last one."""
        # Check if a phase difference was calculated at all. If so, set channels
        # to the appropiate value (last difference calculated).
        if channels is None:
            if 'phase_diff' not in self.steps:
                raise ValueError('Phase difference has not been calculated yet')
                
            channels = self.get_step_info('phase_diff')['channels']
        
        # Check if phase diff was calculated for the required channel. This is 
        # trivial if channels=None
        if not self._check_if_action_was_performed_on_channel('phase_diff', channels):
            raise ValueError(f'Phase difference has not been calculated for channel {channels} yet')
        
        K_name = (channels + '_K').strip('_') 
        return np.nanmean(self._df[ K_name ])
    
    @property
    def peaks(self):
        return self._peaks
    @peaks.setter
    def peaks(self, peak_pair):
        if self.peaks is not None:
            print("You've already calculated peaks before. Overriding the previous one.")
        self._peaks = peak_pair
   
    def get_peak_pos(self, inx):
        assert inx in (1,2), 'inx must be either 1 or 2'
        assert 'findpeaks' in self.steps, 'You must find peaks first'
        return self._df.times.values[self.peaks[inx-1]]
   
    def get_peak_values(self, inx, channels=None):
        """ Returns the value of the series given in channel at the peaks that 
        have been calculated. Note you can query the value of a channel for 
        which the peaks have not been calculated, in which case the function 
        will warn and continue working. in this context, channel refers to the 
        name of the channel that reflects the operations that have been done on 
        it, and not the channel index, which is 'inx'. """
        
        assert inx in (1,2), 'inx must be either 1 or 2'
        assert 'findpeaks' in self.steps, 'You must find peaks first'
        
        if channels is not None and not self._check_if_action_was_performed_on_channel('findepakes', channels):
            print('WARNING: findpeaks was not performed on this channel')
        
        if channels is None: # defalt to calculating peaks on the channel where findpeaks was done
            channels = self.get_step_info('findpeaks')['channels']
        ch_pair = self._get_channels(channels)
        return ch_pair[inx-1].values[self.peaks[inx-1]]
    
    def get_avg_period(self, inx=None):
        """ Calculate the average (mean) period for channel 1 or 2 if inx=1 or
        inx=1, or the average across channels if inx=None"""
        
        assert inx in (1,2, None), 'inx must be either 1 or 2, or None'
        
        if inx is None:
            ch1_period = np.mean( np.diff(self.get_peak_pos(1)))
            ch2_period = np.mean( np.diff(self.get_peak_pos(2)))
            return np.mean((ch1_period, ch2_period))
        else:
            return np.mean( np.diff(self.get_peak_pos(inx)))
        
    def get_periods(self, inx):

        peak_pos = self.get_peak_pos(inx)
        periods = np.diff(peak_pos)
        period_times = peak_pos[:-1] + np.diff(peak_pos ) / 2
        
        return period_times, periods
        
    def get_instant_period(self, inx, outlier_mode_proportion=1.8, **hist_kwargs):
        """ Calculate a trend for the value of the period of the recording as
        time passes. Return a linear fit to all the period values calculated"""
    
        # find the mode of the periods        
        period_times, periods = self.get_periods(inx)
        # counts, bins = np.histogram(periods, **hist_kwargs)
        # bin_centers = bins[:-1] + np.diff(bins) / 2
        # period_mode = bin_centers[np.argmax(counts)]
        period_mode = calc_mode(periods)    
    
        #use the mode to select period values that are not too far from it    
        valid_period_inxs = periods < outlier_mode_proportion*period_mode
        valid_periods = periods[valid_period_inxs]
        
        #calculate the trend and return that as an instant period value
        P = np.polynomial.Polynomial
        trend_poly = P.fit(period_times[valid_period_inxs], valid_periods, 1)
        
        return trend_poly(self._df.times) 
    
    def get_crossings(self, inx, edge, threshold=5, peak_min_distance=0.5, channels=None):
        """
        Calculate the point at which the signal crosses the threshold at each
        rising edge or each falling edge, depending on the value of 'edge'.

        Parameters
        ----------
        inx : int
            1 or 2, the channel for which periods are being calculated.
        edge : str
            either 'falling' or 'rising'. The kind of edge for which the period
            is calculated. If kind='rising', the function will look for a rising
            edge crossing before the peak. If kind='falling', the crossing will
            happen after the peak.
        threshold : float, optional
            Value of the threshold at which the crossing is calculated. 
            Additionally, if a peak value falls under the threshold, that peak
            is ignored. The default is 5, which assumes the data has been 
            detrended.
        peak_min_distance : floar, optional
            The minimum distance between peaks, in units of average distance
            between peaks. After one peak is used, any subsequent peaks closer
            than this value will be ignored. The default is 0.5.

        Returns
        -------
        crossings : numpy array
            Array containing indexes at which the requested crossings happen
        """
        
        
        assert edge in ('rising', 'falling'), 'edge must be one of "rising" or "falling"'
        assert inx in (1,2), 'inx must be either 1 or 2'
        assert 'findpeaks' in self.steps, 'You must find peaks first'
        
        # check if findpeaks was done on the requested channel (for example, gauss_filt)
        # if not, warn and continue
        if channels is not None and not self._check_if_action_was_performed_on_channel('findepakes', channels):
            print('WARNING: findpeaks was not performed on this channel')
        
        # default to using the channel on which findpeaks was performed
        if channels is None: 
            channels = self.get_step_info('findpeaks')['channels']
        ch_pair = self._get_channels(channels)
        data = ch_pair[inx-1].values
        
        # retrieve peak data
        peaks = self.peaks[inx-1]
        mean_period_in_points = round(np.mean(np.diff(peaks)))
        
        crossings = []
        prev_peak = -np.inf # to prime the first peak
        
        # iterate over all peaks
        for peak in peaks:
            
            # skip maxima that are too low
            if data[peak] < threshold:
                continue
             
             # skip maxima that are too close together
            if peak - prev_peak < mean_period_in_points * peak_min_distance:
                continue
            
            if edge == 'rising':
                # find rising edge point (before peak)
                interval = data[:peak]
                try:
                    cross = np.nonzero(interval < threshold)[0][-1]
                except IndexError: 
                # this raises when we have the first peak and don't cross the threshold before the start of the signal
                    cross = 0
                                   
            else:
                # find falling edge point (after peak)    
                starting_point = min(peak + int(mean_period_in_points * peak_min_distance), len(data))
                interval = data[:starting_point]
                cross = np.nonzero(interval > threshold)[0][-1]
            
            crossings.append(cross)
            prev_peak = peak
        
        if not crossings: # no peaks were found over the threshold
            raise ValueError(f"It's likely no peaks were found over the given threshold value of {threshold}. Or maybe something else is wrong. Are you sure you have peaks?")

        crossings = np.asarray(crossings)
        return crossings
    
    
    def get_multi_crossings(self, inx, edge, threshold=5, threshold_var=3, peak_min_distance=0.5, channels=None):
        """
        Calculate the point at which the signal crosses the threshold at each
        rising edge or each falling edge, depending on the value of 'edge'.

        Parameters
        ----------
        inx : int
            1 or 2, the channel for which periods are being calculated.
        edge : str
            either 'falling' or 'rising'. The kind of edge for which the period
            is calculated. If kind='rising', the function will look for a rising
            edge crossing before the peak. If kind='falling', the crossing will
            happen after the peak.
        threshold : float, optional
            Value of the threshold at which the crossing is calculated. 
            Additionally, if a peak value falls under the threshold, that peak
            is ignored. The default is 5, which assumes the data has been 
            detrended.
        threshold_var : floar, optional
            Value by which the threshold will be upwards and downwards to find 
            further corssings. The default is 3.
        peak_min_distance : floar, optional
            The minimum distance between peaks, in units of average distance
            between peaks. After one peak is used, any subsequent peaks closer
            than this value will be ignored. The default is 0.5.

        Returns
        -------
        crossings : numpy array
            Array containing indexes at which the requested crossings happen
        """
        
        
        assert edge in ('rising', 'falling'), 'edge must be one of "rising" or "falling"'
        assert inx in (1,2), 'inx must be either 1 or 2'
        assert 'findpeaks' in self.steps, 'You must find peaks first'
        
        # check if findpeaks was done on the requested channel (for example, gauss_filt)
        # if not, warn and continue
        if channels is not None and not self._check_if_action_was_performed_on_channel('findepakes', channels):
            print('WARNING: findpeaks was not performed on this channel')
        
        # default to using the channel on which findpeaks was performed
        if channels is None: 
            channels = self.get_step_info('findpeaks')['channels']
        ch_pair = self._get_channels(channels)
        data = ch_pair[inx-1].values
        
        # retrieve peak data
        peaks = self.peaks[inx-1]
        mean_period_in_points = round(np.mean(np.diff(peaks)))
        
        crossings = []
        other_crossings = []
        prev_peak = -np.inf # to prime the first peak
        multiple_thresholds = np.linspace(0, threshold_var, 11)[1:]
        # iterate over all peaks
        for peak in peaks:
            
            # skip maxima that are too low
            if data[peak] < threshold:
                continue
             
             # skip maxima that are too close together
            if peak - prev_peak < mean_period_in_points * peak_min_distance:
                continue
            
            this_other_crossings = []
            if edge == 'rising':
                # find rising edge point (before peak)
                interval = data[:peak]
                try:
                    cross = np.nonzero(interval < threshold)[0][-1]
                        
                except IndexError: 
                # this raises when we have the first peak and don't cross the threshold before the start of the signal
                    cross = 0

                # find corossings around this crossing                
                if cross == 0:
                    # if we are handling the first one, skip searching
                    other_crossings.append(np.full((multiple_thresholds.size*2+1,), np.nan))
                else:
    
                    # find individual crossings
                    # first find crossings below
                    start = max(cross - int(mean_period_in_points * peak_min_distance), 0)
                    short_interval = data[start:cross]
                    for th in multiple_thresholds[::-1]:
                        th = threshold - th
                        
                        other_cross_array = np.nonzero(short_interval < th)[0]
                        if other_cross_array.size == 0:
                            other_cross = np.nan
                        else:
                            # last point bellow threshold
                            other_cross = other_cross_array[-1]
                            other_cross += start #redefine the 0th inxdex
                                                    
                        this_other_crossings.append(other_cross)
                    
                    # append the crossing at the middle
                    this_other_crossings.append(cross)
                    
                    # and now crossings above
                    short_interval = data[cross:peak]
                    for th in multiple_thresholds:    
                        th += threshold
                        
                        other_cross_array = np.nonzero(short_interval > th)[0]
                        if other_cross_array.size == 0:
                            other_cross = np.nan
                        else:
                            # first point above threshold
                            other_cross = other_cross_array[0]
                            other_cross += cross #redefine the 0th inxdex
                        
                        this_other_crossings.append(other_cross)
                    
                    other_crossings.append(np.asarray(this_other_crossings))
                        
            else:
                raise NotImplementedError('There is no multi crossing detection for falling edge yet')
                # find falling edge point (after peak)    
                starting_point = min(peak + int(mean_period_in_points * peak_min_distance), len(data))
                interval = data[:starting_point]
                cross = np.nonzero(interval > threshold)[0][-1]
            
            crossings.append(cross)
            prev_peak = peak
        
        if not crossings: # no peaks were found over the threshold
            raise ValueError(f"It's likely no peaks were found over the given threshold value of {threshold}. Or maybe something else is wrong. Are you sure you have peaks?")

        crossings = np.asarray(crossings)
        other_crossings = np.asarray(other_crossings)
        return crossings, other_crossings
    
    
    def get_edge_periods(self, inx, edge, threshold=5, peak_min_distance=0.5):
        """
        Calculate the period as the distance between points at which consecutive
        cycles cross the threshold. The crossing must be either a rising or 
        falling edge, depending on "kind". To perform this action, the user must
        already have performed a findpeaks actions. The crossings are calculated
        with respect to peaks of the signal. Peaks that are under the threshold
        or are too close together are ignored.

        Parameters
        ----------
        inx : int
            1 or 2, the channel for which periods are being calculated.
        edge : str
            either 'falling' or 'rising'. The kind of edge for which the period
            is calculated. If kind='rising', the function will look for a rising
            edge crossing before the peak. If kind='falling', the crossing will
            happen after the peak.
        threshold : float, optional
            Value of the threshold at which the crossing is calculated. 
            Additionally, if a peak value falls under the threshold, that peak
            is ignored. The default is 5, which assumes the data has been 
            detrended.
        peak_min_distance : float, optional
            The minimum distance between peaks, in units of average distance
            between peaks. After one peak is used, any subsequent peaks closer
            than this value will be ignored. The default is 0.5.

        Returns
        -------
        period_times, periods : tuple
            Tuple containg two arrays: time at which the period is calculated (at
            which the crossing happens) and value of the interval between two 
            consecutive crossings.
        
        """
        data = self._df
        crossings = self.get_crossings(inx, edge, threshold, peak_min_distance)
        
        crossing_times = data.times.values[crossings]
        period_times = crossing_times[:-1] + np.diff(crossing_times ) / 2
        periods = np.diff(crossing_times)
        
        return period_times, periods 


    def get_multi_edge_periods(self, inx, edge, threshold=5, threshold_var=3, peak_min_distance=0.5):
        """
        Calculate the period as the distance between points at which consecutive
        cycles cross the threshold. The crossing must be either a rising or 
        falling edge, depending on "kind". To perform this action, the user must
        already have performed a findpeaks actions. The crossings are calculated
        with respect to peaks of the signal. Peaks that are under the threshold
        or are too close together are ignored.

        Parameters
        ----------
        inx : int
            1 or 2, the channel for which periods are being calculated.
        edge : str
            either 'falling' or 'rising'. The kind of edge for which the period
            is calculated. If kind='rising', the function will look for a rising
            edge crossing before the peak. If kind='falling', the crossing will
            happen after the peak.
        threshold : float, optional
            Value of the threshold at which the crossing is calculated. 
            Additionally, if a peak value falls under the threshold, that peak
            is ignored. The default is 5, which assumes the data has been 
            detrended.
        peak_min_distance : float, optional
            The minimum distance between peaks, in units of average distance
            between peaks. After one peak is used, any subsequent peaks closer
            than this value will be ignored. The default is 0.5.

        Returns
        -------
        period_times, periods : tuple
            Tuple containg two arrays: time at which the period is calculated (at
            which the crossing happens) and value of the interval between two 
            consecutive crossings.
        
        """
        data = self._df
        crossings, multi_crossings = self.get_multi_crossings(inx, edge, threshold, threshold_var, peak_min_distance)
        
        # handle the nans like so:
        temp = data.times.values.copy()
        # add an inf to the end of the array
        temp = np.append(temp, np.nan)
        # replace the nans in crossings with -1 to reference the added nan
        multi_crossings[np.isnan(multi_crossings)] = -1
        multi_crossings = multi_crossings.astype(int)
        
        # the time point in the middle of the oscillation
        crossing_times = temp[crossings]
        period_times = crossing_times[:-1] + np.diff(crossing_times ) / 2
        
        # get array of periods
        multi_crossing_times = temp[multi_crossings]
        all_periods = np.diff(multi_crossing_times, axis=0)

        # filter out rows where we have all nans, to avoid empty slices
        all_nans = np.all(np.isnan(all_periods), axis=1)
        all_periods = all_periods[~all_nans]
        period_times = period_times[~all_nans]

        # the average, std and ptp of each period
        periods = np.nanmean(all_periods, axis=1)
        period_err = np.nanstd(all_periods, axis=1)
        period_ptp = (np.nanmax(all_periods, axis=1) - np.nanmin(all_periods, axis=1)) / 2
        
        return period_times, periods, period_err, period_ptp

   
    def downsample(self, downsampling_rate=10):
        """
        Downsamples the data by skipping datapoints. Returns a new dataframe.

        Parameters
        ----------
        downsampling_rate : int, optional
            How many points to skip when downsampling. The default is 10.

        Returns
        -------
        New DataFrame with downsampled data

        """
        manually_copy_attrs = 'rec_datetime' , '_twochannel', 'interval'
        downsampled = self._df[::downsampling_rate].copy()
        
        # make sure to keep metadata and processing steps info
        add_run_info(downsampled, self._df.metadata.file)
        
        for attr in manually_copy_attrs:
            setattr(downsampled.metadata, attr, getattr(self._df.metadata, attr))
        for attr in self._all_info_attributes:
            setattr(downsampled.process, attr, getattr(self, attr))
        
        # record what we did and update the sampling rate
        downsampled.process._add_process_entry('downsampling', downsampling_rate=downsampling_rate)
        downsampled.metadata.sampling_rate = self._df.metadata.sampling_rate / downsampling_rate
        downsampled.reset_index(drop=True, inplace=True)
        
        return downsampled
    
    
    def poly_detrend(self, degree=5, keep_og=False, channels=''):
        """
        Use a polynomial to detrend the data.

        Parameters
        ----------
        degree : int, optional
            Degree of the polynomial used to fit the data. Default is 5.
        keep_og : Bool, optional
            Whether to keep the original column or overwride it. The default is
            False.
        channels : str, optional
            A string describing what channel to apply the funciton on. The 
            default is ''.

        Returns
        -------
        None.

        """
        
        action_name = 'pdetrend'
        data = self._df
        
        t = data.times
        ch1, ch2 = self._get_channels(channels)
        P = np.polynomial.Polynomial
        
        # some filter methods leave behind nans in the data, that raises a LinAlgError
        nan_locs = np.isnan(ch1)
        
        # fit the data
        ch1_trend_poly = P.fit(t[~nan_locs], ch1[~nan_locs], degree)
        ch2_trend_poly = P.fit(t[~nan_locs], ch2[~nan_locs], degree)
        
        # remove the trend from the data, this reintroduces the nans
        y1dtr = ch1 - ch1_trend_poly(t)
        y2dtr = ch2 - ch2_trend_poly(t)
        
        # save processed data
        self._save_processed_data(y1dtr, y2dtr, keep_og, channels, action_name)
        self._add_process_entry(action_name, degree=degree, keep_og=keep_og, channels=channels)
        self.trends = ch1_trend_poly, ch2_trend_poly
        
    def magnitude_detrend(self, keep_og=False, channels=''):
        """
        Use the magnitude calculated through a Hilbert transform to "detrend" a
        signal by dividing it by its magnitude (envelope).

        Parameters
        ----------
        keep_og : Bool, optional
            Whether to keep the original column or overwride it. The default is
            False.
        channels : str, optional
            The channel to apply the detrend to. You need to have calculated 
            the phase and magnitude for the target channel already. If not, 
            this function will do it. Note the limitations of this calculation
            in calc-magnitude_and_phases. The default is ''.

        Returns
        -------
        None.

        """
        
        action_name = 'mdetrend'
        
        if not self._check_if_action_was_performed_on_channel('hilbert', channels):
            self.calc_magnitudes_and_phases(channels)
        
        ch1, ch2 = self._get_channels(channels)
        ch1_mag, ch2_mag = self._get_channels(channels + '_magnitude')
        
        ch1_dtr = ch1 / ch1_mag
        ch2_dtr = ch2 / ch2_mag
        
        self._save_processed_data(ch1_dtr, ch2_dtr, keep_og, channels, action_name)
        self._add_process_entry(action_name, keep_og=keep_og, channels=channels)
        
    def average_detrend(self, outlier_mode_proportion=1.8, keep_og=False, channels=''):
        """
        Calculates a moving average of the signal and uses that to detrend it.
        The moving average is calculated over a window of varying size, in
        particular of size given by the length of a period in the neighborhood
        of the point being evaluated. The length of the period is extracted from
        a linear interpolation of period values from a find_peaks operation. If
        no find_peaks operation was done to to the requeted channel, one is 
        performed.

        Parameters
        ----------
        outlier_mode_proportion : float, optional
            After calculating the periods, the mode of the periods is computed.
            Only the periods that fall within outlier_mode_proportion times the
            value of the mode will be used, the rest will be just discarded. 
            The default is 1.8.
        keep_og : Bool, optional
            Whether to keep the original column or overwride it. The default is
            False.
        channels : str, optional
            The channel to apply the detrend to. You need to have ran a find 
            peals operation on the target channel already. If not, this 
            function will do it. Note the limitations of this calculation in 
            calc-magnitude_and_phases. The default is ''.

        Returns
        -------
        None.

        """        
        
        action_name = 'adetrend'
        data = self._df
        
        if not self._check_if_action_was_performed_on_channel('findpeaks', channels):
            self.find_peaks(channels=channels)        
            
        ch_data = self._get_channels(channels)

        trendlines = []
        for ch in [1,2]:        
            period_times, periods = self.get_periods(ch)
    
            # calculate the mode
            counts, bins = np.histogram(periods)
            bin_centers = bins[:-1] + np.diff(bins) / 2
            period_mode = bin_centers[np.argmax(counts)]
    
            # find periods that fall close to the mode to exclude points where we skipped a pulse
            valid_period_inxs = periods < outlier_mode_proportion * period_mode
            continuos_periods = np.interp(data.times, period_times[valid_period_inxs], periods[valid_period_inxs])
            continuos_periods *= data.metadata.sampling_rate
    
            # calculate the trendline
            nanlocs = np.isnan(ch_data[ch-1])
            trendline = self.varying_size_rolling_average(ch_data[ch-1].values[~nanlocs], continuos_periods[~nanlocs], samesize=True)
            trendlines.append(trendline)
                
        # save trends
        self._save_processed_data(*trendlines, keep_og=True, channels=channels, action='average')
        
        #detrend data
        trendlines = self._get_channels(channels + '_average')
        detrended = [ch - tr for ch, tr in zip(ch_data, trendlines)]
        
        #save detrended data
        self._save_processed_data(*detrended, keep_og, channels, action_name)
        self._add_process_entry(action_name, outlier_mode_proportion=outlier_mode_proportion, keep_og=keep_og, channels=channels)
        
    @staticmethod
    def varying_size_rolling_average(x, sizes, samesize=False, truncate_ends=True):
        """Calculate the rolling average (or 'moving mean') of the given data with
        a window that has a different size in each position. Sizes and x need to be
        the same size, and both need to be 1d arrays. If samesize is True, the output
        array will be prepended and apended with values to match the input size. If
        truncate_ends=True, the first few and las entries that don't fit in a window
        will be truncated."""
        
        assert len(x.shape)==1 and len(sizes.shape)==1, 'x and sizes but be 1D'
        assert x.size == sizes.size, "Sizes of x and sizes must be the same"
        
        # Calculate the halfsizes of the windows, having a bias towards the lower half
        bots = np.floor(sizes/2).astype(int)
        tops = np.ceil(sizes/2).astype(int)
        c = np.cumsum(x)
        
        ret = []
        first = bots[0] if truncate_ends else 0
        for i, (w, bot, top) in enzip(sizes[first:], bots[first:], tops[first:]):
            
            j = i + first
            # if I'm reaching too far to the left, truncate
            if j < bot:
                bot = j
                w = bot + top
            # if I'm reaching too far to the right, stop
            if j + top >= x.size:
                if truncate_ends:
                    break
                else:
                    top = x.size - 1 - j
                    w = bot + top
            
            ret.append( (c[j+top] - c[j-bot])/w )
        
        if samesize and x.size != len(ret):
            ret_array = np.ones_like(x) * ret[-1]
            ret_array[:first] = ret[0]
            ret_array[first:first+len(ret)] = ret
        else:
            ret_array = np.asarray(ret)
            
        return ret_array
        
        
    def abf_gauss_filt(self, abf, sigma=20, keep_og=False, channels=''):
        """
        Fitlers data using a gaussian filter. Since this uses the gaussian 
        filter built into pyabf, it requires the abf object, so it can only be
        done upon initialization.

        Parameters
        ----------
        abf : pyabf.abf
            abf object.
        sigma : float, optional
            Sigma for the gaussian filter in milliseconds. The default is 20.
        keep_og : Bool, optional
            Whether to keep the original column or overwride it. The default is
            False.
        channels : str, optional
            A string describing what channel to apply the funciton on. The 
            default is ''.

        Returns
        -------
        None.

        """
        
        action_name = 'gfilt'
        
        # set the abf object to load data with filters
        pyabf.filter.gaussian(abf, sigma, channel=0)
        abf.setSweep(sweepNumber=0, channel=0)        
        ch1_filt = abf.sweepY

        # if there's only one channel, no need to set the second one (we can't)
        if abf.channelCount > 1:
            pyabf.filter.gaussian(abf, sigma, channel=1)
            abf.setSweep(sweepNumber=0, channel=1)
            ch2_filt = abf.sweepY
        else:
            # cut data, if needed
            interval = self._df.metadata.interval
            if interval is not None and interval != 'todo':
                start_min, end_min = map(float, interval.split('-'))
                start = int(start_min * 60 * abf.sampleRate)
                end = int(end_min * 60 * abf.sampleRate)

                ch1_filt = ch1_filt[start:end]
                
            ch2_filt = ch1_filt.copy()
            
        
        self._save_processed_data(ch1_filt, ch2_filt, keep_og, channels, action_name)
        self._add_process_entry(action_name, sigma=sigma, keep_og=keep_og, channels=channels)
        
    
    def gaussian_filter(self, sigma_ms=20, keep_og=False, channels=''):
        """
        Uses a gaussian kernel to filter data. This gives a result that is 
        absolutely equivalent to the built-in abf gaussian filter, but runs a 
        bit slower. Use this if you no longer have the abf object.

        Parameters
        ----------
        sigma_ms : float, optional
            Sigma in units of milliseconds. The default is 20.
        keep_og : Bool, optional
            Whether to keep the original column or overwride it. The default is
            False.
        channels : str, optional
            A string describing what channel to apply the funciton on. The 
            default is ''.

        Returns
        -------
        None.

        """                
        action_name = 'gfilt2'
        data = self._df
        ch1, ch2 = self._get_channels(channels)
        
        # some filter methods leave behind nans in the data, that raises a LinAlgError
        nan_locs = np.isnan(ch1)
        
        # calculate the sigma in untis of datapoints
        sampling_rate = data.metadata.sampling_rate
        sigma_points = (sigma_ms / 1000) * sampling_rate
        
        ch1_filt = gaussian_filter1d(ch1[~nan_locs], sigma_points)
        ch2_filt = gaussian_filter1d(ch2[~nan_locs], sigma_points)
        
        self._save_processed_data(ch1_filt, ch2_filt, keep_og, channels, action_name)
        self._add_process_entry(action_name, sigma_ms=sigma_ms, keep_og=keep_og, channels=channels)        
    
    
    def lowpass_filter(self, filter_order=2, frequency_cutoff=10, keep_og=False, channels=''):
        """
        Filter the data in channels using a lowpass butterworth filter. The order
        and frequency cutoff value can be set. It uses a forwards and a 
        backwards pass of the filter, resulting in an effective filter order
        that is twice filter_order.

        Parameters
        ----------
        filter_order : int, optional
            Order of the filter. The default is 2.
        frequency_cutoff : float, optional
            Frequency at which the filter drops by 3dB. The default is 10Hz.
        keep_og : Bool, optional
            Whether to keep the original column or overwride it. The default is
            False.
        channels : str, optional
            A string describing what channel to apply the funciton on. The 
            default is ''.

        Returns
        -------
        None.

        """
        
        action_name = 'lpfilt'
        data = self._df
        ch1, ch2 = self._get_channels(channels)
        
        # some filter methods leave behind nans in the data, that raises a LinAlgError
        nan_locs = np.isnan(ch1)
                
        sampling_rate = data.metadata.sampling_rate
        sos = signal.butter(filter_order, frequency_cutoff, btype='lowpass', output='sos', fs=sampling_rate)
        ch1_filt = signal.sosfiltfilt(sos, ch1[~nan_locs])
        ch2_filt = signal.sosfiltfilt(sos, ch2[~nan_locs])
        
        self._save_processed_data(ch1_filt, ch2_filt, keep_og, channels, action_name)
        self._add_process_entry(action_name, filter_order=filter_order, frequency_cutoff=frequency_cutoff, keep_og=keep_og, channels=channels)  
        
    
    def highpass_filter(self, filter_order=2, frequency_cutoff=0.1, keep_og=False, channels=''):
        """
        Filter the data in channels using a highpass butterworth filter. The 
        order and frequency cutoff value can be set. It uses a forwards and a 
        backwards pass of the filter, resulting in an effective filter order
        that is twice filter_order.

        Parameters
        ----------
        filter_order : int, optional
            Order of the filter. The default is 2.
        frequency_cutoff : float, optional
            Frequency at which the filter drops by 3 dB. The default is 0.1Hz.
        keep_og : Bool, optional
            Whether to keep the original column or overwride it. The default is
            False.
        channels : str, optional
            A string describing what channel to apply the funciton on. The 
            default is ''.

        Returns
        -------
        None.

        """
        
        action_name = 'hpfilt'
        data = self._df
        ch1, ch2 = self._get_channels(channels)
        
        # some filter methods leave behind nans in the data, that raises a LinAlgError
        nan_locs = np.isnan(ch1)
                
        sampling_rate = data.metadata.sampling_rate
        sos = signal.butter(filter_order, frequency_cutoff, btype='highpass', output='sos', fs=sampling_rate)
        ch1_filt = signal.sosfiltfilt(sos, ch1[~nan_locs])
        ch2_filt = signal.sosfiltfilt(sos, ch2[~nan_locs])
        
        self._save_processed_data(ch1_filt, ch2_filt, keep_og, channels, action_name)
        self._add_process_entry(action_name, filter_order=filter_order, frequency_cutoff=frequency_cutoff, keep_og=keep_og, channels=channels)  
        
    
    def cross_correlation(self, channels=''):
        """
        Calculate the corss-correlation of the data in channel 1 against that 
        in channel 2. Use the processed data as defined by "channels". Return 
        both the lags in units of the time vector in the data and the cross
        correlation.

        Parameters
        ----------
        channels : str, optional
            A string describing what channel to apply the funciton on. The 
            default is ''.

        Returns
        -------
        lags : array
            Lag values at which the correlation was calculated, in units of the
            time vector in the data.
        corr : array
            Normalized cross-correlation of the data.

        """
        
        ch1, ch2 = self._get_channels(channels)
        sampling_rate = self._df.metadata.sampling_rate
        
        corr = signal.correlate(ch1, ch2)
        corr /= np.max(corr)

        lags = signal.correlation_lags(ch1.shape[0], ch2.shape[0]) / sampling_rate
    
        return lags, corr
    
    def multi_cross_correlation_lag(self, channels='', bits=None, length=None):
        """
        Calculate the cross correlation lag for the run as it evolves over time.
        To that end, split the run into multiple chunks and calculate the lag
        on each chunk. Define either the ammount of bits to make or the length
        (in seconds) of each bit. Can't do both.        

        Parameters
        ----------
        channels : str, optional
            A string describing what channel to apply the funciton on. The 
            default is ''.
        bits : int, optional
            Ammount of bits into which to chop the data. Calculate the cross
            correlation lag on each bit. The default is None.
        length : float, optional
            Length (in seconds) of each bit into which the data gets chopped.
            Calculat the cross correlation lag on each bit. The default is None.

        Returns
        -------
        times, lags : arrays
            Arrays containing the lag at each instant and the timepoint at 
            which it was calculated. The time corresponds to the center of the 
            interval used.

        """

        ch1, ch2 = self._get_channels(channels)
        sampling_rate = self._df.metadata.sampling_rate
        N = len(ch1)
        
        step_length = self._validate_bits_and_length(bits, length, N, sampling_rate)
        
        # calculate the correlation
        times, lags = [], []
        for n in range(0, N, step_length):
            # calculate cross correlation
            theslice = slice(n, min(n + step_length, N))
            
            corr = signal.correlate(ch1[theslice], ch2[theslice])
            corr /= np.max(corr)
        
            size = theslice.stop - theslice.start
            lags_array = signal.correlation_lags(size, size) / sampling_rate
            
            # calculate time and lag
            lag = lags_array[np.argmax(corr)]
            time = (theslice.start + theslice.stop) / 2 / sampling_rate
            
            times.append(time)
            lags.append(lag)
    
        times = np.asarray(times)
        lags = np.asarray(lags)
        
        return times, lags

                
    def calc_magnitudes_and_phases(self, channels=''):
        """
        Calculates the phase and magnitude of the timeseries in channels using 
        the hilbert transform.

        Parameters
        ----------
        channels : str, optional
            A string describing what channel to apply the funciton on. The 
            default is ''.

        Returns
        -------
        None.

        """
        
        action_name = 'hilbert'
        ch1, ch2 = self._get_channels(channels)
        
        ch1_mag, ch1_ph = self.calc_magnitudes_and_phases_in_one_channel(ch1)
        ch2_mag, ch2_ph = self.calc_magnitudes_and_phases_in_one_channel(ch2)
        
        self._save_processed_data(ch1_ph, ch2_ph, keep_og=True, channels=channels, action='phase')
        self._save_processed_data(ch1_mag, ch2_mag, keep_og=True, channels=channels, action='magnitude')
        self._add_process_entry(action_name, channels=channels)  
        
    @staticmethod
    def calc_magnitudes_and_phases_in_one_channel(x):
        """Calculate the phae and magnitude of a timeseries using the Hilbert
        transform. Assumes the data is already detrended, with mean 0."""
        
        nan_locs = np.isnan(x)
        
        analytic = signal.hilbert(x[~nan_locs])
        phases_short = np.unwrap(np.angle(analytic))
        magnitudes_short = np.absolute((analytic))
        
        magnitudes = np.full(x.shape, np.nan)
        phases = np.full(x.shape, np.nan)
        
        magnitudes[~nan_locs] = magnitudes_short
        phases[~nan_locs] = phases_short
        
        return magnitudes, phases
    
    
    def calc_phase_difference(self, channels=''):
        """
        Calculate the phase difference order parameter K as the cosine of the
        difference of the phase of channel 1 vs channel 2. If the phases had not
        been calculated so far, calculate them.

        Returns
        -------
        None.

        """
        
        data = self._df
        
        if not self._check_if_action_was_performed_on_channel('hilbert', channels):
            self.calc_magnitudes_and_phases(channels)
        
        ch1, ch2 = self._get_channels(channels + '_phase')
        K = np.sin(ch1 - ch2)
        data[ (channels + '_K').strip('_') ] = K
        
        self._add_process_entry('phase_diff', channels=channels)
  
    
    def calc_corr_lag(self, channels=''):
        """
        Calculate the lag using the corosscorrelation of the channels. The lag
        is computed as the point at which the maximum of the correlation 
        happens. Lag is always of channel 1 with respect to channel 2.

        Parameters
        ----------
        channels : str, optional
            A string describing what channel to apply the funciton on. The 
            default is ''.

        Returns
        -------
        lag : float
            Lag of channel 1 with respect to channel 2 in units of the time
            vector used.

        """
        
        lags, corr = self.cross_correlation(channels)
        
        return lags[np.argmax(corr)]
    
    
    def baseline(self, channels='', drop_quantile=0.5):
        """
        Uses baseline_in_one_channel to calculate the baseline of the data 
        given in channel. See that function for a more detailed description.

        Parameters
        ----------
       channels : str, optional
           A string describing what channel to apply the funciton on. The 
           default is ''.
        drop_quantile : float, optional
            Quantile under which to drop the minima. Should be between 0 and 1.
            0.5 means drop everything over the median. The default is 0.5.

        Returns
        -------
        (float, float)
            Value of the baseline for each channel. The value is repeated (but)
            not calculated twice) if the data is single channel.
        """
        
        ch1, ch2 = self._get_channels(channels)
        
        *_, minima1 = self.baseline_in_one_channel(ch1, drop_quantile)
        
        if self._df.metadata.twochannel:
            *_, minima2 = self.baseline_in_one_channel(ch2, drop_quantile)
        else:
            minima2 = minima1
            
        return minima1.mean(), minima2.mean()      
    
    def multi_baseline(self, channels='', drop_quantile=0.5, bits=None, length=None):
        """
        Calculate the local baseline value for each channel. "local" is here 
        defined by cutting the data into multiple pices and calculating it for 
        each pice individually. The pices do not overlap. The user can either 
        decide how many pices to cut the data into using 'bits', or how long 
        should each pice be (in seconds) using 'length'.

        Parameters
        ----------
        channels : str, optional
            A string describing what channel to apply the funciton on. The 
            default is ''.
        drop_quantile : float, optional
            Quantile under which to drop the minima. Should be between 0 and 1.
            0.5 means drop everything over the median. The default is 0.5.
        bits : int, optional
            Ammount of bits into which to chop the data. Calculate the baseline
            on each bit. The default is None.
        length : float, optional
            Length (in seconds) of each bit into which the data gets chopped.
            Calculat the baseline on each bit. The default is None.

        Returns
        -------
        times : array
            times at which the baselines where calculated.
        baselines1 : array
            baselines of channel 1.
        baselines2 : TYPE
            baselines of channel 2.

        """
        
        ch1, ch2 = self._get_channels(channels)
        sampling_rate = self._df.metadata.sampling_rate
        N = len(ch1)
        
        step_length = self._validate_bits_and_length(bits, length, N, sampling_rate)
        
        # calculate the baselines
        baselines1, baselines2 = [], []
        for ch, b in zip((ch1, ch2), (baselines1, baselines2)):
            times = []
            for n in range(0, N, step_length):
                theslice = slice(n, min(n+step_length, N))
                *_, minima = self.baseline_in_one_channel(ch.values[theslice], drop_quantile)
                b.append(minima.mean())
            
                time = (theslice.start + theslice.stop)/2 / sampling_rate
                times.append(time)
                
            if not self._df.metadata.twochannel:
                baselines2 = baselines1
                break
        
        times = np.asarray(times)
        baselines1 = np.asarray(baselines1)
        baselines2 = np.asarray(baselines2)
        
        return times, baselines1, baselines2

    
    @staticmethod
    def baseline_in_one_channel(ch, drop_quantile=0.5):
        """
        Finds the local minima of the data in ch and filters out all the ones
        that are over the given drop_quantile. It returns all the local minima 
        and the index at which they happen, and the local minima under the 
        requested quartile, as well as the indexes at which they happen.

        Parameters
        ----------
        ch : array
            Data over which to find the baselines.
        drop_quantile : float, optional
            Quantile under which to drop the minima. Should be between 0 and 1.
            0.5 means drop everything over the median. The default is 0.5.

        Returns
        -------
        min_inx : array of ints
            indexes at which the minima happen.
        minima : array
            values of all the local minima.
        filtered_min_inx : array of ints
            indexes at which minima that are under the requested quantile happen.
        filtered_minima : array
            values of the local minima under the given quantile.
        """
        
        min_inx, _ = signal.find_peaks(ch)
        minima = ch[min_inx]
                
        # find the requested quantile of the minima
        minima = ch[min_inx]
        minima_quantile = np.quantile(minima, drop_quantile)
        
        # keep only the minima under the requested quantile
        filtered_min_inx = min_inx[minima<=minima_quantile]
        filtered_minima = ch[filtered_min_inx]
        
        # return filtered_minima.mean()
        
        return min_inx, minima, filtered_min_inx, filtered_minima        
        
    
    def find_peaks(self, prominence=5, period_percent=0.6, channels=''):
        """
        Finds the peaks in the sata saved in channels. See find_peaks_in_one_channel
        for more info.

        Parameters
        ----------
        channels : str, optional
            A string describing what channel to apply the funciton on. The 
            default is ''.

        Returns
        -------
        None.

        """
        
        action_name = 'findpeaks'
        ch1, ch2 = self._get_channels(channels)
        
        # first channel data
        ch1_peak_indexes = self.find_peaks_in_one_channel(self._df.times, ch1, prominence, period_percent)
        # do second only if it's a two channel signal
        if self._df.metadata.twochannel:
            ch2_peak_indexes = self.find_peaks_in_one_channel(self._df.times, ch2, prominence, period_percent)    
        else:
            ch2_peak_indexes = ch1_peak_indexes 
        
        self.peaks = (ch1_peak_indexes, ch2_peak_indexes)
        self._add_process_entry(action_name, prominence=prominence, period_percent=period_percent, channels=channels)
        
    
    @staticmethod
    def find_peaks_in_one_channel(times, ch, prominence=5, period_percent=0.6):
        """
        Find the peaks in the signal given by (times, ch). The signal is assumed
        to be fairly clean (a gaussian fitler with sigma=100ms seems to be good
        enough). To do so it does three findpeaks passes:
            1. Find (all) peaks that are above 0mv (assumes a detrended signal)
            2. Find peaks that fall above a given threshold. The threshold is 
            calculated from the data of the previous pass using an otsu 
            thresholding method to discriminate high peaks and spurious peaks.
            The otsu threshold will only be used if it falls between two maxima
            of the distribution of peaks, since it tends to give useless values
            when the distribution has only one maximum.
            3. Find peaks that lie at least some distance away from the previous
            peak. The distance is calculated as period_percent% of the mode of 
            the period duration, as given by the previous pass.

        Parameters
        ----------
        times : array
            time vector.
        ch : aray
            data vector.

        Returns
        -------
        p_inx : array
            Indexes where the peaks happen.

        """
        
        ## first pass, with threshold 0mv
        p_inx, _ = signal.find_peaks(ch, height=0)
        # peaks = ch.values[p_inx]
        
        ## second pass, with threshold given by otsu
        # threshold = Processors.get_threshold(peaks)
        # p_inx, _ = signal.find_peaks(ch, height=threshold)
        # peaks = ch.values[p_inx]
        t_peaks = times.values[p_inx]

        ## third pass, with minimum distance between peaks
        counts, bins = np.histogram(np.diff(t_peaks))
        bin_centers = bins[:-1] + np.diff(bins) / 2
        period_mode = bin_centers[ np.argmax(counts) ]
        distance_points = int(period_mode * period_percent / (times[1] - times[0]))
        # p_inx, _ = signal.find_peaks(ch, height=threshold, distance=distance_points)
        p_inx, _ = signal.find_peaks(ch, distance=distance_points, prominence=prominence)
                
        return p_inx
    
    @staticmethod
    def get_threshold(peaks, fallback_threshold=4):
        """Use the otsu method to calculate the threshold for peaks. If the 
        value is too big (more than twice the fallback threshold), don't accept
        it. If the distribution of peaks has only one maximum, don't accept it.
        If the distribution has two or mode maxima (including the first point) 
        and the threshold doesn't fall between those maxima, don't accept it.
        """
        threshold = filters.threshold_otsu(peaks)
        
        if threshold > 2 * fallback_threshold:
            return 2 * fallback_threshold
        
        # we will only accept otsu's threshold if we cna detect two peaks in the 
        # distribution and the threshold falls between them
        counts, bins = np.histogram(peaks, bins='auto')
        bin_centers = bins[:-1] + np.diff(bins) / 2
        maxima, _ = signal.find_peaks(counts)
        # make sure the other peak is not the first point of the distribution
        if all(counts[0]>c for c in counts[1:3]):
            maxima = np.array( (0, *maxima) )

        # if only one maximum was detected, we fallback
        if len(maxima) < 2:
            return fallback_threshold
        
        # if too many maxima were found, keep the largest two
        if len(maxima) > 2:    
            maxima = sort_by(maxima, counts[maxima])[-2:]
            maxima.sort()    

        # if at least two maxima were found, accept threshold only if it lies between them
        if not( bin_centers[maxima[0]] < threshold < bin_centers[maxima[1]]):
            return fallback_threshold
        
        return threshold
    
    def _save_processed_data(self, ch1, ch2, keep_og, channels, action):
        """
        Write the (processed) data form ch1 and ch2 into the corresponding 
        channels, takeing care to replace or keep the original data as 
        needed. If the input data is smaller than the original data, it will
        assume the missing data is due to nans during the calculation and append
        the new data where the original had no nans. This will error out if the
        reason for the size mismatch was differente.

        Parameters
        ----------
        ch1, ch2 : data
            The processed data.
        channels : str
            What channels were targeted.
        keep_og : Bool
            Whether to keep the original channels or not.
        action : str
            what type of processing was performed.

        Returns
        -------
        None.

        """
        
        og_channel_names = self._get_channel_names(channels)
        
        # handle the case where the input had nans
        if ch1.size != self._df.ch1.size:
            ch1_test, ch2_test = self._get_channels(channels)
            nan_locs1 = np.isnan(ch1_test)
            nan_locs2 = np.isnan(ch2_test)
            
            if np.sum(~nan_locs1) != ch1.size or np.sum(~nan_locs2) != ch2.size:
                raise ValueError("The size of the input data couldn't be matched to the non nan values in the original data")
            
            ch1_nans = np.full(ch1_test.shape, np.nan)
            ch1_nans[~nan_locs1] = ch1
            ch1 = ch1_nans
            
            ch2_nans = np.full(ch1_test.shape, np.nan)
            ch2_nans[~nan_locs2] = ch2
            ch2 = ch2_nans
        
        # define behaviour regarding column overwritting
        if keep_og:
            ch1_write, ch2_write = [ch+'_'+action for ch in og_channel_names]
        else:
            ch1_write, ch2_write = og_channel_names
                
        # write columns
        self._df[ch1_write] = ch1
        self._df[ch2_write] = ch2
      
    
    def _add_process_entry(self, action, **kwargs):
        """
        Add an entry to the processing list detailing what was done and what
        parameter were used.

        Parameters
        ----------
        action : str
            The action that was performed, i.e. the name of the processing step.
        **kwargs : 
            The arguments to the processing step.

        Returns
        -------
        None.
        
        """
        
        step = {'step':action, **kwargs}
        self.info = step
        
    def _get_channel_names(self, channels):
        """
        Return the names of the channel pair that match channels.

        Parameters
        ----------
        channels : str
            A string describing the channel. For example, 'detrend' will return
            'ch1_detrend', 'ch2_detrend'. An empty string will return the default
            channels.

        Returns
        -------
        Tuple of channel names.

        """
            
        if not isinstance(channels, str):
            raise TypeError('channels must be a string')
        
        channel_options = set(re.sub(r'ch\d_?', '', x) for x in self._df.columns if x != 'times')
        channels = channels.strip('_') # strip leading '_' in case there were any
        if channels not in channel_options:
            print(f'{channels} is not an available channel. Choose one of {channel_options}. Returning default raw channels.')
            channels = ''
        
        ch1_name = 'ch1_'+channels if channels else 'ch1'
        ch2_name = 'ch2_'+channels if channels else 'ch2'
        
        return ch1_name, ch2_name
        
    def _get_channels(self, channels):
        """
        Gets the series corresponding to the channels defined by channels. Uses
        _get_channel_names to figure out what those channels are.

        Parameters
        ----------
        channels : str
            see _get_channel_names.

        Returns
        -------
        Tuple of channel series.
        """
        ch1_name, ch2_name = self._get_channel_names(channels)
        return self._df[ch1_name], self._df[ch2_name]

    def get_step_info(self, step, last=True):
        """
        Returns the dictionary that stores the arguments corresponding to the 
        queried processing step. If multiple steps with the same name were
        registered, it returns the last one if last=True, or all of them 
        otherwise. 

        Parameters
        ----------
        step : str
            step name.
        last : Bool, optional
            Whether to return only the last step with the corresponding name,
            or all the matching ones. If last=True, return type is a dict, if
            last=False, return type is a tuple of dicts. The default is True.

        Returns
        -------
        step_info: dict, tuple of dicts

        """
        if step not in self.steps:
            raise ValueError(f'The step "{step}" has not been performed yet.')
            
        matches = (len(self.steps) - 1 - i for i, x in enumerate(reversed(self.steps)) if x==step)
        if last:
            return self.info[next(matches)]
        else:
            return tuple( reversed( [self.info[inx] for inx in matches] ) )

    def _check_if_action_was_performed_on_channel(self, action: str, channels: str) -> bool:
        """
        Checks if an action was performed on a given channel and returns a 
        boolean result. The caller has to decide what to do with this 
        information.
        """
    
        if action not in self.steps:
            return False
        
        #check if the action found was done on the correct channel
        steps = self.get_step_info(action, last=False)
        if all(channels!=step['channels'] for step in steps):
            return False # I know I can just return the result of the all call, but this is more explicit
        else:
            return True

    @staticmethod
    def _validate_bits_and_length(bits, length, N, sampling_rate):
        """
        Intended to use in functions where the data is going to be chopped up 
        into pices and the function can decide if the pices are defined by a
        total count or by their duration. Both can't be defined at the same time

        Parameters
        ----------
        bits : int
            How many pices to chop the data into.
        length : float
            How long should each pice of data be (in seconds).
        N : int
            Size of the data to chop into pices.

        Returns
        -------
        step_length : int
            how many points long is the step needed to conform with either bits
            of length.            
        """
        
        if bits is None and length is None:
            raise ValueError('You must give bits or length')
        if bits is not None and length is not None:
            raise ValueError("You can't define both bits and length. Choose one.")
        
        # calculate the length of the step 
        if length is not None:
            step_length = int(length * sampling_rate)
        if bits is not None:
            assert isinstance(bits, numbers.Integral)
            step_length = N // bits
            
        return step_length
    
#%% Aux functions

def load_any_data(file, *args, **kwargs):
    """ A thin wrapper to handle both single channel and dual channel files"""

    abf = pyabf.ABF(file)
    if abf.channelCount == 1:
        print('single channel data')
        return load_single_channel(file, *args, **kwargs)
    elif abf.channelCount == 2:
        print('dual channel data')
        if 'interval' in kwargs:
            # remove the interval argument if for some reason we passed it
            del kwargs['interval']
        return load_data(file, *args, **kwargs)
    else:
        raise NotImplementedError(f"Can't handle files with {abf.channelCount} channels")

def load_data(file, gauss_filter=True, override_raw=True):
    """
    Loads the ABF file, extracts the data from channels 1 and 2 in the sweep
    0 (only sweep in these files) and the times array. The times array will 
    always start at t=0. Pack everything into a pandas.DataFrame.
    The dataFrame used has two custom accessors that store metadata for the run
    and a bunch of processing functions. data.processing.info will store the
    processing steps done to the data.

    Parameters
    ----------
    file : str or Path-like
        Path to the data.
    gauss_filter : Bool, optional
        Whether to apply a gaussian filter. The default is True.
    override_raw : Bool, optional
        Whether to have the gaussia-filtered data override the original data. 
        This only has effect when gauss_filter=True. The default is True.

    Returns
    -------
    data : pandas.DataFrame
        The loaded data, optionally pre-processed with a gaussian filter.

    """
    # Load data
    abf = pyabf.ABF(file)
    # Set sweep and channel, extract tmes and data
    abf.setSweep(sweepNumber=0, channel=0)
    times = abf.sweepX
    ch1 = abf.sweepY
    # reset weep with new channel, extract data
    abf.setSweep(sweepNumber=0, channel=1)
    ch2 = abf.sweepY
    
    # Pack everything into a DataFrame
    times -= times[0]
    data = pd.DataFrame(data={'times':times, 'ch1':ch1, 'ch2':ch2})
    
    add_run_info(data, file)
    data.metadata.rec_datetime = abf.abfDateTime
    
    if gauss_filter:
        data.process.abf_gauss_filt(abf, sigma=20, keep_og = not override_raw)
    return data

def load_single_channel(file, interval=None, gauss_filter=True, override_raw=True):
    """
    NOTE: This function is a kind of wrapper to load single channel files using
    the two channel processors. It copies the only channel onto both channels 
    of the object, so the processors will be highly inefficient until they are
    rewritten.
    
    Loads the ABF file, extracts the data from channel 1 in the sweep 0 (only 
    sweep in these files) and the times array. The times array will always 
    start at t=0. Pack everything into a pandas.DataFrame.
    The dataFrame used has two custom accessors that store metadata for the run
    and a bunch of processing functions. data.processing.info will store the
    processing steps done to the data.

    Parameters
    ----------
    file : str or Path-like
        Path to the data.
    interval : 'str'
        A string indicating what part of the recording to use. It should be 
        either "todo", or two numbers separated by a dash. For example '1-4.5'
        indicates that the data should only be used between minute 1 and minute
        4.5. Setting it to None will use thre whole dataseries. Default is None.
    gauss_filter : Bool, optional
        Whether to apply a gaussian filter. The default is True.
    override_raw : Bool, optional
        Whether to have the gaussia-filtered data override the original data. 
        This only has effect when gauss_filter=True. The default is True.

    Returns
    -------
    data : pandas.DataFrame
        The loaded data, optionally pre-processed with a gaussian filter.

    """
    
    # Load data
    abf = pyabf.ABF(file)
    # Set sweep and channel, extract tmes and data
    abf.setSweep(sweepNumber=0, channel=0)
    times = abf.sweepX
    ch1 = abf.sweepY
    
    # Cut data at required points
    if interval is not None and interval != 'todo':
        start_min, end_min = map(float, interval.split('-'))
        start = int(start_min * 60 * abf.sampleRate)
        end = int(end_min * 60 * abf.sampleRate)

        times = times[start:end]        
        ch1 = ch1[start:end]
    
    # Pack everything into a DataFrame
    # repeat ch1 data into ch2
    times -= times[0]    
    data = pd.DataFrame(data={'times':times, 'ch1':ch1, 'ch2':ch1})
    
    add_run_info(data, file)
    data.metadata.interval = interval
    data.metadata._twochannel = False
    data.metadata.rec_datetime = abf.abfDateTime
    
    # filter data if needed
    if gauss_filter:
        data.process.abf_gauss_filt(abf, sigma=20, keep_og = not override_raw)
    
    return data

def add_run_info(data, file):
    """ Add the file path and pair guide info to the DataFrame"""
    
    # Unpack the information of this run from the pair guide
    # Note that getting the data from the dataframe into a comfortable format is hell    
    # I ended up having to iterate over the rows in a named tuple and extrating the info from there
    
    pair_guide_file = file.parent / 'par_guide.xlsx'
    pair_guide = pd.read_excel(pair_guide_file)
    pair_guide['name'] = pair_guide.name.astype(str)
    # pair_guide = pair_guide.set_index('name')
    run_info = next(pair_guide[pair_guide.name == file.stem].itertuples())
    
    # convert run info into a dict; skip first element, because it's the index
    run_info = {k:v for k, v in zip(pair_guide.columns, run_info[1:])}
    
    data.metadata.guide = run_info
    data.metadata.file = file
    
    data.metadata.sampling_rate = data.metadata.raw_sampling_rate
        
    
def hodges_lehmann_distance_estimator(group1, group2):
    """ Calculate the Hodges-Lehmann estimator, which represents an unbiased
    non parametric estimator for a distributions location parameter, which in
    symmetric distributions represents the median. In this case, we calculate 
    the median distance between the two empirical distributions by finding the
    median difference bwetween all pairs of the two gropus of data.
    If CI is True, then also calculate the bootsrapped confidence interval for
    the median difference.
    """
    
    differences = [x1-x2 for x1, x2 in itertools.product(group1, group2)]
    return np.median(differences)

def hodges_lehmann_distance_confidence_interval(group1, group2, bootstrap_samples=1000):
    """ Calculate the confidence bootstrapped interval for the Hodges-Lehmann 
    distance estimator. 
    """
    
    bs_result = stats.bootstrap([group1, group2], hodges_lehmann_distance_estimator, 
                                method='basic',
                                n_resamples=bootstrap_samples, 
                                vectorized=False)
    
    # plt.hist(bs_result.bootstrap_distribution, bins='auto')
        
    CI = bs_result.confidence_interval
    
    return CI
    
 
def find_valid_periods(period_times, period_values, threshold=1.8, passes=1):
    """
    Find indexes for points with "valid" periods. Valid here means points where
    we assume a period was skipped either by the algorithm or the cell. On 
    first pass it calculates the mode of the periods and considers valid only 
    periods within 1.8 times the mode. On the second one, it does a linear fitting for the periods

    Parameters
    ----------
    period_times, period_values : array-like
        (time, value) pairs for the periods to process.
    threshold : float, ptional
        Any period that is threshold times away from the mode of the periods 
        will be regarded as invalid. Default is 1.8.
    passes : int, optional
        How many passes to do. First apss is only calculating the mode. Second 
        pass does a linear regression and calculates deviations form there. 
        Subsequent passes repeat these two steps. Default is 1.

    Returns
    -------
    valid_indexes : array
        Indexes of the valid periods.

    """
    
    assert isinstance(passes, int) and passes > 0
    
    period_mode = calc_mode(period_values)
    valid_indexes = period_values <= period_mode * threshold
    
    # stop here if we only check mode
    if passes == 1:
        return valid_indexes
    
    trend = np.polynomial.Polynomial.fit(period_times[valid_indexes], period_values[valid_indexes], deg=1)
    detrended_periods = period_values - trend(period_times) + trend(0)
    
    period_mode = calc_mode(detrended_periods)
    valid_indexes = detrended_periods <= period_mode * threshold
    
    if passes == 2:
        return valid_indexes
    
    return find_valid_periods(period_times, period_values, threshold, passes=passes-2)


def linear_fit_error(x, yerr):
    """
    Returns a function that estimates the uncertainty in values estimated by 
    evaluating the a linear fit done over data given by pairs (x, y), where y
    is estimated to have a standard error yerr.

    Parameters
    ----------
    x : array-like
        Independent variable over which the observations were done.
    yerr : floar
        estiamted standard error for the observations.
    
    Returns
    -------
    callable
        Estimation of the uncertainty of the linear fit.
    """
    x = np.asarray(x)
    N = len(x)
    
    return lambda x_eval: np.sqrt( yerr**2/N * (1 + ((x_eval - x.mean())/x.std())**2 ) )


def make_scalebar(ax, text=True):
    """
    Draw a scalebar onto the plot. It retrieves the ticks from ax and draws a 
    vertical and horizontal scalebar using the smallest difference in the axis
    ticks. It's calculated from the ticks and not the lims so tha the value is 
    always a reasonable integer. Optionally, the text argument labels the 
    scale bars.

    Parameters
    ----------
    ax : matplotlib.Axis
        Axis onto which to draw the scalebar.
    text : bool, optional
        Decides whether to label the scalebars with the scales (no units). The 
        default is True.

    Raises
    ------
    ValueError
        If the axes don't ahve enough ticks (at least two).

    Returns
    -------
    None.

    """
    
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    
    if len(xticks) < 2 or len(yticks) <2:
        raise ValueError("Can't built scale bars with so few ticks")
    
    # plot bars
    x_scale = np.diff(xticks)[0]
    y_scale = np.diff(yticks)[0]
    
    ax.plot([xticks[1], xticks[1]+x_scale], [yticks[1], yticks[1]], 'k')
    ax.plot([xticks[1], xticks[1]], [yticks[1], yticks[1]+y_scale], 'k')
    
    if text:
        ax.text(xticks[1] + x_scale/2, yticks[1]-y_scale/5, str(x_scale), ha='center', va='top')
        ax.text(xticks[1] - x_scale/5, yticks[1]+y_scale/2, str(y_scale), ha='right')
    
    
#%% Analysis functions (repeated in Processors)


def polydetrend(times, data, degree=5):
    """ Polynomic detrend"""
    
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
    """Gaussian filter built into the abf object"""
    
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
    """Gaussian filter "from scratch" (using scipy filters)"""
    
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
    """ Find peaks with a  bunch of extra arguments and double passes"""
    
    ## first pass, with threshold 0mv
    p_inx, _ = signal.find_peaks(data, height=0)
    
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
    p_inx, _ = signal.find_peaks(data, distance=distance_points, prominence=prominence)
    
    return p_inx

   
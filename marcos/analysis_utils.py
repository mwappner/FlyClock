#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 15:35:04 2023

@author: marcos
"""

# from pathlib import Path
import re

import numpy as np
from scipy import signal, interpolate
from scipy.ndimage import gaussian_filter1d
from skimage import filters
import pandas as pd

import pyabf
import pyabf.filter

from utils import sort_by, enzip

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
        self._ch1 = guide_info.ch1
        self._ch2 = guide_info.ch2
        self._pair = guide_info.par
        self._raw_sampling_rate = guide_info.samplerate
        self._duration_min = guide_info._6 # this is the duration in minutes, encoded like this because namedtuples
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
        assert inx in (1,2), 'inx must be either 1 or 2'
        assert 'findpeaks' in self.steps, 'You must find peaks first'
        
        if channels is not None and not self._check_if_action_was_performed_on_channel('findepakes', channels):
            print('WARNING: findpeaks was not performed on this channel')
        
        if channels is None:
            channels = self.get_step_info('findpeaks')['channels']
        ch_pair = self._get_channels(channels)
        return ch_pair[inx-1].values[self.peaks[inx-1]]
    
    def get_avg_period(self, inx=None):
        assert inx in (1,2, None), 'inx must be either 1 or 2, or None'
        assert 'findpeaks' in self.steps, 'You must find peaks first'
        
        if inx is None:
            ch1_period = np.mean( np.diff(self.get_peak_pos(1)))
            ch2_period = np.mean( np.diff(self.get_peak_pos(2)))
            return np.mean((ch1_period, ch2_period))
        else:
            return np.mean( np.diff(self.get_peak_pos(inx)))
    def get_periods(self, inx):
        assert inx in (1,2, None), 'inx must be either 1 or 2, or None'
        assert 'findpeaks' in self.steps, 'You must find peaks first'
        
        peak_pos = self.get_peak_pos(inx)
        periods = np.diff(peak_pos)
        period_times = peak_pos[:-1] + np.diff(peak_pos ) / 2
        
        return period_times, periods
    
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
        downsampled = self._df[::downsampling_rate].copy()
        
        # make sure to keep metadata and processing steps info
        add_run_info(downsampled, self._df.metadata.file)
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
        
        pyabf.filter.gaussian(abf, sigma, channel=0)
        pyabf.filter.gaussian(abf, sigma, channel=1)
        
        abf.setSweep(sweepNumber=0, channel=0)        
        ch1_filt = abf.sweepY

        abf.setSweep(sweepNumber=0, channel=1)
        ch2_filt = abf.sweepY
        
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
        Filter the data in challes using a lowpass butterworth filter. The order
        and frequency cutoff value can be set.

        Parameters
        ----------
        filter_order : int, optional
            Order of the filter. The default is 2.
        frequency_cutoff : float, optional
            Frequency at which the filter drops by 10? dB. The default is 10.
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
        ch1_filt = signal.sosfilt(sos, ch1[~nan_locs])
        ch2_filt = signal.sosfilt(sos, ch2[~nan_locs])
        
        self._save_processed_data(ch1_filt, ch2_filt, keep_og, channels, action_name)
        self._add_process_entry(action_name, filter_order=filter_order, frequency_cutoff=filter_order, keep_og=keep_og, channels=channels)  
        

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
        K = np.cos(ch1 - ch2)
        data[ (channels + '_K').strip('_') ] = K
        
        self._add_process_entry('phase_diff', channels=channels)
  
    
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
        
        ch1_peak_indexes = self.find_peaks_in_one_channel(self._df.times, ch1)
        ch2_peak_indexes = self.find_peaks_in_one_channel(self._df.times, ch2)
        
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
        channels = channels.strip('_') # strip leasing '_' in case there were any
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
            return False
        else:
            return True

def load_data(file, gauss_filter=True, override_raw=True):
    """
    Loads the ABF file, extracts the data from channels 1 and 2 in the sweep
    0 (only sweep in these files) and the times array. The times array will 
    always start at t=0. Pack everything into a pandas.DataFrame.
    The ddataFrame used has two custo accessors that store metadata for the run
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
    run_info = next(pair_guide[pair_guide.name == file.stem].itertuples())
    
    data.metadata.guide = run_info
    data.metadata.file = file
    
    data.metadata.sampling_rate = data.metadata.raw_sampling_rate
        
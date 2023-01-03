#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 15:35:04 2023

@author: marcos
"""

# from pathlib import Path
import re

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import pandas as pd

import pyabf
import pyabf.filter

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
    
    def __init__(self, pandas_obj):
        # save the dataframe as an object to later access
        self._df = pandas_obj
        
        self._info = None
        self._trends = None
    
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
        return self.trends[inx-1]
    def get_trend(self, inx):
        assert inx in (1,2), 'inx must be either 1 or 2'
        return self.trends[inx-1](self._df.times)
    
    @property
    def phase_difference(self):
        if 'K' not in self._df.columns:
            raise ValueError('Phase difference has not been calculated yet')
        return np.nanmean(self._df.K)
    
    
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
        for step in self.info:
            downsampled.process.info = step
        if self.trends is not None:
            downsampled.process.trends = self.trends
        
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
        
        action_name = 'detrend'
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


    def calc_phase(self, channels=''):
        """
        Calculates the phase of the timeseries in channels using the hilbert
        transform.

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
        data = self._df
        ch1, ch2 = self._get_channels(channels)
        
        ch1_mag, ch1_ph = self.calc_magnitudes_and_phases(ch1)
        ch2_mag, ch2_ph = self.calc_magnitudes_and_phases(ch2)
        
        data['ch1_phase'] = ch1_ph
        data['ch2_phase'] = ch2_ph
        
        self._add_process_entry(action_name, channels=channels)  
        
    @staticmethod
    def calc_magnitudes_and_phases(x):
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
    
    def calc_phase_difference(self):
        
        data = self._df
        
        if 'ch1_phase' not in data.columns:
            self.calc_phase()
        
        K = np.cos(data.ch1_phase - data.ch2_phase)
        data['K'] = K
        
        self._add_process_entry('phase_diff')

    def _save_processed_data(self, ch1, ch2, keep_og, channels, action):
        """
        Write the (processed) data form ch1 and ch2 into the corresponding 
        channels, takeing care to replace or keep the original data as 
        needed.

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
            ch1_test, _ = self._get_channels(channels)
            nan_locs = np.isnan(ch1_test)
            
            ch1_nans = np.full(ch1_test.shape, np.nan)
            ch1_nans[~nan_locs] = ch1
            ch1 = ch1_nans
            
            ch2_nans = np.full(ch1_test.shape, np.nan)
            ch2_nans[~nan_locs] = ch2
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
        
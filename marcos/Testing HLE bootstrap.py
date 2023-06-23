#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 16:58:03 2023

@author: marcos
"""


import itertools
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
   
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
    
    plt.hist(bs_result.bootstrap_distribution, bins='auto')
        
    CI = bs_result.confidence_interval
    
    return CI

def test_estimator(g1, g2):
    plt.figure(figsize=[6.4 , 3.34], constrained_layout=True)

    ax1 = plt.subplot(1,2,1)
    plt.hist(g1, bins='auto')
    plt.hist(g2, bins='auto', alpha=0.6)

    ax2 = plt.subplot(1,2,2)

    hle = hodges_lehmann_distance_estimator(g1, g2)
    mdiff = np.median(g1) - np.median(g2)
    interval = hodges_lehmann_distance_confidence_interval(g1, g2)
    print(f'{hle:.3f}, {mdiff:.3f}')
    print(interval)

    ax1.set_title(f'Original distributions\nestimator={hle:.3f}')
    ax2.set_title(f'Bootstrapped distr. for the estimator\nCI: [{interval.low:.3f}, {interval.high:.3f}]')

    significant = not (interval.low <= 0 <= interval.high)
    plt.gcf().suptitle(f'{"" if significant else "NO "} significant difference')

rng = np.random.default_rng()

X1 = rng.normal(0, 1, 300)
X2 = rng.normal(0, 0.2, 256)
X3 = rng.normal(1, 0.2, 200)
X4 = rng.normal(1.05, 2, 321)

#%%
test_estimator(X1, X2)
test_estimator(X1, X3)
test_estimator(X2, X3)
test_estimator(X1, X4)
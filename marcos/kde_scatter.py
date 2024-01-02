#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 16:24:07 2023

@author: marcos
"""

from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def kde_scatter(i, x, horizontal_scale=0.1, ax=None, alpha=0.2, rasterized=None, **kw):
    """
    Make a scatter plot for all the values in x. The horizontal coordinate is 
    given by the value i, the vertical coordinate by the value of x. Every 
    point is randomly offset horizontally. The offset ammount is randomly 
    chosen from a normal distribution with a scale that reflects the value of 
    the estimated density function of x at that point. The density function of
    x is estimated using a gaussian kernel density estimator. The overall 
    effect is that, for sufficiently large samplesizes, the scattered cloud has
    a shape that resembles that of the (estiamted) density function.

    Parameters
    ----------
    i : float
        Horizontal coordinate around which the scatter cloud will generate.
    x : array-like
        Data to generate the scatter plot.
    horizontal_scale : float, optional
        Decides how wide each scattered cloud should be. Assuming this function 
        will be called multiple times and that the values of i will be 
        increasing integers each time, 0.1 is a good value to generate decently
        sized clouds. Lower values produce noarrower clouds and vice-versa. The
        default is 0.1.
    ax : matplotlib Axes or None, optional
        Axes onto which to plot the scatter plot. If ax is None, the current 
        active axis is selected. This does not open a figure if none is open,
        so you will have to do that yourself. The default is None.
    alpha : float in [0, 1], optional
        Alpha value  (transparency) of the plotted dots. When plotting multiple 
        points, using an alpha value lower than one is recommended, to prevent
        outliers from distorting the shape of the distribution by chance. The 
        default is 0.2.
    rasterized : Bool or None, optional
        Whether to rasterize the resulting image. When exporting figures as 
        vector images, if the iamge has too many points, editing the resulting 
        file cna be hard. In such cases, it may be better to rasterize the 
        scatter plot and make sure to save it with a high dpi. If rasterized is
        None, then the plot will be rasterized when there are more than 100 
        points in the dataset. The default is None.
    *kw : 
        Other keword arguments to be passed to the plotting function.

    Returns
    -------
    None.

    """
    
    x = np.asarray(x)
    
    kde = stats.gaussian_kde(x)
    max_val = kde(x).max()
        
    if ax is None:
        ax = plt.gca()
    
    # rastrize image if there are too many points
    if rasterized is None:
        rasterized = x.size>100
    
    ax.plot(np.random.normal(i, kde(x)/max_val * horizontal_scale, size=len(x)), x,
                          '.', alpha=alpha, rasterized=rasterized, **kw)
    
def kde_multi_scatter(data, horizontal_scale=0.1, ax=None, alpha=0.2, rasterized=None, **kw):
    """
    A set of measurements using the kde_scatter method. Data should be either 
    an itnerable where each element is a dataset, a dictionary or a pandas
    DataFrame. This function will repeatedly call kde_scatter for each element
    in the iterable, column in the dataframe or entry in the dictionary. In the 
    latter two cases the function will also rename the horizontal axis to 
    reflect the category names in the DataFrame or dictionary.

    See kde_scatter for a description of the other parameters.
    """
    import pandas as pd
    
    if isinstance(data, dict):
        values = data.values()
        names = list(data.keys())
    elif isinstance(data, pd.DataFrame):
        values = data.values.T
        names = data.columns
    else:
        values = data
        names = None
    
    if ax is None:
        ax = plt.gca()
    
    for i, x in enumerate(values):
        kde_scatter(i, x, horizontal_scale, ax, alpha, rasterized, **kw)
        
    if names is not None:
        positions = list(range(len(values)))
        ax.set_xticks(positions)
        ax.set_xticklabels(names)
    
### Example usage of kde_scatter   

rng = np.random.default_rng()
fig, ax = plt.subplots()

# generate fake data and plot it, using two differnet distributions
x = rng.power(5, 2000)
kde_scatter(0, x)

x = rng.power(2, 1000) + rng.normal(1, 0.2, 1000)
kde_scatter(1, x)

### Example uisage of kde_multi_scatter

# generate fake data
rng = np.random.default_rng()
data = {
        'power5': rng.power(5, 3000),
        'power2': rng.power(2, 3000) + rng.normal(1, 0.2, 3000),
        'normal': rng.normal(0, 1, 3000), 
        'binormal': [*rng.normal(-1, 0.3, 1000), *rng.normal(-3, 1, 2000)]
    }
data = pd.DataFrame(data)

# make figure
fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(6, 7), constrained_layout=True)

# kde plot
kde_multi_scatter(data, ax=ax1)
ax1.set_title('kde scatter plot')

# two more standard plots to compare
sns.swarmplot(data[::10], size=2, ax=ax2)
ax2.set_title('seaborn swarm plot (10% of points)')
sns.stripplot(data, size=2, ax=ax3)
ax3.set_title('seaborn strip plot')

# turn on horizontal grid lines
for ax in (ax1, ax2, ax3):
    ax.grid(axis='y')
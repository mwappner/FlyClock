#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 18:31:13 2023

@author: marcos
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

N = 20
a = 1.2
b = 3

xerr = 0.1
yerr = 1

x = np.linspace(5, 15, N)
y = a * x + b + rng.normal(0, yerr, N)
x += rng.normal(0, xerr, N)

plt.errorbar( x, y, yerr, xerr, fmt = '.')
trend = np.polynomial.Polynomial.fit(x, y, deg=1)


yfit_err = lambda x_eval: np.sqrt( yerr**2/N * (1 + ((x_eval - x.mean())/x.std())**2 ) )

yfit = trend(x)
plt.plot(x, yfit, 'k')
plt.plot(x, yfit + yfit_err(x), 'r')
plt.plot(x, yfit - yfit_err(x), 'r')
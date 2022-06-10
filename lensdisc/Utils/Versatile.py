# -*- coding: utf-8 -*-
# @Author: lshuns
# @Date:   2021-07-08 21:04:06
# @Last Modified by:   lshuns
# @Last Modified time: 2021-07-09 14:44:24

# some useful functions

import numpy as np
import pandas as pd
import scipy.interpolate as si

def SmoothingFunc(x, y, dx_interpolate, outlier_sigma=3, rolling_window=5):
    """
    Smoothing the row data by removing outliers.
    This is achieved by:
        1. remove outliers with rolling median using given sigma threshold
        2. fitting rolling median with interp1d to get smoothed curve
    """
    # get the rolling median centered in each point
    rolling_median = np.array(pd.Series(y).rolling(rolling_window, center=True).median().fillna(method='bfill').fillna(method='ffill'))
    # get the deviation for each point
    rolling_sigma = np.abs(y-rolling_median)
    # get the median difference
    sigma_median = np.nanmedian(rolling_sigma)

    # drop outliers
    mask_tmp = rolling_sigma <= outlier_sigma*sigma_median
    x = x[mask_tmp]
    y = y[mask_tmp]

    # get new median
    rolling_median = np.array(pd.Series(y).rolling(rolling_window, center=True).median().fillna(method='bfill').fillna(method='ffill'))

    # interpolate
    x_sampled = np.arange(x[0], x[-1], dx_interpolate)
    # f = si.interp1d(x, y, kind='cubic')
    f = si.interp1d(x, rolling_median, kind='cubic')

    y_sampled = f(x_sampled)


    return x_sampled, y_sampled

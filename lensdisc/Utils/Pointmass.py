#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 10:09:37 2020

@author: mrgr
"""

import numpy as np
import scipy.special as sc
from mpmath import hyp1f1
import cmath

def Pointmass(w,y):
    
    '''
    function returning the analytical solution for the poinmass lens.
    Parameters
    ----------
    w: frequency to calculate the amplification factor at
    y: impact parameter
    
    Returns
    ----------
    Fan : amplification factor, complex array
    '''

    xm=(y+(y**2 + 4)**(1/2))/2
    phim=(xm-y)**2/2 - np.log(xm)
    expon= np.exp(np.pi*w/4 + 1j*w/2* (np.log(w/2) -2* phim ) )
    Fan= np.complex(expon*sc.gamma(1-1j/2*w)*hyp1f1(1j/2*w, 1, 1j/2*w*y**2 ))

    return Fan



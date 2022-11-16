#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 14:56:14 2021

@author: mrgr

"""
# ----------------------------------------------------------------------------
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

from .DiffInt import DInt2D,DInt1D
from .Utils.Versatile import SmoothingFunc
from .Utils.PN_wav import PN_waveFunc
from .Utils.PlotModels import PutLayout
from .Utils.Pointmass import Pointmass
from .Utils.ConfigUnits import PutUnits, RemoveUnits
from .HistCounting.HistCounting import HistMethod
from .Levin.Levin import LevinMethod

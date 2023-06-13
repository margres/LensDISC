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

from lensdisc.DiffInt import DInt2D,DInt1D
from lensdisc.Utils.Versatile import SmoothingFunc
from lensdisc.Utils.PN_wav import PN_waveFunc
from lensdisc.Utils.PlotModels import PutLayout
from lensdisc.Utils.Pointmass import Pointmass_analytical
from lensdisc.Utils.ConfigUnits import PutUnits, RemoveUnits
from lensdisc.HistCounting.HistCounting import HistMethod
from lensdisc.Levin.Levin import LevinMethod

import numpy as np
from .RunConfigFile import ParseConfig
import os
from astropy import units as u
from astropy.constants import G,c, M_sun

G=G.to('Mpc^3/(s^2*kg)')
c=c.to('Mpc/s')
configs_dict=ParseConfig()
ERadInfo=configs_dict['lens_configs']
M=ERadInfo['M']*M_sun
D_ls=ERadInfo['D_ls']*u.Mpc
D_l=ERadInfo['D_l']*u.Mpc
D_s=ERadInfo['D_s']*u.Mpc

def Einstein_radius():

    return (4*G*M*D_ls/(c**2*D_l*D_s))**(1/2)

def ConvFactor():
    return (Einstein_radius()**2/(D_ls))*(D_s*D_l)

def PutUnits(w):
    w_units=w*c/ConvFactor()
    return np.array(w_units)

def RemoveUnits(w):

    w_unitless=w/u.s*ConvFactor()/c
    return np.array(w_unitless)

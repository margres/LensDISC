import numpy as np
from .RunConfigFile import ParseConfig
import os
from astropy import units as u

def PutUnits(w):
    from astropy.constants import G,c, M_sun
    #configuring Einstein radius
    configs_dict=ParseConfig()
    ERadInfo=configs_dict['lens_configs']
    M=ERadInfo['M']*M_sun
    D_ls=ERadInfo['D_ls']*u.kpc
    D_l=ERadInfo['D_l']*u.kpc
    D_s=ERadInfo['D_s']*u.kpc
    G=G.to('kpc^3/(s^2*kg)')
    c=c.to('kpc/s')
    E_rad=(4*G*M*D_ls/(c**2*D_l*D_s))**(1/2) # *(206264.8062471)               #(4*G*M*D_ls/(c**2*D_l*D_s))**(1/2)
    conv_factor=(E_rad**2/(D_ls))*(D_s*D_l)
    w_units=w*c/conv_factor
    return np.array(w_units)

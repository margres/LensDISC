# -*- coding: utf-8 -*-
# @Author: lshuns
# @Date:   2021-07-08 21:08:59
# @Last Modified by:   lshuns
# @Last Modified time: 2021-07-09 15:05:29

### solve the diffraction integral in virture of Fourier transform & histogram counting
###### reference: Nakamura 1999; Ulmer 1995

######### Coordinates convention:
############# the origin is set in the centre of perturbation (external shear) and the x-aixs is parallel to the direction of shear, that means gamma2 = 0
############# the source is set in the origin (xs=0)

import os
import sys
import numpy as np
import pandas as pd
from scipy import signal
from lensdisc.Utils.Versatile import SmoothingFunc
from lensdisc.Images.Images import TFunc, dTFunc, Images
import time
from lensdisc.Utils.PlotModels import PlotHistCounting

class TreeClass(object):
    """
    A tree class saving node information.

    Parameters
    ----------
    dt_require: float
        Sampling step in time delay function.
    tlims: a list of floats
        Sampling limits [t_min, t_max] in time delay function.
    """

    def __init__(self, dt_require, tlims):

        # general information
        self.dt_require = dt_require
        self.tlims = tlims

        # initial empty good nodes
        self.good_nodes = pd.DataFrame({'x1': [],
                                        'x2': [],
                                        't': [],
                                        'weights': []
                                        })

    def SplitFunc(self, weights, x1_list, x2_list, T_list, dT_list):
        """
        Split nodes into bad or good based on the errors of t values.
        """

        # ++++++++++++ calculate error based on gradient
        error_list = dT_list*(weights**0.5)
        flag_error = (error_list < self.dt_require)

        # +++++++++++++ check if T within the target range
        T_min_list = T_list - error_list
        T_max_list = T_list + error_list
        flag_tlim = (T_min_list < self.tlims[1]) & (T_max_list > self.tlims[0])

        # +++++++++++++ total flag
        flag_bad = flag_tlim & np.invert(flag_error)
        flag_good = flag_tlim & flag_error

        # ++++++++++++++ good node information saved to DataFrame
        tmp = pd.DataFrame(data={
            'x1': x1_list[flag_good],
            'x2': x2_list[flag_good],
            't': T_list[flag_good],
            'weights': weights*np.ones_like(x1_list[flag_good])
            })
        self.good_nodes = self.good_nodes.append(tmp, ignore_index=True)

        # ++++++++++++++ bad node using simple directory
        self.bad_nodes = [x1_list[flag_bad], x2_list[flag_bad], T_list[flag_bad], dT_list[flag_bad]]

def FtcFunc(images_info, t_list):
    """
    Calculate the singular part of F(t)

    Parameters
    ----------
    images_info: DataFrame
        All images' information.
    t_list: numpy array
        Sampling of t where Ftc being calculated.
    """

    # make a copy
    images_info = images_info.copy()

    # shift t
    images_info['tI'] -= np.amin(images_info['tI'].values)

    Ftc = np.zeros_like(t_list)
    for index, row in images_info.iterrows():
        tmp = np.zeros_like(t_list)
        # min
        if row['typeI'] == 'min':
            tmp[t_list>=row['tI']] = 2.*np.pi*(row['muI']**0.5)
            # print(">>>> a min image")
        # max
        elif row['typeI'] == 'max':
            tmp[t_list>=row['tI']] = -2.*np.pi*(row['muI']**0.5)
            # print(">>>> a max image")
        # saddle
        elif row['typeI'] == 'saddle':
            tmp = -2.*((-row['muI'])**0.5)*np.log(np.absolute(t_list-row['tI']))
            # print(">>>> a saddle image")
        else:
            raise Exception("Unsupported image type {:} !".format(row['typeI']))

        Ftc += tmp

    return Ftc

def FwcFunc(images_info, w_list):
    """
    Calculate the singular part of F(w)

    Parameters
    ----------
    images_info: DataFrame
        All images' information.
    w_list: numpy array
        Sampling of w where Fwc being calculated.
    """

    # make a copy
    images_info = images_info.copy()

    # shift t
    images_info['tI'] -= np.amin(images_info['tI'].values)

    Fwc = np.zeros_like(w_list, dtype='cfloat')
    for index, row in images_info.iterrows():
        # min
        if row['typeI'] == 'min':
            Fwc += (row['muI']**0.5) * np.exp(1j*w_list*row['tI'])
            # print(">>>> a min image")
        # max
        elif row['typeI'] == 'max':
            Fwc += (row['muI']**0.5) * np.exp(1j*w_list*row['tI'] - 1j*np.pi)
            # print(">>>> a max image")
        # saddle
        elif row['typeI'] == 'saddle':
            Fwc += ((-row['muI'])**0.5) * np.exp(1j*w_list*row['tI'] - 1j*np.pi*0.5)
            # print(">>>> a saddle image")
        else:
            raise Exception("Unsupported image type {:} !".format(row['typeI']))

    return Fwc

def FtHistFunc(xL12, lens_model, kappa=0, gamma=0, tlim_list=[0.5, 10, 100], dt_list=[1e-3, 1e-2, 1e-1]):
    """
    Calculate F(t) with histogram counting

    Parameters
    ----------
    xL12: a list of 1-d numpy arrays [xL1, xL2]
        lens center position, coordinates in the lens plane.
    lens_model: str
        Lens model, supported model ('point', 'SIS').
    kappa: float (optional, default=0)
        Convergence of external shear.
    gamma: float (optional, default=0)
        Shear of external shear.
    tlim_list: a list of float (optional, default=[0.5, 10, 100])
        Sampling limits in time delay function
        (the actual limits are defined on the top of image time delay: tmax = tI_max+tlim).
    dt_list: a list of float (optional, default=[1e-3, 1e-2, 1e-1])
        Sampling accuracy in different time bin as defined by tilm_list
    """

    # calculate the images
    nimages, xI12, muI, tI, typeI = Images(xL12, lens_model, kappa, gamma, return_mu=True, return_T=True)

    print('Found {} images'.format(nimages))
    p=[(round(x,3), round(y,3))for (x,y) in zip(xI12[0],xI12[1])]
    print('with positions', p)
    print('magnification',  [round(m, 3) for m in muI])
    print('time delay', [round(t, 3) for t in tI])


    # collect image info
    images_info = pd.DataFrame(data=
                    {
                    'xI1': xI12[0],
                    'xI2': xI12[1],
                    'muI': muI,
                    'tI': tI,
                    'typeI': typeI
                    })

    # initial guess of space bounds
    ## x1
    xI1_min = np.amin(images_info['xI1'].values)
    xI1_max = np.amax(images_info['xI1'].values)
    dxI1 = xI1_max - xI1_min
    boundI_x1 = [xI1_min-dxI1, xI1_max+dxI1]
    ## x2
    xI2_min = np.amin(images_info['xI2'].values)
    xI2_max = np.amax(images_info['xI2'].values)
    dxI2 = xI2_max - xI2_min
    boundI_x2 = [xI2_min-dxI2, xI2_max+dxI2]

    # t bounds from images
    tImin = np.amin(images_info['tI'].values)
    tImax = np.amax(images_info['tI'].values)

    # +++ extend bounds until meeting tmax
    ## tmax is set on the top of tImax
    tmax = tImax + tlim_list[-1]
    while True:
        N_bounds = 1000
        # build the bounds
        tmp2 = np.linspace(boundI_x1[0], boundI_x1[1], N_bounds)
        x1_test = np.concatenate([
                        tmp2,                            # top
                        np.full(N_bounds, boundI_x1[1]), # right
                        tmp2,                            # bottom
                        np.full(N_bounds, boundI_x1[0])  # left
                        ])

        tmp2 = np.linspace(boundI_x2[0], boundI_x2[1], N_bounds)
        x2_test = np.concatenate([
                        np.full(N_bounds, boundI_x2[1]), # top
                        tmp2,                            # right
                        np.full(N_bounds, boundI_x2[0]), # bottom
                        tmp2                             # left
                        ])
        # evaluate t
        T_tmp = TFunc([x1_test, x2_test], xL12, lens_model, kappa, gamma)
        # break condition
        if np.amin(T_tmp) > tmax:
            break
        # extend bounds
        boundI_x1 = [boundI_x1[0]-0.5*dxI1, boundI_x1[1]+0.5*dxI1]
        boundI_x2 = [boundI_x2[0]-0.5*dxI2, boundI_x2[1]+0.5*dxI2]

    # +++ initial grid
    ## initial steps
    N_x = 5000
    ## initial nodes
    x1_node = np.linspace(boundI_x1[0], boundI_x1[1], N_x)
    x2_node = np.linspace(boundI_x2[0], boundI_x1[1], N_x)
    dx1 = x1_node[1]-x1_node[0]
    dx2 = x2_node[1]-x2_node[0]
    x1_grid, x2_grid = np.meshgrid(x1_node, x2_node)
    x1_list = x1_grid.flatten()
    x2_list = x2_grid.flatten()
    T_list = TFunc([x1_list, x2_list], xL12, lens_model, kappa, gamma)
    ## gradient for error calculation
    dtdx1, dtdx2 = dTFunc([x1_list, x2_list], xL12, lens_model, kappa, gamma)
    dT_list = np.sqrt(np.square(dtdx1)+np.square(dtdx2))

    # +++ hist counting
    t_list_final = []
    Ft_list_final = []
    ## loop over different regions
    for i_reg, tlim in enumerate(tlim_list):
        if i_reg == 0:
            tmin_bin = 0
        else:
            # last bin's max is this bin's min
            tmin_bin = tmax_bin
        ## tmax is set on the top of tImax
        tmax_bin = tImax + tlim

        ## accuracy in this bin
        dt = dt_list[i_reg]

        ## build Tree
        Tree = TreeClass(dt, [tmin_bin, tmax_bin])
        Tree.SplitFunc(dx1*dx2, x1_list, x2_list, T_list, dT_list)

        # iterate until bad_nodes is empty
        idx = 0
        Ft_list = 0
        dx1_tmp = dx1
        dx2_tmp = dx2
        while len(Tree.bad_nodes[0]):
            idx +=1
            # print('loop', idx)

            # bad nodes
            N_bad = len(Tree.bad_nodes[0])
            # print('number of bad_nodes', N_bad)

            # +++ subdivide bad nodes' region
            # each bad node being subdivided to 4 small ones
            x1_bad = np.repeat(Tree.bad_nodes[0], 4)
            x2_bad = np.repeat(Tree.bad_nodes[1], 4)
            # new nodes
            x1_list_tmp = x1_bad + np.tile([-0.25*dx1_tmp, 0.25*dx1_tmp, -0.25*dx1_tmp, 0.25*dx1_tmp], N_bad)
            x2_list_tmp = x2_bad + np.tile([0.25*dx1_tmp, 0.25*dx1_tmp, -0.25*dx1_tmp, -0.25*dx1_tmp], N_bad)

            # +++ calculate t & dt
            T_list_tmp = TFunc([x1_list_tmp, x2_list_tmp], xL12, lens_model, kappa, gamma)
            # gradient for error calculation
            dtdx1, dtdx2 = dTFunc([x1_list_tmp, x2_list_tmp], xL12, lens_model, kappa, gamma)
            dT_list_tmp = np.sqrt(np.square(dtdx1)+np.square(dtdx2))
            ##
            time1 = time.time()

            # +++ split good and bad
            dx1_tmp *= 0.5
            dx2_tmp *= 0.5
            Tree.SplitFunc(dx1_tmp*dx2_tmp, x1_list_tmp, x2_list_tmp, T_list_tmp, dT_list_tmp)

        # re-set origin of t
        Tree.good_nodes['t'] -= tImin

        # hist counting
        tmin_bin_true = np.min(Tree.good_nodes['t'].values)
        tmax_bin_true = np.max(Tree.good_nodes['t'].values)
        N_bins = int((tmax_bin_true-tmin_bin_true)/dt)
        Ft_list, bin_edges = np.histogram(Tree.good_nodes['t'].values, N_bins, weights=Tree.good_nodes['weights'].values/dt)
        t_list = (bin_edges[1:]+bin_edges[:-1])/2.

        # collect
        t_list_final.append(t_list[5:-5])
        Ft_list_final.append(Ft_list[5:-5])

    # merge together
    t_list_final = np.concatenate(t_list_final)
    Ft_list_final = np.concatenate(Ft_list_final)

    # calculate signular part
    Ftc = FtcFunc(images_info, t_list_final)
    
    # remove signular part
    Ftd = Ft_list_final - Ftc

    return t_list_final, Ftd, Ft_list_final, images_info

def HistMethod(xL12, lens_model, kappa=0, gamma=0, verbose = True, wlim=30., tlim_list=[0.5, 10, 100], dt_list=[1e-3, 1e-2, 1e-1], plot_histcounting=False):
    """
    main function for histogram counting method.

    Parameters
    ----------
    xL12: a list of 1-d numpy arrays [xL1, xL2]
        lens center position, coordinates in the lens plane.
    lens_model: str
        Lens model, supported model ('point').
    kappa: float (optional, default=0)
        Convergence of external shear.
    gamma: float (optional, default=0)
        Shear of external shear.
    wlim: float (optional, default=30)
        the maximum w value for wave effects, up which assume F(w)=Fc(w).
    tlim_list: a list of float (optional, default=[0.5, 10, 100])
        Sampling limits in time delay function
        (the actual limits are defined on the top of image time delay: tmax = tI_max+tlim).
    dt_list: a list of float (optional, default=[1e-3, 1e-2, 1e-1])
        Sampling accuracy in different time bin as defined by tlim_list
    """
    if verbose :
        print("Hist Method with: lens - {}; x - {}; kappa - {}; gamma -{}".format(lens_model,xL12, kappa, gamma ))
        print('Running...')
        
    start = time.time()

    # final accuracy determined by the minimum one
    dt = dt_list[0]

    # +++ 1. F(t) from histogram counting
    t_ori, Ftd_ori, Ft_ori, images_info = FtHistFunc(xL12, lens_model, kappa=kappa, gamma=gamma, tlim_list=tlim_list, dt_list=dt_list)

    #if plot_histcounting==True:
    #if True:
    #    np.save('t_ori',t_ori)
    #    np.save('Ftd_ori',Ftd_ori)
    #    np.save('Ft_ori',Ft_ori)
        #PlotHistCounting(t_ori,Ftd_ori,out='HistCount_smoothed' )
        #PlotHistCounting(t_ori,Ft_ori,out='HistCount_tot' )
        
    # +++ 2. smoothing the original results
    t, Ftd = SmoothingFunc(t_ori, Ftd_ori, dt, outlier_sigma=3, rolling_window=5)

    # +++ 3. add negative t values
    ## which only contains critical contributions
    ## the lowest t should equal to -t_max for numerical accuracy
    t_low = np.arange(-t[-1], t[0], dt)
    Ftc_low = -1. * FtcFunc(images_info, t_low)
    t = np.concatenate([t_low[:-1], t])
    Ftd = np.concatenate([Ftc_low[:-1], Ftd])

    # +++ 4. smooth again
    t_final, Ftd_final = SmoothingFunc(t, Ftd, dt, outlier_sigma=3, rolling_window=5)
    N_t = len(t_final)

    # +++ 5. FFT
    ## note: Ftd_final is real, so first half of the FFT series gives usable information
    ##      you can either remove the second half of ifft results, or use a dedicated function ihfft
    Fw = np.fft.ihfft(Ftd_final)
    ## multiply back what is divided
    Fw *= N_t
    ## multiply sampling interval to transfer sum to integral
    Fw *= dt
    ## the corresponding frequency
    freq = np.fft.rfftfreq(N_t, d=dt)
    w = freq*2.*np.pi

    # +++ 6. constants
    t2 = t_final[-1]
    t1 = t_final[0]
    Ft2 = Ftd_final[-1]
    Ft1 = Ftd_final[0]
    first_term = np.exp(1j*w*t2)*Ft2 - np.exp(1j*w*t1)*Ft1
    Fw = first_term/(2*np.pi) + Fw*w/(2j*np.pi)

    # +++ 7. get away artificial signals by picking local min
    index = signal.argrelextrema(np.abs(Fw), np.less_equal, order=3)[0]
    Fw = Fw[index]
    w = w[index]
    ## remove first artificial point
    Fw = Fw[1:]
    w = w[1:]

    # ++++ 8. only use certain range to avoid numerical errors
    ### up which assume to be negalible diffraction effects
    Fw[w >= wlim] = 0

    # ++++ 9. combine with critical contributions
    Fwc = FwcFunc(images_info, w)
    Fw += Fwc
    if verbose:
        print('finished in', round(time.time()-start,2),'s' )

    return w, Fw, Fwc
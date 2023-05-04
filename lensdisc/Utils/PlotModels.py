#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:38:02 2021

@author: mrgr
"""
import numpy as np
import matplotlib.pyplot as plt
from lensdisc.Utils.RunConfigFile import ParseConfig



configs_dict=ParseConfig()
config_plot=configs_dict['plot_configs']
grid=config_plot['grid']

def PutLayout():
    '''
    # default plotting options
    fig_width_pt  = 3.*246.0                                          # get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0/72.27                                        # convert pt to inch
    golden_mean   = (np.sqrt(5.)-1.0)/2.0                             # aesthetic ratio
    fig_width     = fig_width_pt*inches_per_pt                       # width in inches
    fig_height    = fig_width*golden_mean                            # height in inches
    fig_size      = [fig_width,fig_height]
    '''
    plt.tight_layout()
    plt.rcParams["figure.figsize"] = (10,10)
    params = {'font.family': 'DejaVu Sans',
              'font.serif': 'Computer Modern Raman',
              'axes.labelsize': 24,
              'axes.titlesize': 24,
              'xtick.labelsize' : 22,
              'ytick.labelsize' : 22,
              'font.size':20,
              'axes.grid' : grid,
              'text.usetex': True,
              'savefig.dpi' : 100,
              'legend.fontsize':18,
              'lines.markersize':5
              #'figure.figsize': fig_size
             }
    #plt.rcParams.update(params)
    # update rcParams with the user defined ones
    try:
        usr_params = kwargs['rcParams']
        for key in usr_params.keys():
            usr_value = usr_params[key]
            params.update({key: usr_value})
    except:
        pass
    plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command
    plt.rcParams.update(params)


def PutLabels (x_label, y_label, title=None):

    PutLayout()
    plt.figure()
    #plt.axis('square')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def AddFigText(x,y,text):
    plt.figtext(x, y, text, fontsize=18)

'''
def PlotRes(w,Fw_,out):
    ax1.plot(w, np.abs(Fw), '.-', label='LensDISC')
    ax1.xscale('log')
    #plt.xlim([0.1, 100])
    plt.legend(loc='upper left')
    plt.savefig(out+'.png',dpi=300)
    #plt.show(block=False)
    plt.close()

def PlotPhase(w,Fw,out):
    plt.plot(w, np.angle(Fw), '.-', label='LensDISC')
    plt.xscale('log')
    #plt.xlim([0.1, 100])
    plt.legend(loc='upper left')
    plt.savefig(out+'.png',dpi=300)
    #plt.show(block=False)
    plt.close()
'''


def PlotHistCounting(t,Ft, out):
    PutLayout()
    plt.plot(t,Ft, '.-')
    #plt.legend(loc='upper left')
    plt.savefig(out+'.png',dpi=300)
    print('saved in out')
    #plt.show(block=False)
    plt.close()
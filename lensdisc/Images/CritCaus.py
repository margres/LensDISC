#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 12:33:09 2020

@author: mrgr

Functions to plot critical curves and caustics
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from lensdisc.Utils.PostProcess import Filename
from lensdisc.Utils.PlotModels import PutLabels, PutLayout,AddFigText


def DistMatrix( dpsid11,dpsid22,dpsid12, kappa,gamma):

    '''
    Calculates the distortion matrix, used to find the images.

    Parameters
    ----------
    dpsid11: array float, double derivative in x
    dpsid22: array float, double derivative in y
    dpsid12: array float, mixed derivative in x and y
    kappa: float (optional, default=0)
        Convergence of external shear.
    gamma: float (optional, default=0).
        Shear of external shear.

    '''

    # Jacobian matrixF
    j11 = 1. - kappa - gamma - dpsid11
    j22 = 1. - kappa + gamma - dpsid22
    j12 = -dpsid12
    detA=j11*j22-j12*j12

    return detA


def SIScore(x12,kappa,gamma,fact, caustics=False):

    '''
    Calculates first derivatives to calculate distortion matrix for PseudoSIS lens
    and second derivatives to understand the type of the singularity.

    Parameters
    ----------
    x12: float list, lens - source position in x y coord.
    kappa: float (optional, default=0)
        Convergence of external shear.
    gamma: float (optional, default=0).
        Shear of external shear.
    fact: list, float. Parameters of the lens
    caustics: boolean, if True returns the first derivatives (containing the perturbed part too)
    '''


    a=fact[0]
    b=fact[1]

    x1=x12[0]
    x2=x12[1]

    #psi= a*sqrt(x1**2+x2**2+b**2)

    #first derivative
    dx12dx22 = np.sqrt(x1**2.+x2**2.+b**2.)
    dpsi1 = a*x1/(dx12dx22)
    dpsi2 = a*x2/(dx12dx22)

    if caustics==True:

        x1_per = x1*(kappa+gamma)
        x2_per = x2*(kappa-gamma)

        return np.column_stack((x1_per+dpsi1, x2_per+dpsi2))

    #second derivatives

    dx12pdx22_32 = (x1**2.+ x2**2.+b**2)**(3/2)
    # d^2psi/dx1^2
    dpsid11 = (b**2+x2**2)/dx12pdx22_32
    # d^2psi/dx2^2
    dpsid22 = (b**2+x1**2)/dx12pdx22_32
    # d^2psi/dx1dx2
    dpsid12 = -x1*x2/dx12pdx22_32

    detA = DistMatrix(dpsid11,dpsid22,dpsid12, kappa,gamma)

    return detA



def point(x12,kappa, gamma, fact,caustics=False):

    '''
    Calculates first derivatives to calculate distortion matrix for point mass lens
    and second derivatives to understand the type of the singularity.

    Parameters
    ----------
    x12: float list, lens - source position in x y coord.
    kappa: float (optional, default=0)
        Convergence of external shear.
    gamma: float (optional, default=0).
        Shear of external shear.
    fact: list, float. Parameters of the lens
    caustics: boolean,  if True returns the first derivatives (containing the perturbed part too)
    '''

    x1 = x12[0]
    x2 = x12[1]

    #first derivative

    dx12dx22 = x1**2.+x2**2
    dpsi1 = x1/dx12dx22
    dpsi2 = x2/dx12dx22

    if caustics==True:
        x1_per = x1*(kappa+gamma)
        x2_per = x2*(kappa-gamma)

        return  np.column_stack((x1_per+dpsi1, x2_per+dpsi2))
        #return np.column_stack((dpsi1,dpsi2))


    #second derivatives

    dx22mdx12 = x2**2.-x1**2.
    dx12pdx22_2 = (x1**2.+x2**2.)**2.

    # d^2psi/dx1^2
    dpsid11 = dx22mdx12/dx12pdx22_2
    # d^2psi/dx2^2
    dpsid22 = -dpsid11
    # d^2psi/dx1dx2
    dpsid12 = -2*x1*x2/dx12pdx22_2

    detA = DistMatrix(dpsid11,dpsid22,dpsid12, kappa,gamma)

    return detA

def softenedpowerlaw(x12,kappa, gamma, fact,caustics=False):

    '''
    Calculates first derivatives to calculate distortion matrix for softened power-law lens
    and second derivatives to understand the type of the singularity.

    Parameters
    ----------
    x12: float list, lens - source position in x y coord.
    kappa: float (optional, default=0)
        Convergence of external shear.
    gamma: float (optional, default=0).
        Shear of external shear.
    fact: list, float. Parameters of the lens
    caustics: boolean, if True returns the first derivatives (containing the perturbed part too)
    '''

    a=fact[0]
    b=fact[1]
    c=fact[2]
    p=fact[3]

    x1 = x12[0]
    x2 = x12[1]


    dpsi1=a*p*x1*(b**2+(x1**2+x2**2)/c**2)**(p/2 - 1)/c**2
    dpsi2=a*p*x2*(b**2+(x1**2+x2**2)/c**2)**(p/2 - 1)/c**2

    if caustics==True:
        #perturbation contribution
        x1_per = x1*(kappa+gamma)
        x2_per = x2*(kappa-gamma)

        return  np.column_stack((x1_per+dpsi1, x2_per+dpsi2))

    #second derivatives

    ppart=(b**2+ (x1**2+x2**2))
    dpsid11=a*p*ppart**(p/2. - 1.) + 2*a*(p/2. - 1.)*p*x1**2.*ppart**(p/2. - 2.)
    dpsid22=a*p*ppart**(p/2. - 1.) + 2*a*(p/2. - 1.)*p*x2**2.*ppart**(p/2. - 2.)
    dpsid12=2*a*ppart**(p/2. - 2.)*p*x1*x2*(p/2. - 1.)

    detA = DistMatrix(dpsid11,dpsid22,dpsid12, kappa,gamma)

    return detA


def softenedpowerlawkappa(x12,kappa, gamma, fact,caustics=False):

    '''
    Calculates first derivatives to calculate distortion matrix for softened power-law mass distribution lens
    and second derivatives to understand the type of the singularity.

    Parameters
    ----------
    x12: float list, lens - source position in x y coord.
    kappa: float (optional, default=0)
        Convergence of external shear.
    gamma: float (optional, default=0).
        Shear of external shear.
    fact: list, float. Parameters of the lens
    caustics: boolean,  if True returns the first derivatives (containing the perturbed part too)
    '''

    a,b,p =fact[0],fact[1],fact[3]

    x1 = x12[0]
    x2 = x12[1]


    x1x22= x1**2+x2**2


    if p==0:
        dpsi1= (a**2*x1*np.log((b**2 + x1x22)/b**2))/x1x22
        dpsi2= (a**2*x2*np.log((b**2 +  x1x22)/b**2))/x1x22
        dpsid11= (2*a**2*x1**2)/(x1x22*(b**2 + x1x22)) - (a**2*(x1**2 - x2**2)*np.log(1 + x1x22/b**2))/x1x22**2
        dpsid22= (2*a**2*x2**2)/(x1x22*(b**2 + x1x22)) - (a**2*(-x1**2 + x2**2)*np.log(1 + x1x22/b**2))/x1x22**2
        dpsid12= (2*a**2*x1*x2*(x1x22/(b**2 +x1x22) - np.log(1 + x1x22/b**2)))/x1x22**2
    else:

        dpsi1= (x1*a**(2 - p)*((b**2 + x1x22)**(p/2) - b**p))/(p *x1x22)
        dpsi2= (x2*a**(2 - p)*((b**2 + x1x22)**(p/2) - b**p))/(p *x1x22)

        dpsid11=(a**(2 - p) *((x1x22)**(p/2)*(b**2*(x2**2 - x1**2) + (p - 1)*x1**4 + p*x1**2*x2**2 + x2**4)*(b**2/(x1x22)+1)**(p/2) + b**p*(x1 - x2)*(x1 + x2)*(b**2 + x1**2 + x2**2)))/(p*(x1x22)**2*(b**2 + x1x22))
        dpsid22=(a**(2 - p)*((x1x22)**(p/2)*(b**2*(x1 - x2)*(x1 + x2) + p*x1**2*x2**2 + (p-1)*x2**4+x1**4)*(b**2/(x1**2+x2**2)+1)**(p/2)-b**p*(x1-x2)*(x1+x2)*(b**2+x1x22)))/(p*x1x22**2*(b**2+x1**2+x2**2))
        dpsid12=(x1*x2*a**(2-p)*(((p - 2)*x1x22 - 2*b**2)*(b**2 + x1x22)**(p/2) + 2 *b**p *(b**2 + x1x22)))/(p*x1x22**2*(b**2 + x1x22))


    if caustics==True:
        #perturbation contribution
        x1_per = x1*(kappa+gamma)
        x2_per = x2*(kappa-gamma)

        return  np.column_stack((x1_per+dpsi1, x2_per+dpsi2))

    detA = DistMatrix(dpsid11, dpsid22, dpsid12, kappa,gamma)

    return detA


def LensEq(x12,kappa, gamma, fact, lens_model):

    '''
    Calculates the lens equation
    Parameters
    ----------
    x12: float list, lens - source position in x y coord.
    kappa: float (optional, default=0)
        Convergence of external shear.
    gamma: float (optional, default=0).
        Shear of external shear.
    fact: list, float. Parameters of the lens
    lens_model: lens type

    '''

    x1 = x12[0]
    x2 = x12[1]

    alpha=eval(lens_model)((x1,x2),kappa, gamma, fact, caustics=True)
    beta=np.column_stack((x1,x2))-alpha

    return beta


def PlotCurves(xS12,xL12,kappa,gamma,lens_model,fact,path):

    '''
    Plots critical and caustics for a given lens.

    Parameters
    ----------
    xS12: list float, source (x,y) position
    xL12: list float, lens (x,y) position
    kappa: float (optional, default=0)
        Convergence of external shear.
    gamma: float (optional, default=0).
        Shear of external shear.
    fact: list, float. Parameters of the lens
    lens_model: lens type
    path: folder where to save images
    '''
    xL1 = xL12[0]
    xL2 = xL12[1]


    xS1 = xS12[0]
    xS2 = xS12[1]

    a=fact[0]
    b=fact[1]
    p=fact[3]


    coord='$x_1$='+str(xL1)+' $x_2$='+str(xL2)
    shear='$\gamma$='+str(gamma)+' $\kappa$='+str(kappa)


    if lens_model=='softenedpowerlaw' and p>1.7:
        xy_lin=np.linspace(-500,500,1000)
    else:
        xy_lin=np.linspace(-10,10,1000)
    X,Y = np.meshgrid(xy_lin, xy_lin)

    if lens_model == 'SIS':
    	lens_model='SIScore'



    crit_curv=eval(lens_model)((X,Y), kappa, gamma, fact)

    plt.figure(dpi=100)
    PutLabels('$x_1$','$x_2$')

    cp = plt.contour(X,Y, crit_curv,[ 1e-7], colors='k',linestyles= '-', linewidths=0.1)

    #I get the coordinates of the contour plot
    xyCrit_all = cp.collections[0].get_paths()
    #print(np.shape(xyCrit_all))

    for i in range(np.shape(xyCrit_all)[0]):
        figLim=0
        #xyCrit = cp.collections[0].get_paths()[i]
        xyCrit = xyCrit_all[i].vertices
        xCrit=xyCrit[:,0]
        yCrit=xyCrit[:,1]
        plt.plot(xCrit,yCrit,'k--',linewidth=0.7)

        xyCaus=LensEq((xCrit,yCrit), kappa, gamma, fact, lens_model)
        #print(np.shape(xyCaus))

        if xyCaus[:,0].any() <1e-5 and xyCaus[:,1].any() <1e-5 :
            plt.scatter(0,0, s=15, c='k', marker='o')
        plt.plot(xyCaus[:,0],xyCaus[:,1],'k-',linewidth=0.7)
        #plt.plot(xyCaus[:,0],xyCaus[:,0],'k-',linewidth=0.7)

        if lens_model!='point' and lens_model!='SIScore':
            plt.title(str(lens_model)+' - b='+str(b)+' p='+str(p))
        elif lens_model=='SIScore' :
            plt.title(str(lens_model)+' - b='+str(b))


        tmp=np.max(xyCrit_all[i].vertices)
        if tmp>figLim:
            figLim=tmp

    plt.plot(np.nan,np.nan,'k--',label='critical curves',linewidth=0.7)
    plt.plot(np.nan,np.nan,'k-',label='caustics',linewidth=0.7)

    AddFigText(.7, .2, coord)
    if gamma !=0 or kappa!=0:
        AddFigText(.7, .2-0.05, shear)

    #I always plot the critical curves and caustics with the lens in the middle
    if xL1!=0 and xL2!=0:
        xS1=xS1-xL1
        xS2=xS2-xL1
        xL1=0
        xL2=0

    plt.scatter(xL1, xL2, marker='x',color='r', label='lens')

    try:
        for x1,x2 in zip(xS1,xS2):
            plt.scatter(x1, x2, marker='*')
        plt.plot(np.nan,np.nan, marker='*',label='source', color='k', linestyle='None')
    except:
        plt.scatter(xS1, xS2, marker='*',color='tab:blue', label='source')

    #plt.axis('square')
    plt.xlim(-figLim-0.5, figLim+0.5)
    plt.ylim(-figLim-0.5, figLim+0.5)
    plt.legend(loc=3)

    try:
        y =round((xS1**2+xS2**2)**(0.5),2)
    except:
        y='various'

    add_info=Filename(lens_model,y, fact, kappa, gamma)
    plt.tight_layout()
    plt.savefig(path+'CritCaus_'+add_info+'.png')
    #plt.show(block=False)
    plt.close()

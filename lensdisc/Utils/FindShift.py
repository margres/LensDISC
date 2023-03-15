#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 11:11:15 2020

@author: mrgr
"""

import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

import numpy as np
import scipy.optimize as op
import scipy.special as ss
import warnings



def FindCrit(y,lens_model,fact):

    '''
    Finds the x positions (lens plane) of the crittical points of the time delay (images)
    Parameters
    ----------
    y: float, impact parameter position
    lens_model: string, type of lens
    fact: list, lens parameters

    '''

    rmaxlist=[5,20,50,100,1000]

    for rmax in rmaxlist:
        N= 100
        rmin = 0.001

        xt = np.linspace(rmin,rmax,N)
        xt_res=[]

        for i in range(N-1):
            if eval(lens_model)(xt[i],y,fact)* eval(lens_model)(xt[i+1],y,fact)<=0:
                tmp=op.brentq(eval(lens_model), xt[i],xt[i+1], args=(y,fact))
                xt_res.append(tmp)
                #print('x minimum: ',tmp)
                #break
                #try

        try:
            foo = tmp
            #print('x image' ,xt_res)
            break
        except NameError or UnboundLocalError:
            xt_res.append(0)
            if rmax==rmaxlist[-1]:
                warnings.warn('Could not find the value of the first image with y = '+str(y))
                return 0 #no images

    return xt_res

def SIS(x,y,fact):
    '''
    Singular isothermal Sphere potential derivative 1D (x is one dimensional vector).

    Parameters
    ----------
    x: coordinate in the lens plane
    y: float, impact parameter position


    '''
    a=fact[0]
    #derivative of the 1D potential
    phip= a
    #derivative of the time delay
    tp= (x - y) - phip

    return tp
    
def SIScore (x,y,fact):
    '''
    Singular isothermal Sphere potential derivative 1D (x is one dimensional vector).

    Parameters
    ----------
    x: coordinate in the lens plane
    y: float, impact parameter position
    fact: list, lens parameters

    '''

    a=fact[0]
    b=fact[1]

    #derivative of the 1D potential
    phip= a*x/(np.sqrt(b**2+x**2))
    #derivative of the time delay
    tp= (x - y) - phip

    return tp

def point (x,y,fact):

    '''
    Point lens potential derivative 1D (x is one dimensional vector).

    Parameters
    ----------
    x: coordinate in the lens plane
    y: float, impact parameter position
    fact: list, lens parameters

    '''


    #derivative of the 1D potential
    phip= 1./x
    #derivative of the time delay
    tp= (x - y) - phip

    return tp


def softenedpowerlaw(x,y,fact):

    '''
    Softened power law lens potential derivative 1D (x is one dimensional vector).

    Parameters
    ----------
    x: coordinate in the lens plane
    y: float, impact parameter position
    fact: list, lens parameters

    '''

    a,b,p=fact[0], fact[1], fact[3]

    phip= a*p*x*(b**2. + x**2.)**(p/2. - 1.)

    tp= (x - y) - phip

    return tp

def softenedpowerlawkappa(x,y,fact):

    '''
    softened power law potential derivative 1D (x is one dimensional vector).

    Parameters
    ----------
    x: coordinate in the lens plane
    y: float, impact parameter position
    fact: list, lens parameters

    '''

    a,b,p=fact[0], fact[1],fact[3]

    if p==0 and b!=0:
        phip=a**2./x * np.log(1.+ x**2./b**2.)
    elif p<4:
        phip=a**(2.-p)/(p*x) * (x**p*(1.+b**2./x**2.)**(p/2.) - b**p)
        #phip= a**(2 - p)/(p*x)  * ((b**2+x**2)**p/2 - b**p)
    else:
        raise Exception('Unsupported lens model')

    tp= (x - y) - phip

    return tp


def TimeDelay(x,y,fact,lens_model):

    '''
    It finds the time at which the images create and takes the first one (in time) as the 0th of the time.

    Parameters
    ----------
    x: coordinate in the lens plane
    y: float, impact parameter position
    fact: list, lens parameters
    lens_model: string, name of the lens model
    '''
    a,b,c,p=fact[0], fact[1], fact[2],fact[3]

    if x==0:
        phi=0

    elif lens_model == 'SIScore':
        phi =  a * np.sqrt(b**2+x**2)
    elif lens_model == 'point':
        phi = np.log(x)

    elif lens_model == 'softenedpowerlaw':
        phi=a*(x**2/c**2+b**2)**(p/2) - a*b**p

    elif lens_model == 'softenedpowerlawkappa':

        if p>0 and b==0:
            phi= 1/p**2 * a**(2-p) *x**p

        elif b!=0 and p!=0:
            if x==0:
                t1=0
            else:
                t1= 1./p**2. * a**(2.-p)*x**p *ss.hyp2f1(-p/2., -p/2., 1.-p/2., -b**2./x**2.)

            t2= 1./p*a**(2.-p)*b**p*np.log(x/b)
            t3= 1./(2.*p) * a**(2.-p)*b**p*(np.euler_gamma-ss.digamma(-p/2.))
            phi= t1 - t2 - t3

    else:
        raise Exception("Unsupported lens model !")

    psi_m = -(0.5*(abs(x-y))**2. - phi)

    return psi_m

def FirstImage(y,fact,lens_model):

    '''
    We have to scale everything in respect to the first image.
    If we have one more value of x at which we have images we
    need to realize which is the one related to the first
    (in the time domain) image.

    Parameters
    ----------
    y: float, impact parameter position
    fact: list, lens parameters
    lens_model: string, name of the lens model
    '''

    xlist=FindCrit(y,lens_model,fact)
    tlist=[]
    try:
        #if there is more than one image
        for x in xlist:
            t=TimeDelay(x,y,fact,lens_model)
            if np.isnan(t)==False:
                #print('t',t)
                tlist.append (t)
        t=np.min(tlist)
        #print('tlist',tlist)
    except:
        #only one image
        t=TimeDelay(xlist,y,fact,lens_model)
    #print('phi_m:',t)
    return t



if __name__ == '__main__':

    a=1
    b=0.5
    c=1
    p=1.8
    fact=[a,b,c,p]
    y=np.sqrt(2*0.1**2)
    lens_model='point'


    t = FirstImage(y,fact,lens_model)
    print('phi_m:',t)

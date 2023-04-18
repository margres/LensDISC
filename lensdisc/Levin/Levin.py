 # -*- coding: utf-8 -*-
# @Author: lshuns
# @Date:   2020-05-01 21:13:39
# @Last Modified by:   lshuns
# @Last Modified time: 2020-05-12 09:41:00

### Levin method for solving highly oscillatory one-dimensional integral
###### Target form: int_0^Infinity dx x J0(rx) exp(i G(x))
###### First do variable change (x=x/(1-x)): int_0^1 dx (x/(1-x)**3) J0(r*(x/(1-x))) exp(i G(x/(1-x)))
###### Transfer to: L * c = [x/(1-x)**3, 0, 0, 0] for cos, or [0, x/(1-x)**3, 0, 0] for sin

import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1


import numpy as np
import scipy.special as ss
from ..Utils.FindShift import FirstImage
#import mpmath
import warnings
import time
import matplotlib.pyplot as plt
import sys

def ChebyshevTFunc_simple_x(x, size):
    """
    Chebyshev polynomials of the first kind used as basis functions
        return simple list with form: [T0(x), T1(x),...]
    Parameters
    ----------
    x: float
        the variable
    size: positive int
        length of requested series.
    """

    # first two
    T0 = 1.
    T1 = x
    if size==1:
        return np.array([T0])
    if size==2:
        return np.array([T0, T1])

    # generate the rest of series with recurrence relation
    Tlist = np.zeros(size)
    Tlist[:2] = [T0, T1]
    Tn1 = T0
    Tn2 = T1
    for i in range(2, size):
        tmp = 2.*x*Tn2 - Tn1
        Tlist[i] = tmp
        #
        Tn1 = Tn2
        Tn2 = tmp

    #print(Tlist)

    return Tlist


def ChebyshevTFunc(xmin, xmax, size):
    """
    Chebyshev polynomials of the first kind used as basis functions
        return matrix with form: [[T0(x1), T1(x1),...],
                                    [T0(x2), T1(x2),...],
                                    ...,
                                    [T0(xn), T1(xn),...]]
    Parameters
    ----------
    xmin: float
        the lower bound of integrated range
    xmax: float
        the upper bound of integrated range
    size: positive int
        length of requested series.
    """

    # Chebyshev nodes used as collocation points
    ilist = np.arange(size) + 1
    xlist = np.cos(((1. - 2.*ilist + 2.*size)*np.pi)/(4.*size))**2. * (xmax-xmin) + xmin

    # first two
    Tlist0 = np.ones_like(xlist, dtype=float)
    Tlist1 = xlist
    if size==1:
        return xlist, np.vstack([Tlist0,]).transpose()
    if size==2:
        return xlist, np.vstack([Tlist0, Tlist1]).transpose()

    # generate the rest of series with recurrence relation
    Tlist = [Tlist0, Tlist1]
    Tlistn1 = Tlist0
    Tlistn2 = Tlist1
    for i in range(2, size):
        tmp = 2.*xlist*Tlistn2 - Tlistn1
        Tlist.append(tmp)
        #
        Tlistn1 = Tlistn2
        Tlistn2 = tmp
    #print(np.vstack(Tlist).transpose())
    return xlist, np.vstack(Tlist).transpose()


def ChebyshevUFunc(xmin, xmax, size):
    """
    Chebyshev polynomials of the second kind used as the differentiation of basis functions
        return matrix with form: [[U0(x1), U1(x1),...],
                                    [U0(x2), U1(x2),...],
                                    ...,
                                    [U0(xn), U1(xn),...]]
    Parameters
    ----------
    xmin: float
        the lower bound of integrated range
    xmax: float
        the upper bound of integrated range
    size: positive int
        length of requested series.
    """

    # Chebyshev nodes used as collocation points
    ilist = np.arange(size) + 1
    xlist = np.cos(((1. - 2.*ilist + 2.*size)*np.pi)/(4.*size))**2. * (xmax-xmin) + xmin

    # first two
    Ulist0 = np.ones_like(xlist, dtype=float)
    Ulist1 = 2.*xlist
    if size==1:
        return xlist, np.vstack([Ulist0,]).transpose()
    if size==2:
        return xlist, np.vstack([Ulist0, Ulist1]).transpose()

    # generate the rest of series with recurrence relation
    Ulist = [Ulist0, Ulist1]
    Ulistn1 = Ulist0
    Ulistn2 = Ulist1
    for i in range(2, size):
        tmp = 2.*xlist*Ulistn2 - Ulistn1
        Ulist.append(tmp)
        #
        Ulistn1 = Ulistn2
        Ulistn2 = tmp

    return xlist, np.vstack(Ulist).transpose()


def GpFunc(xlist, w, y, lens_model,fact):
    """
    the differentiation of the G(x) function shown in exponential factor
    Parameters
    ----------
    xlist: 1-d numpy array
        the integrated variable
    w: float
        dimensionless frequency from lens problem
    y: float
        impact parameter from lens problem
    lens_model: string
        lens model
    """
    a,b,c,p=fact[0], fact[1], fact[2],fact[3]

    if lens_model == 'SIS':
        gpx = w * (2.*xlist - 1.) / (1. - xlist)**3.

    elif lens_model == 'SIScore':

        sq=(b**2+xlist**2/(c**2*(xlist - 1)**2))**(0.5)
        gpx = w * (a*xlist - c**2*xlist*sq)/(c**2*(xlist - 1)**3*sq)

    elif lens_model == 'point':
        #gpx=-w*(0.5*xlist**2 - 2*xlist +1)/((1-xlist)**3 *xlist)
        gpx=-w*(1 - 2*xlist)/((1-xlist)**3*xlist)

    elif lens_model == 'softenedpowerlaw':

        #ppart=(b**2+xlist**2/(c**2*(xlist - 1)**2))**(p/2 - 1)
        gpx = w *  (xlist*(1. - 1.*a*p*(b**2 + xlist**2/(-1 + xlist)**2)**((-2 + p)/2)))/(1 - xlist)**3


    elif lens_model == 'softenedpowerlawkappa':

        if p!=0:

            gpx=w*(xlist/(1-xlist)**3 -( (a**(2 - p)*b**p)/(p*(xlist- 1)*xlist) - (a**(2 - p)*(xlist/(1 - xlist))**p *((b**2 *(1 - xlist)**2)/xlist**2 + 1)**(p/2))/(p*(xlist - 1)*xlist)) )
            #print(gpx)
            #gpx=(a**(2 - p)*(b**p - (1 + (b**2*(1 - xlist)**2)/xlist**2)**(p/2)*(xlist/(1 - xlist))**p))/(p *(-1 + xlist)*xlist)
        else:
            gpx= w*( xlist/(1-xlist)**3 + a**2*np.log(1 + xlist**2/(b**2*(1 - xlist)**2))/((xlist - 1)*xlist))
    else:
        gpx=0
        raise Exception('Unsupported lens model')

    if np.isnan(gpx).any():
        print(gpx)
        raise Exception('Unsupported value in the derivative')
    #print(gpx)
    return gpx

def WFunc(x, w, y, phim, lens_model,fact):

    """
    oscillatory part of integrand
        it is extended to a vetor to meet Levin requirement w' = A w
    Parameters
    ----------
    x: float
        the integrated variable
    w: float
        dimensionless frequency from lens problem
    y: float
        impact parameter from lens problem
    lens_model:
        lens model
    """

    # variable change
    x = x/(1.-x)

    # Bessel function
    j0v = ss.j0(w*y*x)
    j1v = ss.j1(w*y*x)

    a,b,c,p =fact[0],fact[1],fact[2],fact[3]
    # model-dependant exponential factor
    if lens_model == 'SIS':
        pot=x

    elif lens_model== 'SIScore':

        pot=a*(x**2 + b**2)**(0.5)

    elif lens_model == 'softenedpowerlaw':

        if 0<p<2:

            pot=a*(x**2+b**2)**(p/2) - a*b**p

        else:
             raise Exception('Unsupported model. P has to be in the range (0,2)')


    elif lens_model == 'softenedpowerlawkappa':

        #isothermal power law for p=0
        #modified Hubble model for p= 0
        #Plummer model for p =-2

        if p>0 and b==0:
            #eq 28
            pot=1/p**2 * a**(2-p) *x**p

        elif p<1 and b!=0 and p!=0:

            t1= 1/p**2 * a**(2-p)*(x+1e-5)**p *ss.hyp2f1(-p/2, -p/2, 1-p/2, -b**2/(x+1e-5)**2)
            t2= 1/p*a**(2-p)*b**p*np.log((x+1e-5)/b)
            t3= 1/(2*p) * a**(2-p)*b**p*(np.euler_gamma-ss.digamma(-p/2))
            pot=t1 - t2 - t3

        else:
            raise Exception('Unsupported model. If b=0, p>0 otherwise p<1 when b!=0.')


    elif lens_model == 'point':
        pot=np.log(x+1e-6)


    gx= w*(0.5*x**2. - pot + phim)

    # cos + i*sin for complex exponential part
    cosv = np.cos(float(gx))
    sinv = np.sin(float(gx))

    #print(gx)

    if np.isnan(float(gx)).any():
        print(gx)
        raise Exception('Unsupported value in the exponential')


    return np.array([j0v*cosv,
                     j0v*sinv,
                     j1v*cosv,
                     j1v*sinv])

def LevinFunc(xmin, xmax, size, w, y, phim, lens_model, cosORsin,fact):
    """
    Levin method to solve the integral with range (xmin, xmax)
        Using the linear equations
            L * c = rhs
            where
                L = I * u' + A * u
                the form of matrix A is hard-coded, given our integrand feature from axisymmetric lens problem
            where
                rhs = [f(x), 0, 0, 0] for cos
                rhs = [0, f(x), 0, 0] for sin
    Parameters
    ----------
    xmin: float
        the lower bound of integrated range
    xmax: float
        the upper bound of integrated range
    size: positive int
        length of requested series.
    w: float
        dimensionless frequency from lens problem
    y: float
        impact parameter from lens problem
    lens_model: string
        lens model

    cosORsin: string ('cos', 'sin')
        cos part or sin part
    """

    # the basis functions
    xlist, umatrix = ChebyshevTFunc(xmin, xmax, size)
    # differentiation of the basis functions
    if size == 1:
        upmatrix = np.vstack([np.zeros(size),])
    else:
        upmatrix = np.zeros((size, size), dtype=float)
        upmatrix[:,1:] = ChebyshevUFunc(xmin, xmax, size)[1][:,:-1] # first column is zero
        upmatrix *= np.array([np.arange(size),]*size)

    # factor within Bessel function
    r = w*y

    # hard-coded dimension of A matrix
    m = 4
    Lmatrix = np.zeros((m*size, m*size), dtype=float)
    # components for A * u matrix
    #   variable change is already taken into account
    null_matrix = np.zeros((size, size))
    gpx_u_matrix = np.vstack(GpFunc(xlist, w, y, lens_model,fact))*umatrix
    rx_u_matrix = np.vstack(r / (1.-xlist)**2.)*umatrix
    xx_u_matrix = np.vstack(1. / xlist / (1-xlist))*umatrix

    # assign values block by block for L # determined by transposed A matrix
    Lmatrix[:size, :] = np.concatenate((upmatrix, gpx_u_matrix, rx_u_matrix, null_matrix), axis=1)
    Lmatrix[size:2*size, :] = np.concatenate((-1.*gpx_u_matrix, upmatrix, null_matrix, rx_u_matrix), axis=1)
    Lmatrix[2*size:3*size, :] = np.concatenate((-1.*rx_u_matrix, null_matrix, -1.*xx_u_matrix+upmatrix, gpx_u_matrix), axis=1)
    Lmatrix[3*size:, :] = np.concatenate((null_matrix, -1.*rx_u_matrix, -1.*gpx_u_matrix, -1.*xx_u_matrix+upmatrix), axis=1)



    # oscillatory part
    f_osci_min = WFunc(xmin, w, y,phim, lens_model, fact)
    f_osci_max = WFunc(xmax, w, y,phim, lens_model, fact)

    # basis functions
    ulist_min = ChebyshevTFunc_simple_x(xmin, size)
    ulist_max = ChebyshevTFunc_simple_x(xmax, size)


    # right side of linear equations f(x) = (x/(1-x)**3) after variable change
    rhslist = np.zeros(m*size, dtype=float)
    if cosORsin == 'cos':
    #   [f(x), 0, 0, 0] for cos
        rhslist[:size] = xlist/(1.-xlist)**3.
    elif cosORsin == 'sin':
    #   [0, f(x), 0, 0] for sin
        rhslist[size:2*size] = xlist/(1.-xlist)**3.
    else:
        raise Exception("Unsupported cosORsin value !")

    # solve linear equations
    try:
        clist = np.linalg.solve(Lmatrix, rhslist)
    except np.linalg.LinAlgError:
        #raise Exception("LinAlgError")
        return  0
    # print('clist', clist)

    # collocation approximation
    p_min = np.array([  np.dot(clist[:size], ulist_min),
                        np.dot(clist[size:2*size], ulist_min),
                        np.dot(clist[2*size:3*size], ulist_min),
                        np.dot(clist[3*size:], ulist_min)])
    p_max = np.array([  np.dot(clist[:size], ulist_max),
                        np.dot(clist[size:2*size], ulist_max),
                        np.dot(clist[2*size:3*size], ulist_max),
                        np.dot(clist[3*size:], ulist_max)])

    # integral results
    I = np.dot(p_max, f_osci_max) - np.dot(p_min, f_osci_min)
    #print('Imax',np.dot(p_max, f_osci_max) )
    #print('Imin',np.dot(p_min, f_osci_min))



    return I



def InteFunc(w, y, lens_model='SIScore', fact=[1,0,1], size=19, accuracy=1e-2, N_step=50, Niter=int(1e5)):
    """
    Solve the integral
        int_0^1 dx (x/(1-x)**3) J0(r*(x/(1-x))) exp(i G(x/(1-x)))
        Using
            Levin method + adaptive subdivision

    Parameters
    ----------

    w: float
        dimensionless frequency from lens problem

    y: float
        impact parameter from lens problem

    lens_model: string ('SIS'(default), ...)
        lens model 

    size: positive int (default=19)
        length of requested series.
    
    accuracy: float (default=1e-2)
        tolerable accuracy in smoothness

    N_step: int (default=50)
        first guess of steps for subdivision

    Niter: int (default=int(1e5))
        the maximum iterating running
    """

    # +++ adaptive subdivision until meet the required accuracy +++ #
    # hard-coded integral range
    #   by variable change, the whole range is always (0,1)
    xmin = 0.
    xmax = 1.-1e-9

    I_cos_sin = np.zeros(2, dtype=float)
    part_names = ['cos', 'sin']
    
    phim= FirstImage(y,fact,lens_model)
    
    # cos and sin parts should be divided separately 
    for i_name in range(len(part_names)):
        part_name = part_names[i_name]
        I_final = 0.

        # first guess of the step
        dx = (xmax-xmin)/N_step
        a = xmin
        xmid = a + dx
        b = xmid + dx
        I_test1 = LevinFunc(a, xmid, size, w, y, phim, lens_model, part_name, fact)
        I_test2 = LevinFunc(xmid, b, size, w, y, phim, lens_model, part_name, fact)
        # get accuracy by comparing between neighbours
        diff_test = np.absolute(I_test1 - I_test2)

        # loop until the end (xmax)
        while True:
        
            # refine the binning
            for Nrun_left in range(Niter):
                if diff_test < accuracy:
                    break

                # keep split
                b = xmid
                xmid = (a+b)/2.

                I_test1 = LevinFunc(a, xmid, size, w, y, phim, lens_model, part_name, fact)
                I_test2 = LevinFunc(xmid, b, size, w, y, phim, lens_model, part_name, fact)
                # get accuracy by comparing between neighbours
                diff_test = np.absolute(I_test1 - I_test2)

            # accumulate results
            I_final += I_test1
            I_final += I_test2

            ### move forward to right side
            a = b
            b = a + dx
            b = min(b, xmax)
            xmid = (a+b)/2.
            # print('>>>>>>>>> a', a)
            # print('>>>>>>>>> dx', dx)
            # print('>>>>>>>>> b', b)
            I_test1 = LevinFunc(a, b, size, w, y, phim, lens_model, part_name, fact)
            # get accuracy by comparing between neighbours
            diff_test = np.absolute(I_test1 - I_test2)
            I_test2 = 0

            if (b >= xmax):
                I_final += I_test1
                break

        I_cos_sin[i_name] = I_final
        # print("Running part", part_name)
        # print("resulted upper bound", b)

    return I_cos_sin


def InteFunc_simple(w, y, lens_model='SIScore', fact=[1,0,1],size=19):
    """
    Solve the integral
        int_0^1 dx (x/(1-x)**3) J0(r*(x/(1-x))) exp(i G(x/(1-x)))
        Using
            Levin method
            without any optimization
    Parameters
    ----------
    w: float
        dimensionless frequency from lens problem
    y: float
        impact parameter from lens problem
    lens_model: string ('SIS'(default), ...)
        lens model
    size: positive int (default=19)
        length of requested series.
    """


    # hard-coded integral range
    #   by variable change, the whole range is always (0,1)
    xmin = 0.
    xmax = 1.-1e-9
    phim = FirstImage(y,fact,lens_model)

    I_cos_sin = np.zeros(2, dtype=float)
    #part_names = ['cos', 'sin']

    I_cos_sin[0] = LevinFunc(xmin, xmax, size, w, y, phim, lens_model, 'cos',fact)
    I_cos_sin[1] = LevinFunc(xmin, xmax, size, w, y, phim,  lens_model, 'sin',fact)

    return I_cos_sin


def InteFunc_fix_step(w, y, lens_model='SIScore',fact=[1,0,1,1], size=19, N_step=50):
    """
    Solve the integral
        int_0^1 dx (x/(1-x)**3) J0(r*(x/(1-x))) exp(i G(x/(1-x)))
        Using
            Levin method + fixed subdivision
    Parameters
    ----------
    w: float
        dimensionless frequency from lens problem
    y: float
        impact parameter from lens problem
    lens_model: string ('SIS'(default), ...)
        lens model
    size: positive int (default=19)
        length of requested series.

    N_step: int (default=50)
        fixed number of steps for subdivision
    """

    # hard-coded integral range
    #   by variable change, the whole range is always (0,1)
    xmin = 0.
    xmax = 1.-1e-9
    phim = FirstImage(y,fact,lens_model)

    I_cos_sin = np.zeros(2, dtype=float)

    # fix dx for each step
    xbounds_list = np.linspace(xmin, xmax, N_step)
    for i in range(len(xbounds_list)-1):

        I_test0 = LevinFunc(xbounds_list[i], xbounds_list[i+1], size, w, y, phim, lens_model, 'cos',fact)
        # print("sub-result (cos)", I_test0)
        # accumulate results from the last run
        I_cos_sin[0] += I_test0

        I_test0 = LevinFunc(xbounds_list[i], xbounds_list[i+1], size, w, y, phim, lens_model, 'sin',fact)
        # print("sub-result (sin)", I_test0)
        # accumulate results from the last run
        I_cos_sin[1] += I_test0

    return I_cos_sin

def LevinMethod(w,y, lens_model, fact=[1,0,1,1], typesub='Fixed', verbose = True, N_step=50):
    
    '''
    Solve the diffraction integral
    
    Parameters
    ----------
    w: float
        dimensionless frequency from lens problem
    y: float
        impact parameter from lens problem
    lens_model: string ('SIS', 'point', ..)
        lens model
    fact: parameters used for some lenses.
         fact = [a,b,c,p] 
    verbose: print info about what integral is calculating
    N_step: int (default=50)
        fixed number of steps for subdivision
    '''
    
    a,b,c,p =fact[0],fact[1],fact[2],fact[3]

    if verbose:
        print("Levin Method with: lens - {}; x - {}  and subdivision {}".format(lens_model, y, typesub ))
        if lens_model=='softenedpowerlawkappa' or lens_model=='softenedpowerlaw':
            print(f'additional parameters a - {a}; b - {b}; p-{p}')
        elif lens_model=='SIScore':
            print(f"additional parameters a - {a}; b - {b}")

        print('Running...')

    start = time.time()
    if type(w).__name__ =='list':
        w_range=np.round(np.linspace(w[0],w[1],int(w[2])),5)
    elif type(w).__name__ =='ndarray':
        w_range=np.round(w,5)
    else:
        raise Exception('Only array containing frequencies or list with boundaries is accepted')

    Fw=[]
    #time_l=[]

    for w in w_range:

        #print('W',w)
        const = -1j*w*np.exp(1j*w*y**2./2.)

        # ++++++++++++++++++++++++++ optimal with adaptive subdivision
        if typesub=='Fixed':
            I_cos_sin = InteFunc_fix_step(w, y,lens_model, [a,b,c,p], N_step)
        elif typesub=='Adaptive':
            I_cos_sin = InteFunc(w, y,lens_model, [a,b,c,p], N_step)
        elif typesub=='Simple':
            I_cos_sin = InteFunc_simple(w, y,lens_model, [a,b,c,p])
        else:
            raise Exception('Unsupported subdivision type. available: Fixed, Adaptive,Simple')

        #print('I_cos', I_cos_sin[0])
        #print('I_sin', I_cos_sin[1])

        restemp = const * (I_cos_sin[0] + 1j*I_cos_sin[1])
        #print(restemp)
        Fw.append(restemp)
        #time_l.append(time.time()-start)

    Fw=np.asarray(Fw)
    if verbose:
        print('finished in', round(time.time()-start,2),'s' )

    return w_range, Fw


from multiprocessing import Pool
from functools import partial

def parallel_func(w, y, lens_model, fact, typesub, N_step):
    const = -1j * w * np.exp(1j * w * y ** 2. / 2.)

    if typesub == 'Fixed':
        I_cos_sin = InteFunc_fix_step(w, y, lens_model, fact, N_step)
    elif typesub == 'Adaptive':
        I_cos_sin = InteFunc(w, y, lens_model, fact, N_step)
    elif typesub == 'Simple':
        I_cos_sin = InteFunc_simple(w, y, lens_model, fact)
    else:
        raise Exception('Unsupported subdivision type. available: Fixed, Adaptive,Simple')

    restemp = const * (I_cos_sin[0] + 1j * I_cos_sin[1])
    return restemp
    
def LevinMethodparallel(w,y, lens_model, fact=[1,0,1,1], typesub='Fixed', verbose = True, N_step=50, n_processes=6):
    
    '''
    Solve the diffraction integral
    
    Parameters
    ----------
    w: float
        dimensionless frequency from lens problem
    y: float
        impact parameter from lens problem
    lens_model: string ('SIS', 'point', ..)
        lens model
    fact: parameters used for some lenses.
         fact = [a,b,c,p] 
    verbose: print info about what integral is calculating
    N_step: int (default=50)
        fixed number of steps for subdivision
    '''
    
    a,b,c,p =fact[0],fact[1],fact[2],fact[3]

    if verbose:
        print("Levin Method with: lens - {}; x - {}  and subdivision {}".format(lens_model, y, typesub ))
        if lens_model=='softenedpowerlawkappa' or lens_model=='softenedpowerlaw':
            print(f'additional parameters a - {a}; b - {b}; p-{p}')
        elif lens_model=='SIScore':
            print(f"additional parameters a - {a}; b - {b}")

        print('Running...')

    start = time.time()
    if type(w).__name__ =='list':
        w_range=np.round(np.linspace(w[0],w[1],int(w[2])),5)
    elif type(w).__name__ =='ndarray':
        w_range=np.round(w,5)
    else:
        raise Exception('Only array containing frequencies or list with boundaries is accepted')

    Fw=[]
    #time_l=[]
    
    '''
    for w in w_range:

        #print('W',w)
        const = -1j*w*np.exp(1j*w*y**2./2.)

        # ++++++++++++++++++++++++++ optimal with adaptive subdivision
        if typesub=='Fixed':
            I_cos_sin = InteFunc_fix_step(w, y,lens_model, [a,b,c,p], N_step)
        elif typesub=='Adaptive':
            I_cos_sin = InteFunc(w, y,lens_model, [a,b,c,p], N_step)
        elif typesub=='Simple':
            I_cos_sin = InteFunc_simple(w, y,lens_model, [a,b,c,p])
        else:
            raise Exception('Unsupported subdivision type. available: Fixed, Adaptive,Simple')

        #print('I_cos', I_cos_sin[0])
        #print('I_sin', I_cos_sin[1])
    '''
    
    # create a Pool object with the desired number of processes
    pool = Pool(processes=n_processes)
    parallel_func_partial = partial(parallel_func, y=y, lens_model=lens_model, fact=fact, typesub=typesub, N_step=N_step)

    # run the for loop in parallel using the map() method
    Fw = pool.map(parallel_func_partial, w_range)
    
    # clean up the Pool object
    pool.close()
    pool.join()

    Fw=np.asarray(Fw)
    if verbose:
        print('finished in', round(time.time()-start,2),'s' )

    return w_range, Fw

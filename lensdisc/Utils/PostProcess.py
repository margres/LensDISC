import pandas as pd
from .PlotModels import PutLabels, PutLayout, AddFigText, PlotAmplitude, PlotPhase
from .RunConfigFile import ParseConfig
from .ConfigUnits import PutUnits
import matplotlib.pyplot as plt
import numpy as np

configs_dict=ParseConfig()
ERadInfo=configs_dict['lens_configs']
add_units=ERadInfo['add_units']

def Filename(lens_model,y, fact, kappa=0, gamma=0):

    '''
    Name to indentify results for different lens configurations.

    Parameters
    ----------
    y: float, impact parameter position
    lens_model: string, name of the lens model

    '''

    a,b,c,p =fact[0],fact[1],fact[2],fact[3]

    if lens_model=='softenedpowerlaw':

        add_info=lens_model+'_xL_'+str(y)+'_a_'+str(a)+'_b_'+str(b)+ '_p_'+str(p)

    elif lens_model=='softenedpowerlawkappa':

        add_info=lens_model+'_xL_'+str(y)+'_a_'+str(a)+'_b_'+str(b) + '_p_'+str(p)

    elif lens_model=='point':

        add_info=lens_model+'_xL_'+str(y)

    else:
        add_info=lens_model+'_xL_'+str(y)+'_a_'+str(a)+'_b_'+str(b)

    if kappa!=0 or gamma!=0:
        add_info = lens_model + '_xL_'+str(y)+'_kappa_'+str(kappa)+ '_gamma_'+ str(gamma)

    return add_info



def Savetocsv(w,Fw,savetopath, w_units):

    #save
    df = pd.DataFrame(list(zip(w,w_units,np.real(Fw),np.imag(Fw) , np.abs(Fw) )),columns=['w','w [Hz]','real F(w)', 'imag F(w)', 'abs F(w)' ] )
    df.to_csv(savetopath+'.csv', sep='\t', index= False)


def Histpostprocess(w, Fw,Fwc, xL12,lens_model,kappa,gamma, out_2D, plot=True):

    #input variables
    text_shear=' with external shear'
    if gamma!=0 or kappa!=0:
        title=lens_model+text_shear
    else:
        title=lens_model

    xL1,xL2=xL12[0],xL12[1]
    coord='$x_1$='+str(xL1)+'$x_2$='+str(xL2)
    shear='$\gamma$='+str(gamma)+' $\kappa$='+str(kappa)

    #all the file saved with this name
    add_info = Filename(lens_model,xL12, [1,0,1,1], kappa, gamma)
    savetopath=out_2D+add_info

    print('...saving results as ', savetopath)
    if add_units==True:
        Savetocsv(w,Fw,savetopath, PutUnits(w))
        w=PutUnits(w)
        xlabel='$\omega$ [Hz]'
        ylabel='$|F(\omega)|$'
    else:
        xlabel='$w$'
        ylabel='$|F(w)|$'
        #w_units=np.zeros_like(w)
        Savetocsv(w,Fw,savetopath, np.zeros_like(w))


    if plot:
        ## plot Amplitude
        PutLabels(xlabel,ylabel, title)
        plt.plot(w, abs(Fwc), 'silver',label='semi-classical approx')
        AddFigText(.2,.2,coord)
        AddFigText(.2,.2-0.05,shear)
        PlotAmplitude(w,Fw,out_2D+'Fwamp_'+add_info)

        ## plot phase
        PutLabels(xlabel,ylabel, title)
        plt.plot(w, np.angle(Fwc), 'silver',label='semi-classical approx')
        AddFigText(.2,.2, coord)
        AddFigText(.2,.2-0.05,  shear)
        PlotPhase(w,Fw,out_2D+'Fwphase_'+add_info)
        print('Done.')

def LevinPostprocess(w, Fw,xL, lens_model, fact,out_1D, plot):

    add_info = Filename(lens_model,xL, fact)
    savetopath=out_1D + add_info
    print('...saving results as ', savetopath)
    if add_units==True:
        Savetocsv(w,Fw, savetopath, PutUnits(w))
        w=PutUnits(w)
        xlabel='$\omega$ [Hz]'
        ylabel='$|F(\omega)|$'
    else:
        xlabel='$w$'
        ylabel='$|F(w)|$'
        Savetocsv(w,Fw, savetopath, np.zeros_like(w))


    if plot:
        ## plot Amplitude
        PutLabels(xlabel, ylabel, lens_model)
        PlotAmplitude(w,Fw,out_1D+'Fwamp_'+add_info)

        ## plot phase
        PutLabels(xlabel, '$\Phi_F$',lens_model)
        PlotPhase(w,Fw,out_1D+'Fwphase_'+add_info)
        print('Done.')

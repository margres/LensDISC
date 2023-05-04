import pandas as pd
from lensdisc.Utils.PlotModels import PutLabels, PutLayout, AddFigText
from lensdisc.Utils.RunConfigFile import ParseConfig
from lensdisc.Utils.ConfigUnits import PutUnits
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



def Savetocsv(w,Fw,savetopath, w_units, Fwc=None):

    #save
    df = pd.DataFrame(list(zip(w,w_units,np.real(Fw),np.imag(Fw) , np.abs(Fw) )),columns=['w','w [Hz]','real F(w)', 'imag F(w)', 'abs F(w)' ] )
    if Fwc is not None:
        df['real F(w)_semicl'] = np.real(Fwc)
        df['imag F(w)_semicl'] = np.imag(Fwc)
        df['abs F(w)_semicl'] = np.abs(Fwc)

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
        w_unitless= w.copy()
        w=PutUnits(w_unitless)
        Savetocsv(w_unitless,Fw,savetopath, w, Fwc=Fwc)
        xlabel='$\omega$ [Hz]'
        ylabel_amp='$|F(\omega)|$'
        ylabel_ph = '$\Phi_F$(\omega)'
    else:
        xlabel='$w$'
        ylabel_amp= '$|F(w)|$'
        ylabel_ph = '$\Phi_F$(w)'
        #w_units=np.zeros_like(w)
        Savetocsv(w,Fw,savetopath, np.zeros_like(w), Fwc=Fwc)


    if plot:
        PutLayout()
        ### TO DO: add title and kappa gamma info
        fig, (ax1,ax2) = plt.subplots(nrows=2, sharex=True,  figsize=(10, 10)) # frameon=False removes frames
        ax1.plot( w, np.abs(Fwc), c='silver',ls='-', label= 'Semi-classical')
        ax1.plot( w,np.abs(Fw),c='lightcoral',ls='-',alpha=1, label= 'Hist-counting',)
        ax1.set_xscale('log')
        ax1.set_ylabel(ylabel_amp, fontsize=20)
        #ax1.set_title('SIScore x$_L$=0.4 a=1.0 b=0.5')
        #ax1.legend( fontsize=15)


        ax2.plot(w, np.angle(Fwc),c='silver',ls='-', label= 'Semi-classical',alpha=1)
        ax2.plot(w , np.angle(Fw),c='lightcoral',ls='-',alpha=1, label= 'Hist-counting')
        ax2.set_xscale('log')
        ax2.set_ylabel(ylabel_ph, fontsize=20)
        #ax2.legend( fontsize=15)
        ax2.set_xlabel(xlabel,fontsize=20)
        plt.savefig(out_2D+'Fw_'+add_info+'.png',dpi=100)
        plt.close()
        plt.legend(loc = 'lower center', bbox_to_anchor = (0, -0.01, 1, 1),
           bbox_transform = plt.gcf().transFigure)
        plt.show(block=False)
        plt.close()

    print('Done.')


def LevinPostprocess(w, Fw,xL, lens_model, fact,out_1D, plot):

    add_info = Filename(lens_model,xL, fact)
    savetopath=out_1D + add_info
    print('...saving results as ', savetopath)
    if add_units==True:
        w_unitless= w.copy()
        w=PutUnits(w_unitless) #so we keep the same name for the final thing to plot
        Savetocsv(w_unitless,Fw, savetopath, w)
        xlabel='$\omega$ [Hz]'
        ylabel_amp='$|F(\omega)|$'
        ylabel_ph = '$\Phi_F(\omega)$'
    else:
        xlabel='$w$'
        ylabel_amp='$|F(w)|$'
        ylabel_ph = '$\Phi_F(w)$'
        Savetocsv(w,Fw, savetopath, np.zeros_like(w))


    if plot:
        PutLayout()
        fig, (ax1,ax2) = plt.subplots(nrows=2, sharex=True,  figsize=(10, 10)) # frameon=False removes frames
        ## TO DO:add title
        ax1.plot(w, np.abs(Fw), c='skyblue',ls='-',alpha=1)
        ax1.set_xscale('log')
        ax1.set_ylabel(ylabel_amp, fontsize=20)
        #ax1.set_title('SIScore x$_L$=0.4 a=1.0 b=0.5')
        #ax1.legend( fontsize=15)

        ax2.plot(w, np.angle(Fw), c='skyblue',ls='-',alpha=1)
        ax2.set_xscale('log')
        ax2.set_ylabel(ylabel_ph, fontsize=20)
        #ax2.legend( fontsize=15)
        ax2.set_xlabel(xlabel,fontsize=20 )
        plt.savefig(out_1D+'Fw_'+add_info+'.png',dpi=100)
        plt.show(block=False)
        plt.close()
        '''
        ## plot Amplitude
        PutLabels(xlabel, ylabel, lens_model)
        PlotAmplitude(w,Fw,out_1D+'Fwamp_'+add_info)

        ## plot phase
        PutLabels(xlabel, '$\Phi_F$',lens_model)
        PlotPhase(w,Fw,out_1D+'Fwphase_'+add_info)
        '''
    print('Done.')

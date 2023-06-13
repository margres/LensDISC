### module to generate an example configuration file

import os
import re
import shutil
import logging
import configparser
import distutils.util
import datetime
import logging
import sys


def ParseConfig():


    # ++++++++++++++ 1. config file
    config = configparser.ConfigParser(allow_no_value=True,
                                    inline_comment_prefixes='#',
                                    empty_lines_in_values=False,
                                    interpolation=configparser.ExtendedInterpolation())

    run_tag = os.path.realpath(__file__)
    path_conf='config.ini'               #os.path.join(run_tag, 'config.ini')
    ## existence
    try:
    	with open(path_conf) as f:
      	  config.read(path_conf)

    except IOError:
        print ("No configuration file provided! \n"
               "----> Creating a new one \n")
        GenerateExampleConfig(path_conf)

        #print("Execution stopped. Run again to start the process. ")
        #print("REMEMBER: If you need to change lens parameters modify the file config.ini ")
        #sys.exit()
        config.read(path_conf)

    # ++++++++++++++ 2. some general info (required by all tasks)
    ##  work dir
    config_paths = config['Paths']
    out_dir = config_paths.get('out_dir')
    #out_dir = os.path.join(run_tag, out_dir)

    if not os.path.exists(out_dir) and 'docs' not in out_dir :
        os.mkdir(out_dir)
        print(f'all outputs will be saved to {out_dir}')

    ### levin
    out_levin = config_paths.get('out_levin')
    out_levin  = os.path.join(out_dir, out_levin)

    if not os.path.exists(out_levin) and 'docs' not in out_levin :
        os.mkdir(out_levin)
    ### hist
    out_hist = config_paths.get('out_hist')
    out_hist  = os.path.join(out_dir, out_hist)
    if not os.path.exists(out_hist) and 'docs' not in out_hist:
        os.mkdir(out_hist)

    ### combine to a dictionary
    work_dirs = {'main': out_dir,
                    'levin': out_levin,
                    'hist': out_hist}


    ##  Hist info
    #check conditions on input vlaues
    config_hist = config['HistInfo']
    if config_hist.getfloat('kappa')+ config_hist.getfloat('gamma') >= 1:
        raise ValueError('The sum of kappa and gamma has to be lower than 1.')


    if config_hist.getfloat('kappa')<0 or config_hist.getfloat('gamma') < 0:
        raise ValueError('Kappa and/or gamma have to be positive.')

    if config_hist.get('lens_model') not in ['point', 'SIS']:
        raise ValueError('Lens model {} not available! Pick one between: \n  point or SIS .'.format(config_hist.get('lens_model')))


    hist_configs = {'model': config_hist.get('lens_model'),
                    'kappa': config_hist.getfloat('kappa'),
                    'gamma': config_hist.getfloat('gamma'),
                    'xL': [float(x.strip()) for x in config_hist.get('xL').split(',')]}

    ## Lens Info
    config_lens = config['LensInfo']
    D_l  = config_lens.getfloat('D_l')
    D_s = config_lens.getfloat('D_s')
    D_ls = D_s -D_l
    lens_configs = {'add_units' :config_lens.getboolean('add_units'),
                        'M': config_lens.getfloat('M'),
                        'D_l': D_l ,
                        'D_ls': D_ls,
                        'D_s': D_s }


    ## Levin info
    config_levin = config['LevinInfo']

    if config_levin.get('lens_model') not in [ 'point', 'SIS','SIScore' , 'softenedpowerlaw', 'softenedpowerlawkappa']:
        raise ValueError('Lens model {} not available! Pick one in between: \n  point, SIScore , softenedpowerlaw, softenedpowerlawkappa .'.format(config_levin.get('lens_model')))

    if config_levin.getfloat('xL') <0:
        raise ValueError('The readial distance must to be positive.')
    if int(config_levin.getfloat('N_step'))<0:
        raise ValueError('The amount of steps must be positive.')
    if config_levin.get('typesub') not in ['Fixed', 'Adaptive'] :
        raise ValueError('Wrong type of subdivision.s')

    levin_configs = {'model': config_levin.get('lens_model'),
                        'xL': config_levin.getfloat('xL'),
                        'w': [float(w.strip()) for w in config_levin.get('w').split(',')],
                        'a': config_levin.getfloat('a'),
                        'b': config_levin.getfloat('b'),
                        'p': config_levin.getfloat('p'),
                        'typesub': config_levin.get('typesub'),
                        'N_step': config_levin.getint('N_step') }

    ## Plot info
    config_plot = config['PlotInfo']
    plot_configs = {'grid':config_plot.getboolean('grid'),
                    'showPlotFinal': config_plot.getboolean('savePlotFinal'),
                    'showPlotCritCaus': config_plot.getboolean('savePlotCritCaus'),
                    'style':config_plot.get('style')}

    ## === dictionary for collecting all config info
    configs_dict = { 'work_dirs':work_dirs,'lens_configs':lens_configs,'hist_configs': hist_configs, 'levin_configs': levin_configs, 'plot_configs' : plot_configs}

    return configs_dict

def GenerateExampleConfig (file_name):
    '''
    An example configuration file
    '''

    config = f"# Example configuration file for LensDisc\n\
\n\n\
################################## Paths ################################################\n\
[Paths]\n\n\
out_dir =  ./lensdisc_output/                          # main directory for all the outputs \n\
out_levin = 1D/                                # path results Levin method \n\
out_hist = 2D/                                 # path results Histogram method \n\
\n\n\
################################## LensInfo ################################################\n\
[LensInfo]\n\n\
add_units=False                                # Boolean, if False the results will have unitless frequencies.\n\
                                               # If False the results will be dimensionless \n\
M = 50.0                                       # mass of the lens in solar masses \n\
D_l = 10.0                                     # distance of the lens in Mpc \n\
D_s = 100.0                                    # distance of the source in Mpc \n\
z = 0                                          # redshift of the lens \n\
\n\n\
################################## LevinInfo ##############################################\n\
[LevinInfo]\n\n\
lens_model= SIScore                            # String. Lens model, supported models: \n\
						                       # point, SIS, SIScore , softenedpowerlaw, softenedpowerlawkappa. \n\
                                               # str \n\
xL =       0.1                                 # radial distance lens-source - in units of Einstein radius  \n\
                                               # positive float \n\
w = 0.001,100,1000		               		   # frequency range, start, stop, step \n\
#lens parameters \n\
a =         1.0                                # Amplitude parameter. \n\
                                               # float (default=1)\n\
b =         0                                  # Core value\n\
                                               # float (default=0, range=[0,1])\n\
p =         1.0                                # power law value \n\
                                               # float (default=1) \n\
                                               # softenedpowerlawkappa p<1. softenedpowerlaw range (0,2) \n\
typesub = Fixed                                # type of subdivision (Adaptive or Fixed), default Fixed. \n\
                                               # Fixed divides integral in fixed steps, can be less accurate for certain models but faster. \n\
N_step = 50                                    # For fixed subdivision, amount of steps in the range. For Adaptive first guess of steps. \n\
                                               # integer (default=50) \n\
\n\n\
################################## HistInfo ##############################################\n\
[HistInfo]\n\n\
lens_model= point                              # String. Lens model, supported models: point, SIS. \n\
xL =       0.1,0.1                             # lens position, source is in the center - coordinates in the source plane \n\
                                               # in units of Einstein radius \n\
#external shear \n\
#N.B. the value of kappa + gamma has to be smaller than 1, i.e. in the range [0,1) \n\
\n\
kappa =         0.0                            # Convergence of external shear\n\
                                               # float (positive number, default=0)\n\
gamma =         0.0                            # Shear of external shear.\n\
                                               # float (positive number, default=0)\n\
\n\n\
################################## Plots ##############################################\n\
[PlotInfo]\n\n\
grid = True                                     # Gridding in the plot.\n\
savePlotFinal = True                            # Boolean [True or False], Show and save plots of Diffraction Integral or not. \n\
savePlotCritCaus = False                        # Boolean [True or False], Show and save plot of Critical and Caustic or not. \n\
style = ggplot                                  # string, style of the plots. \n\
\n\n\
"
    # write out the example config file
    with open(file_name, 'w') as configfile:
        configfile.write(config)
    print(f'An example configuration file `{file_name}` has been generated in the current directory.')
    print("REMEMBER: If you need to change lens parameters modify the file config.ini ")


if __name__ == '__main__':

	ParseConfig()


import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

import numpy as np
from .Utils.RunConfigFile import ParseConfig
from .HistCounting.HistCounting_updated import HistMethod
from .Levin.Levin import LevinMethod
from .Utils.PostProcess import LevinPostprocess,Histpostprocess
from .Images.Images import TContourplot
from .Images.CritCaus import PlotCurves
import sys


def ConfWork():
	# configuring working directories
	configs_dict=ParseConfig()
	work_path=configs_dict['work_dirs']
	config_plot=configs_dict['plot_configs']
	showPlotFinal=config_plot['showPlotFinal']
	showoPlotCritCaus=config_plot['showPlotCritCaus']

	return configs_dict,work_path,config_plot,showPlotFinal,showoPlotCritCaus

def DInt2D():

	configs_dict,work_path,config_plot,showPlotFinal,showoPlotCritCaus = ConfWork()
	model_info2D=configs_dict['hist_configs']
	xL12=model_info2D['xL']
	kappa=model_info2D['kappa']
	gamma=model_info2D['gamma']
	lens_model=model_info2D['model']
	out_2D=work_path['hist']+lens_model+'/'

	if not os.path.exists(out_2D):
		os.mkdir(out_2D)

	if showoPlotCritCaus:
		print(f'Saving Critical Curves-Caustics and Contour plot in {out_2D}.')
		xS12=[0,0] #source in the center by definition
		PlotCurves(xS12,xL12,kappa,gamma,lens_model,[1,0,1,1], out_2D)
		print(xL12)
		TContourplot(xL12,kappa,gamma, lens_model, out_2D)

	w, Fw, Fwc = HistMethod(xL12,lens_model,kappa,gamma)
	Histpostprocess(w, Fw,Fwc, xL12,lens_model,kappa,gamma, out_2D,showPlotFinal)

def DInt1D():

	configs_dict,work_path,config_plot,showPlotFinal,showoPlotCritCaus = ConfWork()
	model_info1D=configs_dict['levin_configs']
	xL=model_info1D['xL']
	lens_model=model_info1D['model']
	out_1D=work_path['levin']+lens_model+'/'
	typesub=model_info1D['typesub']
	fact=[model_info1D['a'],model_info1D['b'],1,model_info1D['p']]
	w = model_info1D['w']



	if not os.path.exists(out_1D):
		os.mkdir(out_1D)

	if showoPlotCritCaus:
		print(f'Saving Critical Curves-Caustics plot in {out_1D}.')
		PlotCurves([xL,0], [0,0], 0, 0, lens_model,fact,out_1D)

	w, Fw = LevinMethod(w, xL, lens_model, fact,typesub)
	LevinPostprocess(w, Fw,xL, lens_model, fact, out_1D,showPlotFinal)

if __name__ == "__main__":
	s=sys.argv[1]

	print("REMEMBER: If you need to change lens parameters modify the file config.ini ")
	if s=='2D':
		DI2Int()
	elif s=='1D':
		DI1Int()
	else:
		raise ValueError('Give the type of integral you want to run. \n We ONLY accept 2D or 1D. \n REMEMBER: If you need to change other parameters modify the file config.ini :)')

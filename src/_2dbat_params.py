#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#|-----------------------------------------|
#| _2dbat_params.py
#|-----------------------------------------|
#| by Se-Heon Oh
#| Dept. of Physics and Astronomy
#| Sejong University, Seoul, South Korea
#|-----------------------------------------|

import sys
import numpy as np
import yaml

#  ______________________________________________________  #
# [______________________________________________________] #
# [ global parameters
# _______________________________________________________  #
global _inputDataCube
global _is, _ie, _js, _je
global parameters
global nparams
global ngauss
global ndim
global max_ngauss
global gfit_results
global _x
global nchannels

global _params, r_galaxy_plane


global _tr_model_vf, _wi_2d

#  ______________________________________________________  #
# [______________________________________________________] #
# [ 2dbat parameters
# _______________________________________________________  #

def read_configfile(configfile):
    with open(configfile, "r") as file:
        _params = yaml.safe_load(file)
    return _params

def default_params():
    _params = {
    #+++++++++++++++++++++++++++++++++
    # directories
    # 2dbat main
    'main_2dbat':'/Users/seheon/research/projects/2dbat-PI-dev/2dbat-PI-dev/src',  # <---- UPDATE HERE

    # working directory where the input data cube is
    #'wdir':'/Users/seheon/research/projects/2dbat-PI-dev/2dbat-PI-dev/src/analysis/vcc667',  # <---- UPDATE HERE
    'wdir':'/Users/seheon/research/projects/2dbat-PI-dev/2dbat-PI-dev/src/analysis/vcc1992',  # <---- UPDATE HERE
    #'wdir':'/Users/seheon/research/projects/2dbat-PI-dev/2dbat-PI-dev/src/analysis',  # <---- UPDATE HERE

    # output directory
    '_2dbatdir':'output',

    #+++++++++++++++++++++++++++++++++
    # input 2D velocity field to fit
    #'input_vf':'vfs/sgfit.G3_1.1.fits',  # <---- UPDATE HERE
    #'input_vf':'test2.fits',  # <---- UPDATE HERE
    #'input_vf':'test2.fits',  # <---- UPDATE HERE
    #'input_vf':'vcc667_sgfit1_masked.fits',  # <---- UPDATE HERE
    'input_vf':'vcc1992_sgfit1_masked.fits',  # <---- UPDATE HERE
    #'input_vf':'NGC0289/sgfit.G3_1.1_crop_trim.fits',  # <---- UPDATE HERE

    #+++++++++++++++++++++++++++++++++
    # input 2D velocity field : VLOS lower/upper limits
    '_vlos_lower':-10000,  # <---- UPDATE HERE in km/s
    '_vlos_upper':10000,  # <---- UPDATE HERE in km/s

    #+++++++++++++++++++++++++++++++++
    # 2D tilted-ring parametres
    'sigma_fitting':'free',  # <---- fixed or free
    'xpos_fitting':'free',  # <---- fixed or free
    'ypos_fitting':'free',  # <---- fixed or free
    'vsys_fitting':'free',  # <---- fixed or free
    'pa_fitting':'free',  # <---- fixed or free
    'incl_fitting':'free',  # <---- fixed or free
    'vrot_fitting':'free',  # <---- fixed or free
    'vrad_fitting':'fixed',  # <---- fixed or free

    'r_galaxy_plane_s':0, # : used for PA normalisation (0 ~ 1)
    'r_galaxy_plane_e':20, # : used for PA normalisation (0 ~ 1)
    'ring_w':4,
    'nrings':100,
    'nrings_reliable':10,
    'nrings_intp':50,

    'free_angle':10,
    'ri':0,
    'ro':1000,
    'cosine_weight_power':1, # 0: uniform,  1: |cosine(theta)| or 2: cos(theta)^2

    # 1D tilted-ring fit param bounds 
    'sigma_bounds_width':50, # bound width for SIGMA in km/s
    'xpos_bounds_width':2, # bound width for XPOS in pixels
    'ypos_bounds_width':2, # bound width for YPOS in pixels
    'vsys_bounds_width':10, # bound width for VSYS in km/s
    'pa_bounds_width':80, # bound width for PA in degre
    'incl_bounds_width':20, # bound width for INCL in degree
    'vrot_bounds_width':100, # bound width for VROT in km/s
    'vrad_bounds_width':20, # bound width for VRAD in km/s

    # ----------------------------------
    # PA-BS
    'n_pa_bs_knots_inner':0,
    'k_pa_bs':3, # 1:linear, 2:quadractic, 3:cubic

    'n_coeffs_pa_bs':999,
    'pa_bs_min':0, # min PA value over the radii : used for PA normalisation (0 ~ 1)
    'pa_bs_max':360, # min PA value over the radii : used for PA normalisation (0 ~ 1)
    'pa_bs_bounds':20, # pa_bs bounds in deg

    # ----------------------------------
    # INCL-BS
    'n_incl_bs_knots_inner':0,
    'k_incl_bs':0, # 1:linear, 2:quadractic, 3:cubic

    'n_coeffs_incl_bs':999,
    'incl_bs_min':0, # min PA value over the radii : used for INCL normalisation (0 ~ 1)
    'incl_bs_max':90, # min PA value over the radii : used for INCL normalisation (0 ~ 1)
    'incl_bs_bounds':50, # incl_bs bounds in deg

    # ----------------------------------
    # VROT-BS
    'n_vrot_bs_knots_inner':4,
    'k_vrot_bs':1, # 1:linear, 2:quadractic, 3:cubic

    'n_coeffs_vrot_bs':999,
    'vrot_bs_bounds':100, # vrot_bs bounds in km/s

    # ----------------------------------
    # VRAD-BS
    'n_vrad_bs_knots_inner':0,
    'k_vrad_bs':0, # 1:linear, 2:quadractic, 3:cubic

    'n_coeffs_vrad_bs':999,
    'vrad_bs_bounds':50, # vrot_bs bounds in km/s

    #+++++++++++++++++++++++++++++++++
    # init
    'naxis1':0,
    'naxis2':0,
    'naxis3':0,
    'cdelt1':0,
    'cdelt2':0,
    'cdelt3':0,
    'vel_min':0, # km/s
    'vel_max':0, # km/s
    '_rms_med':0,
    '_bg_med':0,

    '_xc_el':0,
    '_yc_el':0,
    '_vsys_el':0,
    '_theta_el':0,
    '_i_el':0,
    '_r_max_el':0,

    '_sigma_init':10,
    '_xpos_init':20,
    '_ypos_init':20,
    '_vsys_init':173,
    '_pa_init':45,
    '_incl_init':50,
    '_vrot_init':30,
    '_vrad_init':0,
    #+++++++++++++++++++++++++++++++++

    # grids
    'x_grid_2d':1, # <---- UPDATE HERE
    'y_grid_2d':1, # <---- UPDATE HERE
    'x_grid_tr':1, # <---- UPDATE HERE
    'y_grid_tr':1, # <---- UPDATE HERE


    #  ______________________________________________________  #
    # [______________________________________________________] #
    # [ dynesty parameters
    # _______________________________________________________  #
    '_dynesty_class_2d':'static', # (recommended)
    #'_dynesty_class_2d':'dynamic', # (dynamic): posterior only

    '_dynesty_class_tr':'static', # (recommended)
    #'_dynesty_class_tr':'dynamic', # (dynamic): posterior only

    'print_progress_tr':'',
    'print_progress_2d':'True',
    'nlive':100,
    #'sample':'auto', # or auto
    #'sample':'unif', # or auto
    'sample':'rwalk', # or auto
    #'sample':'slice', # or auto
    #'sample':'rslice', # or auto
    #'sample':'hslice', # or auto
    'dlogz_tr':10000000.01,
    'dlogz_2d':0.01,
    'maxiter_tr':999999,
    'maxiter':999999,
    'maxcall':999999,
    'update_interval':1000, # ===> 2.0 x nlilve
    'vol_dec':0.2,
    'vol_check':2,
    'facc':0.5,
    'fmove':0.5,
    'walks':5,
    'max_move':100,
    #'bound':'single', # or 'single' for unimodal  :: if it complains about sampling efficiency, 'multi' is recommended..
    #'bound':'cubes', # or 'single' for unimodal  :: if it complains about sampling efficiency, 'multi' is recommended..
    'bound':'single', # or 'single' for unimodal  :: if it complains about sampling efficiency, 'multi' is recommended..
    #'bound':'none', # or 'single' for unimodal  :: if it complains about sampling efficiency, 'multi' is recommended..


    'first_tr_run':999, # first TR run?

    #  ______________________________________________________  #
    # [______________________________________________________] #
    # [ priors
    # _______________________________________________________  #
    # parameters for the first 2nd gaussian fit : put a wide x range
    # x
    'x_lowerbound_gfit':0.1,
    'x_upperbound_gfit':0.9,
    # _______________________________________________________  #
    # parameters for the current and subsequent N-Gaussian fits
    'nsigma_prior_range_gfit':3.0, # for the N-gaussians in the N+1 gaussian fit
    # sigma
    'sigma_prior_lowerbound_factor':0.2,
    'sigma_prior_upperbound_factor':2.0,
    # bg
    'bg_prior_lowerbound_factor':0.2,
    'bg_prior_upperbound_factor':2.0,
    # x
    'x_prior_lowerbound_factor':5.0, # --> X_min - lowerbound_factor * STD_max
    'x_prior_upperbound_factor':5.0, # --> X_max + upperbound_factor * STD_max
    # std
    'std_prior_lowerbound_factor':0.1,
    'std_prior_upperbound_factor':3.0,
    # p
    'p_prior_lowerbound_factor':0.05,
    'p_prior_upperbound_factor':1.0,

    #  ______________________________________________________  #
    # [______________________________________________________] #
    # parallellisation parameters
    'ncolm_per_core':'',
    'nsegments_nax2':'',
    'num_cpus_tr_ray':2,  # <---- UPDATE HERE
    'num_cpus_tr_dyn':4,  # <---- UPDATE HERE
    'num_cpus_2d_dyn':8,  # <---- UPDATE HERE
    }
    
    return _params

#-- END OF SUB-ROUTINE____________________________________#


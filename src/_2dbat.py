#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#|-----------------------------------------|
#| _2dbat.py
#|-----------------------------------------|
#| 2024. Jan. 24
#|-----------------------------------------|
#| by Se-Heon Oh
#| Dept. of Physics and Astronomy
#| Sejong University, Seoul, South Korea
#|-----------------------------------------|


#|-----------------------------------------|
# Python 3 compatability
from __future__ import division, print_function

#|-----------------------------------------|
# system functions
import time, sys, os
from datetime import datetime

#|-----------------------------------------|
# python packages
import numpy as np
from numpy import array
import psutil
from multiprocessing import cpu_count

import fitsio

from scipy.interpolate import BSpline, splrep, splev, PPoly
from scipy import optimize
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

#|-----------------------------------------|
# import ray
import ray

#|-----------------------------------------|
# load 2dbat-PI modules
#|-----------------------------------------|
# _params.py
from _2dbat_params import default_params, read_configfile
global _x, _params, r_galaxy_plane, _wi_2d

#|-----------------------------------------|
# _dynesty_sampler.py
from _nested_sampler import run_dynesty_sampler_uniform_priors
from _nested_sampler import run_dynesty_sampler_optimal_priors
from _nested_sampler import derive_rms_npoints
from _nested_sampler import make_vlos_model_vf_given_dyn_params
from _nested_sampler import make_vlos_model_vf_given_dyn_params_trfit_final_vrot
from _nested_sampler import extract_vrot_bs_tr_rings_given_dyn_params

from _nested_sampler import run_nested_sampler_trfit_2d, derive_vlos_model_2d, bspline_ncoeffs_tck, set_phi_bounds

from _nested_sampler import write_fits_images, evaluate_constant_spline_vectorized
from _nested_sampler import find_maximum_radius
from _nested_sampler import extract_tr2dfit_params

#|-----------------------------------------|
# _fits_io.py
from _fits_io import read_datacube, moment_analysis, estimate_init_values, find_area_tofit, trfit_ring_by_ring, bspline_fit_to_1d, trfit_2d, derive_rad_profiles, set_params_fit_option
from _fits_io import min_val, max_val, set_vrot_bs_coefficients_bounds, set_pa_bs_coefficients_bounds, set_incl_bs_coefficients_bounds, ellipsefit_2dpoints, trfit_ring_by_ring_final

from _plot import print_2dbat, write_2dbat_t1, write_2dbat_t2, plot_write_2dbat_all
#|-----------------------------------------|
# import make_dirs
from _dirs_files import make_dirs

from operator import mul

import logging
#|-----------------------------------------|

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def main():
    # read the input datacube
    start = datetime.now()
    logging.basicConfig(level=logging.ERROR)

    if len(sys.argv) == 3:
        print("")
        print("")
        print(91*"_")
        print(91*"")
        print(" :: 2dbat.py usage ::")
        print(91*"")
        print(" usage-1: running 2dbat.py with 2dbat_params file")
        print(" > python3 2dbat.py [ARG1: _2dbat_params.yaml] [ARG2: running-number]")
        print(" e.g.,")
        print(" > python3 2dbat.py _2dbat_ngc5194.yaml 1")
        print("")
        print("")
        configfile = sys.argv[1]
        _2dbat_run_i = int(sys.argv[2])
        _2dbat_run_i_pre = 999
        _params = read_configfile(configfile)

    elif len(sys.argv) == 4:
        print("")
        print("")
        print(91*"_")
        print(91*"")
        print(" :: 2dbat.py usage ::")
        print(91*"")
        print(" usage-2: running 2dbat.py with 2dbat_params file")
        print(" > python3 2dbat.py [ARG1: _2dbat_params.yaml] [ARG2: running-number] [ARG3: pre-running-number]")
        print(" e.g.,")
        print(" > python3 2dbat.py _2dbat_ngc5194.yaml 2 1")
        print("")
        print("")
        configfile = sys.argv[1]
        _2dbat_run_i = int(sys.argv[2])
        _2dbat_run_i_pre = int(sys.argv[3])
        _params = read_configfile(configfile)
    
    else:
        print(" > python3 2dbat.py [ARG1: _2dbat_params.yaml] [ARG2: running-number]")
        print(" e.g.,")
        print(" > python3 2dbat.py _2dbat_ngc5194.yaml 1")


    n_cpus = int(_params['num_cpus_tr_ray'])
    ray.init(num_cpus=n_cpus, logging_level=40)
    num_cpus = psutil.cpu_count(logical=False)


    # 1.
    #----------------------------------------------------------
    #----------------------------------------------------------
    # Find the largest area of the input velocity field to fit
    # --> _params['_vf_area_tofit']
    #_input_vf_tofit_grid1, _input_vf_tofit_tr, _input_vf_tofit_2d, _tr_model_vf, _input_int_w, _input_vdisp, _input_int_vdisp_w, _input_int_vdisp_w_50percentile, _input_int, combined_vfe_res_w = find_area_tofit(_params, _2dbat_run_i, _2dbat_run_i_pre)
    _input_vf_tofit_grid1, _input_vf_tofit_tr, _input_vf_tofit_2d, _tr_model_vf, _input_int_w, _input_vdisp, _input_int_vdisp_w, _input_int_vdisp_w_50percentile, _input_int, combined_vfe_res_w = find_area_tofit(_params, _2dbat_run_i, _2dbat_run_i_pre, 1)

    # ----------------------------------------
    # rough guess of initial ellipse params for ellipse fit
    _xpos_t = _params['naxis1'] / 2.
    _ypos_t = _params['naxis2'] / 2.
    _pa_t = 45 
    _incl_t = 45
    ri = 0
    ro = _params['naxis1'] 
    side = 0 # both side
    # ----------------------------------------
    _naxis1 = _params['naxis1']
    _naxis2 = _params['naxis2']
    _wi_2d = np.full((_naxis2, _naxis1), fill_value=np.nan, dtype=np.float64)
    _r_galaxy_plane_2d = np.full((_naxis2, _naxis1), fill_value=np.nan, dtype=np.float64)

    _xpos_el, _ypos_el, _vsys_el, _pa_el, _incl_el, _r_max_el = ellipsefit_2dpoints(_input_vf_tofit_grid1, _input_int_vdisp_w_50percentile, _wi_2d, _params, _xpos_t, _ypos_t, _pa_t, _incl_t, ri, ro, side)

    _xpos_t = _xpos_el
    _ypos_t = _ypos_el
    _pa_t = _pa_el
    _incl_t = _incl_el


    _xpos_el, _ypos_el, _vsys_el, _pa_el, _incl_el, _r_max_el = ellipsefit_2dpoints(_input_vf_tofit_grid1, _input_int_vdisp_w_50percentile, _wi_2d, _params, _xpos_t, _ypos_t, _pa_t, _incl_t, ri, ro, side)

    _params['_xpos_el'] = _xpos_el
    _params['_ypos_el'] = _ypos_el
    _params['_vsys_el'] = _vsys_el
    _params['_pa_el'] = _pa_el
    _params['_incl_el'] = _incl_el
    _params['_r_max_el'] = _r_max_el

    _params['r_galaxy_plane_e_geo'], _wt_2d_geo = find_maximum_radius(_input_vf_tofit_grid1, _xpos_t, _ypos_t, _pa_t, _incl_t, ri, _r_max_el*2, 1, _params, _2dbat_run_i)
    _params['r_galaxy_plane_e'] = _params['r_galaxy_plane_e_geo']* 1.0 


    # -----------------------------------------------------------------
    # -----------------------------------------------------------------
    n_coeffs_pa_bs, tck_pa_bs = bspline_ncoeffs_tck(_params, 'pa', _params['nrings_intp'])
    n_coeffs_incl_bs, tck_incl_bs = bspline_ncoeffs_tck(_params, 'incl', _params['nrings_intp'])
    n_coeffs_vrot_bs, tck_vrot_bs = bspline_ncoeffs_tck(_params, 'vrot', _params['nrings_intp'])
    _params['n_coeffs_pa_bs'] = n_coeffs_pa_bs
    _params['n_coeffs_incl_bs'] = n_coeffs_incl_bs
    _params['n_coeffs_vrot_bs'] = n_coeffs_vrot_bs

    n_coeffs_vrad_bs = 0
    tr_params_bounds = np.zeros((8 + n_coeffs_pa_bs + n_coeffs_incl_bs + n_coeffs_vrot_bs + n_coeffs_vrad_bs, 2))
    nrings_reliable = int(_params['r_galaxy_plane_e'] / _params['ring_w']) + 1
    _params['nrings_reliable'] = nrings_reliable

    # -----------------------------------------------------------------
    # -----------------------------------------------------------------
    if n_coeffs_pa_bs != 0: # not constant

        for _nbs in range(0, n_coeffs_pa_bs):
            # ellipse geo bounds
            tr_params_bounds[7+1+_nbs, 0] = _params['_pa_el'] - _params['pa_bounds_width'] 
            tr_params_bounds[7+1+_nbs, 1] = _params['_pa_el'] + _params['pa_bounds_width'] 
#            print(_pa_bs_nbs_l, _pa_bs_nbs_u, (_pa_bs_nbs_l + _pa_bs_nbs_u)/2.)

    if n_coeffs_incl_bs != 0: # not constant

        for _nbs in range(0, n_coeffs_incl_bs):

            tr_params_bounds[7+1+n_coeffs_pa_bs+_nbs, 0] = _params['_incl_el'] - _params['incl_bounds_width']
            tr_params_bounds[7+1+n_coeffs_pa_bs+_nbs, 1] = _params['_incl_el'] + _params['incl_bounds_width']

            if tr_params_bounds[7+1+n_coeffs_pa_bs+_nbs, 0] < 0: tr_params_bounds[7+1+n_coeffs_pa_bs+_nbs, 0] = 0
            if tr_params_bounds[7+1+n_coeffs_pa_bs+_nbs, 1] > 89: tr_params_bounds[7+1+n_coeffs_pa_bs+_nbs, 1] = 89

        # dummy prior for the first coefficient of INCL : this will be overwritten with the value of the 2nd coefficient in log-like function
        tr_params_bounds[7+1+n_coeffs_pa_bs+0, 0] = _params['_incl_el']
        tr_params_bounds[7+1+n_coeffs_pa_bs+0, 1] = _params['_incl_el'] + 0.1


    # bounds
    tr_params_bounds[0, 0] = 0 # sigma0
    tr_params_bounds[0, 1] = 0 + _params['sigma_bounds_width'] # sigma1 
    if tr_params_bounds[0, 0] < 0: tr_params_bounds[0, 0] = 0

    # FOR 2D FIT : strong centre constraints
    tr_params_bounds[1, 0] = _params['_xpos_el'] - _params['ring_w']*1.0
    tr_params_bounds[1, 1] = _params['_xpos_el'] + _params['ring_w']*1.0
    tr_params_bounds[2, 0] = _params['_ypos_el'] - _params['ring_w']*1.0
    tr_params_bounds[2, 1] = _params['_ypos_el'] + _params['ring_w']*1.0


    tr_params_bounds[3, 0] = _params['_vsys_el'] - _params['vsys_bounds_width']
    tr_params_bounds[3, 1] = _params['_vsys_el'] + _params['vsys_bounds_width']

    tr_params_bounds[4, 0] = _params['_pa_el'] - _params['pa_bounds_width']
    tr_params_bounds[4, 1] = _params['_pa_el'] + _params['pa_bounds_width']

    tr_params_bounds[5, 0] = _params['_incl_el'] - _params['incl_bounds_width']
    tr_params_bounds[5, 1] = _params['_incl_el'] + _params['incl_bounds_width']
    if tr_params_bounds[5, 0] < 0: tr_params_bounds[5, 0] = 0
    if tr_params_bounds[5, 1] > 89: tr_params_bounds[5, 1] = 89

    tr_params_bounds[6, 0] = _params['vrot0_bs']
    tr_params_bounds[6, 1] = _params['vrot1_bs']
    if tr_params_bounds[6, 0] < 0: tr_params_bounds[6, 0] = 0

    tr_params_bounds[7, 0] = 0 - _params['vrad_bounds_width']
    tr_params_bounds[7, 1] = 0 + _params['vrad_bounds_width']
    # -----------------------------------------------------------------



    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # PA bspline coefficients : INTERPOLATED SPLREP
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # ----------------------------------------
    # ------------------------------------------------------
    # PA - bspline : calculate the number of pa coefficients and generate a dummy array for pa
    # derive _pa_bs coeffs via splrep fitting : this is for PA-BS coeffs bounds
    xs_bs = _params['r_galaxy_plane_s']

    # ------------------------------------------------------
    # inverpolation version
    # ------------------------------------------------------
    _ring_t = np.zeros(nrings_reliable*2, dtype=np.float64)

    _ring_t = np.zeros(nrings_reliable*2, dtype=np.float64)
    _sigma_t = np.zeros(nrings_reliable*2, dtype=np.float64)
    _xpos_t = np.zeros(nrings_reliable*2, dtype=np.float64)
    _ypos_t = np.zeros(nrings_reliable*2, dtype=np.float64)
    _vsys_t = np.zeros(nrings_reliable*2, dtype=np.float64)
    _pa_t = np.zeros(nrings_reliable*2, dtype=np.float64)
    _incl_t = np.zeros(nrings_reliable*2, dtype=np.float64)
    _vrot_t = np.zeros(nrings_reliable*2, dtype=np.float64)
    _vrad_t = np.zeros(nrings_reliable*2, dtype=np.float64)

    _sigma_et = np.zeros(nrings_reliable*2, dtype=np.float64)
    _xpos_et = np.zeros(nrings_reliable*2, dtype=np.float64)
    _ypos_et = np.zeros(nrings_reliable*2, dtype=np.float64)
    _vsys_et = np.zeros(nrings_reliable*2, dtype=np.float64)
    _pa_et = np.zeros(nrings_reliable*2, dtype=np.float64)
    _incl_et = np.zeros(nrings_reliable*2, dtype=np.float64)
    _vrot_et = np.zeros(nrings_reliable*2, dtype=np.float64)
    _vrad_et = np.zeros(nrings_reliable*2, dtype=np.float64)


    _ring_t[nrings_reliable-1] = _params['r_galaxy_plane_e'] 

    # generate 1D interpolation function
    scipy_interp1d = interp1d(_ring_t[:nrings_reliable], _pa_t[:nrings_reliable], kind='linear')

    # generate fine gaps
    # _ring_t[0] <-- 0
    # _ring_t[nrings_reliable] <-- _params['r_galaxy_plane_e'] 
    r_fine = np.linspace(_ring_t[0], _ring_t[nrings_reliable-1], _params['nrings_intp'], endpoint=True)

    # do interpolation
    pa_fine = scipy_interp1d(r_fine)

    #xs_bs = _ring_t[0]
    xs_bs = _params['r_galaxy_plane_s']
    xe_bs = _params['r_galaxy_plane_e']
    #n_coeffs_pa_bs, tck_pa_bs = bspline_ncoeffs_tck(_params, 'pa', nrings_reliable)
    n_coeffs_pa_bs, tck_pa_bs = bspline_ncoeffs_tck(_params, 'pa', _params['nrings_intp'])
    # PA-BS coefficients
    # number of inner knots
    if n_coeffs_pa_bs != 0: # not constant
        n_knots_inner = _params['n_pa_bs_knots_inner'] # 0, 1, 2, ...
        k_bs = _params['k_pa_bs'] # 0, 1, 2, ...
        pa_bs_knots_inner = np.linspace(xs_bs, xe_bs, (2+n_knots_inner))[1:-1]
        tck_pa_bs_init_from_trfit = splrep(r_fine, pa_fine, t=pa_bs_knots_inner, k=k_bs)
    else:
        n_knots_inner = 0 # dummy value
        k_bs = 1 # dummy value
        pa_bs_knots_inner = np.linspace(xs_bs, xe_bs, (2+n_knots_inner))[1:-1]
        tck_pa_bs_init_from_trfit = splrep(r_fine, pa_fine, t=pa_bs_knots_inner, k=k_bs)


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # INCL bspline coefficients : INTERPOLATED SPLREP
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # ------------------------------------------------------
    # inverpolation version
    # ------------------------------------------------------
    _ring_t[nrings_reliable-1] = _params['r_galaxy_plane_e'] 

    # generate 1D interpolation function
    scipy_interp1d = interp1d(_ring_t[:nrings_reliable], _incl_t[:nrings_reliable], kind='linear')  # 선형 보간

    # generate fine gaps
    r_fine = np.linspace(_ring_t[0], _ring_t[nrings_reliable-1], _params['nrings_intp'], endpoint=True)

    # do interpolation
    incl_fine = scipy_interp1d(r_fine)

    #xs_bs = _ring_t[0]
    xs_bs = _params['r_galaxy_plane_s']
    xe_bs = _params['r_galaxy_plane_e']
    n_coeffs_incl_bs, tck_incl_bs = bspline_ncoeffs_tck(_params, 'incl', _params['nrings_intp'])
    # INCL-BS coefficients
    # number of inner knots
    if n_coeffs_incl_bs != 0: # not constant
        n_knots_inner = _params['n_incl_bs_knots_inner'] # 0, 1, 2, ...
        k_bs = _params['k_incl_bs'] # 0, 1, 2, ...
        incl_bs_knots_inner = np.linspace(xs_bs, xe_bs, (2+n_knots_inner))[1:-1]
        tck_incl_bs_init_from_trfit = splrep(r_fine, incl_fine, t=incl_bs_knots_inner, k=k_bs)
    else:
        n_knots_inner = 0 # dummy value
        k_bs = 1 # dummy value
        incl_bs_knots_inner = np.linspace(xs_bs, xe_bs, (2+n_knots_inner))[1:-1]
        tck_incl_bs_init_from_trfit = splrep(r_fine, incl_fine, t=incl_bs_knots_inner, k=k_bs)

    if n_coeffs_incl_bs != 0: # not constant
        del_incl = _params['incl_bs_bounds']
        tck_incl_bs_bound_l, tck_incl_bs_bound_u = set_incl_bs_coefficients_bounds(_incl_t, del_incl, _params, _ring_t)

        for _nbs in range(0, n_coeffs_incl_bs):

            tr_params_bounds[7+1+n_coeffs_pa_bs+_nbs, 0] = _params['_incl_el'] - _params['incl_bounds_width']
            tr_params_bounds[7+1+n_coeffs_pa_bs+_nbs, 1] = _params['_incl_el'] + _params['incl_bounds_width']

            if tr_params_bounds[7+1+n_coeffs_pa_bs+_nbs, 0] < 0: tr_params_bounds[7+1+n_coeffs_pa_bs+_nbs, 0] = 0
            if tr_params_bounds[7+1+n_coeffs_pa_bs+_nbs, 1] > 89: tr_params_bounds[7+1+n_coeffs_pa_bs+_nbs, 1] = 89

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # VROT bspline coefficients : INTERPOLATED SPLREP
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # -----------------------------------------------------------------
    # BS coeffs boundaries if b-spline regularization for PA, INCL, and VROT is applied
    if n_coeffs_vrot_bs != 0: # not constant

        for _nbs in range(0, n_coeffs_vrot_bs):

            tr_params_bounds[7+1+n_coeffs_pa_bs+n_coeffs_incl_bs+_nbs, 0] = _params['vrot0_bs']
            tr_params_bounds[7+1+n_coeffs_pa_bs+n_coeffs_incl_bs+_nbs, 1] = _params['vrot1_bs']

    # -----------------------------------------------------------------
    # inverpolation version
    # ------------------------------------------------------
    _ring_t[nrings_reliable-1] = _params['r_galaxy_plane_e'] 
    #_vrot_init_t[nrings_reliable] = _vrot_init_t[nrings_reliable-1]

    # generate 1D interpolation function
    scipy_interp1d = interp1d(_ring_t[:nrings_reliable], _vrot_t[:nrings_reliable], kind='linear')  # 선형 보간

    # generate fine gaps
    r_fine = np.linspace(_ring_t[0], _ring_t[nrings_reliable-1], _params['nrings_intp'], endpoint=True)

    # do interpolation
    vrot_fine = scipy_interp1d(r_fine)

    #xs_bs = _ring_t[0]
    xs_bs = _params['r_galaxy_plane_s']
    xe_bs = _params['r_galaxy_plane_e']
    n_coeffs_vrot_bs, tck_vrot_bs = bspline_ncoeffs_tck(_params, 'vrot', _params['nrings_intp'])
    # VROT-BS coefficients
    # number of inner knots
    if n_coeffs_vrot_bs != 0: # not constant
        n_knots_inner = _params['n_vrot_bs_knots_inner'] # 0, 1, 2, ...
        k_bs = _params['k_vrot_bs'] # 0, 1, 2, ...
        vrot_bs_knots_inner = np.linspace(xs_bs, xe_bs, (2+n_knots_inner))[1:-1]
        tck_vrot_bs_init_from_trfit = splrep(r_fine, vrot_fine, t=vrot_bs_knots_inner, k=k_bs)

    else:
        n_knots_inner = 0 # dummy value
        k_bs = 1 # dummy value
        vrot_bs_knots_inner = np.linspace(xs_bs, xe_bs, (2+n_knots_inner))[1:-1]
        tck_vrot_bs_init_from_trfit = splrep(r_fine, vrot_fine, t=vrot_bs_knots_inner, k=k_bs)



# -------------------------------------------------------------------------
    # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    # -------------------------------------------------------------------------
    # 4. ----------------------------------
    # r: 0.01_r_geo ~ 1.0_r_geo

    fit_opt, ndim, tr_params_priors_init = set_params_fit_option(_params, \
                          'sigma', 'free', _params['_sigma_sectors_med'], \
                         # 'sigma', 'fixed', 0.976, \
                        #  'xpos', 'fixed', 233, \
                        #  'ypos', 'fixed', 239, \
                        #  'vsys', 'free', 1566, \
                         # 'xpos', 'fixed', 101, \
                         # 'ypos', 'fixed', 101, \
                         # 'vsys', 'free', 2198, \
                        #  'xpos', 'fixed', 20, \
                        #  'ypos', 'fixed', 20, \
                        #  'vsys', 'free', 0, \
                          'xpos', 'free', _params['_xpos_el'], \
                          'ypos', 'free', _params['_ypos_el'], \
                          'vsys', 'free', _params['_vsys_el'], \
                          'pa', 'free', _params['_pa_el'], \
                          'incl', 'free', _params['_incl_el'], \
                          #'incl', 'free', 60, \
                          'vrot', 'free', 20, \
                          'vrad', 'fixed', 0, nrings_reliable, 'False')

    _ri = _params['r_galaxy_plane_s']
    _ro = _params['r_galaxy_plane_e']*1.0
    side = 0


    _ij_area_tofit, _tr2dfit_results, _n_dim, fit_opt_2d, std_resample_run  = trfit_2d(_input_vf_tofit_2d, combined_vfe_res_w, \
                                                                                       _ri, _ro, \
                                                                                       _tr_model_vf, \
                                                                                       _input_int_w, \
                                                                                       _input_vdisp, \
                                                                                       _wt_2d_geo, \
                                                                                       _params, \
                                                                                       tr_params_bounds, \
                                                                                       _r_galaxy_plane_2d, \
                                                                                       tck_vrot_bs_init_from_trfit, \
                                                                                       tck_pa_bs_init_from_trfit, \
                                                                                       tck_incl_bs_init_from_trfit, \
                                                                                       _2dbat_run_i, \
                                                                                       'entire')


    _vrot_bs, _vrot_bs_e = extract_vrot_bs_tr_rings_given_dyn_params(_input_vf_tofit_grid1, _input_vf_tofit_2d, _tr2dfit_results, _params, fit_opt_2d, _tr_model_vf, _2dbat_run_i)
    bsfit_vf = make_vlos_model_vf_given_dyn_params(_input_vf_tofit_grid1, _input_vf_tofit_2d, _tr2dfit_results, _params, fit_opt_2d, _tr_model_vf, _2dbat_run_i)


    # The ring params derived from the 2dfit will be also collected in trfit_ring_by_ring_final below
    # This is just for setting up fitting options which are not used for the trfit below 
    _sigma_2dfit, _sigma_e_2dfit, _xpos_2dfit, _xpos_e_2dfit, _ypos_2dfit, _ypos_e_2dfit, _vsys_2dfit, _vsys_e_2dfit, _pa_2dfit, _pa_e_2dfit, _incl_2dfit, _incl_e_2dfit, _vrot_2dfit, _vrot_e_2dfit, _vrad_2dfit, _vrad_e_2dfit \
        = extract_tr2dfit_params(_tr2dfit_results, _params, fit_opt_2d, _ro, 'entire')

    fit_opt, ndim, tr_params_priors_init = set_params_fit_option(_params, \
                          'sigma', 'free', _sigma_2dfit, \
                          'xpos', 'fixed', _xpos_2dfit, \
                          'ypos', 'fixed', _ypos_2dfit, \
                          'vsys', 'fixed', _vsys_2dfit, \
                          'pa', 'fixed', _pa_2dfit, \
                          'incl', 'fixed', _incl_2dfit, \
                          'vrot', 'free', _vrot_2dfit, \
                          'vrad', 'fixed', 0, nrings_reliable, 'False')
    

    # ----------------------------------------
    # ----------------------------------------
    # 5. TRFIT given the rings params from 2D fit : BOTH SIDES
    # ----------------------------------------
    _2dbat_trfit_final_b = trfit_ring_by_ring_final(_input_vf_tofit_tr, _tr_model_vf, _wi_2d, _params, fit_opt, fit_opt_2d, ndim, tr_params_priors_init, _tr2dfit_results, 0, 'True') # both sides
    nrings_reliable_b =int(_2dbat_trfit_final_b[19])
    _params['nrings_reliable'] = nrings_reliable_b

    write_2dbat_t1(_2dbat_trfit_final_b, nrings_reliable_b, _params, _2dbat_run_i, '_2dbat_trfit_temp.txt')
    print("")
    print('-'*50)
    print("+ BOTH SIDES")
    print('-'*50)
    print(tr_params_priors_init)
    print_2dbat(_params, _2dbat_run_i, '_2dbat_trfit_temp.txt')

    trfit_vf = make_vlos_model_vf_given_dyn_params_trfit_final_vrot(_input_vf_tofit_grid1, _input_vf_tofit_2d, _tr2dfit_results, \
                                                                    _2dbat_trfit_final_b[2], \
                                                                    _2dbat_trfit_final_b[3], \
                                                                    _2dbat_trfit_final_b[4], \
                                                                    _2dbat_trfit_final_b[5], \
                                                                    _2dbat_trfit_final_b[6], \
                                                                    _2dbat_trfit_final_b[7], \
                                                                    _2dbat_trfit_final_b[8], \
                                                                    _params, fit_opt_2d, _tr_model_vf, _2dbat_run_i)

    # ----------------------------------------
    # ----------------------------------------
    # 6. TRFIT given the rings params from 2D fit : APPROACHING SIDES
    # ----------------------------------------
    _2dbat_trfit_final_a = trfit_ring_by_ring_final(_input_vf_tofit_tr, _tr_model_vf, _wi_2d, _params, fit_opt, fit_opt_2d, ndim, tr_params_priors_init, _tr2dfit_results, -1, 'True') # receding side
    nrings_reliable_a =int(_2dbat_trfit_final_a[19])
    _params['nrings_reliable'] = nrings_reliable_a

    write_2dbat_t1(_2dbat_trfit_final_a, nrings_reliable_a, _params, _2dbat_run_i, '_2dbat_trfit_temp.txt')
    print("")
    print('-'*50)
    print("+ APPROACHING SIDE")
    print('-'*50)
    print(tr_params_priors_init)
    print_2dbat(_params, _2dbat_run_i, '_2dbat_trfit_temp.txt')




    # ----------------------------------------
    # ----------------------------------------
    # 7. TRFIT given the rings params from 2D fit : RECEDING SIDE 
    # ----------------------------------------
    _2dbat_trfit_final_r = trfit_ring_by_ring_final(_input_vf_tofit_tr, _tr_model_vf, _wi_2d, _params, fit_opt, fit_opt_2d, ndim, tr_params_priors_init, _tr2dfit_results, 1, 'True') # receding side
    nrings_reliable_r =int(_2dbat_trfit_final_r[19])
    _params['nrings_reliable'] = nrings_reliable_r

    write_2dbat_t1(_2dbat_trfit_final_r, nrings_reliable_r, _params, _2dbat_run_i, '_2dbat_trfit_temp.txt')
    print("")
    print('-'*50)
    print("+ RECEDING SIDE")
    print('-'*50)
    print(tr_params_priors_init)
    print_2dbat(_params, _2dbat_run_i, '_2dbat_trfit_temp.txt')


    # ----------------------------------------
    # ----------------------------------------
    # 2DBAT output directory.RUNNING_NUMBER
    # ----------------------------------------
    _dir_2dbat_PI_output = _params['wdir'] + '/' + _params['_2dbatdir'] + ".%d" % _2dbat_run_i
    # If not present
    if not os.path.exists("%s" % _dir_2dbat_PI_output):
        make_dirs("%s" % _dir_2dbat_PI_output)

    write_2dbat_t2(_vrot_bs, _vrot_bs_e, _2dbat_trfit_final_b, nrings_reliable_b, _2dbat_trfit_final_a, nrings_reliable_a, _2dbat_trfit_final_r, nrings_reliable_r, _params, _2dbat_run_i, '_2dbat_trfit_temp.txt')
    plot_write_2dbat_all('_2dbat_trfit_temp.txt', _input_vf_tofit_grid1, bsfit_vf, trfit_vf, _dir_2dbat_PI_output+'/'+'2dbat_plot.png', _params, _2dbat_run_i, '2dbat_trfit_results.txt')

    print("")
    print('-'*50)
    print("+ execution time =", datetime.now() - start)
    print("+ ...completed...")
    print('-'*50)
    print("")

    print("duration =", datetime.now() - start)

    ray.shutdown()
    sys.exit()
#-- END OF SUB-ROUTINE____________________________________________________________#

if __name__ == '__main__':
    main()


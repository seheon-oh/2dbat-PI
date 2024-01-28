


from __future__ import division, print_function

import time, sys, os
from datetime import datetime

import numpy as np
from numpy import array
import psutil
from multiprocessing import cpu_count

import fitsio

from scipy.interpolate import BSpline, splrep, splev, PPoly
from scipy import optimize
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

import ray

from _2dbat_params import default_params, read_configfile
global _x, _params, r_galaxy_plane, _wi_2d

from _nested_sampler import run_dynesty_sampler_uniform_priors
from _nested_sampler import run_dynesty_sampler_optimal_priors
from _nested_sampler import derive_rms_npoints
from _nested_sampler import make_vlos_model_vf_given_dyn_params
from _nested_sampler import make_vlos_model_vf_given_dyn_params_trfit_final_vrot
from _nested_sampler import extract_vrot_bs_tr_rings_given_dyn_params

from _nested_sampler import run_nested_sampler_trfit_2d, derive_vlos_model_2d, bspline_ncoeffs_tck, set_phi_bounds

from _nested_sampler import write_fits_images

from _fits_io import read_datacube, moment_analysis, estimate_init_values, find_area_tofit, trfit_ring_by_ring, bspline_fit_to_1d, trfit_2d, derive_rad_profiles, set_params_fit_option
from _fits_io import min_val, max_val, set_vrot_bs_coefficients_bounds, set_pa_bs_coefficients_bounds, set_incl_bs_coefficients_bounds, ellipsefit_2dpoints, trfit_ring_by_ring_final

from _plot import print_2dbat, write_2dbat_t1, write_2dbat_t2, plot_write_2dbat_all
from _dirs_files import make_dirs

from operator import mul

import logging

def main():
    start = datetime.now()
    logging.basicConfig(level=logging.ERROR)

    if len(sys.argv) == 2:
        print("")
        print("")
        print(91*"-")
        print(" Usage: running _2dbat.py with _2dbat_params.py file")
        print(" e.g.,")
        print(" > python3 _2dbat.py [running-number]")
        print(91*"_")
        print("")
        print("")
        _params = default_params()
        _2dbat_run_i = int(sys.argv[1])

    elif len(sys.argv) == 999:
        print("")
        print("")
        print(91*"_")
        print(91*"")
        print(" :: 2dbat.py usage ::")
        print(91*"")
        print(" usage-1: running 2dbat.py with 2dbat_params file")
        print(" > python3 2dbat.py [ARG1: _2dbat_params.txt]")
        print(" e.g.,")
        print(" > python3 2dbat.py _2dbat_params.ngc2403.txt")
        print("")
        print("")
        configfile = sys.argv[1]
        _params = read_configfile(configfile)
    
    else:
        print(" Usage: running _2dbat.py with _2dbat_params.py file")
        print(" e.g.,")
        print(" > python3 _2dbat.py [running-number]")


    n_cpus = int(_params['num_cpus_tr_ray'])
    ray.init(num_cpus=n_cpus, logging_level=40)
    num_cpus = psutil.cpu_count(logical=False)


    _input_vf_tofit_grid1, _input_vf_tofit_tr, _input_vf_tofit_2d, _tr_model_vf = find_area_tofit(_params)

    _xpos_t = _params['naxis1'] / 2.
    _ypos_t = _params['naxis2'] / 2.
    _pa_t = 45
    _incl_t = 45
    ri = 0
    ro = 1000
    side = 0 # both side
    _naxis1 = _params['naxis1']
    _naxis2 = _params['naxis2']
    _wi_2d = np.full((_naxis2, _naxis1), fill_value=0, dtype=np.float64)

    _xc_el, _yc_el, _vsys_init_el, _theta_el, _i_el, _r_max_el = ellipsefit_2dpoints(_input_vf_tofit_grid1, _wi_2d, _params, _xpos_t, _ypos_t, _pa_t, _incl_t, ri, ro, side)
    _params['_xc_el'] = _xc_el
    _params['_yc_el'] = _yc_el
    _params['_vsys_el'] = _vsys_init_el
    _params['_theta_el'] = _theta_el
    _params['_i_el'] = _i_el
    _params['_r_max_el'] = _r_max_el

    _params['r_galaxy_plane_e'] = _r_max_el

    fit_opt, ndim, tr_params_priors_init = set_params_fit_option(_params, 'sigma', 'free', 10, \
                          'xpos', 'free', _xc_el, \
                          'ypos', 'free', _yc_el, \
                          'vsys', 'free', _vsys_init_el, \
                          'pa', 'free', _theta_el, \
                          'incl', 'free', _i_el, \
                          'vrot', 'free', 50, \
                          'vrad', 'fixed', 0, 100, 0)
    

    _ring_t, \
        _sigma_init_t, \
        _xpos_init_t, \
        _ypos_init_t, \
        _vsys_init_t, \
        _pa_init_t, \
        _incl_init_t, \
        _vrot_init_t, \
        _vrad_init_t, \
        _sigma_init_et, \
        _xpos_init_et, \
        _ypos_init_et, \
        _vsys_init_et, \
        _pa_init_et, \
        _incl_init_et, \
        _vrot_init_et, \
        _vrad_init_et, \
        nrings_reliable = trfit_ring_by_ring(_input_vf_tofit_tr, _tr_model_vf, _wi_2d, _params, fit_opt, ndim, tr_params_priors_init, 'False')

    fit_opt, ndim, tr_params_priors_init = set_params_fit_option(_params, 'sigma', 'free', np.nanmedian(_sigma_init_t[1:-1]), \
                          'xpos', 'fixed', np.nanmedian(_xpos_init_t[1:-1]), \
                          'ypos', 'fixed', np.nanmedian(_ypos_init_t[1:-1]), \
                          'vsys', 'fixed', np.nanmedian(_vsys_init_t[1:-1]), \
                          'pa', 'free', np.nanmedian(_pa_init_t[1:-1]), \
                          'incl', 'free', np.nanmedian(_incl_init_t[1:-1]), \
                          'vrot', 'free', np.nanmedian(_vrot_init_t[1:-1]), \
                          'vrad', 'fixed', np.nanmedian(_vrad_init_t[1:-1]), nrings_reliable, 1)
    _ring_t, \
        _sigma_init_t, \
        _xpos_init_t, \
        _ypos_init_t, \
        _vsys_init_t, \
        _pa_init_t, \
        _incl_init_t, \
        _vrot_init_t, \
        _vrad_init_t, \
        _sigma_init_et, \
        _xpos_init_et, \
        _ypos_init_et, \
        _vsys_init_et, \
        _pa_init_et, \
        _incl_init_et, \
        _vrot_init_et, \
        _vrad_init_et, \
        nrings_reliable = trfit_ring_by_ring(_input_vf_tofit_tr, _tr_model_vf, _wi_2d, _params, fit_opt, ndim, tr_params_priors_init, 'False')

    fit_opt, ndim, tr_params_priors_init = set_params_fit_option(_params, 'sigma', 'free', np.nanmedian(_sigma_init_t[1:-1]), \
                          'xpos', 'free', np.nanmedian(_xpos_init_t[1:-1]), \
                          'ypos', 'free', np.nanmedian(_ypos_init_t[1:-1]), \
                          'vsys', 'free', np.nanmedian(_vsys_init_t[1:-1]), \
                          'pa', 'fixed', np.nanmedian(_pa_init_t[1:-1]), \
                          'incl', 'fixed', np.nanmedian(_incl_init_t[1:-1]), \
                          'vrot', 'free', np.nanmedian(_vrot_init_t[1:-1]), \
                          'vrad', 'fixed', np.nanmedian(_vrad_init_t[1:-1]), nrings_reliable, 1)
    _ring_t, \
        _sigma_init_t, \
        _xpos_init_t, \
        _ypos_init_t, \
        _vsys_init_t, \
        _pa_init_t, \
        _incl_init_t, \
        _vrot_init_t, \
        _vrad_init_t, \
        _sigma_init_et, \
        _xpos_init_et, \
        _ypos_init_et, \
        _vsys_init_et, \
        _pa_init_et, \
        _incl_init_et, \
        _vrot_init_et, \
        _vrad_init_et, \
        nrings_reliable = trfit_ring_by_ring(_input_vf_tofit_tr, _tr_model_vf, _wi_2d, _params, fit_opt, ndim, tr_params_priors_init, 'False')

    fit_opt, ndim, tr_params_priors_init = set_params_fit_option(_params, 'sigma', 'free', np.nanmedian(_sigma_init_t[1:-1]), \
                          'xpos', 'fixed', np.nanmedian(_xpos_init_t[1:-1]), \
                          'ypos', 'fixed', np.nanmedian(_ypos_init_t[1:-1]), \
                          'vsys', 'fixed', np.nanmedian(_vsys_init_t[1:-1]), \
                          'pa', 'free', np.nanmedian(_pa_init_t[1:-1]), \
                          'incl', 'free', np.nanmedian(_incl_init_t[1:-1]), \
                          'vrot', 'free', np.nanmedian(_vrot_init_t[1:-1]), \
                          'vrad', 'fixed', np.nanmedian(_vrad_init_t[1:-1]), nrings_reliable, 1)
    _ring_t, \
        _sigma_init_t, \
        _xpos_init_t, \
        _ypos_init_t, \
        _vsys_init_t, \
        _pa_init_t, \
        _incl_init_t, \
        _vrot_init_t, \
        _vrad_init_t, \
        _sigma_init_et, \
        _xpos_init_et, \
        _ypos_init_et, \
        _vsys_init_et, \
        _pa_init_et, \
        _incl_init_et, \
        _vrot_init_et, \
        _vrad_init_et, \
        nrings_reliable = trfit_ring_by_ring(_input_vf_tofit_tr, _tr_model_vf, _wi_2d, _params, fit_opt, ndim, tr_params_priors_init, 'True')
    

    n_coeffs_pa_bs, tck_pa_bs = bspline_ncoeffs_tck(_params, 'pa', _params['nrings_intp'])
    n_coeffs_incl_bs, tck_incl_bs = bspline_ncoeffs_tck(_params, 'incl', _params['nrings_intp'])
    n_coeffs_vrot_bs, tck_vrot_bs = bspline_ncoeffs_tck(_params, 'vrot', _params['nrings_intp'])

    n_coeffs_vrad_bs = 0
    tr_params_bounds = np.zeros((8 + n_coeffs_pa_bs + n_coeffs_incl_bs + n_coeffs_vrot_bs + n_coeffs_vrad_bs, 2))



    _ring_t[nrings_reliable-1] = _params['r_galaxy_plane_e'] 

    scipy_interp1d = interp1d(_ring_t[:nrings_reliable], _pa_init_t[:nrings_reliable], kind='linear')

    r_fine = np.linspace(_ring_t[0], _ring_t[nrings_reliable-1], _params['nrings_intp'], endpoint=True)

    pa_fine = scipy_interp1d(r_fine)

    xs_bs = _ring_t[0]
    xe_bs = _params['r_galaxy_plane_e']
    n_coeffs_pa_bs, tck_pa_bs = bspline_ncoeffs_tck(_params, 'pa', _params['nrings_intp'])
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




    if n_coeffs_pa_bs != 0: # not constant
        del_pa = _params['pa_bs_bounds']
        tck_pa_bs_bound_l, tck_pa_bs_bound_u = set_pa_bs_coefficients_bounds(_pa_init_t, del_pa, _params, _ring_t)

        for _nbs in range(0, n_coeffs_pa_bs):
            _pa_bs_nbs_l = min_val(tck_pa_bs_bound_l[1][_nbs], tck_pa_bs_bound_u[1][_nbs])
            _pa_bs_nbs_u = max_val(tck_pa_bs_bound_l[1][_nbs], tck_pa_bs_bound_u[1][_nbs])

            tr_params_bounds[7+1+_nbs, 0] = _pa_bs_nbs_l
            tr_params_bounds[7+1+_nbs, 1] = _pa_bs_nbs_u







    _ring_t[nrings_reliable-1] = _params['r_galaxy_plane_e'] 

    scipy_interp1d = interp1d(_ring_t[:nrings_reliable], _incl_init_t[:nrings_reliable], kind='linear')  # 선형 보간

    r_fine = np.linspace(0, _ring_t[nrings_reliable-1], _params['nrings_intp'], endpoint=True)

    incl_fine = scipy_interp1d(r_fine)

    xs_bs = 0
    xe_bs = _params['r_galaxy_plane_e']
    n_coeffs_incl_bs, tck_incl_bs = bspline_ncoeffs_tck(_params, 'incl', _params['nrings_intp'])
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
        tck_incl_bs_bound_l, tck_incl_bs_bound_u = set_incl_bs_coefficients_bounds(_incl_init_t, del_incl, _params, _ring_t)

        for _nbs in range(0, n_coeffs_incl_bs):
            _incl_bs_nbs_l = min_val(tck_incl_bs_bound_l[1][_nbs], tck_incl_bs_bound_u[1][_nbs])
            _incl_bs_nbs_u = max_val(tck_incl_bs_bound_l[1][_nbs], tck_incl_bs_bound_u[1][_nbs])

            tr_params_bounds[7+1+n_coeffs_pa_bs+_nbs, 0] = _incl_bs_nbs_l
            tr_params_bounds[7+1+n_coeffs_pa_bs+_nbs, 1] = _incl_bs_nbs_u

            if tr_params_bounds[7+1+n_coeffs_pa_bs+_nbs, 0] < 0: tr_params_bounds[7+1+n_coeffs_pa_bs+_nbs, 0] = 0
            if tr_params_bounds[7+1+n_coeffs_pa_bs+_nbs, 1] > 89: tr_params_bounds[7+1+n_coeffs_pa_bs+_nbs, 1] = 89

            if tr_params_bounds[7+1+n_coeffs_pa_bs+_nbs, 0] > 89 or tr_params_bounds[7+1+n_coeffs_pa_bs+_nbs, 1] < 0:
               tr_params_bounds[7+1+n_coeffs_pa_bs+_nbs, 0] = 0 
               tr_params_bounds[7+1+n_coeffs_pa_bs+_nbs, 1] = 89



    fit_opt, ndim, tr_params_priors_init = set_params_fit_option(_params, 'sigma', 'free', np.nanmedian(_sigma_init_t[1:-1]), \
                          'xpos', 'fixed', np.nanmedian(_xpos_init_t[1:-1]), \
                          'ypos', 'fixed', np.nanmedian(_ypos_init_t[1:-1]), \
                          'vsys', 'fixed', np.nanmedian(_vsys_init_t[1:-1]), \
                          'pa', 'fixed', np.nanmedian(_pa_init_t[1:-1]), \
                          'incl', 'fixed', np.nanmedian(_incl_init_t[1:-1]), \
                          'vrot', 'free', np.nanmedian(_vrot_init_t[1:-1]), \
                          'vrad', 'fixed', np.nanmedian(_vrad_init_t[1:-1]), nrings_reliable, 1)
    _ring_t, \
        _sigma_init_t, \
        _xpos_init_t, \
        _ypos_init_t, \
        _vsys_init_t, \
        _pa_init_t, \
        _incl_init_t, \
        _vrot_init_t, \
        _vrad_init_t, \
        _sigma_init_et, \
        _xpos_init_et, \
        _ypos_init_et, \
        _vsys_init_et, \
        _pa_init_et, \
        _incl_init_et, \
        _vrot_init_et, \
        _vrad_init_et, \
        nrings_reliable = trfit_ring_by_ring(_input_vf_tofit_tr, _tr_model_vf, _wi_2d, _params, fit_opt, ndim, tr_params_priors_init, 'True')





    _ring_t[nrings_reliable-1] = _params['r_galaxy_plane_e'] 

    scipy_interp1d = interp1d(_ring_t[:nrings_reliable], _vrot_init_t[:nrings_reliable], kind='linear')  # 선형 보간

    r_fine = np.linspace(0, _ring_t[nrings_reliable-1], _params['nrings_intp'], endpoint=True)

    vrot_fine = scipy_interp1d(r_fine)

    xs_bs = 0
    xe_bs = _params['r_galaxy_plane_e']
    n_coeffs_vrot_bs, tck_vrot_bs = bspline_ncoeffs_tck(_params, 'vrot', _params['nrings_intp'])
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







    _sigma_init = _params['_sigma_init']
    _xpos_init = _params['_xpos_init']
    _ypos_init = _params['_ypos_init']
    _vsys_init = _params['_vsys_init']
    _pa_init = _params['_pa_init']
    _incl_init = _params['_incl_init']
    _vrot_init = _params['_vrot_init']
    _vrad_init = _params['_vrad_init']


    _params['n_coeffs_vrot_bs'] = n_coeffs_pa_bs
    _params['n_coeffs_vrot_bs'] = n_coeffs_incl_bs
    _params['n_coeffs_vrot_bs'] = n_coeffs_vrot_bs
    _params['nrings_reliable'] = nrings_reliable


    tr_params_bounds[0, 0] = _sigma_init - _params['sigma_bounds_width'] # sigma0
    tr_params_bounds[0, 1] = _sigma_init + _params['sigma_bounds_width'] # sigma1 
    if tr_params_bounds[0, 0] < 0: tr_params_bounds[0, 0] = 0


    tr_params_bounds[1, 0] = _xpos_init - _params['ring_w']*0.5
    tr_params_bounds[1, 1] = _xpos_init + _params['ring_w']*0.5
    tr_params_bounds[2, 0] = _ypos_init - _params['ring_w']*0.5
    tr_params_bounds[2, 1] = _ypos_init + _params['ring_w']*0.5

    tr_params_bounds[3, 0] = _vsys_init - _params['vsys_bounds_width']
    tr_params_bounds[3, 1] = _vsys_init + _params['vsys_bounds_width']

    tr_params_bounds[4, 0] = _pa_init - _params['pa_bounds_width']
    tr_params_bounds[4, 1] = _pa_init + _params['pa_bounds_width']

    tr_params_bounds[5, 0] = _incl_init - _params['incl_bounds_width']
    tr_params_bounds[5, 1] = _incl_init + _params['incl_bounds_width']
    if tr_params_bounds[5, 0] <= 0: tr_params_bounds[5, 0] = 1
    if tr_params_bounds[5, 1] >= 90: tr_params_bounds[5, 1] = 89

    tr_params_bounds[6, 0] = _vrot_init - _params['vrot_bounds_width']
    tr_params_bounds[6, 1] = _vrot_init + _params['vrot_bounds_width']
    if tr_params_bounds[6, 0] < 0: tr_params_bounds[6, 0] = 0

    tr_params_bounds[7, 0] = _vrad_init - _params['vrad_bounds_width']
    tr_params_bounds[7, 1] = _vrad_init + _params['vrad_bounds_width']






    if n_coeffs_vrot_bs != 0: # not constant
        del_vrot = _params['vrot_bs_bounds']
        tck_vrot_bs_bound_l, tck_vrot_bs_bound_u = set_vrot_bs_coefficients_bounds(_vrot_init_t, del_vrot, _params, _ring_t)

        for _nbs in range(0, n_coeffs_vrot_bs):
            _vrot_bs_nbs_l = min_val(tck_vrot_bs_bound_l[1][_nbs], tck_vrot_bs_bound_u[1][_nbs])
            _vrot_bs_nbs_u = max_val(tck_vrot_bs_bound_l[1][_nbs], tck_vrot_bs_bound_u[1][_nbs])

            tr_params_bounds[7+1+n_coeffs_pa_bs+n_coeffs_incl_bs+_nbs, 0] = _vrot_bs_nbs_l
            tr_params_bounds[7+1+n_coeffs_pa_bs+n_coeffs_incl_bs+_nbs, 1] = _vrot_bs_nbs_u

            if tr_params_bounds[7+1+n_coeffs_pa_bs+n_coeffs_incl_bs+_nbs, 0] < 0: tr_params_bounds[7+1+n_coeffs_pa_bs+n_coeffs_incl_bs+_nbs, 0] = 0
            if tr_params_bounds[7+1+n_coeffs_pa_bs+n_coeffs_incl_bs+_nbs, 1] < 0: tr_params_bounds[7+1+n_coeffs_pa_bs+n_coeffs_incl_bs+_nbs, 1] = 0



    print("")
    print(tr_params_bounds)
    print("")





    fit_opt, ndim, tr_params_priors_init = set_params_fit_option(_params, 'sigma', 'free', np.nanmedian(_sigma_init_t[1:-1]), \
                          'xpos', 'free', np.nanmedian(_xpos_init_t[1:-1]), \
                          'ypos', 'free', np.nanmedian(_ypos_init_t[1:-1]), \
                          'vsys', 'free', np.nanmedian(_vsys_init_t[1:-1]), \
                          'pa', 'free', np.nanmedian(_pa_init_t[1:-1]), \
                          'incl', 'free', np.nanmedian(_incl_init_t[1:-1]), \
                          'vrot', 'free', np.nanmedian(_vrot_init_t[1:-1]), \
                          'vrad', 'fixed', np.nanmedian(_vrad_init_t[1:-1]), nrings_reliable, 1)
    
    _ij_area_tofit, _tr2dfit_results, _n_dim, fit_opt_2d, std_resample_run  = trfit_2d(_input_vf_tofit_2d, _tr_model_vf, _wi_2d, _params, tr_params_bounds, nrings_reliable, tck_vrot_bs_init_from_trfit, tck_pa_bs_init_from_trfit, tck_incl_bs_init_from_trfit, _2dbat_run_i)
    print("r-galaxy:", _params['r_galaxy_plane_e'])
    print("")


    _vrot_bs, _vrot_bs_e = extract_vrot_bs_tr_rings_given_dyn_params(_input_vf_tofit_grid1, _input_vf_tofit_2d, _tr2dfit_results, _params, fit_opt_2d, _tr_model_vf, _2dbat_run_i)

    bsfit_vf = make_vlos_model_vf_given_dyn_params(_input_vf_tofit_grid1, _input_vf_tofit_2d, _wi_2d, _tr2dfit_results, _params, fit_opt_2d, _tr_model_vf, _2dbat_run_i)


    fit_opt, ndim, tr_params_priors_init = set_params_fit_option(_params, \
                          'sigma', 'free', np.nanmedian(_sigma_init_t[1:-1]), \
                          'xpos', 'fixed', np.nanmedian(_xpos_init_t[1:-1]), \
                          'ypos', 'fixed', np.nanmedian(_ypos_init_t[1:-1]), \
                          'vsys', 'fixed', np.nanmedian(_vsys_init_t[1:-1]), \
                          'pa', 'fixed', np.nanmedian(_pa_init_t[1:-1]), \
                          'incl', 'fixed', np.nanmedian(_incl_init_t[1:-1]), \
                          'vrot', 'free', np.nanmedian(_vrot_init_t[1:-1]), \
                          'vrad', 'fixed', np.nanmedian(_vrad_init_t[1:-1]), nrings_reliable, 'True')
    
    _params['dlogz_tr'] = _params['dlogz_2d']


    _2dbat_trfit_final_b = trfit_ring_by_ring_final(_input_vf_tofit_tr, _tr_model_vf, _wi_2d, _params, fit_opt, fit_opt_2d, ndim, tr_params_priors_init, _tr2dfit_results, 0, 'True') # receding side
    nrings_reliable_b =int(_2dbat_trfit_final_b[19])

    write_2dbat_t1(_2dbat_trfit_final_b, nrings_reliable_b, _params, _2dbat_run_i, '_2dbat_trfit_temp.txt')
    print("")
    print('-'*50)
    print("+ BOTH SIDES")
    print('-'*50)
    print_2dbat(_params, _2dbat_run_i, '_2dbat_trfit_temp.txt')

    trfit_vf = make_vlos_model_vf_given_dyn_params_trfit_final_vrot(_input_vf_tofit_grid1, _input_vf_tofit_2d, _wi_2d, _tr2dfit_results, \
                                                                    _2dbat_trfit_final_b[2], \
                                                                    _2dbat_trfit_final_b[3], \
                                                                    _2dbat_trfit_final_b[4], \
                                                                    _2dbat_trfit_final_b[5], \
                                                                    _2dbat_trfit_final_b[6], \
                                                                    _2dbat_trfit_final_b[7], \
                                                                    _2dbat_trfit_final_b[8], \
                                                                    _params, fit_opt_2d, _tr_model_vf, _2dbat_run_i)

    _2dbat_trfit_final_a = trfit_ring_by_ring_final(_input_vf_tofit_tr, _tr_model_vf, _wi_2d, _params, fit_opt, fit_opt_2d, ndim, tr_params_priors_init, _tr2dfit_results, -1, 'True') # receding side
    nrings_reliable_a =int(_2dbat_trfit_final_a[19])

    write_2dbat_t1(_2dbat_trfit_final_a, nrings_reliable_a, _params, _2dbat_run_i, '_2dbat_trfit_temp.txt')
    print("")
    print('-'*50)
    print("+ APPROACHING SIDE")
    print('-'*50)
    print_2dbat(_params, _2dbat_run_i, '_2dbat_trfit_temp.txt')



    _2dbat_trfit_final_r = trfit_ring_by_ring_final(_input_vf_tofit_tr, _tr_model_vf, _wi_2d, _params, fit_opt, fit_opt_2d, ndim, tr_params_priors_init, _tr2dfit_results, 1, 'True') # receding side
    nrings_reliable_r =int(_2dbat_trfit_final_r[19])

    write_2dbat_t1(_2dbat_trfit_final_r, nrings_reliable_r, _params, _2dbat_run_i, '_2dbat_trfit_temp.txt')
    print("")
    print('-'*50)
    print("+ RECEDING SIDE")
    print('-'*50)
    print_2dbat(_params, _2dbat_run_i, '_2dbat_trfit_temp.txt')


    _dir_2dbat_PI_output = _params['wdir'] + '/' + _params['_2dbatdir'] + ".%d" % _2dbat_run_i
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
    ray.shutdown()
    sys.exit()



 








    print("duration =", datetime.now() - start)

if __name__ == '__main__':
    main()


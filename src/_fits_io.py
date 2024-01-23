#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#|-----------------------------------------|
#| _fits_io.py
#|-----------------------------------------|
#| by Se-Heon Oh
#| Dept. of Physics and Astronomy
#| Sejong University, Seoul, South Korea
#|-----------------------------------------|



#|-----------------------------------------|
import numpy as np
import math
import fitsio

from datetime import datetime

import matplotlib.pyplot as plt

import astropy.units as u
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from spectral_cube import SpectralCube
from spectral_cube import BooleanArrayMask

from _nested_sampler import run_nested_sampler_trfit, run_nested_sampler_trfit_2d, derive_vlos_model_2d, derive_vlos_model_2d_sc, bspline_ncoeffs_tck, set_phi_bounds
from _nested_sampler import derive_vlos_model_2d_sc_checkplot, extract_tr2dfit_params
from _nested_sampler import define_tilted_ring, min_val, max_val

#|-----------------------------------------|
# visualize model VF 
import sys
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS
#|-----------------------------------------|




from scipy.interpolate import BSpline, splrep, splev
from scipy import optimize
from scipy.interpolate import interp1d

from itertools import zip_longest
from numba import njit


import ray

from skimage.measure import EllipseModel
from scipy.spatial import distance

# import make_dirs
from _dirs_files import make_dirs
import os


#|-----------------------------------------|
#| 2DBAT routines related 
#|-----------------------------------------|



def ellipsefit_2dpoints(_input_vf_tofit, _wi_2d, _params, xpos, ypos, pa, incl, ri, ro, side):

    npoints_in_a_ring_total_including_blanks, npoints_in_ring, _ij_aring_tofit, _wt_2d = define_tilted_ring(_input_vf_tofit, xpos, ypos, pa, incl, ri, ro, side, _params)
    #print(_ij_aring_tofit[:, 0])
    #print("")
    #print(_ij_aring_tofit[:, 1])

    a_points_t = np.array((_ij_aring_tofit[:, 0], _ij_aring_tofit[:, 1]))
    a_points = np.transpose(a_points_t)

    x = a_points[:, 0]
    y = a_points[:, 1]

    ell = EllipseModel()
    ell.estimate(a_points)
    _xc, _yc, _a, _b, _theta = ell.params
    _i = np.arccos((_b/_a)) * 180. / np.pi # in degree
    _theta = (_theta * 180. / np.pi) + 90 # in degree
    _theta = 10

    #print("center = ",  (_xc, _yc))
    #print("angle of rotation = ",  _theta)
    #print("axes = ", (_a, _b))
    #print("i = ", np.arccos((_b/_a)) * 180. / np.pi  )
    #_centre = np.array((_xc, _yc))
    _centre = [(_xc, _yc)]
    dists = distance.cdist(_centre, a_points, 'euclidean')
    r_max = np.max(dists)

    npoints_in_a_ring_total_including_blanks, npoints_in_ring, _ij_aring_central, _wt_2d = define_tilted_ring(_input_vf_tofit, _xc, _yc, _theta, _i, 0, 0.2*r_max, side, _params)
    _it = _ij_aring_central[:, 0]
    _jt = _ij_aring_central[:, 1]
    vlos = _input_vf_tofit[_jt.astype(int), _it.astype(int)]
    vsys_init = np.median(vlos)

    return _xc, _yc, vsys_init, _theta, _i, r_max

#    from matplotlib.patches import Ellipse
#    import matplotlib.pyplot as plt
#
#    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
#    axs[0].scatter(x,y)
#
#    axs[1].scatter(x, y)
#    axs[1].scatter(_xc, _yc, color='red', s=100)
#    axs[1].set_xlim(x.min(), x.max())
#    axs[1].set_ylim(y.min(), y.max())
#
#    ell_patch = Ellipse((_xc, _yc), 2*_a, 2*_b, _theta*180/np.pi, edgecolor='red', facecolor='none')
#
#    axs[1].add_patch(ell_patch)
#    plt.show()
#
#    sys.exit()



def set_vrot_bs_coefficients_bounds(_vrot_init_t, del_vrot, _params, _ring_t):

    # ------------------------------------------------------
    # inverpolation version
    # ------------------------------------------------------
    nrings_reliable = _params['nrings_reliable']

    xs_bs = 0
    xe_bs = _ring_t[nrings_reliable-1]
    # generate 1D interpolation function
    scipy_interp1d = interp1d(_ring_t[:nrings_reliable], _vrot_init_t[:nrings_reliable], kind='linear')  # 선형 보간

    # generate fine gaps
    r_fine = np.linspace(xs_bs, xe_bs, _params['nrings_intp'], endpoint=True)

    # do interpolation
    vrot_fine = scipy_interp1d(r_fine)

    _vrot_bound_l = np.zeros(vrot_fine.shape[0], dtype=np.float64)
    _vrot_bound_u = np.zeros(vrot_fine.shape[0], dtype=np.float64)

    for _i in range(len(vrot_fine)):

        _vrot_bound_l[_i] = vrot_fine[_i] - del_vrot
        _vrot_bound_u[_i] = vrot_fine[_i] + del_vrot

    n_knots_inner = _params['n_vrot_bs_knots_inner'] # 0, 1, 2, ...
    k_bs = _params['k_vrot_bs'] # 0, 1, 2, ...
    vrot_bs_knots_inner = np.linspace(xs_bs, xe_bs, (2+n_knots_inner))[1:-1]
    tck_vrot_bs_bound_l = splrep(r_fine, _vrot_bound_l, t=vrot_bs_knots_inner, k=k_bs)
    tck_vrot_bs_bound_u = splrep(r_fine, _vrot_bound_u, t=vrot_bs_knots_inner, k=k_bs)

    return tck_vrot_bs_bound_l, tck_vrot_bs_bound_u


#def set_vrot_bs_coefficients_bounds(_vrot_intp, del_vrot, _params, _ring_t):
#    xs_bs = _params['r_galaxy_plane_s']
#    xe_bs = _params['r_galaxy_plane_e']
#
#    _vrot_bound_l = np.zeros(_ring_t.shape[0], dtype=np.float64)
#    _vrot_bound_u = np.zeros(_ring_t.shape[0], dtype=np.float64)
#
#    for _i in range(len(_ring_t)):
#        _ri = _ring_t[_i] 
#        _ro = _ring_t[_i] + _params['ring_w'] 
#
#        _vrot_bound_l[_i] = _vrot_intp[_i] - del_vrot
#        if _i == 0:
#            _vrot_bound_l[_i] = 0
#        _vrot_bound_u[_i] = _vrot_intp[_i] + del_vrot
#
#        if _vrot_bound_l[_i] < 0: _vrot_bound_l[_i] = 0
#        if _vrot_bound_u[_i] < 0: _vrot_bound_u[_i] = 0
#
#    n_knots_inner = _params['n_vrot_bs_knots_inner'] # 0, 1, 2, ...
#    k_bs = _params['k_vrot_bs'] # 0, 1, 2, ...
#    vrot_bs_knots_inner = np.linspace(xs_bs, xe_bs, (2+n_knots_inner))[1:-1]
#    tck_vrot_bs_bound_l = splrep(_ring_t, _vrot_bound_l, t=vrot_bs_knots_inner, k=k_bs)
#    tck_vrot_bs_bound_u = splrep(_ring_t, _vrot_bound_u, t=vrot_bs_knots_inner, k=k_bs)
#
#    return tck_vrot_bs_bound_l, tck_vrot_bs_bound_u


#def set_incl_bs_coefficients_bounds(_incl_intp, del_incl, _params, _ring_t):
#    xs_bs = _params['r_galaxy_plane_s']
#    xe_bs = _params['r_galaxy_plane_e']
#
#    _incl_bound_l = np.zeros(_ring_t.shape[0], dtype=np.float64)
#    _incl_bound_u = np.zeros(_ring_t.shape[0], dtype=np.float64)
#
#    for _i in range(len(_ring_t)):
#        _ri = _ring_t[_i] 
#        _ro = _ring_t[_i] + _params['ring_w'] 
#
#        _incl_bound_l[_i] = _incl_intp[_i] - del_incl
#        _incl_bound_u[_i] = _incl_intp[_i] + del_incl
#
#        if _incl_bound_l[_i] < 0: _incl_bound_l[_i] = 0
#        if _incl_bound_u[_i] < 0: _incl_bound_u[_i] = 0
#
#    n_knots_inner = _params['n_incl_bs_knots_inner'] # 0, 1, 2, ...
#    k_bs = _params['k_incl_bs'] # 0, 1, 2, ...
#    incl_bs_knots_inner = np.linspace(xs_bs, xe_bs, (2+n_knots_inner))[1:-1]
#    tck_incl_bs_bound_l = splrep(_ring_t, _incl_bound_l, t=incl_bs_knots_inner, k=k_bs)
#    tck_incl_bs_bound_u = splrep(_ring_t, _incl_bound_u, t=incl_bs_knots_inner, k=k_bs)
#
#    return tck_incl_bs_bound_l, tck_incl_bs_bound_u

def set_incl_bs_coefficients_bounds(_incl_init_t, del_incl, _params, _ring_t):

    # ------------------------------------------------------
    # inverpolation version
    # ------------------------------------------------------
    nrings_reliable = _params['nrings_reliable']

    xs_bs = 0
    xe_bs = _ring_t[nrings_reliable-1]
    # generate 1D interpolation function
    scipy_interp1d = interp1d(_ring_t[:nrings_reliable], _incl_init_t[:nrings_reliable], kind='linear')  # 선형 보간

    # generate fine gaps
    r_fine = np.linspace(xs_bs, xe_bs, _params['nrings_intp'], endpoint=True)

    # do interpolation
    incl_fine = scipy_interp1d(r_fine)

    _incl_bound_l = np.zeros(incl_fine.shape[0], dtype=np.float64)
    _incl_bound_u = np.zeros(incl_fine.shape[0], dtype=np.float64)

    for _i in range(len(incl_fine)):

        _incl_bound_l[_i] = incl_fine[_i] - del_incl
        _incl_bound_u[_i] = incl_fine[_i] + del_incl

    n_knots_inner = _params['n_incl_bs_knots_inner'] # 0, 1, 2, ...
    k_bs = _params['k_incl_bs'] # 0, 1, 2, ...
    incl_bs_knots_inner = np.linspace(xs_bs, xe_bs, (2+n_knots_inner))[1:-1]
    tck_incl_bs_bound_l = splrep(r_fine, _incl_bound_l, t=incl_bs_knots_inner, k=k_bs)
    tck_incl_bs_bound_u = splrep(r_fine, _incl_bound_u, t=incl_bs_knots_inner, k=k_bs)

    return tck_incl_bs_bound_l, tck_incl_bs_bound_u


#def set_pa_bs_coefficients_bounds(_pa_intp, del_pa, _params, _ring_t):
def set_pa_bs_coefficients_bounds(_pa_init_t, del_pa, _params, _ring_t):

    # ------------------------------------------------------
    # inverpolation version
    # ------------------------------------------------------
    nrings_reliable = _params['nrings_reliable']
    #_ring_t[nrings_reliable] = _params['r_galaxy_plane_e'] 
    #_pa_init_t[nrings_reliable] = _pa_init_t[nrings_reliable-1]

    xs_bs = 0
    xe_bs = _ring_t[nrings_reliable-1]
    # generate 1D interpolation function
    scipy_interp1d = interp1d(_ring_t[:nrings_reliable], _pa_init_t[:nrings_reliable], fill_value=(_pa_init_t[0], _pa_init_t[nrings_reliable-1]), kind='linear')  # linear intp.

    # generate fine gaps
    r_fine = np.linspace(xs_bs, xe_bs, _params['nrings_intp'], endpoint=True)

    # do interpolation
    print(r_fine)
    print(nrings_reliable, xe_bs)
    pa_fine = scipy_interp1d(r_fine)

    _pa_bound_l = np.zeros(pa_fine.shape[0], dtype=np.float64)
    _pa_bound_u = np.zeros(pa_fine.shape[0], dtype=np.float64)

    for _i in range(len(pa_fine)):
        _pa_bound_l[_i] = pa_fine[_i] - del_pa
        _pa_bound_u[_i] = pa_fine[_i] + del_pa

    n_knots_inner = _params['n_pa_bs_knots_inner'] # 0, 1, 2, ...
    k_bs = _params['k_pa_bs'] # 0, 1, 2, ...
    pa_bs_knots_inner = np.linspace(xs_bs, xe_bs, (2+n_knots_inner))[1:-1]
    tck_pa_bs_bound_l = splrep(r_fine, _pa_bound_l, t=pa_bs_knots_inner, k=k_bs)
    tck_pa_bs_bound_u = splrep(r_fine, _pa_bound_u, t=pa_bs_knots_inner, k=k_bs)

    return tck_pa_bs_bound_l, tck_pa_bs_bound_u


#    xs_bs = 0
#    xe_bs = _params['r_galaxy_plane_e']
#    n_coeffs_pa_bs, tck_pa_bs = bspline_ncoeffs_tck(_params, 'pa', _params['nrings_intp'])
#    # PA-BS coefficients
#    # number of inner knots
#    if n_coeffs_pa_bs != 0: # not constant
#        n_knots_inner = _params['n_pa_bs_knots_inner'] # 0, 1, 2, ...
#        k_bs = _params['k_pa_bs'] # 0, 1, 2, ...
#        pa_bs_knots_inner = np.linspace(xs_bs, xe_bs, (2+n_knots_inner))[1:-1]
#        tck_pa_bs_init_from_trfit = splrep(r_fine, pa_fine, t=pa_bs_knots_inner, k=k_bs)
#    else:
#        n_knots_inner = 0 # dummy value
#        k_bs = 1 # dummy value
#        pa_bs_knots_inner = np.linspace(xs_bs, xe_bs, (2+n_knots_inner))[1:-1]
#        tck_pa_bs_init_from_trfit = splrep(r_fine, pa_fine, t=pa_bs_knots_inner, k=k_bs)
#
#
#
#    xs_bs = _params['r_galaxy_plane_s']
#    xe_bs = _params['r_galaxy_plane_e']
#
#    _pa_bound_l = np.zeros(_ring_t.shape[0], dtype=np.float64)
##    _pa_bound_u = np.zeros(_ring_t.shape[0], dtype=np.float64)
#
#    for _i in range(len(_ring_t)):
#        _ri = _ring_t[_i] 
#        _ro = _ring_t[_i] + _params['ring_w'] 
#
#        _pa_bound_l[_i] = _pa_intp[_i] - del_pa
#        _pa_bound_u[_i] = _pa_intp[_i] + del_pa
#
#        #if _pa_bound_l[_i] < 0: _pa_bound_l[_i] = 0
###        #if _pa_bound_u[_i] < 0: _pa_bound_u[_i] = 0
#
#    n_knots_inner = _params['n_pa_bs_knots_inner'] # 0, 1, 2, ...
#    k_bs = _params['k_pa_bs'] # 0, 1, 2, ...
#    pa_bs_knots_inner = np.linspace(xs_bs, xe_bs, (2+n_knots_inner))[1:-1]
#    tck_pa_bs_bound_l = splrep(_ring_t, _pa_bound_l, t=pa_bs_knots_inner, k=k_bs)
#    tck_pa_bs_bound_u = splrep(_ring_t, _pa_bound_u, t=pa_bs_knots_inner, k=k_bs)
#
#    return tck_pa_bs_bound_l, tck_pa_bs_bound_u


def set_pa_bs_coefficients_bounds_constrained_to_incl_vrot(_pa_bs_intp, _vrot_intp, del_vrot, _incl_intp, del_i, _params, _input_vf_tofit, _wi_2d, _ring_t):

    side = 0
    _xpos_init = _params['_xpos_init']
    _ypos_init = _params['_ypos_init']

    xs_bs = _params['r_galaxy_plane_s']
    xe_bs = _params['r_galaxy_plane_e']

    _pa_bound_l = np.zeros(_ring_t.shape[0], dtype=np.float64)
    _pa_bound_u = np.zeros(_ring_t.shape[0], dtype=np.float64)

    for _i in range(len(_ring_t)):

        _ri = _ring_t[_i] 
        _ro = _ring_t[_i] + _params['ring_w'] 

        npoints_in_a_ring_total_including_blanks, npoints_in_ring, _ij_aring_pa_bound, _wt_2d = define_tilted_ring(_input_vf_tofit, _xpos_init, _ypos_init, _pa_bs_intp[_i], _incl_intp[_i], _ri, _ro, side, _params)
        _pa_bound_l[_i], _pa_bound_u[_i] = set_phi_bounds(_pa_bs_intp[_i], _vrot_intp[_i], del_vrot, _incl_intp[_i], del_i, _params, _ij_aring_pa_bound)

        print("ring:", _ring_t[_i], _vrot_intp[_i], "n:", _ij_aring_pa_bound.shape[0], "incl:", _incl_intp[_i], _pa_bound_l[_i], _pa_bs_intp[_i], _pa_bound_u[_i])

    n_knots_inner = _params['n_pa_bs_knots_inner'] # 0, 1, 2, ...
    k_bs = _params['k_pa_bs'] # 0, 1, 2, ...
    pa_bs_knots_inner = np.linspace(xs_bs, xe_bs, (2+n_knots_inner))[1:-1]
    tck_pa_bs_bound_l = splrep(_ring_t, _pa_bound_l, t=pa_bs_knots_inner, k=k_bs)
    tck_pa_bs_bound_u = splrep(_ring_t, _pa_bound_u, t=pa_bs_knots_inner, k=k_bs)

    return tck_pa_bs_bound_l, tck_pa_bs_bound_u


def set_vrot_bs_coefficients_bounds_constrained_to_pa_incl(_pa_bs_intp, _vrot_intp, del_vrot, _incl_intp, del_i, _params, _input_vf_tofit, _wi_2d, _ring_t):

    side = 0
    _xpos_init = _params['_xpos_init']
    _ypos_init = _params['_ypos_init']

    xs_bs = _params['r_galaxy_plane_s']
    xe_bs = _params['r_galaxy_plane_e']

    _vrot_bound_l = np.zeros(_ring_t.shape[0], dtype=np.float64)
    _vrot_bound_u = np.zeros(_ring_t.shape[0], dtype=np.float64)

    for _i in range(len(_ring_t)):

        _ri = _ring_t[_i] 
        _ro = _ring_t[_i] + _params['ring_w'] 

        npoints_in_a_ring_total_including_blanks, npoints_in_ring, _ij_aring_pa_bound, _wt_2d = define_tilted_ring(_input_vf_tofit, _xpos_init, _ypos_init, _pa_bs_intp[_i], _incl_intp[_i], _ri, _ro, side, _params)
        _vrot_bound_l[_i], _vrot_bound_u[_i] = set_vrot_bounds(_pa_bs_intp[_i], _vrot_intp[_i], del_vrot, _incl_intp[_i], del_i, _params, _ij_aring_pa_bound)

        #print("ring:", _ring_t[_i], _vrot_intp[_i], "n:", _ij_aring_pa_bound.shape[0], "incl:", _incl_intp[_i], _pa_bound_l[_i], _pa_bs_intp[_i], _pa_bound_u[_i])

    n_knots_inner = _params['n_vrot_bs_knots_inner'] # 0, 1, 2, ...
    k_bs = _params['k_vrot_bs'] # 0, 1, 2, ...
    vrot_bs_knots_inner = np.linspace(xs_bs, xe_bs, (2+n_knots_inner))[1:-1]
    tck_vrot_bs_bound_l = splrep(_ring_t, _vrot_bound_l, t=vrot_bs_knots_inner, k=k_bs)
    tck_vrot_bs_bound_u = splrep(_ring_t, _vrot_bound_u, t=vrot_bs_knots_inner, k=k_bs)

    return tck_vrot_bs_bound_l, tck_vrot_bs_bound_u


#
#    a = list(tck_pa_bs_bound_l)
#    b = list(tck_pa_bs_bound_u)
#    c = [(a1 + b1)/2. for a1, b1 in zip(a, b)]
#    c[2] = 1
#    print(a)
#    print(b)
#    print(c)
#    xx = tuple(c)
#
#    print(tck_pa_bs_bound_l[1])
#    print(tck_pa_bs_bound_u[1])
#    _p1 = BSpline(*tck_pa_bs_bound_l, extrapolate=True)(_ring_t)
#    _p2 = BSpline(*tck_pa_bs_bound_u, extrapolate=True)(_ring_t)
#    _p3 = BSpline(*xx, extrapolate=True)(_ring_t)
#    print(_pa_bound_l)
#    print(_pa_bound_u)
#    print("")
#
#    # ---------------
#    fig = plt.figure()
#    ax = fig.add_subplot()
#    plt.xlim(0, 50)
#    plt.ylim(-30, 90)
#    plt.scatter(_ring_t, _pa_bound_l, c='b')	
#    plt.plot(_ring_t, _p1, c='b')	
#
#    plt.scatter(_ring_t, _pa_bound_u, c='y')	
#    plt.plot(_ring_t, _p2, c='y')	
#
#    plt.scatter(_ring_t, _pa_init_t, c='g')	
#    plt.plot(_ring_t, _p3, c='g')	
#
#    plt.show()
#    # ---------------




#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
@ray.remote(num_cpus=1)
def define_tilted_ring_i(_input_vf, _xpos, ypos, pa, incl, ri, ro, side, _params):

    naxis1 = _params['naxis1']
    naxis2 = _params['naxis2']

    # ..............................................
    # 1. extract the region to fit with decimals starting from the centre position given
    x0 = 0
    y0 = 0
    x1 = naxis1
    y1 = naxis2

    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    ri_deg = cdelt1*ri
    ro_deg = cdelt1*ro

    # DEG TO RAD
    deg_to_rad = np.pi / 180.

    # ..............................................
    # 0. set free angle
    free_angle = 0
    sine_free_angle = np.fabs(np.sin(free_angle*deg_to_rad))

    sinp = np.sin(deg_to_rad * pa)       # sine of pa. 
    cosp = np.cos(deg_to_rad * pa)       # cosine of pa. 
    sini = np.sin(deg_to_rad * incl)     #   sine of inc. 
    cosi = np.cos(deg_to_rad * incl)     # cosine of inc. 
    a = (1.0 - cosp * cosp * sini * sini)**0.5
    b = (1.0 - sinp * sinp * sini * sini)**0.5

    i0_lo = max_val(0, nint_val(xpos - a * ro_deg / cdelt1))
    i0_up = min_val(naxis1, nint_val( xpos + a * ro_deg / cdelt1))
    j0_lo = max_val(0, nint_val(ypos - b * ro_deg / cdelt2))
    j0_up = min_val(naxis2, nint_val(ypos + b * ro_deg / cdelt2))

    # ..............................................
    # count the total pixels in a given ring
    npoints_in_a_ring_total_including_blanks = 0
    for i0 in range(i0_lo, i0_up):
        for j0 in range(j0_lo, j0_up):
            rx = cdelt1 * i0  # X position in plane of galaxy
            ry = cdelt2 * j0  # Y position in plane of galaxy

            xr = ( - ( rx - cdelt1 * xpos ) * sinp + ( ry - cdelt2 * ypos ) * cosp )
            yr = ( - ( rx - cdelt1 * xpos ) * cosp - ( ry - cdelt2 * ypos ) * sinp ) / cosi
            r = ( xr**2 + yr**2 )**0.5  # distance from centre
            if r < 0.1:
                theta = 0.0
            else:
                theta = math.atan2( yr, xr ) / deg_to_rad # in degree

            costh = np.fabs( np.cos( deg_to_rad * theta ) ) # in radian

            if r > ri_deg and r < ro_deg: # both sides
                npoints_in_a_ring_total_including_blanks += 1

    # ..............................................
    # between ri_deg and ro_deg
    x0 = 0
    y0 = 0
    x1 = naxis1
    y1 = naxis2
		
    # ..............................................
    # Derive weight map
    #_input_vf_weight = np.zeros((naxis2, naxis1), dtype=float)
    wpow = 1
    npoints_in_ring_t = 0
    for i0 in range(i0_lo, i0_up):
        for j0 in range(j0_lo, j0_up):

            rx = cdelt1 * i0  # X position in plane of galaxy
            ry = cdelt2 * j0  # Y position in plane of galaxy
            #if not np.isinf(_input_vf[j0, i0]) and not np.isnan(_input_vf[j0, i0] and _input_vf[j0, i0] > -1E8 and _input_vf[j0, i0] < 1E8):
            if _input_vf[j0, i0] > -1E8 and _input_vf[j0, i0] < 1E8:

                xr = ( - ( rx - cdelt1 * xpos ) * sinp + ( ry - cdelt2 * ypos ) * cosp )
                yr = ( - ( rx - cdelt1 * xpos ) * cosp - ( ry - cdelt2 * ypos ) * sinp ) / cosi
                r = ( xr**2 + yr**2 )**0.5  # distance from centre
                if r < 0.1:
                    theta = 0.0
                else:
                    theta = math.atan2( yr, xr ) / deg_to_rad # in degree

                costh = np.fabs( np.cos ( deg_to_rad * theta ) ) # in radian

                # put weight
                if r > ri_deg and r < ro_deg and costh > sine_free_angle: # both sides
                    #_input_vf_weight[j0, i0] = costh**wpow # weight : note that radial weight doesn't need to be applied as all points within a ring have the same radius.
                    npoints_in_ring_t += 1

    # ..............................................
    # Reset the connected area to fit
    # [0:i, 1:j, 2:tr_vlos_model]
    ij_tilted_ring = np.array([], dtype=np.float64)
    ij_tilted_ring.shape = (0, 3)

    x0 = 0
    y0 = 0
    x1 = naxis1
    y1 = naxis2
    # ..............................................
    npoints_in_ring_t = 0
    for i0 in range(i0_lo, i0_up):
        for j0 in range(j0_lo, j0_up):
            rx = cdelt1 * i0  # X position in plane of galaxy
            ry = cdelt2 * j0  # Y position in plane of galaxy
            #if not np.isinf(_input_vf[j0, i0]) and not np.isnan(_input_vf[j0, i0] and _input_vf[j0, i0] > -1E8 and _input_vf[j0, i0] < 1E8):
            if _input_vf[j0, i0] > -1E8 and _input_vf[j0, i0] < 1E8:

                xr = ( - ( rx - cdelt1 * xpos ) * sinp + ( ry - cdelt2 * ypos ) * cosp )
                yr = ( - ( rx - cdelt1 * xpos ) * cosp - ( ry - cdelt2 * ypos ) * sinp ) / cosi
                r = ( xr**2 + yr**2 )**0.5  # distance from centre
                if r < 0.1:
                    theta = 0.0
                else:
                    theta = math.atan2( yr, xr ) / deg_to_rad # in degree

                costh = np.fabs( np.cos(deg_to_rad * theta) ) # in radian
                if r > ri_deg and r < ro_deg and costh > sine_free_angle:
                    # [0:i, 1:j, 2:tr_vlos_model]
                    ij_point = np.array([[i0, j0, -1E10]])
                    ij_tilted_ring = np.concatenate( (ij_tilted_ring, ij_point) )
                    #ij_tilted_ring = _ij_tilted_ring
                    npoints_in_ring_t += 1
                    
                    #if npoints_in_ring_t == 0:
                    #    ij_tilted_ring[int(npoints_in_ring_t), 0] = i0
                    #    ij_tilted_ring[int(npoints_in_ring_t), 1] = j0
                    #    npoints_in_ring_t += 1
                    #else:
                    #    np.concatenate((ij_tilted_ring, [i0, j0]), axis=0)
                    #    npoints_in_ring_t += 1

    #-- ij_tilted_ring = np.zeros((naxis2*naxis1, 2), dtype=int)

    #print(npoints_in_ring_t)
    #fig = plt.figure()
    #ax = fig.add_subplot()
    #ax.set_aspect('equal', adjustable='box')
    #plt.xlim(0, 40)
    #plt.ylim(0, 40)
    #plt.scatter(ij_tilted_ring[:,0], ij_tilted_ring[:,1])	
    #plt.show()

    return ij_tilted_ring
#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def trfit_ring_by_ring(_input_vf, _tr_model_vf, _wi_2d, _params, fit_opt, ndim, tr_params_priors_init, _intp_index):
    # xpos
    # ypos : ypos[nrings]
    # vsys : vsys[nrings]
    # pa : pa[nrings]
    # incl : incl[nrings]
    # vrot : vrot[nrings]

    nrings = _params['nrings_reliable']
    ring_w = _params['ring_w']

    _sigma = _params['_sigma_init']
    _xpos = _params['_xpos_init']
    _ypos = _params['_ypos_init']
    _vsys = _params['_vsys_init']
    _pa = _params['_pa_init']
    _incl = _params['_incl_init']
    _vrot = _params['_vrot_init']
    _vrad = _params['_vrad_init']

    _ring_t = np.zeros(nrings+2, dtype=np.float)
    _sigma_t = np.zeros(nrings+2, dtype=np.float)
    _xpos_t = np.zeros(nrings+2, dtype=np.float)
    _ypos_t = np.zeros(nrings+2, dtype=np.float)
    _vsys_t = np.zeros(nrings+2, dtype=np.float)
    _pa_t = np.zeros(nrings+2, dtype=np.float)
    _incl_t = np.zeros(nrings+2, dtype=np.float)
    _vrot_t = np.zeros(nrings+2, dtype=np.float)
    _vrad_t = np.zeros(nrings+2, dtype=np.float)

    _sigma_et = np.zeros(nrings+2, dtype=np.float)
    _xpos_et = np.zeros(nrings+2, dtype=np.float)
    _ypos_et = np.zeros(nrings+2, dtype=np.float)
    _vsys_et = np.zeros(nrings+2, dtype=np.float)
    _pa_et = np.zeros(nrings+2, dtype=np.float)
    _incl_et = np.zeros(nrings+2, dtype=np.float)
    _vrot_et = np.zeros(nrings+2, dtype=np.float)
    _vrad_et = np.zeros(nrings+2, dtype=np.float)

    _nrings_reliable_t = np.zeros(nrings+2, dtype=np.float)


    # ---------------------------------------
    # ray.put : speed up
    _input_vf_id = ray.put(_input_vf)
    _tr_model_vf_id = ray.put(_tr_model_vf)
    _wi_2d_id = ray.put(_wi_2d)
    _params_id = ray.put(_params)
    _i = 0
    _i_id = ray.put(_i)

    fit_opt_id = ray.put(fit_opt)
    ndim_id = ray.put(ndim)
    tr_params_priors_init_id = ray.put(tr_params_priors_init)

    # Run parallel processing
    results_ids = [trfit_ring_i.remote(_input_vf_id, _tr_model_vf_id, _wi_2d_id, _params_id, fit_opt_id, ndim_id, tr_params_priors_init_id, _i_id) for _i_id in range(0, nrings)]

    while len(results_ids):

        done_ids, results_ids = ray.wait(results_ids)
        if done_ids:

            ring =  ray.get(done_ids[0])[0]
            _sigma_ti =  ray.get(done_ids[0])[1]
            _xpos_ti =  ray.get(done_ids[0])[2]
            _ypos_ti =  ray.get(done_ids[0])[3]
            _vsys_ti =  ray.get(done_ids[0])[4]
            _pa_ti =  ray.get(done_ids[0])[5]
            _incl_ti =  ray.get(done_ids[0])[6]
            _vrot_ti =  ray.get(done_ids[0])[7]
            _vrad_ti =  ray.get(done_ids[0])[8]

            _sigma_eti =  ray.get(done_ids[0])[9]
            _xpos_eti =  ray.get(done_ids[0])[10]
            _ypos_eti =  ray.get(done_ids[0])[11]
            _vsys_eti =  ray.get(done_ids[0])[12]
            _pa_eti =  ray.get(done_ids[0])[13]
            _incl_eti =  ray.get(done_ids[0])[14]
            _vrot_eti =  ray.get(done_ids[0])[15]
            _vrad_eti =  ray.get(done_ids[0])[16]

            _i_ti =  ray.get(done_ids[0])[17]
            _i_t_reliable =  ray.get(done_ids[0])[18]


            # Compile the fitting results
            # ----------------------------------------
            # ring #
            _ring_t[_i_ti+1] = ring
            # ----------------------------------------
            # sigma
            _sigma_t[_i_ti+1] = _sigma_ti
            _sigma_et[_i_ti+1] = _sigma_eti
            # ----------------------------------------
            # xpos
            _xpos_t[_i_ti+1] = _xpos_ti
            _xpos_et[_i_ti+1] = _xpos_eti
            # ----------------------------------------
            # ypos
            _ypos_t[_i_ti+1] = _ypos_ti
            _ypos_et[_i_ti+1] = _ypos_eti
            # ----------------------------------------
            # vsys
            _vsys_t[_i_ti+1] = _vsys_ti
            _vsys_et[_i_ti+1] = _vsys_eti
            # ----------------------------------------
            # pa
            _pa_t[_i_ti+1] = _pa_ti
            _pa_et[_i_ti+1] = _pa_eti
            # ----------------------------------------
            # incl
            _incl_t[_i_ti+1] = _incl_ti
            _incl_et[_i_ti+1] = _incl_eti
            # ----------------------------------------
            # vrot
            _vrot_t[_i_ti+1] = _vrot_ti
            _vrot_et[_i_ti+1] = _vrot_eti
            # ----------------------------------------
            # vrad
            _vrad_t[_i_ti+1] = _vrad_ti
            _vrad_et[_i_ti+1] = _vrad_eti
            # ----------------------------------------
            # nrings_reliable : _i_ti
            _nrings_reliable_t[_i_ti] = _i_t_reliable


    # <-- out to _ring_t[nrings_reliable_t-1] 
    nrings_reliable = int(np.max(_nrings_reliable_t)) + 1

    #nrings_reliable = int(np.max(_nrings_reliable_t)) + 1 # + 2 to include ring[0] = 0 and _r_galaxy_plane_e
    #print("seheon", _ring_t, _pa_t, _nrings_reliable_t, nrings_reliable)

    if _intp_index == 'True':
        print("here", nrings_reliable)
        print(_ring_t[:nrings_reliable])
        print(_pa_t[:nrings_reliable])

        # at ring[0] == 0
        _ring_t[0] = 0
        _sigma_t[0] = _sigma_t[1]
        _xpos_t[0] = _xpos_t[1]
        _ypos_t[0] = _ypos_t[1]
        _vsys_t[0] = _vsys_t[1]
        _pa_t[0] = _pa_t[1]
        _incl_t[0] = _incl_t[1]
        _vrot_t[0] = _vrot_t[1]
        _vrad_t[0] = _vrad_t[1]

        _sigma_et[0] = _sigma_et[1]
        _xpos_et[0] = _xpos_et[1]
        _ypos_et[0] = _ypos_et[1]
        _vsys_et[0] = _vsys_et[1]
        _pa_et[0] = _pa_et[1]
        _incl_et[0] = _incl_et[1]
        _vrot_et[0] = _vrot_et[1]
        _vrad_et[0] = _vrad_et[1]

        print("here", nrings_reliable)
        print(_ring_t[:nrings_reliable])
        print(_pa_t[:nrings_reliable])

        _ring_t[nrings_reliable] = _ring_t[nrings_reliable-1] + ring_w # --> this will be replaced to _r_galaxy_plane_e after in _2dbat.py 
        _sigma_t[nrings_reliable] = _sigma_t[nrings_reliable-1]
        _xpos_t[nrings_reliable] = _xpos_t[nrings_reliable-1]
        _ypos_t[nrings_reliable] = _ypos_t[nrings_reliable-1]
        _vsys_t[nrings_reliable] = _vsys_t[nrings_reliable-1]
        _pa_t[nrings_reliable] = _pa_t[nrings_reliable-1]
        _incl_t[nrings_reliable] = _incl_t[nrings_reliable-1]
        _vrot_t[nrings_reliable] = _vrot_t[nrings_reliable-1]
        _vrad_t[nrings_reliable] = _vrad_t[nrings_reliable-1]

        _sigma_et[nrings_reliable] = _sigma_et[nrings_reliable-1]
        _xpos_et[nrings_reliable] = _xpos_et[nrings_reliable-1]
        _ypos_et[nrings_reliable] = _ypos_et[nrings_reliable-1]
        _vsys_et[nrings_reliable] = _vsys_et[nrings_reliable-1]
        _pa_et[nrings_reliable] = _pa_et[nrings_reliable-1]
        _incl_et[nrings_reliable] = _incl_et[nrings_reliable-1]
        _vrot_et[nrings_reliable] = _vrot_et[nrings_reliable-1]
        _vrad_et[nrings_reliable] = _vrad_et[nrings_reliable-1]

        # to include ring==_r_galaxy_plane_e (for the purpose of interpolation later) nrings_reliable = nrings_reliable_t + 2
        nrings_reliable += 1 # add extra outer ring
        print("here", nrings_reliable)
        print(_ring_t[:nrings_reliable])
        print(_pa_t[:nrings_reliable])

    return _ring_t[:nrings_reliable], \
        _sigma_t[:nrings_reliable], \
        _xpos_t[:nrings_reliable], \
        _ypos_t[:nrings_reliable], \
        _vsys_t[:nrings_reliable], \
        _pa_t[:nrings_reliable], \
        _incl_t[:nrings_reliable], \
        _vrot_t[:nrings_reliable], \
        _vrad_t[:nrings_reliable], \
        _sigma_et[:nrings_reliable], \
        _xpos_et[:nrings_reliable], \
        _ypos_et[:nrings_reliable], \
        _vsys_et[:nrings_reliable], \
        _pa_et[:nrings_reliable], \
        _incl_et[:nrings_reliable], \
        _vrot_et[:nrings_reliable], \
        _vrad_et[:nrings_reliable], \
        nrings_reliable 

#-- END OF SUB-ROUTINE____________________________________________________________#




#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def trfit_ring_by_ring_final(_input_vf, _tr_model_vf, _wi_2d, _params, fit_opt, fit_opt_2d, ndim, tr_params_priors_init, _tr2dfit_results, side, _intp_index):

    # xpos
    # ypos : ypos[nrings]
    # vsys : vsys[nrings]
    # pa : pa[nrings]
    # incl : incl[nrings]
    # vrot : vrot[nrings]

    n_cpus = int(_params['num_cpus_tr_ray'])

    nrings = 2*_params['nrings_reliable']
    ring_w = _params['ring_w']

    _sigma = _params['_sigma_init']
    _xpos = _params['_xpos_init']
    _ypos = _params['_ypos_init']
    _vsys = _params['_vsys_init']
    _pa = _params['_pa_init']
    _incl = _params['_incl_init']
    _vrot = _params['_vrot_init']
    _vrad = _params['_vrad_init']

    _ring_t = np.zeros(nrings+2, dtype=np.float)

    _sigma_t = np.zeros(nrings+2, dtype=np.float)
    _xpos_t = np.zeros(nrings+2, dtype=np.float)
    _ypos_t = np.zeros(nrings+2, dtype=np.float)
    _vsys_t = np.zeros(nrings+2, dtype=np.float)
    _pa_t = np.zeros(nrings+2, dtype=np.float)
    _incl_t = np.zeros(nrings+2, dtype=np.float)
    _vrot_t = np.zeros(nrings+2, dtype=np.float)
    _vrad_t = np.zeros(nrings+2, dtype=np.float)

    _sigma_et = np.zeros(nrings+2, dtype=np.float)
    _xpos_et = np.zeros(nrings+2, dtype=np.float)
    _ypos_et = np.zeros(nrings+2, dtype=np.float)
    _vsys_et = np.zeros(nrings+2, dtype=np.float)
    _pa_et = np.zeros(nrings+2, dtype=np.float)
    _incl_et = np.zeros(nrings+2, dtype=np.float)
    _vrot_et = np.zeros(nrings+2, dtype=np.float)
    _vrad_et = np.zeros(nrings+2, dtype=np.float)

    _nrings_reliable_t = np.zeros(nrings+2, dtype=np.float)
    _npoints_in_a_ring_total = np.zeros(nrings+2, dtype=np.float)
    _npoints_in_a_ring = np.zeros(nrings+2, dtype=np.float)

    # ---------------------------------------
    # ray.put : speed up
    _input_vf_id = ray.put(_input_vf)
    _tr_model_vf_id = ray.put(_tr_model_vf)
    _wi_2d_id = ray.put(_wi_2d)
    _params_id = ray.put(_params)
    _tr2dfit_results_id = ray.put(_tr2dfit_results)
    _i = 0
    _i_id = ray.put(_i)

    side_id = ray.put(side)

    fit_opt_id = ray.put(fit_opt)
    fit_opt_2d_id = ray.put(fit_opt_2d)
    ndim_id = ray.put(ndim)
    tr_params_priors_init_id = ray.put(tr_params_priors_init)



    # Run parallel processing
    results_ids = [trfit_ring_i_final.remote(_input_vf_id, _tr_model_vf_id, _wi_2d_id, _params_id, fit_opt_id, fit_opt_2d_id, ndim_id, tr_params_priors_init_id, _tr2dfit_results_id, side_id, _i_id) for _i_id in range(0, nrings)]

    while len(results_ids):
        done_ids, results_ids = ray.wait(results_ids)
        if done_ids:

            ring =  ray.get(done_ids[0])[0]

            _sigma_ti =  ray.get(done_ids[0])[1]
            _xpos_ti =  ray.get(done_ids[0])[2]
            _ypos_ti =  ray.get(done_ids[0])[3]
            _vsys_ti =  ray.get(done_ids[0])[4]
            _pa_ti =  ray.get(done_ids[0])[5]
            _incl_ti =  ray.get(done_ids[0])[6]
            _vrot_ti =  ray.get(done_ids[0])[7]
            _vrad_ti =  ray.get(done_ids[0])[8]

            _sigma_eti =  ray.get(done_ids[0])[9]
            _xpos_eti =  ray.get(done_ids[0])[10]
            _ypos_eti =  ray.get(done_ids[0])[11]
            _vsys_eti =  ray.get(done_ids[0])[12]
            _pa_eti =  ray.get(done_ids[0])[13]
            _incl_eti =  ray.get(done_ids[0])[14]
            _vrot_eti =  ray.get(done_ids[0])[15]
            _vrad_eti =  ray.get(done_ids[0])[16]

            _i_ti =  ray.get(done_ids[0])[17]
            _i_t_reliable =  ray.get(done_ids[0])[18]

            npoints_in_a_ring_total_including_blanks =  ray.get(done_ids[0])[19]
            npoints_in_a_ring =  ray.get(done_ids[0])[20]

            # Compile the fitting results
            # ----------------------------------------
            # ring # : ring[0] = 0, ring[1] = ring_s, ring[2] = ring_w ...
            _ring_t[_i_ti+1] = ring
            # ----------------------------------------
            # sigma
            _sigma_t[_i_ti+1] = _sigma_ti
            _sigma_et[_i_ti+1] = _sigma_eti
            # ----------------------------------------
            # xpos
            _xpos_t[_i_ti+1] = _xpos_ti
            _xpos_et[_i_ti+1] = _xpos_eti
            # ----------------------------------------
            # ypos
            _ypos_t[_i_ti+1] = _ypos_ti
            _ypos_et[_i_ti+1] = _ypos_eti
            # ----------------------------------------
            # vsys
            _vsys_t[_i_ti+1] = _vsys_ti
            _vsys_et[_i_ti+1] = _vsys_eti
            # ----------------------------------------
            # pa
            _pa_t[_i_ti+1] = _pa_ti
            _pa_et[_i_ti+1] = _pa_eti
            # ----------------------------------------
            # incl
            _incl_t[_i_ti+1] = _incl_ti
            _incl_et[_i_ti+1] = _incl_eti
            # ----------------------------------------
            # vrot
            _vrot_t[_i_ti+1] = _vrot_ti
            _vrot_et[_i_ti+1] = _vrot_eti
            # ----------------------------------------
            # vrad
            _vrad_t[_i_ti+1] = _vrad_ti
            _vrad_et[_i_ti+1] = _vrad_eti
            # ----------------------------------------
            # nrings_reliable : _i_ti
            _nrings_reliable_t[_i_ti] = _i_t_reliable

            _npoints_in_a_ring_total[_i_ti+1] = npoints_in_a_ring_total_including_blanks
            _npoints_in_a_ring[_i_ti+1] = npoints_in_a_ring

    # <-- out to _ring_t[nrings_reliable_t-1] 
    nrings_reliable = int(np.max(_nrings_reliable_t)) + 1

    #nrings_reliable = int(np.max(_nrings_reliable_t)) + 1 # + 2 to include ring[0] = 0 and _r_galaxy_plane_e
    #print("seheon", _ring_t, _pa_t, _nrings_reliable_t, nrings_reliable)

    if _intp_index == 'True':
        print("here", nrings_reliable)
        print(_ring_t[:nrings_reliable])
        print(_pa_t[:nrings_reliable])

        # at ring[0] == 0
        _ring_t[0] = 0
        _sigma_t[0] = _sigma_t[1]
        _xpos_t[0] = _xpos_t[1]
        _ypos_t[0] = _ypos_t[1]
        _vsys_t[0] = _vsys_t[1]
        _pa_t[0] = _pa_t[1]
        _incl_t[0] = _incl_t[1]
        _vrot_t[0] = 0
        _vrad_t[0] = _vrad_t[1]

        print("here", nrings_reliable)
        print(_ring_t[:nrings_reliable])
        print(_pa_t[:nrings_reliable])

        _ring_t[nrings_reliable] = _ring_t[nrings_reliable-1] + ring_w # --> this will be replaced to _r_galaxy_plane_e after in _2dbat.py 
        _sigma_t[nrings_reliable] = _sigma_t[nrings_reliable-1]
        _xpos_t[nrings_reliable] = _xpos_t[nrings_reliable-1]
        _ypos_t[nrings_reliable] = _ypos_t[nrings_reliable-1]
        _vsys_t[nrings_reliable] = _vsys_t[nrings_reliable-1]
        _pa_t[nrings_reliable] = _pa_t[nrings_reliable-1]
        _incl_t[nrings_reliable] = _incl_t[nrings_reliable-1]
        _vrot_t[nrings_reliable] = _vrot_t[nrings_reliable-1]
        _vrad_t[nrings_reliable] = _vrad_t[nrings_reliable-1]

        # to include ring==_r_galaxy_plane_e (for the purpose of interpolation later) nrings_reliable = nrings_reliable_t + 2
        nrings_reliable += 1 # add extra outer ring
        print("here", nrings_reliable)
        print(_ring_t[:nrings_reliable])
        print(_pa_t[:nrings_reliable])

    return _ring_t[:nrings_reliable], \
        _sigma_t[:nrings_reliable], \
        _xpos_t[:nrings_reliable], \
        _ypos_t[:nrings_reliable], \
        _vsys_t[:nrings_reliable], \
        _pa_t[:nrings_reliable], \
        _incl_t[:nrings_reliable], \
        _vrot_t[:nrings_reliable], \
        _vrad_t[:nrings_reliable], \
        _sigma_et[:nrings_reliable], \
        _xpos_et[:nrings_reliable], \
        _ypos_et[:nrings_reliable], \
        _vsys_et[:nrings_reliable], \
        _pa_et[:nrings_reliable], \
        _incl_et[:nrings_reliable], \
        _vrot_et[:nrings_reliable], \
        _vrad_et[:nrings_reliable], \
        _npoints_in_a_ring_total[:nrings_reliable], \
        _npoints_in_a_ring[:nrings_reliable], \
        nrings_reliable 

#-- END OF SUB-ROUTINE____________________________________________________________#



#def trfit_ring_i(_input_vf, _tr_model_vf, _params, _i, fit_opt, ndim, tr_params_priors_init):
@ray.remote(num_cpus=1)
def trfit_ring_i_final(_input_vf, _tr_model_vf, _wi_2d, _params, fit_opt, fit_opt_2d, ndim, tr_params_priors_init, _tr2dfit_results, side, _i):

    # DEG TO RAD
    deg_to_rad = np.pi / 180.

    ring_w = _params['ring_w']
    nrings_outto_rmax = int(_params['r_galaxy_plane_e'] / ring_w)

    _sigma = _params['_sigma_init']
    _xpos = _params['_xpos_init']
    _ypos = _params['_ypos_init']
    _vsys = _params['_vsys_init']
    _pa = _params['_pa_init']
    _incl = _params['_incl_init']
    _vrot = _params['_vrot_init']
    _vrad = _params['_vrad_init']

    ring_s = ring_w
    ri = ring_s + _i*ring_w - 0.5*ring_w
    ro = ring_s + _i*ring_w + 0.5*ring_w

    #if ri < 0.0: ri = 0.0
    ring = (ri+ro)/2.0

    # ---------------------------------------
    # ---------------------------------------
    # OUTER RINGS CONSTRAINTS
    # DONE INSIDE extract_tr2dfit_params 
    # ---------------------------------------
    # ---------------------------------------
    _sigma, _sigma_e, _xpos, _xpos_e, _ypos, _ypos_e, _vsys, _vsys_e, _pa, _pa_e, _incl, _incl_e, _vrot, _vrot_e, _vrad, _vrad_e \
        = extract_tr2dfit_params(_tr2dfit_results, _params, fit_opt_2d, ring)

    npoints_in_a_ring_total_including_blanks, npoints_in_ring, _ij_aring, _wt_2d = define_tilted_ring(_input_vf, _xpos, _ypos, _pa, _incl, ri, ro, side, _params)
	# the number of pixels unfiltered in each ring
    n_ij_aring = _ij_aring.shape[0]
    print(n_ij_aring, _sigma, _xpos, _ypos, _vsys, _pa, _incl, _vrot, _vrad)

    # derive the fraction of the available data points given the current ellipse ring
    _a = ring # along major axis
    _b = _a * np.cos(_incl*deg_to_rad) # along minor axis
    _l = np.pi * (3.0*(_a + _b) - ((3*_a + _b) * (_a + 3*_b))**0.5) # ellipse perimeter approximation
    _ring_area = _l * ring_w
    _f_npoints = n_ij_aring / _ring_area

    #if n_ij_aring > 2: # dynesty run
    #print(_f_npoints)
    if n_ij_aring > 10 or (n_ij_aring < 10 and _i < nrings_outto_rmax): # dynesty run

        print("ring----- ", _i, ":", n_ij_aring)
        _trfit_results, _n_dim = run_nested_sampler_trfit(_input_vf, _tr_model_vf, _wt_2d, _ij_aring, _params, fit_opt, ndim, tr_params_priors_init)

        # ----------------------------------------
        n_ring_params_free = 0
        # ----------------------------------------
        # sigma
        if _params['sigma_fitting'] == 'free':
            _sigma_t = _trfit_results[n_ring_params_free]
            _sigma_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _sigma_t = _sigma
            _sigma_et = _sigma_e

        # ----------------------------------------
        # xpos
        if _params['xpos_fitting'] == 'free':
            _xpos_t = _trfit_results[n_ring_params_free]
            _xpos_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _xpos_t = _xpos
            _xpos_et = _xpos_e

        # ----------------------------------------
        # ypos
        if _params['ypos_fitting'] == 'free':
            _ypos_t = _trfit_results[n_ring_params_free]
            _ypos_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _ypos_t = _ypos
            _ypos_et = _ypos_e

        # ----------------------------------------
        # vsys
        if _params['vsys_fitting'] == 'free':
            _vsys_t = _trfit_results[n_ring_params_free]
            _vsys_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _vsys_t = _vsys
            _vsys_et = _vsys_e

        # ----------------------------------------
        # pa
        if _params['pa_fitting'] == 'free':
            _pa_t = _trfit_results[n_ring_params_free]
            _pa_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _pa_t = _pa
            _pa_et = _pa_e

        # ----------------------------------------
        # incl
        if _params['incl_fitting'] == 'free':
            _incl_t = _trfit_results[n_ring_params_free]
            _incl_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _incl_t = _incl
            _incl_et = _incl_e

        # ----------------------------------------
        # vrot
        if _params['vrot_fitting'] == 'free':
            _vrot_t = _trfit_results[n_ring_params_free]
            _vrot_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _vrot_t = _vrot
            _vrot_et = _vrot_e

        # ----------------------------------------
        # vrad
        if _params['vrad_fitting'] == 'free':
            _vrad_t = _trfit_results[n_ring_params_free]
            _vrad_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _vrad_t = _vrad
            _vrad_et = _vrad_e

        # ----------------------------------------
        # _i : reliable ring #
        _i_t = _i
        _i_t_reliable = _i

        for p in range(0, n_ring_params_free):
            print("p-", p, "=", _trfit_results[p], "+/-", _trfit_results[ndim+p])
        
    else:
        _sigma_t = _sigma
        _xpos_t = _xpos
        _ypos_t = _ypos
        _vsys_t = _vsys
        _pa_t = _pa
        _incl_t = _incl
        _vrot_t = _vrot
        _vrad_t = _vrad
        _i_t = _i
        _i_t_reliable = -999

        _sigma_et = _sigma_e
        _xpos_et = _xpos_e
        _ypos_et = _ypos_e
        _vsys_et = _vsys_e
        _pa_et = _pa_e
        _incl_et = _incl_e
        _vrot_et = _vrot_e
        _vrad_et = _vrad_e

    return ring, _sigma_t, _xpos_t, _ypos_t, _vsys_t, _pa_t, _incl_t, _vrot_t, _vrad_t, \
                 _sigma_et, _xpos_et, _ypos_et, _vsys_et, _pa_et, _incl_et, _vrot_et, _vrad_et, \
                 _i_t, _i_t_reliable, npoints_in_a_ring_total_including_blanks, npoints_in_ring

#-- END OF SUB-ROUTINE____________________________________________________________#



@ray.remote(num_cpus=1)
def trfit_ring_i(_input_vf, _tr_model_vf, _wi_2d, _params, fit_opt, ndim, tr_params_priors_init, _i):

    # DEG TO RAD
    deg_to_rad = np.pi / 180.

    #ring_s = _params['r_galaxy_plane_s']
    ring_w = _params['ring_w']

    nrings_outto_rmax = int(_params['r_galaxy_plane_e'] / ring_w)

    _sigma = _params['_sigma_init']
    _xpos = _params['_xpos_init']
    _ypos = _params['_ypos_init']
    _vsys = _params['_vsys_init']
    _pa = _params['_pa_init']
    _incl = _params['_incl_init']
    _vrot = _params['_vrot_init']
    _vrad = _params['_vrad_init']

    side = 0
    ring_s = ring_w
    ri = ring_s + _i*ring_w - 0.5*ring_w
    ro = ring_s + _i*ring_w + 0.5*ring_w
    #if ri < 0.0: ri = 0.0
    ring = (ri+ro)/2.0

    npoints_in_a_ring_total_including_blanks, npoints_in_ring, _ij_aring, _wt_2d = define_tilted_ring(_input_vf, _xpos, _ypos, _pa, _incl, ri, ro, side, _params)

	# the number of pixels unfiltered in each ring
    n_ij_aring = _ij_aring.shape[0]

    # derive the fraction of the available data points given the current ellipse ring
    _a = ring # along major axis
    _b = _a * np.cos(_incl*deg_to_rad) # along minor axis
    _l = np.pi * (3.0*(_a + _b) - ((3*_a + _b) * (_a + 3*_b))**0.5) # ellipse perimeter approximation
    _ring_area = _l * ring_w
    _f_npoints = n_ij_aring / _ring_area

    #if n_ij_aring > 10: # dynesty run
    if n_ij_aring > 10 or (n_ij_aring < 10 and _i < nrings_outto_rmax): # dynesty run
        print("ring----- ", _i, ":", n_ij_aring)
        _trfit_results, _n_dim = run_nested_sampler_trfit(_input_vf, _tr_model_vf, _wt_2d, _ij_aring, _params, fit_opt, ndim, tr_params_priors_init)

        # ----------------------------------------
        n_ring_params_free = 0
        # ----------------------------------------
        # sigma
        if _params['sigma_fitting'] == 'free':
            _sigma_t = _trfit_results[n_ring_params_free]
            _sigma_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _sigma_t = _sigma
            _sigma_et = 999

        # ----------------------------------------
        # xpos
        if _params['xpos_fitting'] == 'free':
            _xpos_t = _trfit_results[n_ring_params_free]
            _xpos_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _xpos_t = _xpos
            _xpos_et = 999

        # ----------------------------------------
        # ypos
        if _params['ypos_fitting'] == 'free':
            _ypos_t = _trfit_results[n_ring_params_free]
            _ypos_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _ypos_t = _ypos
            _ypos_et = 999

        # ----------------------------------------
        # vsys
        if _params['vsys_fitting'] == 'free':
            _vsys_t = _trfit_results[n_ring_params_free]
            _vsys_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _vsys_t = _vsys
            _vsys_et = 999

        # ----------------------------------------
        # pa
        if _params['pa_fitting'] == 'free':
            _pa_t = _trfit_results[n_ring_params_free]
            _pa_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _pa_t = _pa
            _pa_et = 999

        # ----------------------------------------
        # incl
        if _params['incl_fitting'] == 'free':
            _incl_t = _trfit_results[n_ring_params_free]
            _incl_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _incl_t = _incl
            _incl_et = 999

        # ----------------------------------------
        # vrot
        if _params['vrot_fitting'] == 'free':
            _vrot_t = _trfit_results[n_ring_params_free]
            _vrot_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _vrot_t = _vrot
            _vrot_et = 999

        # ----------------------------------------
        # vrad
        if _params['vrad_fitting'] == 'free':
            _vrad_t = _trfit_results[n_ring_params_free]
            _vrad_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _vrad_t = _vrad
            _vrad_et = 999

        # ----------------------------------------
        # _i : reliable ring #
        _i_t = _i
        _i_t_reliable = _i

        for p in range(0, n_ring_params_free):
            print("p-", p, "=", _trfit_results[p], "+/-", _trfit_results[ndim+p])


    else:
        _sigma_t = _sigma
        _xpos_t = _xpos
        _ypos_t = _ypos
        _vsys_t = _vsys
        _pa_t = _pa
        _incl_t = _incl
        _vrot_t = _vrot
        _vrad_t = _vrad
        _i_t = _i
        _i_t_reliable = -999

        _sigma_et = 999
        _xpos_et = 999
        _ypos_et = 999
        _vsys_et = 999
        _pa_et = 999
        _incl_et = 999
        _vrot_et = 999
        _vrad_et = 999

    return ring, _sigma_t, _xpos_t, _ypos_t, _vsys_t, _pa_t, _incl_t, _vrot_t, _vrad_t, \
                 _sigma_et, _xpos_et, _ypos_et, _vsys_et, _pa_et, _incl_et, _vrot_et, _vrad_et, _i_t, _i_t_reliable

#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def set_params_fit_option(_params, sigma, sigma_fitoption, sigma_init, \
                          xpos, xpos_fitoption, xpos_init, \
                          ypos, ypos_fitoption, ypos_init, \
                          vsys, vsys_fitoption, vsys_init, \
                          pa, pa_fitoption, pa_init, \
                          incl, incl_fitoption, incl_init, \
                          vrot, vrot_fitoption, vrot_init, \
                          vrad, vrad_fitoption, vrad_init, \
                          nrings_reliable, first_run):

    # ----------------------------------------
    # sigma
    if sigma_fitoption == 'free':
        _params['sigma_fitting'] = 'free'
    else:
        _params['sigma_fitting'] = 'fixed'
    _params['_sigma_init'] = sigma_init

    # ----------------------------------------
    # xpos
    if xpos_fitoption == 'free':
        _params['xpos_fitting'] = 'free'
    else:
        _params['xpos_fitting'] = 'fixed'
    _params['_xpos_init'] = xpos_init

    # ----------------------------------------
    # ypos
    if ypos_fitoption == 'free':
        _params['ypos_fitting'] = 'free'
    else:
        _params['ypos_fitting'] = 'fixed'
    _params['_ypos_init'] = ypos_init

    # ----------------------------------------
    # vsys
    if vsys_fitoption == 'free':
        _params['vsys_fitting'] = 'free'
    else:
        _params['vsys_fitting'] = 'fixed'
    _params['_vsys_init'] = vsys_init

    # ----------------------------------------
    # pa
    if pa_fitoption == 'free':
        _params['pa_fitting'] = 'free'
    else:
        _params['pa_fitting'] = 'fixed'
    _params['_pa_init'] = pa_init

    # ----------------------------------------
    # incl
    if incl_fitoption == 'free':
        _params['incl_fitting'] = 'free'
    else:
        _params['incl_fitting'] = 'fixed'
    _params['_incl_init'] = incl_init

    # ----------------------------------------
    # vrot
    if vrot_fitoption == 'free':
        _params['vrot_fitting'] = 'free'
    else:
        _params['vrot_fitting'] = 'fixed'
    _params['_vrot_init'] = vrot_init

    # ----------------------------------------
    # vrad
    if vrad_fitoption == 'free':
        _params['vrad_fitting'] = 'free'
    else:
        _params['vrad_fitting'] = 'fixed'
    _params['_vrad_init'] = vrad_init


    if first_run == 0:
        _params['first_tr_run'] = 'True'
    else:
        _params['first_tr_run'] = 'False'

    _params['nrings_reliable'] = nrings_reliable




    # ring parameters fitting option: fixed:0 or free:1
    fit_opt = np.zeros(8, dtype=np.int32)

    # number of params to fit
    ndim_t = 0
    if _params['sigma_fitting'] == 'free':
        fit_opt[0] = 1
        ndim_t += 1
    if _params['xpos_fitting'] == 'free':
        fit_opt[1] = 1
        ndim_t += 1
    if _params['ypos_fitting'] == 'free':
        fit_opt[2] = 1
        ndim_t += 1
    if _params['vsys_fitting'] == 'free':
        fit_opt[3] = 1
        ndim_t += 1
    if _params['pa_fitting'] == 'free':
        fit_opt[4] = 1
        ndim_t += 1
    if _params['incl_fitting'] == 'free':
        fit_opt[5] = 1
        ndim_t += 1
    if _params['vrot_fitting'] == 'free':
        fit_opt[6] = 1
        ndim_t += 1
    if _params['vrad_fitting'] == 'free':
        fit_opt[7] = 1
        ndim_t += 1

    ndim = ndim_t

    # tr_params_priors init
    tr_params_priors_init = np.zeros(2*8, dtype=np.float32)

    if _params['first_tr_run'] == 'True':
        xpos0 = _params['_xc_el'] - _params['xpos_bounds_width'] 
        xpos1 = _params['_xc_el'] + _params['xpos_bounds_width'] 
        if xpos0 < 0: xpos0 = 0
        if xpos1 > _params['naxis1']-1: xpos1 = _params['naxis1'] - 2

        ypos0 = _params['_yc_el'] - _params['ypos_bounds_width'] 
        ypos1 = _params['_yc_el'] + _params['ypos_bounds_width'] 
        if ypos0 < 0: ypos0 = 0
        if ypos1 > _params['naxis2']-1: ypos1 = _params['naxis2'] - 2

        vsys0 = _params['_vsys_el'] - _params['vsys_bounds_width'] 
        vsys1 = _params['_vsys_el'] + _params['vsys_bounds_width'] 

        pa0 = _params['_theta_el'] - _params['pa_bounds_width'] 
        pa1 = _params['_theta_el'] + _params['pa_bounds_width'] 
        if pa0 < 0: pa0 = 0
        if pa1 > 360: pa1 = 360

        incl0 = _params['_i_el'] - _params['incl_bounds_width'] 
        incl1 = _params['_i_el'] + _params['incl_bounds_width'] 
        if incl0 <= 0: incl0 = 1
        if incl1 >= 90: incl2 = 89

        sigma0 = 0
        sigma1 = 100

        vrot0 = 0
        vrot1 = 500
    else:
        xpos0 = _params['_xpos_init'] - _params['xpos_bounds_width'] 
        xpos1 = _params['_xpos_init'] + _params['xpos_bounds_width'] 
        if xpos0 < 0: xpos0 = 0
        if xpos1 > _params['naxis1']-1: xpos1 = _params['naxis1'] - 2

        ypos0 = _params['_ypos_init'] - _params['ypos_bounds_width'] 
        ypos1 = _params['_ypos_init'] + _params['ypos_bounds_width'] 
        if ypos0 < 0: ypos0 = 0
        if ypos1 > _params['naxis2']-1: ypos1 = _params['naxis2'] - 2

        vsys0 = _params['_vsys_init'] - _params['vsys_bounds_width'] 
        vsys1 = _params['_vsys_init'] + _params['vsys_bounds_width'] 

        pa0 = _params['_pa_init'] - _params['pa_bounds_width'] 
        pa1 = _params['_pa_init'] + _params['pa_bounds_width'] 
        if pa0 < 0: pa0 = 0
        if pa1 > 360: pa1 = 360

        incl0 = _params['_incl_init'] - _params['incl_bounds_width'] 
        incl1 = _params['_incl_init'] + _params['incl_bounds_width'] 
        if incl0 <= 0: incl0 = 1
        if incl1 >= 90: incl2 = 89

        sigma0 = _params['_sigma_init'] - _params['sigma_bounds_width'] 
        sigma1 = _params['_sigma_init'] + _params['sigma_bounds_width'] 
        if sigma0 < 0: sigma0 = 0

        vrot0 = 0
        vrot1 = 4*_params['_vrot_init']
        if vrot1 < 0: vrot1 = 500

    tr_params_priors_init = [sigma0, xpos0, ypos0, vsys0, pa0, incl0, vrot0, -999, \
                             sigma1, xpos1, ypos1, vsys1, pa1, incl1, vrot1, 999]

    #print("xpos: ", xpos0, xpos1, "ypos: ", ypos0, ypos1, "vsys: ", vsys0, vsys1, "pa: ", pa0, pa1, "incl: ", incl0, incl1)

    return fit_opt, ndim, tr_params_priors_init

#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def trfit_2d(_input_vf, _tr_model_vf, _wi_2d, _params, tr_params_bounds, nrings_reliable, tck_vrot_bs_init_from_trfit, tck_pa_bs_init_from_trfit, tck_incl_bs_init_from_trfit, _2dbat_run_i):
    # xpos
    # ypos : ypos[nrings]
    # vsys : vsys[nrings]
    # pa : pa[nrings]
    # incl : incl[nrings]
    # vrot : vrot[nrings]

    ring_s = 0
    ring_w = 5*_params['r_galaxy_plane_e']

    _sigma = _params['_sigma_init']
    _xpos = _params['_xpos_init']
    _ypos = _params['_ypos_init']
    _pa = _params['_pa_init']
    _incl = _params['_incl_init']
    _vsys = _params['_vsys_init']
    _vrot = _params['_vrot_init']
    _vrad = _params['_vrad_init']
    ri = 0
    ro =  5*_params['r_galaxy_plane_e']
    side = 0

    npoints_total = 0 
    npoints_in_current_ring = 0

    npoints_in_a_ring_total_including_blanks, npoints_in_ring, _ij_aring, _wt_2d = define_tilted_ring(_input_vf, _xpos, _ypos, _pa, _incl, ri, ro, side, _params)

    n_ij_aring = _ij_aring.shape[0]
    r_galaxy_plane = np.zeros(n_ij_aring, dtype=np.float)

    _trfit_results, _n_dim, fit_opt_2d, std_resample_run = run_nested_sampler_trfit_2d(_input_vf, _tr_model_vf, _wt_2d, _ij_aring, _params, tr_params_bounds, nrings_reliable, r_galaxy_plane, tck_vrot_bs_init_from_trfit, tck_pa_bs_init_from_trfit, tck_incl_bs_init_from_trfit, _2dbat_run_i)

    print("")
    print("npoints:", n_ij_aring)
    for p in range(0, _n_dim):
        print("p-", p, "=", _trfit_results[p], "+/-", _trfit_results[p+_n_dim])


    return _ij_aring, _trfit_results, _n_dim, fit_opt_2d, std_resample_run



    #---------------------------------------------------
    # for plotting
    # ring parameters fitting option: fixed:0 or free:1
    fit_opt = np.zeros(8, dtype=np.int32)

    # -------------------------------------
    # NUMBER OF VROT-BS coefficients
    # VROT-BS coefficients
    n_coeffs_vrot_bs, tck_vrot_bs = bspline_ncoeffs_tck(_params, 'vrot', _params['nrings_intp'])
    # -------------------------------------
    # NUMBER OF PA-BS coefficients
    # PA-BS coefficients
    n_coeffs_pa_bs, tck_pa_bs = bspline_ncoeffs_tck(_params, 'pa', _params['nrings_intp'])
    # -------------------------------------
    # NUMBER OF INCL-BS coefficients
    # INCL-BS coefficients
    n_coeffs_incl_bs, tck_incl_bs = bspline_ncoeffs_tck(_params, 'incl', _params['nrings_intp'])


    # number of params to fit
    ndim_t = 0
    if _params['sigma_fitting'] == 'free':
        fit_opt[0] = 1
        ndim_t += 1
    if _params['xpos_fitting'] == 'free':
        fit_opt[1] = 1
        ndim_t += 1
    if _params['ypos_fitting'] == 'free':
        fit_opt[2] = 1
        ndim_t += 1
    if _params['vsys_fitting'] == 'free':
        fit_opt[3] = 1
        ndim_t += 1
    if _params['pa_fitting'] == 'free' and n_coeffs_pa_bs == 0: # constant PA
        fit_opt[4] = 1
        ndim_t += 1
    if _params['incl_fitting'] == 'free' and n_coeffs_incl_bs == 0: # constant INCL
        fit_opt[5] = 1
        ndim_t += 1
    if _params['vrot_fitting'] == 'free' and n_coeffs_vrot_bs == 0: # constant VROT
        fit_opt[6] = 1
        ndim_t += 1
    if _params['vrad_fitting'] == 'free':
        fit_opt[7] = 1
        ndim_t += 1



    # NUMBER OF VROT-BS coefficients
    # VROT-BS coefficients
    #ndim = ndim_t + n_coeffs_vrot_bs 

#    _tr_model_2d, _wi_2d, _r_galaxy_plane_given_params, _pa_bs_given_params, _incl_bs_given_params, _vrot_bs_given_params = derive_vlos_model_2d_sc(_trfit_results, _tr_model_vf, _wi_2d, _ij_aring, fit_opt, _params, nrings_reliable, r_galaxy_plane)
#                            n_coeffs_pa_bs, tck_pa_bs, \
#                            n_coeffs_incl_bs, tck_incl_bs, \
#                            n_coeffs_vrot_bs, tck_vrot_bs, \
#                            _sigma, _xpos, _ypos, _vsys, _pa, _incl, _vrot, _vrad)
#

#    r, p = derive_vlos_model_2d_sc_checkplot_vrot(_trfit_results, _tr_model_vf, _wi_2d, _ij_aring, fit_opt, _params, nrings_reliable, r_galaxy_plane)
#                            n_coeffs_pa_bs, tck_pa_bs, \
#                            n_coeffs_incl_bs, tck_incl_bs, \
#                            n_coeffs_vrot_bs, tck_vrot_bs, \
#                            _sigma, _xpos, _ypos, _vsys, _pa, _incl, _vrot, _vrad)
#
#    #print(npoints_in_ring_t)
#    fig = plt.figure()
#    ax = fig.add_subplot()
#    #ax.set_aspect('equal', adjustable='box')
#    plt.xlim(0, 180)
#    plt.ylim(0, 200)
#    plt.scatter(r, p)	
#    plt.show()


#    r, p = derive_vlos_model_2d_sc_checkplot_pa(_trfit_results, _tr_model_vf, _wi_2d, _ij_aring, fit_opt, _params, nrings_reliable, r_galaxy_plane)
#                            n_coeffs_pa_bs, tck_pa_bs, \
#                            n_coeffs_incl_bs, tck_incl_bs, \
#                            n_coeffs_vrot_bs, tck_vrot_bs, \
#                            _sigma, _xpos, _ypos, _vsys, _pa, _incl, _vrot, _vrad)
#
#    #print(npoints_in_ring_t)
#    fig = plt.figure()
#    ax = fig.add_subplot()
#    #ax.set_aspect('equal', adjustable='box')
#    plt.xlim(0, 90)
#    plt.ylim(0, 200)
#    plt.scatter(r, p)	
#    plt.show()

#    r, p = derive_vlos_model_2d_sc_checkplot_incl(_trfit_results, _tr_model_vf, _wi_2d, _ij_aring, fit_opt, _params, nrings_reliable, r_galaxy_plane)
#                            n_coeffs_pa_bs, tck_pa_bs, \
#                            n_coeffs_incl_bs, tck_incl_bs, \
#                            n_coeffs_vrot_bs, tck_vrot_bs, \
#                            _sigma, _xpos, _ypos, _vsys, _pa, _incl, _vrot, _vrad)
#
#    #print(npoints_in_ring_t)
#    fig = plt.figure()
#    ax = fig.add_subplot()
#    #ax.set_aspect('equal', adjustable='box')
#    plt.xlim(0, 90)
#    plt.ylim(0, 200)
#    plt.scatter(r, p)	
#    plt.show()

#    # Function which converts NumPy array as image
#    #ax =plt.imshow(_tr_model_2d, cmap='rainbow', vmin=0, vmax=260)
#    ax =plt.imshow(_tr_model_2d, cmap='rainbow', vmin=1400, vmax=1800)
#    #ax =plt.imshow(_tr_model_2d, cmap='rainbow', vmin=960, vmax=1080)
#    #ax =plt.imshow(_tr_model_2d, cmap='rainbow')
#    # Adding a color bar to the plot
#    plt.colorbar()
#    plt.show()

    # Function which converts NumPy array as image
    #ax =plt.imshow(_wi_2d, cmap='rainbow', vmin=960, vmax=1080)
    #ax =plt.imshow(_wi_2d, cmap='rainbow')
    # Adding a color bar to the plot
    #plt.colorbar()
    #plt.show()

#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def estimate_init_values(_params, sigma, sigma_init, \
                          xpos, xpos_init, \
                          ypos, ypos_init, \
                          vsys, vsys_init, \
                          pa, pa_init, \
                          incl, incl_init, \
                          vrot, vrot_init, \
                          vrad, vrad_init):
    try:
        # ----------------------------------------
        # sigma
        if _params['sigma_fitting'] == 'free':
            _params['_sigma_init'] = sigma_init
        # ----------------------------------------
        # xpos
        if _params['xpos_fitting'] == 'free':
            _params['_xpos_init'] = xpos_init
        # ----------------------------------------
        # ypos
        if _params['ypos_fitting'] == 'free':
            _params['_ypos_init'] = ypos_init
        # ----------------------------------------
        # vsys
        if _params['vsys_fitting'] == 'free':
            _params['_vsys_init'] = vsys_init
        # ----------------------------------------
        # pa
        if _params['pa_fitting'] == 'free':
            _params['_pa_init'] = pa_init
        # ----------------------------------------
        # incl
        if _params['incl_fitting'] == 'free':
            _params['_incl_init'] = incl_init
        # ----------------------------------------
        # vrot
        if _params['vrot_fitting'] == 'free':
            _params['_vrot_init'] = vrot_init
        # ----------------------------------------
        # vrad
        if _params['vrad_fitting'] == 'free':
            _params['_vrad_init'] = vrad_init
    except:
        pass

    print("initialized...")

#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def find_area_tofit_bg(_params):

    global _tr_model_vf

    with fits.open(_params['wdir'] + '/' + _params['input_vf'], 'update') as hdu:
        try:

            _input_vf = fitsio.read(_params['wdir'] + '/' + _params['input_vf'])
            _naxis1 = hdu[0].header['NAXIS1']
            _naxis2 = hdu[0].header['NAXIS2']
        except:
            pass

    _params['naxis1'] = _naxis1
    _params['naxis2'] = _naxis2

    _tr_model_vf = np.full((_naxis2, _naxis1), fill_value=-1E10, dtype=np.float64)
    #_____________________________________
    #-------------------------------------
    # 1. make VLOS mask
    _input_vf_vlos_LU_masked = np.where((_input_vf > _params['_vlos_lower']), _input_vf, -1E10)
    _input_vf_vlos_LU_masked = np.where((_input_vf < _params['_vlos_upper']), _input_vf_vlos_LU_masked, -1E10)

    return _input_vf_vlos_LU_masked, _tr_model_vf
#-- END OF SUB-ROUTINE____________________________________________________________#







def find_area_tofit(_params):
    global _tr_model_vf

    with fits.open(_params['wdir'] + '/' + _params['input_vf'], 'update') as hdu:
        try:
            _input_vf = hdu[0].data
            _naxis1 = hdu[0].header['NAXIS1']
            _naxis2 = hdu[0].header['NAXIS2']
        except Exception as e:
            print(f"Error: {e}")

    _params['naxis1'] = _naxis1
    _params['naxis2'] = _naxis2

    x_grid_tr, y_grid_tr = _params['x_grid_tr'], _params['y_grid_tr']
    x_grid_2d, y_grid_2d = _params['x_grid_2d'], _params['y_grid_2d']

    _tr_model_vf = np.full((_naxis2, _naxis1), fill_value=np.nan, dtype=np.float64)
    _input_vf_vlos_LU_masked_2d = np.full((_naxis2, _naxis1), fill_value=np.nan, dtype=np.float64)
    _input_vf_vlos_LU_masked_tr = np.full((_naxis2, _naxis1), fill_value=np.nan, dtype=np.float64)
    
    # 1. grid-11 map
    _input_vf_vlos_LU_masked_nogrid = np.where((_input_vf > _params['_vlos_lower']), _input_vf, np.nan)
    _input_vf_vlos_LU_masked_nogrid = np.where((_input_vf < _params['_vlos_upper']), _input_vf_vlos_LU_masked_nogrid, np.nan)


    # Assuming _input_vf, _naxis1, _naxis2, y_grid, x_grid, and _params are defined
    # Create a masked array with default value -1E10

    # --------------------------------------------
    # 1. The map for 2d fit
    # Create grid indices
    x_indices_2d = np.arange(0, _naxis1, x_grid_2d)
    y_indices_2d = np.arange(0, _naxis2, y_grid_2d)

    # Vectorized condition check
    condition_2d = (_input_vf[y_indices_2d[:, None], x_indices_2d] > _params['_vlos_lower']) & (_input_vf[y_indices_2d[:, None], x_indices_2d] < _params['_vlos_upper'])

    # Update the masked array based on the condition
    _input_vf_vlos_LU_masked_2d[y_indices_2d[:, None], x_indices_2d] = np.where(condition_2d, _input_vf[y_indices_2d[:, None], x_indices_2d], np.nan)


    # --------------------------------------------
    # 2. The map for tr fit
    # Create grid indices
    x_indices_tr = np.arange(0, _naxis1, x_grid_tr)
    y_indices_tr = np.arange(0, _naxis2, y_grid_tr)

    # Vectorized condition check
    condition_tr = (_input_vf[y_indices_tr[:, None], x_indices_tr] > _params['_vlos_lower']) & (_input_vf[y_indices_tr[:, None], x_indices_tr] < _params['_vlos_upper'])

    # Update the masked array based on the condition
    _input_vf_vlos_LU_masked_tr[y_indices_tr[:, None], x_indices_tr] = np.where(condition_tr, _input_vf[y_indices_tr[:, None], x_indices_tr], np.nan)

    return _input_vf_vlos_LU_masked_nogrid, _input_vf_vlos_LU_masked_tr, _input_vf_vlos_LU_masked_2d, _tr_model_vf
#-- END OF SUB-ROUTINE_____________________________________________________________



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def bspline_fit_to_1d(_cube_mask_2d, _params, ring_param):

    with fits.open(_params['wdir'] + '/' + _params['input_datacube'], 'update') as hdu:
        _naxis1 = hdu[0].header['NAXIS1']
        try:
            hdu[0].header['RESTFREQ'] = hdu[0].header['FREQ0'] # THIS IS NEEDED WHEN INPUTING FITS PROCESSED WITH GIPSY
        except:
            pass

#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def derive_rad_profiles(_cube_mask_2d, _params, ring_param):

    with fits.open(_params['wdir'] + '/' + _params['input_datacube'], 'update') as hdu:
        _naxis1 = hdu[0].header['NAXIS1']
        try:
            hdu[0].header['RESTFREQ'] = hdu[0].header['FREQ0'] # THIS IS NEEDED WHEN INPUTING FITS PROCESSED WITH GIPSY
        except:
            pass

    print("ok")
#-- END OF SUB-ROUTINE____________________________________________________________#






















#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def read_datacube(_params):
    global _inputDataCube

    with fits.open(_params['wdir'] + '/' + _params['input_datacube'], 'update') as hdu:
        _naxis1 = hdu[0].header['NAXIS1']
        _naxis2 = hdu[0].header['NAXIS2']
        _naxis3 = hdu[0].header['NAXIS3']

        _cdelt1 = hdu[0].header['CDELT1']
        _cdelt2 = hdu[0].header['CDELT2']
        _cdelt3 = hdu[0].header['CDELT3']
        try:
            hdu[0].header['RESTFREQ'] = hdu[0].header['FREQ0'] # THIS IS NEEDED WHEN INPUTING FITS PROCESSED WITH GIPSY
        except:
            pass

        try:
            if(hdu[0].header['CUNIT3']=='M/S' or hdu[0].header['CUNIT3']=='m/S'):
                hdu[0].header['CUNIT3'] = 'm/s'
        except KeyError:
            hdu[0].header['CUNIT3'] = 'm/s'

    _params['naxis1'] = _naxis1   
    _params['naxis2'] = _naxis2  
    _params['naxis3'] = _naxis3   
    _params['cdelt1'] = _cdelt1   
    _params['cdelt2'] = _cdelt2   
    _params['cdelt3'] = _cdelt3   

    cube = SpectralCube.read(_params['wdir'] + '/' + _params['input_datacube']).with_spectral_unit(u.km/u.s) # in km/s

    # normalise velocity-axis to 0-1 scale
    _x = np.linspace(0, 1, _naxis3, dtype=np.float32)
    _vel_min = cube.spectral_axis.min().value
    _vel_max = cube.spectral_axis.max().value
    _params['vel_min'] = _vel_min   
    _params['vel_max'] = _vel_max  

    #  _____________________________________________________________________________  #
    # [_____________________________________________________________________________] #
    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> check cube dimension...]")
    print("[--> naxis1: ", _naxis1)
    print("[--> naxis2: ", _naxis2)
    print("[--> naxis3: ", _naxis3)
    print(" ____________________________________________")
    print("[--> check cube velocity range :: velocities should be displayed in [KM/S] here...]")
    print("[--> If the velocity units are displayed with [km/s] then the input cube fortmat is fine for the 2dbat analysis...]")
    print("[--> The spectral axis of the input data cube should be in m/s ...]")
    print("")
    print("The lowest velocity [km/s]: ", _vel_min)
    print("The highest velocity [km/s]: ", _vel_max)
    print("CDELT3 [m/s]: ", _cdelt3)
    if _cdelt3 < 0:
        print("[--> Spectral axis with decreasing order...]")
    else:
        print("[--> Spectral axis with increasing order...]")
    print("")
    print("")
    #print(_x)

    #_inputDataCube = fitsio.read(_params['wdir'] + _params['input_datacube'], dtype=np.float32)
    _inputDataCube = fitsio.read(_params['wdir'] + '/' + _params['input_datacube'])
    #_spect = _inputDataCube[:,516,488]
    return _inputDataCube, _x

    #plot profile
    #plt.figure(figsize=(12, 5))
    #plt.plot(_x, _spect, color='black', marker='x', 
    #        ls='none', alpha=0.9, markersize=10)
    #plt.plot(_x, _spect, marker='o', color='red', ls='none', alpha=0.7)
    #plt.xlabel('X')
    #plt.ylabel('Y')
    #plt.tight_layout()
    #plt.show()
#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def update_header_cube_to_2d(_hdulist_nparray, _hdu_cube):
    # _hdulist_nparray: numpy array for fits whose header info is updated.
    # _hdu_cube : input data cube whose header info is used for updating the 2d fits

    #_hdulist_nparray[0].header.update(NAXIS1=_hdu[0].header['NAXIS1'])
    #_hdulist_nparray[0].header.update(NAXIS2=_hdu[0].header['NAXIS2'])
    _hdulist_nparray[0].header.insert('NAXIS2', ('CDELT1', _hdu_cube[0].header['CDELT1']), after=True)
    #_hdulist_nparray[0].header.insert('CDELT1', ('CROTA1', _hdu_cube[0].header['CROTA1']), after=True)
    #_hdulist_nparray[0].header.insert('CROTA1', ('CRPIX1', _hdu_cube[0].header['CRPIX1']), after=True)
    _hdulist_nparray[0].header.insert('CDELT1', ('CRPIX1', _hdu_cube[0].header['CRPIX1']), after=True)
    _hdulist_nparray[0].header.insert('CRPIX1', ('CRVAL1', _hdu_cube[0].header['CRVAL1']), after=True)
    _hdulist_nparray[0].header.insert('CRVAL1', ('CTYPE1', _hdu_cube[0].header['CTYPE1']), after=True)
    try:
        _hdulist_nparray[0].header.insert('CTYPE1', ('CUNIT1', _hdu_cube[0].header['CUNIT1']), after=True)
    except:
        _hdulist_nparray[0].header.insert('CTYPE1', ('CUNIT1', 'deg'), after=True)


    _hdulist_nparray[0].header.insert('CUNIT1', ('CDELT2', _hdu_cube[0].header['CDELT2']), after=True)
    #_hdulist_nparray[0].header.insert('CDELT2', ('CROTA2', _hdu_cube[0].header['CROTA2']), after=True)
    #_hdulist_nparray[0].header.insert('CROTA2', ('CRPIX2', _hdu_cube[0].header['CRPIX2']), after=True)
    _hdulist_nparray[0].header.insert('CDELT2', ('CRPIX2', _hdu_cube[0].header['CRVAL2']), after=True)
    _hdulist_nparray[0].header.insert('CRPIX2', ('CRVAL2', _hdu_cube[0].header['CRVAL2']), after=True)
    _hdulist_nparray[0].header.insert('CRVAL2', ('CTYPE2', _hdu_cube[0].header['CTYPE2']), after=True)

    try:
        _hdulist_nparray[0].header.insert('CTYPE2', ('CUNIT2', _hdu_cube[0].header['CUNIT2']), after=True)
    except:
        _hdulist_nparray[0].header.insert('CTYPE2', ('CUNIT2', 'deg'), after=True)

    #_hdulist_nparray[0].header.insert('CUNIT2', ('EPOCH', _hdu_cube[0].header['EPOCH']), after=True)


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def write_fits_seg(_segarray, _segfitsfile):
    hdu = fits.PrimaryHDU(data=_segarray)
    hdu.writeto(_segfitsfile, overwrite=True)
#-- END OF SUB-ROUTINE____________________________________________________________#




#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def write_fits_model_vf(_params):

    with fits.open(_params['wdir'] + '/' + _params['input_datacube'], 'update') as hdu:
        _naxis3 = hdu[0].header['NAXIS3']
        cdelt3 = abs(hdu[0].header['CDELT3'])
        try:
            hdu[0].header['RESTFREQ'] = hdu[0].header['FREQ0']    # in case the input cube is pre-processed using GIPSY
        except:
            pass
    
        try:
            if(hdu[0].header['CUNIT3']=='M/S' or hdu[0].header['CUNIT3']=='m/S'):
                hdu[0].header['CUNIT3'] = 'm/s'
        except KeyError:
            hdu[0].header['CUNIT3'] = 'm/s'


    #_____________________________________
    #-------------------------------------
    # 0. load the input cube
    #cubedata = fitsio.read(_params['wdir'] + _params['input_datacube'], dtype=np.float32)
    _input_cube = SpectralCube.read(_params['wdir'] + '/' + _params['input_datacube']).with_spectral_unit(u.km/u.s, velocity_convention='radio')


    # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
    # set _input_cube.beam_threshold > 0.01, e.g., 0.1 : 10%, normally < 1%
    _input_cube.beam_threshold = 0.1

   
    #_____________________________________
    #-------------------------------------
    # 1. make a mask
    _flux_threshold = _params['mom0_nrms_limit']*_params['_rms_med'] + _params['_bg_med']
    peak_sn_mask = _input_cube > _flux_threshold*u.Jy/u.beam
    #print("flux_threshold:", _flux_threshold)

    # 2. extract profiles > _flux_threhold
    _input_cube_peak_sn_masked = _input_cube.with_mask(peak_sn_mask)

    # 3. extract mom0
    mom0 = _input_cube_peak_sn_masked.with_spectral_unit(u.km/u.s).moment(order=0, axis=0)

    # 4. extrac N
    # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
    # hdulist should be used instead of hdu
    if _input_cube.beam_threshold > 0.09: # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
        _N_masked = np.where(np.isnan(_input_cube_peak_sn_masked.hdulist[0].data), -1E5, _input_cube_peak_sn_masked.hdulist[0].data)
    #_N_masked = np.where(np.isnan(_input_cube_peak_sn_masked.hdu.data), -1E5, _input_cube_peak_sn_masked.hdu.data)

    # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403 : VLA + SINGLE DISH
    # hdulist should be used instead of hdu

	# UNCOMMENT FOR NGC 2403 multi-resolution cube !!!!!!!!!!!!!!!!!!!
    #if _input_cube.beam_threshold > 0.09: # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
    #    _N_masked = np.where(np.isnan(_input_cube_peak_sn_masked.hdulist[0].data), -1E5, _input_cube_peak_sn_masked.hdulist[0].data)
    _N_masked = np.where(np.isnan(_input_cube_peak_sn_masked.hdu.data), -1E5, _input_cube_peak_sn_masked.hdu.data)

    _N_masked = np.where(np.isinf(_N_masked), -1E5, _N_masked)
    _N_masked = np.where(np.isinf(-1*_N_masked), -1E5, _N_masked)
    _N = (_N_masked > -1E5).sum(axis=0)

    # 5. derive integerated rms
    _rms_int = np.sqrt(_N) * _params['_rms_med'] * (cdelt3/1000.)


    # 6. integrated s/n map --> spectralcube array
    _sn_int_map = mom0 / _rms_int
    #print(_params['_rms_med'], _params['_bg_med'])
    #print(mom0)

    # 7. integrated s/n map: numpy array : being returned 

    # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
    # hdulist should be used instead of hdu
    #if _input_cube.beam_threshold > 0.09: # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
    #    _sn_int_map_nparray = _sn_int_map.hdulist[0].data
    _sn_int_map_nparray = _sn_int_map.hdu.data
    _sn_int_map_nparray = np.where(np.isnan(_sn_int_map_nparray), 0, _sn_int_map_nparray)
    _sn_int_map_nparray = np.where(np.isinf(_sn_int_map_nparray), 1E5, _sn_int_map_nparray)
    _sn_int_map_nparray = np.where(np.isinf(-1*_sn_int_map_nparray), -1E5, _sn_int_map_nparray)


    #_____________________________________
    #-------------------------------------
    # make a peak s/n map
    # 1. extract the peak flux map
    # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
    # hdulist should be used instead of hdu
	# UNCOMMENT FOR NGC 2403 multi-resolution cube !!!!!!!!!!!!!!!!!!!
    #if _input_cube.beam_threshold > 0.09: # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
    #    peak_flux_map = _input_cube.hdulist[0].data.max(axis=0)
    peak_flux_map = _input_cube.hdu.data.max(axis=0)
    peak_flux_map = np.where(np.isnan(peak_flux_map), -1E5, peak_flux_map)
    peak_flux_map = np.where(np.isinf(peak_flux_map), 1E5, peak_flux_map)
    peak_flux_map = np.where(np.isinf(-1*peak_flux_map), -1E5, peak_flux_map)

    # 2. peak s/n map
    peak_sn_map = (peak_flux_map - _params['_bg_med']) / _params['_rms_med']


    #-------------------------------------
    # write fits
    # moment0
    mom0.write('test1.mom0.fits', overwrite=True)

    # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
    # hdulist should be used instead of hdu
    #if _input_cube.beam_threshold > 0.09: # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
    #    _sn_int_map.hdulist[0].header['BUNIT'] = 's/n'
    _sn_int_map.hdu.header['BUNIT'] = 's/n'
    # write fits
    # _sn_int_map
    _sn_int_map.write('test1.sn_int.fits', overwrite=True)
    #print('moment0_unit:', mom0.unit)

    return peak_sn_map, _sn_int_map_nparray
#-- END OF SUB-ROUTINE____________________________________________________________#










#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def moment_analysis(_params):

    with fits.open(_params['wdir'] + '/' + _params['input_datacube'], 'update') as hdu:
        _naxis3 = hdu[0].header['NAXIS3']
        cdelt3 = abs(hdu[0].header['CDELT3'])
        try:
            hdu[0].header['RESTFREQ'] = hdu[0].header['FREQ0']    # in case the input cube is pre-processed using GIPSY
        except:
            pass
    
        try:
            if(hdu[0].header['CUNIT3']=='M/S' or hdu[0].header['CUNIT3']=='m/S'):
                hdu[0].header['CUNIT3'] = 'm/s'
        except KeyError:
            hdu[0].header['CUNIT3'] = 'm/s'


    #_____________________________________
    #-------------------------------------
    # 0. load the input cube
    #cubedata = fitsio.read(_params['wdir'] + _params['input_datacube'], dtype=np.float32)
    _input_cube = SpectralCube.read(_params['wdir'] + '/' + _params['input_datacube']).with_spectral_unit(u.km/u.s, velocity_convention='radio')


    # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
    # set _input_cube.beam_threshold > 0.01, e.g., 0.1 : 10%, normally < 1%
    _input_cube.beam_threshold = 0.1

   
    #_____________________________________
    #-------------------------------------
    # 1. make a mask
    _flux_threshold = _params['mom0_nrms_limit']*_params['_rms_med'] + _params['_bg_med']
    peak_sn_mask = _input_cube > _flux_threshold*u.Jy/u.beam
    #print("flux_threshold:", _flux_threshold)

    # 2. extract profiles > _flux_threhold
    _input_cube_peak_sn_masked = _input_cube.with_mask(peak_sn_mask)

    # 3. extract mom0
    mom0 = _input_cube_peak_sn_masked.with_spectral_unit(u.km/u.s).moment(order=0, axis=0)

    # 4. extrac N
    # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
    # hdulist should be used instead of hdu
    if _input_cube.beam_threshold > 0.09: # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
        _N_masked = np.where(np.isnan(_input_cube_peak_sn_masked.hdulist[0].data), -1E5, _input_cube_peak_sn_masked.hdulist[0].data)
    #_N_masked = np.where(np.isnan(_input_cube_peak_sn_masked.hdu.data), -1E5, _input_cube_peak_sn_masked.hdu.data)

    # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403 : VLA + SINGLE DISH
    # hdulist should be used instead of hdu

	# UNCOMMENT FOR NGC 2403 multi-resolution cube !!!!!!!!!!!!!!!!!!!
    #if _input_cube.beam_threshold > 0.09: # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
    #    _N_masked = np.where(np.isnan(_input_cube_peak_sn_masked.hdulist[0].data), -1E5, _input_cube_peak_sn_masked.hdulist[0].data)
    _N_masked = np.where(np.isnan(_input_cube_peak_sn_masked.hdu.data), -1E5, _input_cube_peak_sn_masked.hdu.data)

    _N_masked = np.where(np.isinf(_N_masked), -1E5, _N_masked)
    _N_masked = np.where(np.isinf(-1*_N_masked), -1E5, _N_masked)
    _N = (_N_masked > -1E5).sum(axis=0)

    # 5. derive integerated rms
    _rms_int = np.sqrt(_N) * _params['_rms_med'] * (cdelt3/1000.)


    # 6. integrated s/n map --> spectralcube array
    _sn_int_map = mom0 / _rms_int
    #print(_params['_rms_med'], _params['_bg_med'])
    #print(mom0)

    # 7. integrated s/n map: numpy array : being returned 

    # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
    # hdulist should be used instead of hdu
    #if _input_cube.beam_threshold > 0.09: # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
    #    _sn_int_map_nparray = _sn_int_map.hdulist[0].data
    _sn_int_map_nparray = _sn_int_map.hdu.data
    _sn_int_map_nparray = np.where(np.isnan(_sn_int_map_nparray), 0, _sn_int_map_nparray)
    _sn_int_map_nparray = np.where(np.isinf(_sn_int_map_nparray), 1E5, _sn_int_map_nparray)
    _sn_int_map_nparray = np.where(np.isinf(-1*_sn_int_map_nparray), -1E5, _sn_int_map_nparray)


    #_____________________________________
    #-------------------------------------
    # make a peak s/n map
    # 1. extract the peak flux map
    # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
    # hdulist should be used instead of hdu
	# UNCOMMENT FOR NGC 2403 multi-resolution cube !!!!!!!!!!!!!!!!!!!
    #if _input_cube.beam_threshold > 0.09: # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
    #    peak_flux_map = _input_cube.hdulist[0].data.max(axis=0)
    peak_flux_map = _input_cube.hdu.data.max(axis=0)
    peak_flux_map = np.where(np.isnan(peak_flux_map), -1E5, peak_flux_map)
    peak_flux_map = np.where(np.isinf(peak_flux_map), 1E5, peak_flux_map)
    peak_flux_map = np.where(np.isinf(-1*peak_flux_map), -1E5, peak_flux_map)

    # 2. peak s/n map
    peak_sn_map = (peak_flux_map - _params['_bg_med']) / _params['_rms_med']


    #-------------------------------------
    # write fits
    # moment0
    mom0.write('test1.mom0.fits', overwrite=True)

    # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
    # hdulist should be used instead of hdu
    #if _input_cube.beam_threshold > 0.09: # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
    #    _sn_int_map.hdulist[0].header['BUNIT'] = 's/n'
    _sn_int_map.hdu.header['BUNIT'] = 's/n'
    # write fits
    # _sn_int_map
    _sn_int_map.write('test1.sn_int.fits', overwrite=True)
    #print('moment0_unit:', mom0.unit)

    return peak_sn_map, _sn_int_map_nparray
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def moment_analysis_alternate(_params):

    with fits.open(_params['wdir'] + '/' + _params['input_datacube'], 'update') as hdu:
        _naxis3 = hdu[0].header['NAXIS3']
        cdelt3 = abs(hdu[0].header['CDELT3'])
        try:
            hdu[0].header['RESTFREQ'] = hdu[0].header['FREQ0']    # in case the input cube is pre-processed using GIPSY
        except:
            pass
    
        try:
            if(hdu[0].header['CUNIT3']=='M/S' or hdu[0].header['CUNIT3']=='m/S'):
                hdu[0].header['CUNIT3'] = 'm/s'
        except KeyError:
            hdu[0].header['CUNIT3'] = 'm/s'
    
    
    #cubedata = fitsio.read(_params['wdir'] + _params['input_datacube'], dtype=np.float32)
    cubedata = fitsio.read(_params['wdir'] + '/' + _params['input_datacube'])
   
    # peak s/n
    _chan_linefree1 = np.mean(fits.open(_params['wdir'] + '/' + _params['input_datacube'])[0].data[0:int(_naxis3*0.05):1, :, :], axis=0) # first 5% channels
    _chan_linefree2 = np.mean(fits.open(_params['wdir'] + '/' + _params['input_datacube'])[0].data[int(_naxis3*0.95):_naxis3-1:1, :, :], axis=0) # last 5% channels
    _chan_linefree = (_chan_linefree1 + _chan_linefree2)/2.

    # masking nan, inf, -inf
    _chan_linefree = np.where(np.isnan(_chan_linefree), -1E5, _chan_linefree)
    _chan_linefree = np.where(np.isinf(_chan_linefree), 1E5, _chan_linefree)
    _chan_linefree = np.where(np.isinf(-1*_chan_linefree), -1E5, _chan_linefree)
    #print(_chan_linefree.shape)
    _mean_bg, _median_bg, _std_bg = sigma_clipped_stats(_chan_linefree, sigma=3.0)
    #print(_mean_bg, _median_bg, _std_bg)
    # use _params['_rms_med'] instead of _std_bg which tends to be lower

    _flux_threshold = _params['mom0_nrms_limit']*_params['_rms_med'] + _median_bg
    _input_cube = SpectralCube.read(_params['wdir'] + '/' + _params['input_datacube']).with_spectral_unit(u.km/u.s, velocity_convention='radio')

    # make a mask
    peak_sn_mask = _input_cube > _flux_threshold*u.Jy/u.beam
    # extract lines > peak_sn
    _input_cube_peak_sn_masked = _input_cube.with_mask(peak_sn_mask)
    # extract mom0
    mom0 = _input_cube_peak_sn_masked.with_spectral_unit(u.km/u.s).moment(order=0, axis=0)

    # extract the peak flux map
    peak_flux_map = _input_cube.hdu.data.max(axis=0)
    peak_flux_map = np.where(np.isnan(peak_flux_map), -1E5, peak_flux_map)
    peak_flux_map = np.where(np.isinf(peak_flux_map), 1E5, peak_flux_map)
    peak_flux_map = np.where(np.isinf(-1*peak_flux_map), -1E5, peak_flux_map)


    # make a peak s/n map
    peak_sn_map = (peak_flux_map - _median_bg) / _params['_rms_med']
    #print("peak sn")
    #print(peak_sn_map)

    _N_masked = np.where(np.isnan(_input_cube_peak_sn_masked.hdu.data), -1E5, _input_cube_peak_sn_masked.hdu.data)
    _N_masked = np.where(np.isinf(_N_masked), -1E5, _N_masked)
    _N_masked = np.where(np.isinf(-1*_N_masked), -1E5, _N_masked)

    #print(_N_masked)
    _N = (_N_masked > -1E5).sum(axis=0)
    #print(_N)
    _rms_int = np.sqrt(_N) * _params['_rms_med'] * (_params['cdelt3']/1000.)

    print(mom0)
    print(_rms_int)
    # integrated s/n map: spectralcube array
    _sn_int_map = mom0 / _rms_int
    #print("int sn")
    #print(_sn_int_map)
    # integrated s/n map: numpy array : being returned 
    #print(_sn_int_map)

    _sn_int_map_nparray = _sn_int_map.hdu.data
    _sn_int_map_nparray = np.where(np.isnan(_sn_int_map_nparray), 0, _sn_int_map_nparray)
    _sn_int_map_nparray = np.where(np.isinf(_sn_int_map_nparray), 1E5, _sn_int_map_nparray)
    _sn_int_map_nparray = np.where(np.isinf(-1*_sn_int_map_nparray), -1E5, _sn_int_map_nparray)

    # write fits
    # moment0
    mom0.write('test1.mom0.fits', overwrite=True)
    _sn_int_map.hdu.header['BUNIT'] = 's/n'
    # write fits
    # _sn_int_map
    _sn_int_map.write('test1.sn_int.fits', overwrite=True)
    #print('moment0_unit:', mom0.unit)

    return peak_sn_map, _sn_int_map_nparray
#-- END OF SUB-ROUTINE____________________________________________________________#
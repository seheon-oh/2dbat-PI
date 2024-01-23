#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#|-----------------------------------------|
#| _dynesty_sampler.py
#|-----------------------------------------|
#|
#| version history
#| v1.0 (2022 Dec 25)
#|
#|-----------------------------------------|
#| by Se-Heon Oh
#| Dept. of Physics and Astronomy
#| Sejong University, Seoul, South Korea
#|-----------------------------------------|

#|-----------------------------------------|
# system functions
import time, sys, os
import io
import contextlib


#|-----------------------------------------|
from re import A, I
import sys
import numpy as np
import numpy.ma as ma
from numpy import sum, exp, log, pi
from numpy import linalg, array, sum, log, exp, pi, std, diag, concatenate


import scipy
from scipy.interpolate import BSpline, splrep, splev
from scipy import optimize

from scipy.optimize import minimize

from itertools import zip_longest
from numba import njit
from scipy.optimize import fsolve
from scipy.interpolate import interp1d

from scipy.optimize import brentq
from scipy.optimize import newton
from scipy.optimize import root_scalar

from scipy.optimize import newton

#from solve_r_galaxy_plane import solve_r_galaxy_plane_jit_pa_bs_incl_const_test


from collections import Counter

import matplotlib.pyplot as plt
import math

#|-----------------------------------------|
# TEST 
import numba
from numba import njit
from numba import jit
from numba import vectorize, float64, jit, cuda
import numba as nb
from numba import njit, prange

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

from joblib import Parallel, delayed


#|-----------------------------------------|
import dynesty
from dynesty import NestedSampler
from dynesty import DynamicNestedSampler
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
from dynesty.utils import resample_equal

import gc
import ray
import pickle


import multiprocessing as mp
import dynesty.pool as dypool


from astropy.io import fits


#|-----------------------------------------|
# import make_dirs
from _dirs_files import make_dirs

# Set the number of threads for Julia
num_threads = 8  # Set this to the number of threads you want to use
os.environ["JULIA_NUM_THREADS"] = str(num_threads)

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def max_val(x, y):
    if x > y:
        return x
    else:
        return y
#-- END OF SUB-ROUTINE____________________________________________________________#

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def min_val(x, y):
    if x < y:
        return x
    else:
        return y
#-- END OF SUB-ROUTINE____________________________________________________________#

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def nint_val(x):
    if x > 0.0:
        return (int)(x+0.5)
    else:
        return (int)(x-0.5)
#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def define_tilted_ring(_input_vf, xpos, ypos, pa, incl, ri, ro, side, _params):

    naxis1 = _params['naxis1']
    naxis2 = _params['naxis2']
    _wt_2d = np.full((naxis2, naxis1), fill_value=0, dtype=np.float)

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
    free_angle = _params['free_angle']
    sine_free_angle = np.fabs(np.sin(free_angle*deg_to_rad))

    sinp = np.sin(deg_to_rad * pa)       # sine of pa. 
    cosp = np.cos(deg_to_rad * pa)       # cosine of pa. 
    sini = np.sin(deg_to_rad * incl)     #   sine of inc. 
    cosi = np.cos(deg_to_rad * incl)     # cosine of inc. 
    a = (1.0 - cosp * cosp * sini * sini)**0.5
    b = (1.0 - sinp * sinp * sini * sini)**0.5

    if np.isnan(a) or np.isinf(a): a = naxis1
    if np.isnan(b) or np.isinf(b): b = naxis2

    #i0_lo = max_val(0, nint_val(xpos - a * ro_deg / cdelt1))
    #i0_up = min_val(naxis1, nint_val( xpos + a * ro_deg / cdelt1))
    #j0_lo = max_val(0, nint_val(ypos - b * ro_deg / cdelt2))
    #j0_up = min_val(naxis2, nint_val(ypos + b * ro_deg / cdelt2))

    i0_lo = 0
    i0_up = naxis1
    j0_lo = 0
    j0_up = naxis2

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
    # 1. npoints_in_a_ring_total_including_blanks 
    # Generate a grid of i0 and j0 values
    # ..............................................
    i0, j0 = np.meshgrid(range(i0_lo, i0_up), range(j0_lo, j0_up), indexing='ij')

    # Calculate positions
    rx = cdelt1 * i0
    ry = cdelt2 * j0

    # Rotate and scale the coordinates
    xr = (- (rx - cdelt1 * xpos) * sinp + (ry - cdelt2 * ypos) * cosp)
    yr = (- (rx - cdelt1 * xpos) * cosp - (ry - cdelt2 * ypos) * sinp) / cosi

    # Compute distance from the center and theta
    r = np.sqrt(xr**2 + yr**2)
    theta = np.where(r < 0.1, 0.0, np.arctan2(yr, xr) / deg_to_rad) # in degrees

    # Compute cosine of theta
    costh = np.abs(np.cos(deg_to_rad * theta)) # in radians

    # Count points within the specified ring

    if side == -1: # approaching side
        npoints_in_a_ring_total_including_blanks_app = np.sum((r > ri_deg) & (r < ro_deg) & (np.fabs(theta) >= 90.0))
        #print(ri_deg, ro_deg, npoints_in_a_ring_total_including_blanks_app)
    elif side == 1: # receding side
        npoints_in_a_ring_total_including_blanks_rec = np.sum((r > ri_deg) & (r < ro_deg) & (np.fabs(theta) < 90.0))
        #print(ri_deg, ro_deg, npoints_in_a_ring_total_including_blanks_rec)
    elif side == 0: # both sides
        npoints_in_a_ring_total_including_blanks_both = np.sum((r > ri_deg) & (r < ro_deg))
        #print(ri_deg, ro_deg, npoints_in_a_ring_total_including_blanks_both)


    # ..............................................
    # between ri_deg and ro_deg
    x0 = 0
    y0 = 0
    x1 = naxis1
    y1 = naxis2
		
#    # ..............................................
#    # Derive weight map
#    #_input_vf_weight = np.zeros((naxis2, naxis1), dtype=float)
#    wpow = 1
#    npoints_in_ring_t = 0
#    for i0 in range(i0_lo, i0_up):
#        for j0 in range(j0_lo, j0_up):
#
#            rx = cdelt1 * i0  # X position in plane of galaxy
#            ry = cdelt2 * j0  # Y position in plane of galaxy
#            #if not np.isinf(_input_vf[j0, i0]) and not np.isnan(_input_vf[j0, i0] and _input_vf[j0, i0] > -1E8 and _input_vf[j0, i0] < 1E8):
#            if _input_vf[j0, i0] > -1E8 and _input_vf[j0, i0] < 1E8:
#
#                xr = ( - ( rx - cdelt1 * xpos ) * sinp + ( ry - cdelt2 * ypos ) * cosp )
#                yr = ( - ( rx - cdelt1 * xpos ) * cosp - ( ry - cdelt2 * ypos ) * sinp ) / cosi
#                r = ( xr**2 + yr**2 )**0.5  # distance from centre
#                if r < 0.1:
#                    theta = 0.0
#                else:
#                    theta = math.atan2( yr, xr ) / deg_to_rad # in degree
#
#                costh = np.fabs( np.cos ( deg_to_rad * theta ) ) # in radian
#
#                # put weight
#                if r > ri_deg and r < ro_deg and costh > sine_free_angle: # both sides
#                    #_input_vf_weight[j0, i0] = costh**wpow # weight : note that radial weight doesn't need to be applied as all points within a ring have the same radius.
#                    npoints_in_ring_t += 1

    # ..............................................
    # 2. npoints_in_ring_t
    # Generate a grid of i0 and j0 values
    # ..............................................
    i0, j0 = np.meshgrid(range(i0_lo, i0_up), range(j0_lo, j0_up), indexing='ij')

    # Filter valid points based on _input_vf values
    valid_points = (_input_vf[j0, i0] > _params['_vlos_lower']) & (_input_vf[j0, i0] < _params['_vlos_upper'])

    # Calculate positions
    rx = cdelt1 * i0
    ry = cdelt2 * j0

    # Rotate and scale the coordinates
    xr = (- (rx - cdelt1 * xpos) * sinp + (ry - cdelt2 * ypos) * cosp)
    yr = (- (rx - cdelt1 * xpos) * cosp - (ry - cdelt2 * ypos) * sinp) / cosi

    # Compute distance from the center and theta
    r = np.sqrt(xr**2 + yr**2)
    theta = np.where(r < 0.1, 0.0, np.arctan2(yr, xr) / deg_to_rad) # in degrees

    # Compute cosine of theta
    costh = np.abs(np.cos(deg_to_rad * theta)) # in radians

    # Apply conditions for counting pixels
    #weighted_pixels = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle)

    # Count the pixels that meet all conditions
    #npoints_in_ring_t = np.sum(weighted_pixels)

    if side == -1: # approaching side
        weighted_pixels = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle) & (np.fabs(theta) >= 90.0)
        npoints_in_ring_t_app = np.sum(weighted_pixels)
        #print(ri_deg, ro_deg, npoints_in_ring_t_app)
    elif side == 1: # receding side
        weighted_pixels = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle) & (np.fabs(theta) < 90.0)
        npoints_in_ring_t_rec = np.sum(weighted_pixels)
        #print(ri_deg, ro_deg, npoints_in_ring_t_rec)
    elif side == 0: # both sides
        weighted_pixels = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle)
        npoints_in_ring_t_both = np.sum(weighted_pixels)
        #print(ri_deg, ro_deg, npoints_in_ring_t_both)

#    # ..............................................
#    # Reset the connected area to fit
#    # [0:i, 1:j, 2:tr_vlos_model]
#    ij_tilted_ring = np.array([], dtype=np.float64)
#    ij_tilted_ring.shape = (0, 3)
#
#    x0 = 0
#    y0 = 0
#    x1 = naxis1
#    y1 = naxis2
#    # ..............................................
#    npoints_in_ring_t = 0
#    for i0 in range(i0_lo, i0_up):
#        for j0 in range(j0_lo, j0_up):
#            rx = cdelt1 * i0  # X position in plane of galaxy
#            ry = cdelt2 * j0  # Y position in plane of galaxy
#            #if not np.isinf(_input_vf[j0, i0]) and not np.isnan(_input_vf[j0, i0] and _input_vf[j0, i0] > -1E8 and _input_vf[j0, i0] < 1E8):
#            if _input_vf[j0, i0] > -1E8 and _input_vf[j0, i0] < 1E8:
#
#                xr = ( - ( rx - cdelt1 * xpos ) * sinp + ( ry - cdelt2 * ypos ) * cosp )
#                yr = ( - ( rx - cdelt1 * xpos ) * cosp - ( ry - cdelt2 * ypos ) * sinp ) / cosi
#                r = ( xr**2 + yr**2 )**0.5  # distance from centre
#                if r < 0.1:
#                    theta = 0.0
#                else:
#                    theta = math.atan2( yr, xr ) / deg_to_rad # in degree
#
#                costh = np.fabs( np.cos(deg_to_rad * theta) ) # in radian
#                if r > ri_deg and r < ro_deg and costh > sine_free_angle:
#                    # [0:i, 1:j, 2:tr_vlos_model]
#                    ij_point = np.array([[i0, j0, -1E10]])
#                    ij_tilted_ring = np.concatenate( (ij_tilted_ring, ij_point) )
#                    #ij_tilted_ring = _ij_tilted_ring
#                    npoints_in_ring_t += 1
#                    
#                    #if npoints_in_ring_t == 0:
#                    #    ij_tilted_ring[int(npoints_in_ring_t), 0] = i0
#                    #    ij_tilted_ring[int(npoints_in_ring_t), 1] = j0
#                    #    npoints_in_ring_t += 1
#                    #else:
#                    #    np.concatenate((ij_tilted_ring, [i0, j0]), axis=0)
#                    #    npoints_in_ring_t += 1
#
#    #-- ij_tilted_ring = np.zeros((naxis2*naxis1, 2), dtype=int)
#
#    #print(npoints_in_ring_t)
#    #fig = plt.figure()
#    #ax = fig.add_subplot()
#    #ax.set_aspect('equal', adjustable='box')
#    #plt.xlim(0, 40)
#    #plt.ylim(0, 40)
#    #plt.scatter(ij_tilted_ring[:,0], ij_tilted_ring[:,1])	
#    #plt.show()

    # ..............................................
    # 3. Reset the connected area to fit
    # [0:i, 1:j, 2:tr_vlos_model]
    # Generate a grid of i0 and j0 values
    # ..............................................
    i0, j0 = np.meshgrid(range(i0_lo, i0_up), range(j0_lo, j0_up), indexing='ij')

    # Filter valid points based on _input_vf values
    valid_points = (_input_vf[j0, i0] > _params['_vlos_lower']) & (_input_vf[j0, i0] < _params['_vlos_upper'])

    # Calculate positions
    rx = cdelt1 * i0
    ry = cdelt2 * j0

    # Rotate and scale the coordinates
    xr = (- (rx - cdelt1 * xpos) * sinp + (ry - cdelt2 * ypos) * cosp)
    yr = (- (rx - cdelt1 * xpos) * cosp - (ry - cdelt2 * ypos) * sinp) / cosi

    # Compute distance from the center and theta
    r = np.sqrt(xr**2 + yr**2)
    theta = np.where(r < 0.1, 0.0, np.arctan2(yr, xr) / deg_to_rad)  # in degrees

    # Compute cosine of theta
    costh = np.abs(np.cos(deg_to_rad * theta))  # in radians
    # compute weight: 0-uniform, 1, or 2
    cosine_weight = costh ** _params['cosine_weight_power']

    # Apply conditions for selecting points
    selected_points = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle)

    if side == -1: # approaching side
        selected_points = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle) & (np.fabs(theta) >= 90.0)
    elif side == 1: # receding side
        selected_points = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle) & (np.fabs(theta) < 90.0)
    elif side == 0: # both sides
        selected_points = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle)

    # Extract the coordinates of the selected points
    i0_selected = i0[selected_points]
    j0_selected = j0[selected_points]

    # Create an array of points with the desired format
    ij_tilted_ring = np.stack((i0_selected.flatten(), j0_selected.flatten(), np.full(i0_selected.size, np.nan)), axis=-1)

    _wt_2d[j0_selected, i0_selected] = cosine_weight[selected_points]

    if side == -1: # approaching
        return npoints_in_a_ring_total_including_blanks_app, npoints_in_ring_t_app, ij_tilted_ring, _wt_2d
    elif side == 1: # receding
        return npoints_in_a_ring_total_including_blanks_rec, npoints_in_ring_t_rec, ij_tilted_ring, _wt_2d
    elif side == 0: # both
        return npoints_in_a_ring_total_including_blanks_both, npoints_in_ring_t_both, ij_tilted_ring, _wt_2d

#-- END OF SUB-ROUTINE____________________________________________________________#




def set_phi_bounds(phi1, vrot1, del_vrot, i1, del_i, _params, ij_aring):

    deg_to_rad = np.pi / 180.

    xpos = _params['_xpos_init']
    ypos = _params['_ypos_init']

    _phi_p = phi1 
    x1 = ij_aring[:, 0]
    y1 = ij_aring[:, 1]

    _i_p1 = i1 - del_i
    _i_p2 = i1 + del_i
    _vrot_p1 = vrot1 - del_vrot
    _vrot_p2 = vrot1 + del_vrot

    if _vrot_p1 <= 0:
        _vrot_p1 = 1E-13 
        _vrot_p2 = 1E-2

    _A = (y1 - ypos)
    _B = -(x1 - xpos)

    _phi_pp = np.zeros(4)

    # -----------------------------------
    # combination 1
    _P = (1./np.sin(_i_p1*deg_to_rad)) * (vrot1 * np.sin(i1*deg_to_rad) / _vrot_p1)
    _C = _P * (-(x1 - xpos) * np.sin(phi1*deg_to_rad) + (y1 - ypos) * np.cos(phi1*deg_to_rad))

    # do optimization
    result = minimize(of1, _phi_p, args=(_A, _B, _C))

    # estimated _phi_p in degree
    _phi_pp[0] = result.x[0]

    # -----------------------------------
    # combination 2
    _P = (1./np.sin(_i_p1*deg_to_rad)) * (vrot1 * np.sin(i1*deg_to_rad) / _vrot_p2)
    _C = _P * (-(x1 - xpos) * np.sin(phi1*deg_to_rad) + (y1 - ypos) * np.cos(phi1*deg_to_rad))

    # do optimization
    result = minimize(of1, _phi_p, args=(_A, _B, _C))

    # estimated _phi_p in degree
    _phi_pp[1] = result.x[0]

    # -----------------------------------
    # combination 3
    _P = (1./np.sin(_i_p2*deg_to_rad)) * (vrot1 * np.sin(i1*deg_to_rad) / _vrot_p1)
    _C = _P * (-(x1 - xpos) * np.sin(phi1*deg_to_rad) + (y1 - ypos) * np.cos(phi1*deg_to_rad))

    # do optimization
    result = minimize(of1, _phi_p, args=(_A, _B, _C))

    # estimated _phi_p in degree
    _phi_pp[2] = result.x[0]

    # -----------------------------------
    # combination 4
    _P = (1./np.sin(_i_p2*deg_to_rad)) * (vrot1 * np.sin(i1*deg_to_rad) / _vrot_p2)
    _C = _P * (-(x1 - xpos) * np.sin(phi1*deg_to_rad) + (y1 - ypos) * np.cos(phi1*deg_to_rad))

    # do optimization
    result = minimize(of1, _phi_p, args=(_A, _B, _C))

    # estimated _phi_p in degree
    _phi_pp[3] = result.x[0]

    #print(_phi_pp.min(), _phi_pp.max())

    return _phi_pp.min(), _phi_pp.max()




def of1(_phi_p, _A, _B, _C):
    deg_to_rad = np.pi / 180.
    return np.sum((_C - (_A * np.cos(_phi_p*deg_to_rad) + _B * np.sin(_phi_p*deg_to_rad)))**2)





#@ray.remote(num_cpus=1)
def solve_r_galaxy_plane_newton_org(fx_r_galaxy_plane, _dyn_params, fit_opt, _params, ij_aring, nrings_reliable, _xpos, _ypos, _nxy):

    cdelt1 = 1
    cdelt2 = 1
    #del_xi = (ray.get(ij_aring)[_nxy, 0] - _xpos)*cdelt1 # calculate x
    #del_yi = (ray.get(ij_aring)[_nxy, 1] - _ypos)*cdelt2 # calculate y
    del_xi = (ij_aring[_nxy, 0] - _xpos)*cdelt1 # calculate x
    del_yi = (ij_aring[_nxy, 1] - _ypos)*cdelt2 # calculate y
    _r_galaxy_plane_i_init = 3*(del_xi**2 + del_yi**2)**0.5

    return optimize.newton(fx_r_galaxy_plane, _r_galaxy_plane_i_init, args=(_dyn_params, fit_opt, _params, ij_aring[_nxy, 0], ij_aring[_nxy, 1], nrings_reliable), tol=1E-1, maxiter=10000, disp=False)




@ray.remote(num_cpus=1)
def solve_r_galaxy_plane_newton(fx_r_galaxy_plane, _dyn_params, fit_opt, _params, ij_aring, nrings_reliable, _xpos, _ypos, _nxy):

    n_cpus = int(_params['num_cpus_2d_dyn'])
    cdelt1 = 1
    cdelt2 = 1
    #del_xi = (ray.get(ij_aring)[_nxy, 0] - _xpos)*cdelt1 # calculate x
    #del_yi = (ray.get(ij_aring)[_nxy, 1] - _ypos)*cdelt2 # calculate y

    #roots = optimize.newton(fx_r_galaxy_plane, range(_r_galaxy_plane_i_init, args=(_dyn_params, fit_opt, _params, ij_aring[_nxy, 0], ij_aring[_nxy, 1], nrings_reliable), tol=1E-1, maxiter=10000, disp=False)
    #roots = optimize.newton(fx_r_galaxy_plane, np.arange(0, _r_galaxy_plane_i_init, (_r_galaxy_plane_i_init/4.)), args=(_dyn_params, fit_opt, _params, ij_aring[_nxy, 0], ij_aring[_nxy, 1], nrings_reliable), rtol=1E-1, tol=0.5E-1, maxiter=10000, disp=True)
    #roots = optimize.newton(fx_r_galaxy_plane, x0=_r_galaxy_plane_i_init, args=(_dyn_params, fit_opt, _params, ij_aring[_nxy, 0], ij_aring[_nxy, 1], nrings_reliable), tol=4, maxiter=10000, disp=True)
    if _nxy == n_cpus-1:
        _is = _nxy*int(ij_aring.shape[0]/n_cpus)
        _ie = int(ij_aring.shape[0])

        del_xi = (ij_aring[_is:_ie, 0] - _xpos)*cdelt1 # calculate x
        del_yi = (ij_aring[_is:_ie, 1] - _ypos)*cdelt2 # calculate y
        _r_galaxy_plane_i_init = 1*(del_xi**2 + del_yi**2)**0.5

        roots = optimize.newton(fx_r_galaxy_plane, x0=_r_galaxy_plane_i_init, args=(_dyn_params, fit_opt, _params, ij_aring[_is:_ie, 0], ij_aring[_is:_ie, 1], nrings_reliable), tol=1, maxiter=10000, disp=True)
    else:
        _is = _nxy*int(ij_aring.shape[0]/n_cpus)
        _ie = (_nxy+1)*int(ij_aring.shape[0]/n_cpus)

        del_xi = (ij_aring[_is:_ie, 0] - _xpos)*cdelt1 # calculate x
        del_yi = (ij_aring[_is:_ie, 1] - _ypos)*cdelt2 # calculate y
        _r_galaxy_plane_i_init = 1*(del_xi**2 + del_yi**2)**0.5

        roots = optimize.newton(fx_r_galaxy_plane, x0=_r_galaxy_plane_i_init, args=(_dyn_params, fit_opt, _params, ij_aring[_is:_ie, 0], ij_aring[_is:_ie, 1], nrings_reliable), tol=1, maxiter=10000, disp=True)

    #root, count = Counter(roots).most_common(1)[0]

    return roots, _nxy




#def solve_r_galaxy_plane_newton_sc(fx_r_galaxy_plane, _dyn_params, fit_opt, _params, ij_aring, nrings_reliable, _xpos, _ypos, _pa, _incl, _nxy, \
#                                   n_coeffs_pa_bs, tck_pa_bs, n_coeffs_incl_bs, tck_incl_bs, n_coeffs_vrot_bs, tck_vrot_bs):
def solve_r_galaxy_plane_newton_sc(fit_opt, _params, ij_aring, _xpos, _ypos, _pa, _incl, _nxy, \
                                   n_coeffs_pa_bs, tck_pa_bs, n_coeffs_incl_bs, tck_incl_bs):

    cdelt1 = 1
    cdelt2 = 1
    n_cpus = 1
    deg_to_rad = np.pi / 180.

    if _nxy == n_cpus-1:
        _is = _nxy*int(ij_aring.shape[0]/n_cpus)
        _ie = int(ij_aring.shape[0])

        del_xi = (ij_aring[_is:_ie, 0] - _xpos)*cdelt1 # calculate x
        del_yi = (ij_aring[_is:_ie, 1] - _ypos)*cdelt2 # calculate y

        # ---------------
        # ellipse fit results: init
        _pa_el = _params['_theta_el']
        _incl_el = _params['_i_el']

        cosp1 = np.cos(deg_to_rad*_pa_el)
        sinp1 = np.sin(deg_to_rad*_pa_el)
        cosi1 = np.cos(deg_to_rad*_incl_el) #cosine
        sini1 = np.sin(deg_to_rad*_incl_el) # sine

        x_galaxy_plane = (-del_xi * sinp1 + del_yi * cosp1) # x in plane of galaxy
        y_galaxy_plane = (-del_xi * cosp1 - del_yi * sinp1) / cosi1 # y in plane of galaxy
        _r_galaxy_plane_i_init = (x_galaxy_plane**2 + y_galaxy_plane**2)**0.5

        _x = ij_aring[_is:_ie, 0]
        _y = ij_aring[_is:_ie, 1]

        # 1. const PA + const INCL
        if fit_opt[4] == 1 and n_coeffs_pa_bs == 0 and fit_opt[5] == 1 and n_coeffs_incl_bs == 0:
            roots = solve_r_galaxy_plane_jit_pa_const_incl_const_test(_x, _y, _xpos, _ypos, _pa, _incl, _r_galaxy_plane_i_init, _params['r_galaxy_plane_e'])
            return roots, _nxy

        # 2. bspline PA + const INCL
        if fit_opt[4] == 0 and n_coeffs_pa_bs != 0 and fit_opt[5] == 1 and n_coeffs_incl_bs == 0:
            roots = solve_r_galaxy_plane_jit_pa_bs_incl_const_test(_x, _y, _xpos, _ypos, tck_pa_bs, _incl, _r_galaxy_plane_i_init, _params['r_galaxy_plane_e'])
            #roots = solve_r_galaxy_plane_jit_pa_bs_incl_const_test_gpu(_x, _y, _xpos, _ypos, tck_pa_bs, _incl, _r_galaxy_plane_i_init, _params['r_galaxy_plane_e'])
            return roots, _nxy

        # 3. const PA + bspline INCL
        if fit_opt[4] == 1 and n_coeffs_pa_bs == 0 and fit_opt[5] == 0 and n_coeffs_incl_bs != 0:
            roots = solve_r_galaxy_plane_jit_pa_const_incl_bs_test(_x, _y, _xpos, _ypos, _pa, tck_incl_bs, _r_galaxy_plane_i_init, _params['r_galaxy_plane_e'])
            return roots, _nxy

        # 4. bspline PA + bspline INCL
        if fit_opt[4] == 0 and n_coeffs_pa_bs != 0 and fit_opt[5] == 0 and n_coeffs_incl_bs != 0:
            roots = solve_r_galaxy_plane_jit_pa_bs_incl_bs_test(_x, _y, _xpos, _ypos, tck_pa_bs, tck_incl_bs, _r_galaxy_plane_i_init, _params['r_galaxy_plane_e'])
            return roots, _nxy

    else:
        _is = _nxy*int(ij_aring.shape[0]/n_cpus)
        _ie = (_nxy+1)*int(ij_aring.shape[0]/n_cpus)

        del_xi = (ij_aring[_is:_ie, 0] - _xpos)*cdelt1 # calculate x
        del_yi = (ij_aring[_is:_ie, 1] - _ypos)*cdelt2 # calculate y
        print("CHECK FIT OPTIONS: BSPLINE PA | BSPLINE INCL | CONST PA | CONST INCL")


        #_r_galaxy_plane_i_init = 1*(del_xi**2 + del_yi**2)**0.5
        #roots = optimize.newton(fx_r_galaxy_plane, x0=_r_galaxy_plane_i_init, args=(_dyn_params, fit_opt, _params, ij_aring[_is:_ie, 0], ij_aring[_is:_ie, 1], nrings_reliable), tol=0.5, maxiter=10000, disp=True)

    #root, count = Counter(roots).most_common(1)[0]

    return roots, _nxy



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  ----------------------------------------------------
#  ----------------------------------------------------
#  1. solve_r_galaxy_plane_jit_pa_const_incl_const_test
#  ----------------------------------------------------

@jit(nopython=True)
def fx_r_galaxy_plane_jit_pa_const_incl_const(x, i, deg_to_rad, _pa, _incl, del_x, del_y):
    cosi1 = np.cos(deg_to_rad * _incl) # RADIAN
    _pa_x0 = _pa * deg_to_rad # RADIAN
    left_side = (-del_x[i] * np.sin(_pa_x0) + del_y[i] * np.cos(_pa_x0))**2 + \
                ((-del_x[i] * np.cos(_pa_x0) - del_y[i] * np.sin(_pa_x0)) / cosi1)**2
    return left_side - x**2

def find_root_pa_const_incl_const(i, deg_to_rad, _pa, _incl, del_x, del_y, _r_galaxy_plane_i_init, _r_max):
    _a = 0.5 * _r_galaxy_plane_i_init[i]
    _b = min(1.5 * _r_galaxy_plane_i_init[i], _r_max)

    for _ in range(100):
        try:
            result = root_scalar(fx_r_galaxy_plane_jit_pa_const_incl_const, args=(i, deg_to_rad, _pa, _incl, del_x, del_y), bracket=(_a, _b), method='brentq')
            return result.root
        except ValueError:  # if not found, increase the bounds
            _a -= 0.2 * abs(_a)
            if _a < 0:
                _a = 0
            _b += 0.2 * abs(_b)
    return np.nan  # if not found


def solve_r_galaxy_plane_jit_pa_const_incl_const_test(_x, _y, _xpos, _ypos, _pa, _incl, _r_galaxy_plane_i_init, _r_max):
    deg_to_rad = np.pi / 180.0

    del_x = _x - _xpos
    del_y = _y - _ypos

    solutions = np.empty_like(del_x, dtype=float)

    for i in range(del_x.shape[0]):
        solutions[i] = find_root_pa_const_incl_const(i, deg_to_rad, _pa, _incl, del_x, del_y, _r_galaxy_plane_i_init, _r_max)

    return solutions





# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  ----------------------------------------------------
#  2. _pasolve_r_galaxy_plane_jit_pa_bs_incl_const
#  ----------------------------------------------------


# GPU에 사용되지 않는 복잡한 루트 탐색 로직을 CPU에서 처리하는 함수
def find_root_cpu(i, deg_to_rad, x_values, bspline_values_pa, _incl, del_x, del_y, _r_galaxy_plane_i_init, _r_max):
    _a = 0.5 * _r_galaxy_plane_i_init[i]
    _b = min(1.5 * _r_galaxy_plane_i_init[i], _r_max)
    for _ in range(100):
        try:
            result = root_scalar(fx_r_galaxy_plane_jit_pa_bs_incl_const, args=(i, deg_to_rad, x_values, bspline_values_pa, _incl, del_x, del_y), bracket=(_a, _b), method='brentq')
            return result.root
        except ValueError:
            _a -= 0.2 * abs(_a)
            if _a < 0:
                _a = 0
            _b += 0.2 * abs(_b)
    return np.nan

# GPU에서 실행될 간단한 계산 로직을 포함하는 함수
@cuda.jit
def solve_r_galaxy_plane_gpu(_x, _y, _xpos, _ypos, tck_pa_bs, _incl, _r_galaxy_plane_i_init, _r_max, solutions):
    idx = cuda.grid(1)
    if idx < _x.size:
        deg_to_rad = np.pi / 180.0
        bspline = BSpline(*tck_pa_bs)
        x_values = np.linspace(0, int(_r_max), int(_r_max))
        bspline_values_pa = bspline(x_values)

        del_x = _x[idx] - _xpos
        del_y = _y[idx] - _ypos

        # GPU에서 실행되므로 복잡한 루트 탐색 로직은 제외됨
        solutions[idx] = find_root_cpu(idx, deg_to_rad, x_values, bspline_values_pa, _incl, del_x, del_y, _r_galaxy_plane_i_init, _r_max)

# 메인 함수
def solve_r_galaxy_plane_jit_pa_bs_incl_const_test_gpu(_x, _y, _xpos, _ypos, tck_pa_bs, _incl, _r_galaxy_plane_i_init, _r_max):
    solutions = np.empty_like(_x, dtype=float)
    threadsperblock = 12
    blockspergrid = (_x.size + (threadsperblock - 1)) // threadsperblock
    solve_r_galaxy_plane_gpu[blockspergrid, threadsperblock](_x, _y, _xpos, _ypos, tck_pa_bs, _incl, _r_galaxy_plane_i_init, _r_max, solutions)
    return solutions






@jit(nopython=True)
def fx_r_galaxy_plane_jit_pa_bs_incl_const(x, i, deg_to_rad, x_values, bspline_values_pa, _incl, del_x, del_y):
    cosi1 = np.cos(deg_to_rad * _incl) # RADIAN
    _pa_x0 = np.interp(x, x_values, bspline_values_pa) * deg_to_rad # RADIAN, Interpolate bspline values

    left_side = (-del_x[i] * np.sin(_pa_x0) + del_y[i] * np.cos(_pa_x0))**2 + \
                ((-del_x[i] * np.cos(_pa_x0) - del_y[i] * np.sin(_pa_x0)) / cosi1)**2
    return left_side - x**2

def find_root_pa_bs_incl_const(i, deg_to_rad, x_values, bspline_values_pa, _incl, del_x, del_y, _r_galaxy_plane_i_init, _r_max):
    _a = 0.5 * _r_galaxy_plane_i_init[i]
    _b = min(1.5 * _r_galaxy_plane_i_init[i], _r_max)

    for _ in range(500):
        try:
            result = root_scalar(fx_r_galaxy_plane_jit_pa_bs_incl_const, args=(i, deg_to_rad, x_values, bspline_values_pa, _incl, del_x, del_y), bracket=(_a, _b), method='brentq')
            return result.root
        except ValueError:  # if not found, increase the bounds
            _a -= 0.2 * abs(_a)
            if _a < 0:
                _a = 0
            _b += 0.2 * abs(_b)
    return np.nan  # 최대 반복 횟수 동안 근을 찾지 못한 경우

def solve_r_galaxy_plane_jit_pa_bs_incl_const_test(_x, _y, _xpos, _ypos, tck_pa_bs, _incl, _r_galaxy_plane_i_init, _r_max):
    deg_to_rad = np.pi / 180.0

    bspline = BSpline(*tck_pa_bs) # DEGREE

    # Precompute bspline values
    x_values = np.linspace(0, int(_r_max), int(_r_max))  # Adjust the number of points as needed
    bspline_values_pa = bspline(x_values) # DEGREE

    del_x = _x - _xpos
    del_y = _y - _ypos

    solutions = np.empty_like(del_x, dtype=float)

    for i in range(del_x.shape[0]):
        solutions[i] = find_root_pa_bs_incl_const(i, deg_to_rad, x_values, bspline_values_pa, _incl, del_x, del_y, _r_galaxy_plane_i_init, _r_max)

    return solutions




# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  ----------------------------------------------------
#  3. solve_r_galaxy_plane_jit_pa_const_incl_bs
#  ----------------------------------------------------
@jit(nopython=True)
def fx_r_galaxy_plane_jit_pa_const_incl_bs(x, i, deg_to_rad, x_values, _pa, bspline_values_incl, del_x, del_y):
    _pa_x0 = _pa * deg_to_rad # RADIAN
    _incl_x0 = np.interp(x, x_values, bspline_values_incl) * deg_to_rad # RADIAN, Interpolate bspline values
    cosi1 = np.cos(_incl_x0)

    left_side = (-del_x[i] * np.sin(_pa_x0) + del_y[i] * np.cos(_pa_x0))**2 + \
                ((-del_x[i] * np.cos(_pa_x0) - del_y[i] * np.sin(_pa_x0)) / cosi1)**2
    return left_side - x**2

def find_root_pa_const_incl_bs(i, deg_to_rad, x_values, _pa, bspline_values_incl, del_x, del_y, _r_galaxy_plane_i_init, _r_max):
    _a = 0.5 * _r_galaxy_plane_i_init[i]
    _b = min(1.5 * _r_galaxy_plane_i_init[i], _r_max)

    for _ in range(100):
        try:
            result = root_scalar(fx_r_galaxy_plane_jit_pa_const_incl_bs, args=(i, deg_to_rad, x_values, _pa, bspline_values_incl, del_x, del_y), bracket=(_a, _b), method='brentq')
            return result.root
        except ValueError:  # if not found, increase the bounds
            _a -= 0.2 * abs(_a)
            if _a < 0:
                _a = 0
            _b += 0.2 * abs(_b)
    return np.nan  # if not found

def solve_r_galaxy_plane_jit_pa_const_incl_bs_test(_x, _y, _xpos, _ypos, _pa, tck_incl_bs, _r_galaxy_plane_i_init, _r_max):
    deg_to_rad = np.pi / 180.0

    bspline_incl = BSpline(*tck_incl_bs) # DEGREE

    # Precompute bspline values
    x_values = np.linspace(0, int(_r_max), int(_r_max))  # Adjust the number of points as needed
    bspline_values_incl = bspline_incl(x_values) # DEGREE

    del_x = _x - _xpos
    del_y = _y - _ypos

    solutions = np.empty_like(del_x, dtype=float)

    for i in range(del_x.shape[0]):
        solutions[i] = find_root_pa_const_incl_bs(i, deg_to_rad, x_values, _pa, bspline_values_incl, del_x, del_y, _r_galaxy_plane_i_init, _r_max)

    return solutions

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  ----------------------------------------------------
#  4. solve_r_galaxy_plane_jit_pa_bs_incl_bs
#  ----------------------------------------------------
# Numba JIT-compiled function


@jit(nopython=True)
def fx_r_galaxy_plane_jit_pa_bs_incl_bs(x, i, deg_to_rad, x_values, bspline_values_pa, bspline_values_incl, del_x, del_y):
    _pa_x0 = np.interp(x, x_values, bspline_values_pa) * deg_to_rad # RADIAN, Interpolate bspline values
    _incl_x0 = np.interp(x, x_values, bspline_values_incl) * deg_to_rad # RADIAN, Interpolate bspline values
    cosi1 = np.cos(_incl_x0)

    left_side = (-del_x[i] * np.sin(_pa_x0) + del_y[i] * np.cos(_pa_x0))**2 + \
                ((-del_x[i] * np.cos(_pa_x0) - del_y[i] * np.sin(_pa_x0)) / cosi1)**2
    return left_side - x**2

def find_root_pa_bs_incl_bs(i, deg_to_rad, x_values, bspline_values_pa, bspline_values_incl, del_x, del_y, _r_galaxy_plane_i_init, _r_max):
    _a = 0.5 * _r_galaxy_plane_i_init[i]
    _b = min(1.5 * _r_galaxy_plane_i_init[i], _r_max)

    for _ in range(100):
        try:
            result = root_scalar(fx_r_galaxy_plane_jit_pa_bs_incl_bs, args=(i, deg_to_rad, x_values, bspline_values_pa, bspline_values_incl, del_x, del_y), bracket=(_a, _b), method='brentq')
            return result.root
        except ValueError:  # if not found, increase the bounds
            _a -= 0.2 * abs(_a)
            if _a < 0:
                _a = 0
            _b += 0.2 * abs(_b)
    return np.nan  # if not found

def solve_r_galaxy_plane_jit_pa_bs_incl_bs_test(_x, _y, _xpos, _ypos, tck_pa_bs, tck_incl_bs, _r_galaxy_plane_i_init, _r_max):
    deg_to_rad = np.pi / 180.0

    bspline_pa = BSpline(*tck_pa_bs) # degree
    bspline_incl = BSpline(*tck_incl_bs) # degree

    # Precompute bspline values
    x_values = np.linspace(0, int(_r_max), int(_r_max))  # Adjust the number of points as needed
    bspline_values_pa = bspline_pa(x_values) # DEGREE
    bspline_values_incl = bspline_incl(x_values) # DEGREE

    del_x = _x - _xpos
    del_y = _y - _ypos

    solutions = np.empty_like(del_x, dtype=float)

    for i in range(del_x.shape[0]):
        solutions[i] = find_root_pa_bs_incl_bs(i, deg_to_rad, x_values, bspline_values_pa, bspline_values_incl, del_x, del_y, _r_galaxy_plane_i_init, _r_max)

    return solutions





#  ----------------------------------------------------
# PRIVATE NEWTON method
@jit(nopython=True)
#@vectorize(['float64(float64, float64, float64, float64, float64, float64, float64, float64)'], target='parallel')
def fx_r_galaxy_plane_vectorized(r_galaxy_plane, x_galaxy_plane, y_galaxy_plane):
    fx = (x_galaxy_plane**2 + y_galaxy_plane**2)**0.5 - r_galaxy_plane
    return fx

#@jit(nopython=True) # slow
#def newton_method_vectorized(f, x0, x_galaxy_plane, y_galaxy_plane, deg_to_rad, tol=0.1, max_iter=1000, h=1e-3):
#    for _ in range(max_iter):
#        fx = f(x0, x_galaxy_plane, y_galaxy_plane)
#        dfx = (f(x0 + h, x_galaxy_plane, y_galaxy_plane) - fx) / h
#        x0 -= fx / dfx
#        
#        if np.all(np.abs(fx) < tol):
#            break
#    return x0

@jit(nopython=True) # fast
def secant_vectorized_jitted(f, x0, x1, x_galaxy_plane, y_galaxy_plane):
    tol = 0.1
    max_iter = 1000
    x_new = x0
    for _ in range(max_iter):
        f_x0 = f(x0, x_galaxy_plane, y_galaxy_plane)
        f_x1 = f(x1, x_galaxy_plane, y_galaxy_plane)

        # Compute the new estimate of the root
        x_new = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)

        # Check for convergence
        if np.all(np.abs(x_new - x1) < tol):
            return x_new

        # Update for the next iteration
        x0, x1 = x1, x_new
    return x_new # not converged




@njit(fastmath=True)
def solve_r_galaxy_plane(r_galaxy_plane, del_x, del_y, cosp1, sinp1, cosi1, sini1):
    tol = 0.1
    max_iter = 100  # Adjust this as needed for better convergence
    x0s = 0.9 * r_galaxy_plane
    x1s = 1.1 * r_galaxy_plane
    for _ in range(max_iter):
        f_x0 = fx_r_galaxy_plane(x0s, del_x, del_y, cosp1, sinp1, cosi1, sini1)
        f_x1 = fx_r_galaxy_plane(x1s, del_x, del_y, cosp1, sinp1, cosi1, sini1)

        # Compute new estimates
        new_x0s = x1s - f_x1 * (x1s - x0s) / (f_x1 - f_x0)

        # Check for convergence
        if np.all(np.abs(new_x0s - x0s) < tol):
            return new_x0s

        # Update for the next iteration
        x0s, x1s = new_x0s, x0s

    return x0s








@jit(nopython=True, fastmath=True)
#def secant_vectorized_jitted_pa_const_incl_const(f, x0, x1, tol, max_iter, _x, _y, _xpos, _ypos, cosp1, sinp1, cosi1, sini1):
def secant_vectorized_jitted_pa_const_incl_const_new(f, x0, x1, _x, _y, _xpos, _ypos, cosp1, sinp1, cosi1, sini1):
    tol = 0.1
    max_iter = 100  # Adjust this as needed for better convergence

    del_x = (_x - _xpos)
    del_y = (_y - _ypos)
    x_galaxy_plane = (-del_x * sinp1 + del_y * cosp1)
    y_galaxy_plane = (-del_x * cosp1 - del_y * sinp1) / cosi1

    #f_x0 = f(x0, _x, _y, _xpos, _ypos, cosp1, sinp1, cosi1, sini1)
    #f_x1 = f(x1, _x, _y, _xpos, _ypos, cosp1, sinp1, cosi1, sini1)
    f_x0 = f(x0, x_galaxy_plane, y_galaxy_plane)
    f_x1 = f(x1, x_galaxy_plane, y_galaxy_plane)
    
    for _ in range(max_iter):
        if f_x0 * f_x1 < 0.0:
            # Bisection
            x_mid = (x0 + x1) / 2
            #f_mid = f(x_mid, _x, _y, _xpos, _ypos, cosp1, sinp1, cosi1, sini1)
            f_mid = f(x_mid, x_galaxy_plane, y_galaxy_plane)

            if np.abs(f_mid) < tol:
                return x_mid

            if f_mid * f_x0 > 0:
                x0, f_x0 = x_mid, f_mid
            else:
                x1, f_x1 = x_mid, f_mid
        else:
            # Secant or inverse quadratic interpolation
            df = f_x1 - f_x0
            if df == 0:
                return x1

            x_new = x1 - f_x1 * (x1 - x0) / df
            #f_new = f(x_new, _x, _y, _xpos, _ypos, cosp1, sinp1, cosi1, sini1)
            f_new = f(x_new, x_galaxy_plane, y_galaxy_plane)

            if np.abs(f_new) < tol:
                return x_new

            x0, x1 = x1, x_new
            f_x0, f_x1 = f_x1, f_new

        if np.abs(x1 - x0) < tol:
            return (x0 + x1) / 2

    return (x0 + x1) / 2









@jit(nopython=True)
def solve_r_galaxy_plane_pa_const_incl_const(_x, _y, _xpos, _ypos, _pa, _incl, _r_galaxy_plane_i_init):
    deg_to_rad = np.pi / 180.0

    cdelt1 = 1.0
    cdelt2 = 1.0

    del_x = (_x - _xpos) * cdelt1
    del_y = (_y - _ypos) * cdelt2

    cosp1 = np.cos(deg_to_rad * _pa)
    sinp1 = np.sin(deg_to_rad * _pa)
    cosi1 = np.cos(deg_to_rad * _incl)
    sini1 = np.sin(deg_to_rad * _incl)

    x_galaxy_plane = (-del_x * sinp1 + del_y * cosp1)
    y_galaxy_plane = (-del_x * cosp1 - del_y * sinp1) / cosi1

    x0s = 0.9*_r_galaxy_plane_i_init
    x1s = 1.1*_r_galaxy_plane_i_init

    return secant_vectorized_jitted(fx_r_galaxy_plane_vectorized, x0s, x1s, x_galaxy_plane, y_galaxy_plane)
#  ----------------------------------------------------




















#  ----------------------------------------------------
# SCIPY.OPTIMIZE.NEWTON method
def solve_r_galaxy_plane_jit_pa_const_incl_const(_x, _y, _xpos, _ypos, _pa, _incl, _r_galaxy_plane_i_init):

    deg_to_rad = np.pi / 180.0
    cdelt1 = 1.0
    cdelt2 = 1.0

    del_x = (_x - _xpos) * cdelt1
    del_y = (_y - _ypos) * cdelt2

    cosp1 = np.cos(deg_to_rad * _pa)
    sinp1 = np.sin(deg_to_rad * _pa)
    cosi1 = np.cos(deg_to_rad * _incl)
    sini1 = np.sin(deg_to_rad * _incl)

    x_galaxy_plane = (-del_x * sinp1 + del_y * cosp1)
    y_galaxy_plane = (-del_x * cosp1 - del_y * sinp1) / cosi1
    _r_galaxy_plane_part = (x_galaxy_plane**2 + y_galaxy_plane**2)**0.5

    r_galaxy_plane_roots = optimize.newton(fx_r_galaxy_plane_jit, x0=_r_galaxy_plane_i_init, args=(_r_galaxy_plane_part, x_galaxy_plane, y_galaxy_plane), tol=0.1, maxiter=1000)
    return r_galaxy_plane_roots

@jit(nopython=True, parallel=True)
def fx_r_galaxy_plane_jit(r_galaxy_plane, _r_galaxy_plane_part, x_galaxy_plane, y_galaxy_plane):
    fx = _r_galaxy_plane_part - r_galaxy_plane
    return fx
#  ----------------------------------------------------

















def solve_r_galaxy_plane_jit_pa_const_incl_bs(_x, _y, _xpos, _ypos, _pa, tck_incl_bs, _r_galaxy_plane_i_init):
    x0s = 0.9*_r_galaxy_plane_i_init
    x1s = 1.1*_r_galaxy_plane_i_init
    r_galaxy_plane_roots = secant_vectorized_pa_const_incl_bs(fx_r_galaxy_plane_jit, x0s, x1s, _x, _y, _xpos, _ypos, _pa, tck_incl_bs)
    return r_galaxy_plane_roots




def solve_r_galaxy_plane_jit_pa_bs_incl_bs(_x, _y, _xpos, _ypos, tck_pa_bs, tck_incl_bs, _r_galaxy_plane_i_init):
    x0s = 0.9*_r_galaxy_plane_i_init
    x1s = 1.1*_r_galaxy_plane_i_init
    r_galaxy_plane_roots = secant_vectorized_pa_bs_incl_bs(fx_r_galaxy_plane_jit, x0s, x1s, _x, _y, _xpos, _ypos, tck_pa_bs, tck_incl_bs)
    return r_galaxy_plane_roots


@jit(nopython=True)
def secant_vectorized_pa_const_incl_const(f, x0, x1, _x, _y, _xpos, _ypos, _pa, _incl):
    return secant_vectorized_jitted(f, x0, x1, _x, _y, _xpos, _ypos, _pa, _incl)

#@jit(nopython=True)
def secant_vectorized_pa_const_incl_bs(f, x0, x1, _x, _y, _xpos, _ypos, _pa, tck_incl_bs):
    _incl = BSpline(*tck_incl_bs, extrapolate=True)(x1)
    return secant_vectorized_jitted(f, x0, x1, _x, _y, _xpos, _ypos, _pa, _incl)



#@jit(nopython=True)
def secant_vectorized_pa_bs_incl_bs(f, x0, x1, _x, _y, _xpos, _ypos, tck_pa_bs, tck_incl_bs):
    _pa = BSpline(*tck_pa_bs, extrapolate=True)(x1)
    _incl = BSpline(*tck_incl_bs, extrapolate=True)(x1)
    return secant_vectorized_jitted(f, x0, x1, _x, _y, _xpos, _ypos, _pa, _incl)


#@jit(nopython=True, parallel=True)
#def secant_vectorized_jitted(f, x0, x1, _x, _y, _xpos, _ypos, _pa, _incl):
#    tol = 0.1
#    max_iter = 1
#    x_new = x0
#    for _ in range(max_iter):
#        f_x0 = f(x0, _x, _y, _xpos, _ypos, _pa, _incl)
#        f_x1 = f(x1, _x, _y, _xpos, _ypos, _pa, _incl)
#
#        # Compute the new estimate of the root
#        x_new = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
#
#        # Check for convergence
#        if np.all(np.abs(x_new - x1) < tol):
#            return x_new
#
#        # Update for the next iteration
#        x0, x1 = x1, x_new
#    return x_new



















@jit(nopython=True, parallel=True, cache=True)
def secant_method(f, x0, x1, args, tol=0.1, max_iter=1000):
    f_x0 = f(x0, *args)
    f_x1 = f(x1, *args)

    for _ in range(max_iter):
        if np.all(np.abs(f_x1) < tol):  # Check if all elements are within tolerance
            return x1

        x_next = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)

        # Update values for the next iteration
        x0, x1 = x1, x_next
        f_x0, f_x1 = f_x1, f(x1, *args)

    return x1

























@ray.remote(num_cpus=1)
def solve_r_galaxy_plane_fsolve(fx_r_galaxy_plane, _dyn_params, fit_opt, _params, ij_aring, nrings_reliable, _xpos, _ypos, _nxy):

    cdelt1 = 1
    cdelt2 = 1
    #del_xi = (ray.get(ij_aring)[_nxy, 0] - _xpos)*cdelt1 # calculate x
    #del_yi = (ray.get(ij_aring)[_nxy, 1] - _ypos)*cdelt2 # calculate y
    del_xi = (ij_aring[_nxy, 0] - _xpos)*cdelt1 # calculate x
    del_yi = (ij_aring[_nxy, 1] - _ypos)*cdelt2 # calculate y
    _r_galaxy_plane_i_init = 1*(del_xi**2 + del_yi**2)**0.5

    #roots = optimize.newton(fx_r_galaxy_plane, range(_r_galaxy_plane_i_init, args=(_dyn_params, fit_opt, _params, ij_aring[_nxy, 0], ij_aring[_nxy, 1], nrings_reliable), tol=1E-1, maxiter=10000, disp=False)
    roots = fsolve(fx_r_galaxy_plane, x0=_r_galaxy_plane_i_init, xtol=10, maxfev=10, factor=0.1, args=(_dyn_params, fit_opt, _params, ij_aring[_nxy, 0], ij_aring[_nxy, 1], nrings_reliable))
    return roots

def r_galaxy_plane_min_given_bounds(x, _x, _y, _xpos, _ypos):

    deg_to_rad = np.pi / 180.

    cdelt1 = 1
    cdelt2 = 1
    del_x = (_x - _xpos)*cdelt1 # calculate x
    del_y = (_y - _ypos)*cdelt2 # calculate y

    # x[0] : pa
    # x[1] : incl
    cosp1 = np.cos(deg_to_rad*x[0])
    sinp1 = np.sin(deg_to_rad*x[0])

    cosi1 = np.cos(deg_to_rad*x[1]) #cosine
    sini1 = np.sin(deg_to_rad*x[1]) # sine

    x_galaxy_plane = (-del_x * sinp1 + del_y * cosp1) # x in plane of galaxy
    y_galaxy_plane = (-del_x * cosp1 - del_y * sinp1) / cosi1 # y in plane of galaxy

    r_galaxy_plane = (x_galaxy_plane**2 + y_galaxy_plane**2)**0.5
    return r_galaxy_plane

def r_galaxy_plane_max_given_bounds(x, _x, _y, _xpos, _ypos):

    deg_to_rad = np.pi / 180.

    cdelt1 = 1
    cdelt2 = 1
    del_x = (_x - _xpos)*cdelt1 # calculate x
    del_y = (_y - _ypos)*cdelt2 # calculate y

    # x[0] : pa
    # x[1] : incl
    cosp1 = np.cos(deg_to_rad*x[0])
    sinp1 = np.sin(deg_to_rad*x[0])

    cosi1 = np.cos(deg_to_rad*x[1]) #cosine
    sini1 = np.sin(deg_to_rad*x[1]) # sine

    x_galaxy_plane = (-del_x * sinp1 + del_y * cosp1) # x in plane of galaxy
    y_galaxy_plane = (-del_x * cosp1 - del_y * sinp1) / cosi1 # y in plane of galaxy

    r_galaxy_plane = (x_galaxy_plane**2 + y_galaxy_plane**2)**0.5
    return (1.0 / r_galaxy_plane)



def fx_r_galaxy_plane(r_galaxy_plane, _dyn_params, fit_opt, _params, _x, _y, nrings_reliable, _xpos, _ypos, tck_pa_bs, tck_incl_bs):
    # args[0] (_params) : r_galaxy_plane (passed by user): lower or upper limit
    # args[1] (_dyn_params): 2dbat TR parameters (passed by dynesty)
    # args[2] (ij_aring): (i, j coordinates of the fitting area) (passed by user)
    # args[3] (fit_opt) : 2dbat tr parameters fitting option (passed by user)
    # args[4] (_params) : 2dbat parameters (passed by user)
    # return fx(r_galaxy_plane) : 

    # _________________________________________
    # arguments
    # _________________________________________
    # 1) r_galaxy_plane : lower or upper limit
    #
    # 2) _dyn_params = args[1] : dynesty params passed
    #   ---------------------
    #   2dbat tr parameters
    #   ---------------------
    #   _dyn_params[0] : sigma
    #   _dyn_params[1] : xpos
    #   _dyn_params[2] : ypos
    #   _dyn_params[3] : vsys
    #   _dyn_params[4] : pa
    #   _dyn_params[5] : incl
    #   _dyn_params[6] : vrot
    #   _dyn_params[7] : vrad

    #   _dyn_params[8] : vrot_c0_bs
    #   _dyn_params[9] : vrot_c1_bs
    #   _dyn_params[10] : vrot_c2_bs

    # 
    # 3) i, j : coordinate x, y
    # 4) fit_opt : trfit fitting option
    # 5) _params : 2dbat parameters
    # _________________________________________

    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    deg_to_rad = np.pi / 180.


    _pa_bs = BSpline(*tck_pa_bs, extrapolate=True)(r_galaxy_plane) # <-- tck_pa_bs[1] coeffs (0 ~ 360)
    _incl_bs = BSpline(*tck_incl_bs, extrapolate=True)(r_galaxy_plane) # <-- tck_incl_bs[1] coeffs (0 ~ 90)

    #_pa_bs = 130
    #_incl_bs = 60
    

    cdelt1 = 1
    cdelt2 = 1
    del_x = (_x - _xpos)*cdelt1 # calculate x
    del_y = (_y - _ypos)*cdelt2 # calculate y

    cosp1 = np.cos(deg_to_rad*_pa_bs)
    sinp1 = np.sin(deg_to_rad*_pa_bs)

    cosi1 = np.cos(deg_to_rad*_incl_bs) #cosine
    sini1 = np.sin(deg_to_rad*_incl_bs) # sine

    x_galaxy_plane = (-del_x * sinp1 + del_y * cosp1) # x in plane of galaxy
    y_galaxy_plane = (-del_x * cosp1 - del_y * sinp1) / cosi1 # y in plane of galaxy

    fx = (x_galaxy_plane**2 + y_galaxy_plane**2)**0.5 - r_galaxy_plane

    return fx



def solve_r_galaxy_plane_newton_sc_test(fx_r_galaxy_plane, _dyn_params, fit_opt, _params, ij_aring, nrings_reliable, _xpos_init, _ypos_init, _pa_init, _incl_init, _nxy):

    cdelt1 = 1
    cdelt2 = 1
    n_cpus = 1
    deg_to_rad = np.pi / 180.

    _is = 0
    _ie = 100

    del_xi = (ij_aring[_is:_ie, 0] - _xpos_init)*cdelt1 # calculate x
    del_yi = (ij_aring[_is:_ie, 1] - _ypos_init)*cdelt2 # calculate y

    cosp1 = np.cos(deg_to_rad*_pa_init)
    sinp1 = np.sin(deg_to_rad*_pa_init)
    cosi1 = np.cos(deg_to_rad*_incl_init) #cosine
    sini1 = np.sin(deg_to_rad*_incl_init) # sine

    x_galaxy_plane = (-del_xi * sinp1 + del_yi * cosp1) # x in plane of galaxy
    y_galaxy_plane = (-del_xi * cosp1 - del_yi * sinp1) / cosi1 # y in plane of galaxy

    _r_galaxy_plane_i_init = (x_galaxy_plane**2 + y_galaxy_plane**2)**0.5

    # ------------------------------------------------------
    # PA - bspline : calculate the number of pa coefficients and generate a dummy array for pa
    n_coeffs_pa_bs, tck_pa_bs = bspline_ncoeffs_tck(_params, 'pa', _params['nrings_intp'])
    
    # ------------------------------------------------------
    # INCL - bspline
    n_coeffs_incl_bs, tck_incl_bs = bspline_ncoeffs_tck(_params, 'incl', _params['nrings_intp'])

    # ------------------------------------------------------
    # VROT - bspline : calculate the number of pa coefficients and generate a dummy array for pa
    n_coeffs_vrot_bs, tck_vrot_bs = bspline_ncoeffs_tck(_params, 'vrot', _params['nrings_intp'])

    _xpos = 10
    _ypos = 10

    n_ring_params_free = 0 # starting from sigma

    # derive _pa_bs via BSpline using the generated coefficients from dynesty
    if n_coeffs_pa_bs != 0:
        for _nbs in range(0, n_coeffs_pa_bs):
            tck_pa_bs[1][_nbs] = _dyn_params[n_ring_params_free + _nbs]

    # -------------------------------------
    # _incl_bs
    # derive _incl_bs via BSpline using the generated coefficients from dynesty
    if n_coeffs_incl_bs != 0:
        for _nbs in range(0, n_coeffs_incl_bs):
            tck_incl_bs[1][_nbs] = _dyn_params[n_ring_params_free + n_coeffs_pa_bs + _nbs]

    roots = optimize.newton(fx_r_galaxy_plane, x0=_r_galaxy_plane_i_init, args=(_dyn_params, fit_opt, _params, ij_aring[_is:_ie, 0], ij_aring[_is:_ie, 1], nrings_reliable, _xpos, _ypos, tck_pa_bs, tck_incl_bs), tol=1000, maxiter=10000, disp=None)

    return roots

def fx_r_galaxy_plane_test(r_galaxy_plane, _x, _y, _xpos, _ypos, tck_pa_bs, tck_incl_bs):

    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    deg_to_rad = np.pi / 180.


    _pa_bs = BSpline(*tck_pa_bs, extrapolate=True)(r_galaxy_plane) # <-- tck_pa_bs[1] coeffs (0 ~ 360)
    _incl_bs = BSpline(*tck_incl_bs, extrapolate=True)(r_galaxy_plane) # <-- tck_incl_bs[1] coeffs (0 ~ 90)

    del_x = (_x - _xpos)*cdelt1 # calculate x
    del_y = (_y - _ypos)*cdelt2 # calculate y

    cosp1 = np.cos(deg_to_rad*_pa_bs)
    sinp1 = np.sin(deg_to_rad*_pa_bs)

    cosi1 = np.cos(deg_to_rad*_incl_bs) #cosine
    sini1 = np.sin(deg_to_rad*_incl_bs) # sine

    x_galaxy_plane = (-del_x * sinp1 + del_y * cosp1) # x in plane of galaxy
    y_galaxy_plane = (-del_x * cosp1 - del_y * sinp1) / cosi1 # y in plane of galaxy

    fx = (x_galaxy_plane**2 + y_galaxy_plane**2)**0.5 - r_galaxy_plane

    return fx


def bspline_ncoeffs_tck(_params, tr_param, nrings_reliable):

    try: 
        xs_bs = _params['r_galaxy_plane_s']
        xe_bs = _params['r_galaxy_plane_e']
        #nrings = _params['nrings']
        nrings = nrings_reliable
        x_bs = np.linspace(xs_bs, xe_bs, nrings, endpoint=True)
        y_bs = np.linspace(xs_bs, xe_bs, nrings, endpoint=True) # dummy y_bs for generating dummy tck
    except:
        pass

    if tr_param == 'vrot':

        # VROT-BS coefficients
        # note:
        # number of knots = number of coefficients (number of B-spline basis functions) + degree + 1
        # number of knots = 2 x (degree + 1) + number of inner knots
        # --> number of coefficients = degree + 1 + number of inner knots

        # number of inner knots
        n_knots_inner = _params['n_vrot_bs_knots_inner'] # 0, 1, 2, ...
        #vrot_bs_knots = np.linspace(xs_bs, xe_bs, (2+n_knots))[1:-1]
        # vrot_bs inner knot vector
        vrot_bs_knots_inner = np.linspace(xs_bs, xe_bs, (2+n_knots_inner))[1:-1]
        # k_bs = 3 (cubic)
        k_bs = _params['k_vrot_bs'] # 1, 2, ...
        # number of cofficients = len(knots_bs) + 2*(k_bs+1) - k_bs - 1 
        #n_coeffs_vrot_bs = len(vrot_bs_knots) + 2*(k_bs+1) - k_bs - 1
        #n_coeffs_vrot_bs = len(vrot_bs_knots) + k_bs+1
        n_coeffs_vrot_bs = k_bs + 1 + n_knots_inner

        if(n_coeffs_vrot_bs == 1): # constant vrot
            n_coeffs_vrot_bs = 0
            tck_vrot_bs =  0
            return n_coeffs_vrot_bs, tck_vrot_bs
        else:
            # make a dummy array for vrot_bs coefficients
            tck_vrot_bs = splrep(x_bs, y_bs, t=vrot_bs_knots_inner, k=k_bs)
            return n_coeffs_vrot_bs, tck_vrot_bs
    
    elif tr_param == 'pa':

        # PA-BS coefficients
        # note:
        # number of knots = number of coefficients (number of B-spline basis functions) + degree + 1
        # number of knots = 2 x (degree + 1) + number of inner knots
        # --> number of coefficients = degree + 1 + number of inner knots

        # number of inner knots
        n_knots_inner = _params['n_pa_bs_knots_inner'] # 0, 1, 2, ...
        pa_bs_knots_inner = np.linspace(xs_bs, xe_bs, (2+n_knots_inner))[1:-1]
        # k_bs = 3 (cubic)
        k_bs = _params['k_pa_bs'] # 1, 2, ...
        # number of cofficients = len(knots_bs) + 2*(k_bs+1) - k_bs - 1 
        #n_coeffs_pa_bs = len(pa_bs_knots) + 2*(k_bs+1) - k_bs - 1
        n_coeffs_pa_bs = k_bs + 1 + n_knots_inner

        if(n_coeffs_pa_bs == 1): # constant pa
            n_coeffs_pa_bs = 0
            tck_pa_bs =  0
            return n_coeffs_pa_bs, tck_pa_bs
        else:
            # make a dummy array for pa_bs coefficients
            tck_pa_bs = splrep(x_bs, y_bs, t=pa_bs_knots_inner, k=k_bs)
            return n_coeffs_pa_bs, tck_pa_bs

    elif tr_param == 'incl':

        # INCL-BS coefficients
        # note:
        # number of knots = number of coefficients (number of B-spline basis functions) + degree + 1
        # number of knots = 2 x (degree + 1) + number of inner knots
        # --> number of coefficients = degree + 1 + number of inner knots

        # number of inner knots
        n_knots_inner = _params['n_incl_bs_knots_inner'] # 0, 1, 2, ...
        incl_bs_knots_inner = np.linspace(xs_bs, xe_bs, (2+n_knots_inner))[1:-1]
        # k_bs = 3 (cubic)
        k_bs = _params['k_incl_bs'] # 1, 2, ...
        # number of cofficients = len(knots_bs) + 2*(k_bs+1) - k_bs - 1 
        #n_coeffs_incl_bs = len(incl_bs_knots) + 2*(k_bs+1) - k_bs - 1
        n_coeffs_incl_bs = k_bs + 1 + n_knots_inner

        if(n_coeffs_incl_bs == 1): # constant incl
            n_coeffs_incl_bs = 0
            tck_incl_bs =  0
            return n_coeffs_incl_bs, tck_incl_bs
        else:
            # make a dummy array for incl_bs coefficients
            tck_incl_bs = splrep(x_bs, y_bs, t=incl_bs_knots_inner, k=k_bs)
            return n_coeffs_incl_bs, tck_incl_bs

    elif tr_param == 'vrad':

        # VRAD-BS coefficients
        # note:
        # number of knots = number of coefficients (number of B-spline basis functions) + degree + 1
        # number of knots = 2 x (degree + 1) + number of inner knots
        # --> number of coefficients = degree + 1 + number of inner knots

        # number of inner knots
        n_knots_inner = _params['n_vrad_bs_knots_inner'] # 0, 1, 2, ...
        vrad_bs_knots_inner = np.linspace(xs_bs, xe_bs, (2+n_knots_inner))[1:-1]
        # k_bs = 3 (cubic)
        k_bs = _params['k_vrad_bs'] # 1, 2, ...
        # number of cofficients = len(knots_bs) + 2*(k_bs+1) - k_bs - 1 
        #n_coeffs_vrad_bs = len(vrad_bs_knots) + 2*(k_bs+1) - k_bs - 1
        n_coeffs_vrad_bs = k_bs + 1 + n_knots_inner

        if(n_coeffs_vrad_bs == 1): # constant vrad 
            n_coeffs_vrad_bs = 0
            tck_vrad_bs =  0
            return n_coeffs_vrad_bs, tck_vrad_bs
        else:
            # make a dummy array for vrad_bs coefficients
            tck_vrad_bs = splrep(x_bs, y_bs, t=vrad_bs_knots_inner, k=k_bs)
            return n_coeffs_vrad_bs, tck_vrad_bs


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def loglike_trfit(*args):

    # args[0] : 2dbat tr parameters (passed by dynesty)
    # args[1] : _input_vf to fit (passed by user)
    # args[2] : _tr_model_vf (passed by user) : this is a blank array which will be filled up below
    # args[3] : _wt_2d (passed by user) : weight array
    # args[4] : ij_aring (i, j coordinates of the fitting area) (passed by user)
    # args[5] (fit_opt) : 2dbat tr parameters fitting option (passed by user)
    # args[6] (_params) : 2dbat config parameters (passed by user)
    # args[7] : pa, incl, vrot, vrad b-spline order / knots (passed by user)
    # <-- these arguments (1, 2, 3 and 4) are passed by logl_args=[....] in NestedSampler() 

    #---------------------
    # args[0] : default arg passed by dynesty
    # 2dbat tr parameters generated via nested sampling
    #---------------------
    # args[0][0] : sigma
    # args[0][1] : xpos
    # args[0][2] : ypos
    # args[0][3] : vsys
    # args[0][4] : pa
    # args[0][5] : incl
    # args[0][6] : vrot
    # args[0][7] : vrad
    # ...
    #_____________________

    #---------------------
    # args[1] : _input_vf to fit
    # 2d array
    #---------------------

    #---------------------
    # args[2] : _tr_model_vf 
    # 2d array (blank)
    #---------------------

    #---------------------
    # args[3] : ij_aring
    # (i, j coordinates of the fitting area) 
    #---------------------

    #---------------------
    # args[4] : fit_opt
    # 2dbat tr parameters fitting option
    #---------------------
    # args[4][0] : sigma free or fixed
    # args[4][1] : xpos free or fixed 
    # args[4][2] : ypos free or fixed
    # args[4][3] : vsys free or fixed
    # args[4][4] : pa free or fixed
    # args[4][5] : incl free or fixed
    # args[4][6] : vrot free or fixed
    # args[4][7] : vrad free or fixed
    #_____________________

    #---------------------
    # args[5] : bspline_opt
    # b-spline orders & knots
    #---------------------
    # args[5][0] : pa b-spline order ?
    # args[5][1] : pa b-spline knots ?
    # args[5][2] : incl b-spline order ?
    # args[5][3] : incl b-spline knots ?
    # args[5][4] : vrot b-spline order ?
    # args[5][5] : vrot b-spline knots ?
    # args[5][6] : vrad b-spline order ?
    # args[5][7] : vrad b-spline knots ?
    #_____________________

    #global _tr_model_vf
    #print(args[1])

    #npoints = args[3].shape[0]
    sigma = args[0][0] # loglikelihoood sigma: dynesty default params[0]

    # tilted-ring vlos model
    #tr_model_vf = tr_vlos_model(args[0], args[2], args[3], args[4])
    #_ij_aring_vlos = derive_vlos_model(_dyn_params, ij_aring, fit_opt)

    # nuree99
    tr_model_vf_new = derive_vlos_model(args[0], args[2], args[4], args[5], args[6])


    #print(tr_model_vf[22, 23], tr_model_vf[22, 22] )

    # args[0] : 2dbat tr parameters (passed by dynesty)
    # args[1] : _input_vf to fit (passed by user)
    # args[2] : _tr_model_vf (passed by user) : this is a blank array which will be filled up below
    # args[3] : ij_aring (i, j coordinates of the fitting area) (passed by user)
    # args[4] (fit_opt) : 2dbat tr parameters fitting option (passed by user)
    # args[5] : pa, incl, vrot, vrad b-spline order / knots (passed by user)

    #print(args[3][3, 1].astype(int), args[3][3, 0].astype(int), tr_model_vf[args[3][3, 1].astype(int), args[3][3, 0].astype(int)], args[1][args[3][3, 1].astype(int), args[3][3, 0].astype(int)] )

    res = np.array([np.where( \
                        #(tr_model_vf_new > -1E8) & (args[1] > -1E8), \
                        (tr_model_vf_new > args[6]['_vlos_lower']) & \
                        (tr_model_vf_new < args[6]['_vlos_upper']) & \
                        (args[1] > args[6]['_vlos_lower']) & \
                        (args[1] < args[6]['_vlos_upper']),
                        (tr_model_vf_new - args[1]), 0.0)])[0]
    
    npoints = np.count_nonzero(res)
    loglike = -0.5 * (
        np.sum( ((res / sigma) * args[3]) ** 2) + npoints * np.log(2 * np.pi * sigma**2))
        #np.sum((res / sigma) ** 2) + npoints * np.log(2 * np.pi * sigma**2))
    
    return loglike


    #chi2_ma = np.array([np.where( \
    #                    (tr_model_vf_new[:, :] > -1E9) & (args[1][:, :] > -1E9), \
    #                    ((-1.0 / (2*sigma**2)) * ((tr_model_vf_new[:, :] - args[1][:, :])**2)), 0.0)])[0]

    #npoints = np.count_nonzero(chi2_ma)
    #log_n_sigma = -0.5*npoints*log(2.0*np.pi) - 1.0*npoints*log(sigma)

    #chi2_ma = ma.masked_less((-1.0 / (2*sigma**2)) * ((tr_model_vf - args[1])**2), -10000000)
    #chi2 = chi2_ma.sum()
    #chi2 = sum((-1.0 / (2*sigma**2)) * ((gfit - args[1])**2))
    #print("seheon", chi2_ma.shape, chi2)

    #return log_n_sigma + chi2

#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def loglike_trfit_2d(*args):

# nuree2
    # args[0] : 2dbat tr parameters (passed by dynesty)
    # args[1] : _input_vf to fit (passed by user)
    # args[2] : _tr_model_vf (passed by user) : this is a blank array which will be filled up below
    # args[3] : ij_aring (i, j coordinates of the fitting area) (passed by user)
    # args[4] (fit_opt) : 2dbat tr parameters fitting option (passed by user)
    # args[5] : pa, incl, vrot, vrad b-spline order / knots (passed by user)
    # args[6] : _params : 2dbat params (passed by user)
    # args[7] : nrings_reliable (passed by user)
    # args[8] : r_galaxy_plane_dummy (passed by user)

    # args[10] : n_coeffs_pa_bs (passed by user)
    # args[11] : tck_pa_bs (passed by user)
    # args[12] : n_coeffs_incl_bs (passed by user)
    # args[13] : tck_incl_bs (passed by user)
    # args[14] : n_coeffs_vrot_bs (passed by user)
    # args[15] : tck_vrot_bs (passed by user)    # args[9] : _wi_2d (passed by user)



    # <-- these arguments (1, 2, 3 and 4) are passed by logl_args=[....] in NestedSampler() 

    #---------------------
    # args[0] : default arg passed by dynesty
    # 2dbat tr parameters generated via nested sampling
    #---------------------
    # args[0][0] : sigma
    # args[0][1] : xpos
    # args[0][2] : ypos
    # args[0][3] : vsys
    # args[0][4] : pa
    # args[0][5] : incl
    # args[0][6] : vrot
    # args[0][7] : vrad
    # ...
    #_____________________

    #---------------------
    # args[1] : _input_vf to fit
    # 2d array
    #---------------------

    #---------------------
    # args[2] : _tr_model_vf 
    # 2d array (blank)
    #---------------------

    #---------------------
    # args[3] : ij_aring
    # (i, j coordinates of the fitting area) 
    #---------------------

    #---------------------
    # args[4] : fit_opt
    # 2dbat tr parameters fitting option
    #---------------------
    # args[4][0] : sigma free or fixed
    # args[4][1] : xpos free or fixed 
    # args[4][2] : ypos free or fixed
    # args[4][3] : vsys free or fixed
    # args[4][4] : pa free or fixed
    # args[4][5] : incl free or fixed
    # args[4][6] : vrot free or fixed
    # args[4][7] : vrad free or fixed
    #_____________________

    #---------------------
    # args[5] : bspline_opt
    # b-spline orders & knots
    #---------------------
    # args[5][0] : pa b-spline order ?
    # args[5][1] : pa b-spline knots ?
    # args[5][2] : incl b-spline order ?
    # args[5][3] : incl b-spline knots ?
    # args[5][4] : vrot b-spline order ?
    # args[5][5] : vrot b-spline knots ?
    # args[5][6] : vrad b-spline order ?
    # args[5][7] : vrad b-spline knots ?
    #_____________________

    #global _tr_model_vf
    #print(args[1])

    #npoints = args[3].shape[0]
    sigma = args[0][0] # loglikelihoood sigma: dynesty default params[0]

    # tilted-ring vlos model
    #tr_model_vf = tr_vlos_model(args[0], args[2], args[3], args[4])
    #_ij_aring_vlos = derive_vlos_model(_dyn_params, ij_aring, fit_opt)
    #tr_model_vf, wi_2d, _r_galaxy_plane_given_params, _pa_bs_given_params, _incl_bs_given_params, _vrot_bs_given_params = derive_vlos_model_2d_sc(args[0], args[2], args[9], args[3], args[4], args[6], args[7], args[8])

    tr_model_vf = derive_vlos_model_2d_sc(args[0], args[2], args[9], args[3], args[4], args[6], args[7], args[8], args[10], args[11], args[12], args[13], args[14], args[15])

    # args[0] : 2dbat tr parameters (passed by dynesty)
    # args[1] : _input_vf to fit (passed by user)
    # args[2] : _tr_model_vf (passed by user) : this is a blank array which will be filled up below
    # args[3] : ij_aring (i, j coordinates of the fitting area) (passed by user)
    # args[4] (fit_opt) : 2dbat tr parameters fitting option (passed by user)
    # args[5] : pa, incl, vrot, vrad b-spline order / knots (passed by user)

    #print(args[3][3, 1].astype(int), args[3][3, 0].astype(int), tr_model_vf[args[3][3, 1].astype(int), args[3][3, 0].astype(int)], args[1][args[3][3, 1].astype(int), args[3][3, 0].astype(int)] )

    #chi2_ma = np.array([np.where( \
    #                    (tr_model_vf[:, :] > -1E9) & (args[1][:, :] > -1E9), \
    #                    ((-1.0 / (2*sigma**2)) * ((tr_model_vf[:, :] - args[1][:, :])**2)), 0.0)])[0]

    #chi2_ma = np.array([np.where( \
    #                    (tr_model_vf > -1E9) & (args[1] > -1E9), \
    #                    ((tr_model_vf - args[1])**2)*wi_2d, 0.0)])[0]

    #_r_min = np.min(_r_galaxy_plane_given_params)
    #_r_max = np.max(_r_galaxy_plane_given_params)
#
#    _pa_min = np.min(_pa_bs_given_params)
#    _pa_max = np.max(_pa_bs_given_params)
#
#    _incl_min = np.min(_incl_bs_given_params)
#    _incl_max = np.max(_incl_bs_given_params)
#
#    _vrot_min = np.min(_vrot_bs_given_params)
#    _vrot_max = np.max(_vrot_bs_given_params)

    #_vlos_min = np.min(args[1][xx[0].astype(int), xx[1].astype(int)])
    #_vlos_max = np.max(args[1][xx[0].astype(int), xx[1].astype(int)])
    #print(_vlos_min, _vlos_max)

    #print(_r_min, _r_max, _pa_min, _pa_max, _incl_min, _incl_max, _vrot_min, _vrot_max)

#    if _r_min < 0 or _r_max > 1.3*args[6]['r_galaxy_plane_e'] \
#        or _pa_min < -20 or _pa_max > 80 \
#        or _incl_min < 10 or _incl_max > 80 \
#        or _vrot_min < 0 or _vrot_max > 150:
#        return -np.inf


#    else:
#        chi2_ma = np.array([np.where( \
#                            (tr_model_vf > -1E9) & (args[1] > -1E9), \
#                            ((tr_model_vf - args[1])**2), 0.0)])[0]
#
#        npoints = np.count_nonzero(chi2_ma)
#        log_n_sigma = -0.5*npoints*log(2.0*np.pi) - 1.0*npoints*log(sigma)
#
#        #chi2_ma = ma.masked_less((-1.0 / (2*sigma**2)) * ((tr_model_vf - args[1])**2), -10000000)
#        chi2 = ( -1.0 / (2*(sigma**2)) ) * chi2_ma.sum()
#        #chi2 = sum((-1.0 / (2*sigma**2)) * ((gfit - args[1])**2))
#        #print("seheon", chi2_ma.shape, chi2)
#
#        if np.isnan(log_n_sigma + chi2) or np.isinf(log_n_sigma + chi2):
#            return -np.inf
#        else:
#            return log_n_sigma + chi2

#    else:

#        residsq = (tr_model_vf  - args[1])**2 / sigma**2
#        residsq = np.array([np.where( \
#                            (tr_model_vf > -1E9) & (args[1] > -1E9), \
#                            residsq, 0.0)])[0]
#        loglike1 = -0.5 * np.sum(residsq)
#        loglike2 = -0.5 * np.sum(np.log(2 * np.pi * sigma**2))
#
#        loglike = loglike1 + loglike2
#    
#        if not np.isfinite(loglike):
#            loglike = -1e300
#            
#        return loglike

    #if _r_min < 0 or _r_max > 1.3*args[6]['r_galaxy_plane_e']:
    #if  _pa_min < -50 or _pa_max > 100 \
    #if _vlos_min < -10000000000 or _vlos_max > 10000000:
    #    #print("INCL issue")
    #    return -np.inf 
    
    #if _r_min < 0 or _r_max > 1.3*args[6]['r_galaxy_plane_e'] \
    #    or _vrot_min < 0 or _vrot_max > 150:
    #    return -np.inf

    #else:





#    res = np.array([np.where( \
#                        (tr_model_vf > -1E9) & (args[1] > -1E9), \
#                        (tr_model_vf - args[1]), 0.0)])[0]
#    
#    npoints = np.count_nonzero(res)
#    loglike = -0.5 * (
#        np.sum((res / sigma) ** 2) + npoints * np.log(2 * np.pi * sigma**2))
#    
#    #if np.min(res) != 0.0 and np.min(res) > -200 and np.max(res) < 200:
#    if np.min(res) != 0.0:
#        return loglike
#    else:
#        return -np.inf


    #if np.any(np.isnan(tr_model_vf)):
    #    return -np.inf

    # args[9]: _wt_2d
    res = np.array([np.where( \
                        #(tr_model_vf > -1E8) & (args[1] > -1E8), \
                        (tr_model_vf > args[6]['_vlos_lower']) & \
                        (tr_model_vf < args[6]['_vlos_upper']) & \
                        (args[1] > args[6]['_vlos_lower']) & \
                        (args[1] < args[6]['_vlos_upper']),
                        (tr_model_vf - args[1]), 0.0)])[0]
    
    npoints = np.count_nonzero(res)
    loglike = -0.5 * (
        np.sum(( (res / sigma) * args[9] ) ** 2) + npoints * np.log(2 * np.pi * sigma**2))
        #np.sum((res / sigma) ** 2) + npoints * np.log(2 * np.pi * sigma**2))

    return loglike 


#    else:
#        res = np.array([np.where( \
#                            (tr_model_vf > -1E9) & (args[1] > -1E9), \
#                            (tr_model_vf - args[1]), 0.0)])[0]
#
#        npoints = np.count_nonzero(res)
#        return -0.5 * (
#            np.sum((res / sigma) ** 2) + npoints * np.log(2 * np.pi * sigma**2))


#    else:
#
#        chi2_ma = np.array([np.where( \
#                            (tr_model_vf > -1E9) & (args[1] > -1E9), \
#                            ((tr_model_vf - args[1])**2), 0.0)])[0]
#
#        npoints = np.count_nonzero(chi2_ma)
#        log_n_sigma = -0.5*npoints*log(2.0*np.pi) - 1.0*npoints*log(sigma)
#
#        #chi2_ma = ma.masked_less((-1.0 / (2*sigma**2)) * ((tr_model_vf - args[1])**2), -10000000)
#        chi2 = ( -1.0 / (2*(sigma**2)) ) * chi2_ma.sum()
#        #chi2 = sum((-1.0 / (2*sigma**2)) * ((gfit - args[1])**2))
#        #print("seheon", chi2_ma.shape, chi2)
#
#        return log_n_sigma + chi2


#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def	tr_vlos_model_org(_dyn_params, _tr_model_vf, ij_aring, fit_opt):
    # args[0] (_dyn_params): 2dbat tr parameters (passed by dynesty)
    # args[1] (_tr_model_vf): blank array which will be filled up below (passed by user)
    # args[2] : ij_aring (i, j coordinates of the fitting area) (passed by user)
    # args[3] (fit_opt) : 2dbat tr parameters fitting option (passed by user)

    npoints = ij_aring.shape[0]

    #naxis1 = 41
    #naxis2 = 41
    #tr_model_vf = np.full((naxis1, naxis2), fill_value=1E90, dtype=np.float64)

    # here !!!!! : nuree
    for n in range(npoints):
        i = ij_aring[n, 0]
        j = ij_aring[n, 1]

        #print("seheon9", i, j, _dyn_params, fit_opt)
		# derive vlos model
        tr_vlos_model = derive_vlos_model(_dyn_params, i, j, fit_opt)

        _tr_model_vf[j, i] = tr_vlos_model

    return _tr_model_vf

#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
#def	tr_vlos_model(_dyn_params, _tr_model_vf, ij_aring, fit_opt):
#    # args[0] (_dyn_params): 2dbat tr parameters (passed by dynesty)
#    # args[1] (_tr_model_vf): blank array which will be filled up below (passed by user)
#    # args[2] : ij_aring (i, j coordinates of the fitting area) (passed by user)
#    # args[3] (fit_opt) : 2dbat tr parameters fitting option (passed by user)
#
#    npoints = ij_aring.shape[0]
#
#    #naxis1 = 41
#    #naxis2 = 41
#    #tr_model_vf = np.full((naxis1, naxis2), fill_value=1E90, dtype=np.float64)
#
#    _ij_aring_vlos = derive_vlos_model(_dyn_params, ij_aring, fit_opt)
#
#
#
#
#    # here !!!!! : nuree
#    for n in range(npoints):
#        i = ij_aring[n, 0]
#        j = ij_aring[n, 1]

#        #print("seheon9", i, j, _dyn_params, fit_opt)
#		# derive vlos model
#        tr_vlos_model = derive_vlos_model(_dyn_params, i, j, fit_opt)
#
#        _tr_model_vf[j, i] = tr_vlos_model
#
#    return _tr_model_vf
#
##-- END OF SUB-ROUTINE____________________________________________________________#
        
#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def derive_vlos_model(_dyn_params, _tr_model_vf, ij_aring, fit_opt, _params):
    # args[0] (_dyn_params): 2dbat tr parameters (passed by dynesty)
    # args[1] (_tr_model_vf): blank array which will be filled up below (passed by user)
    # args[2] (ij_aring): (i, j coordinates of the fitting area) (passed by user)
    # args[3] (fit_opt) : 2dbat tr parameters fitting option (passed by user)
    # args[4] (_params) : 2dbat config parameters (passed by user)
    # return _tr_model_vf : vlos updated

    # _________________________________________
    # arguments
    # _________________________________________
    # 1) _dyn_params = args[0] : dynesty params passed
    #   ---------------------
    #   2dbat tr parameters
    #   ---------------------
    #   _dyn_params[0] : sigma
    #   _dyn_params[1] : xpos
    #   _dyn_params[2] : ypos
    #   _dyn_params[3] : vsys
    #   _dyn_params[4] : pa
    #   _dyn_params[5] : incl
    #   _dyn_params[6] : vrot
    #   _dyn_params[7] : vrad
    # 
    # 2) _tr_model_vf : blank array
    # 3) i, j : coordinate x, y
    # 4) fit_opt : trfit fitting option
    # _________________________________________

    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    deg_to_rad = np.pi / 180.

    n_ring_params_free = 0 # starting from sigma
    # ------------------------------------------------------
    # sigma
    if fit_opt[0] == 1: # sigma fixed:0 free:1
        _sigma = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _sigma = _params['_sigma_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # ------------------------------------------------------
    # XPOS
    if fit_opt[1] == 1: # xpos fixed:0 free:1
        _xpos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _xpos = _params['_xpos_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # YPOS
    if fit_opt[2] == 1: # ypos fixed:0 free:1
        _ypos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _ypos = _params['_ypos_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # VSYS
    if fit_opt[3] == 1: # vsys fixed:0 free:1
        _vsys = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vsys = _params['_vsys_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # PA
    if fit_opt[4] == 1: # pa fixed:0 free:1
        _pa = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _pa = _params['_pa_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # INCL
    if fit_opt[5] == 1: # incl fixed:0 free:1
        _incl = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _incl = _params['_incl_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # VROT
    if fit_opt[6] == 1: # vrot fixed:0 free:1
        _vrot = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrot = _params['_vrot_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # VRAD
    if fit_opt[7] == 1: # vrad fixed:0 free:1
        _vrad = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrad = _params['_vrad_init'] # fixed to the previous fitting result

    #print("seheon10", _xpos, _ypos, _vsys, _pa, _incl, _vrot, _vrad)
    # Derive radii on the galaxy plane in pixel, arcsec and pc units */
    cdelt1 = 1
    cdelt2 = 1


    cosp1 = np.cos(deg_to_rad*_pa)
    sinp1 = np.sin(deg_to_rad*_pa)

    cosi1 = np.cos(deg_to_rad*_incl) #cosine
    sini1 = np.sin(deg_to_rad*_incl) # sine

    del_x = (ij_aring[:, 0] - _xpos)*cdelt1 # calculate x
    del_y = (ij_aring[:, 1] - _ypos)*cdelt2 # calculate y

    x_galaxy_plane = (-del_x * sinp1 + del_y * cosp1) # x in plane of galaxy
    y_galaxy_plane = (-del_x * cosp1 - del_y * sinp1) / cosi1 # y in plane of galaxy

    r_galaxy_plane = (x_galaxy_plane**2 + y_galaxy_plane**2)**0.5
    cost1 = x_galaxy_plane / r_galaxy_plane # cosine of angle in plane of galaxy 
    sint1 = y_galaxy_plane / r_galaxy_plane # sine of angle in plane of galaxy

    #ij_aring[:,2] = _vsys + (_vrot * cost1 + _vrad * sint1) * sini1 # model LOS velocity

    #_tr_model_vf[:, :] = -1E10
    #_tr_model_vf[ij_aring[:, 1].astype(int), ij_aring[:, 0].astype(int)] = ij_aring[:, 2]
    _tr_model_vf[ij_aring[:, 1].astype(int), ij_aring[:, 0].astype(int)] = _vsys + (_vrot * cost1 + _vrad * sint1) * sini1 # model LOS velocity

    return _tr_model_vf

#-- END OF SUB-ROUTINE____________________________________________________________#

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def derive_vlos_model_2d(_dyn_params, _tr_model_vf, ij_aring, fit_opt, _params, nrings_reliable, r_galaxy_plane):

    # nuree4
    # args[0] (_dyn_params): 2dbat tr parameters (passed by dynesty)
    # args[1] (_tr_model_vf): blank array which will be filled up below (passed by user)
    # args[2] (ij_aring): (i, j coordinates of the fitting area) (passed by user)
    # args[3] (fit_opt) : 2dbat tr parameters fitting option (passed by user)
    # args[4] (_params) : 2dbat tr parameters fitting option (passed by user)
    # args[5] (_r_galaxy_plane_init_guess) : (passed by user)
    # return _tr_model_vf : vlos updated

    # _________________________________________
    # arguments
    # _________________________________________
    # 1) _dyn_params = args[0] : dynesty params passed
    #   ---------------------
    #   2dbat tr parameters
    #   ---------------------
    #   _dyn_params[0] : sigma
    #   _dyn_params[1] : xpos
    #   _dyn_params[2] : ypos
    #   _dyn_params[3] : vsys
    #   _dyn_params[4] : pa
    #   _dyn_params[5] : incl
    #   _dyn_params[6] : vrot
    #   _dyn_params[7] : vrad

    #   _dyn_params[8] : vrot_c0_bs
    #   _dyn_params[9] : vrot_c1_bs
    #   _dyn_params[10] : vrot_c2_bs

    # 
    # 2) _tr_model_vf : blank array
    # 3) i, j : coordinate x, y
    # 4) fit_opt : trfit fitting option
    # _________________________________________
    n_cpus = int(_params['num_cpus_2d_dyn'])

    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    deg_to_rad = np.pi / 180.

    n_ring_params_free = 0 # starting from sigma
    # ------------------------------------------------------
    # sigma
    if fit_opt[0] == 1: # sigma fixed:0 free:1
        _sigma = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _sigma = _params['_sigma_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # ------------------------------------------------------
    # XPOS
    if fit_opt[1] == 1: # xpos fixed:0 free:1
        _xpos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _xpos = _params['_xpos_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # YPOS
    if fit_opt[2] == 1: # ypos fixed:0 free:1
        _ypos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _ypos = _params['_ypos_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # VSYS
    if fit_opt[3] == 1: # vsys fixed:0 free:1
        _vsys = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vsys = _params['_vsys_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # PA
    if fit_opt[4] == 1: # pa fixed:0 free:1
        _pa = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _pa = _params['_pa_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # INCL
    if fit_opt[5] == 1: # incl fixed:0 free:1
        _incl = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _incl = _params['_incl_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # VROT
    if fit_opt[6] == 1: # vrot fixed:0 free:1
        _vrot = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrot = _params['_vrot_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # VRAD
    if fit_opt[7] == 1: # vrad fixed:0 free:1
        _vrad = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrad = 0 # fixed to the previous fitting result


    # ------------------------------------------------------
    # VROT - bspline
    n_coeffs_vrot_bs, tck_vrot_bs = bspline_ncoeffs_tck(_params, 'vrot', _params['nrings_intp'])
    # ------------------------------------------------------
    # PA - bspline
    n_coeffs_pa_bs, tck_pa_bs = bspline_ncoeffs_tck(_params, 'pa', _params['nrings_intp'])
    # ------------------------------------------------------
    # INCL - bspline
    n_coeffs_incl_bs, tck_incl_bs = bspline_ncoeffs_tck(_params, 'incl', _params['nrings_intp'])


    # ---------------------------------------
    # Derive the minimum r_galaxy_plane given (xpos, ypos)
    # This is used for initial values in the newton method below
    cdelt1 = 1
    cdelt2 = 1
    del_x = (ij_aring[:, 0] - _xpos)*cdelt1 # calculate x
    del_y = (ij_aring[:, 1] - _ypos)*cdelt2 # calculate y
    #_r_galaxy_plane_init = (del_x**2 + del_y**2)**0.5 # actually minimum r_galaxy_plane given PA / INCL
    #r_galaxy_plane = (del_x**2 + del_y**2)**0.5 # actually minimum r_galaxy_plane given PA / INCL
    # ---------------------------------------

    # ---------------------------------------
    # ---------------------------------------
    # ray.put : speed up
    fx_r_galaxy_plane_id = ray.put(fx_r_galaxy_plane)
    _dyn_params_id = ray.put(_dyn_params)
    fit_opt_id = ray.put(fit_opt)
    _params_id = ray.put(_params)
    ij_aring_id = ray.put(ij_aring)
    nrings_reliable_id = ray.put(nrings_reliable)
    _xpos_id = ray.put(_xpos)
    _ypos_id = ray.put(_ypos)
    _nxy = 0
    _nxy_id = ray.put(_nxy)
    #r_galaxy_plane = ray.get([solve_r_galaxy_plane_newton.remote(fx_r_galaxy_plane, _dyn_params_id, fit_opt_id, _params_id, ij_aring_id, nrings_reliable_id, _xpos_id, _ypos_id, _nxy) for _nxy in range(0, ij_aring.shape[0])])

    #_r_galaxy_plane_init = (del_x**2 + del_y**2)**0.5 # actually minimum r_galaxy_plane given PA / INCL
    #r_galaxy_plane = (del_x**2 + del_y**2)**0.5 # actually minimum r_galaxy_plane given PA / INCL
    #results_ids = [ray.remote(optimize.newton(fx_r_galaxy_plane, _r_galaxy_plane_init, args=(_dyn_params, fit_opt, _params, ij_aring[_nxy, 0], ij_aring[_nxy, 1], nrings_reliable), tol=1E-1, maxiter=10000, disp=True)) for _nxy in range(0, ij_aring.shape[0])]
    #results_ids = [solve_r_galaxy_plane_fsolve.remote(fx_r_galaxy_plane, _dyn_params_id, fit_opt_id, _params_id, ij_aring_id, nrings_reliable_id, _xpos_id, _ypos_id, _nxy) for _nxy in range(0, ij_aring.shape[0])]
    #results_ids = [solve_r_galaxy_plane_newton.remote(fx_r_galaxy_plane, _dyn_params_id, fit_opt_id, _params_id, ij_aring_id, nrings_reliable_id, _xpos_id, _ypos_id, _nxy) for _nxy in range(0, ij_aring.shape[0])]

    results_ids = [solve_r_galaxy_plane_newton.remote(fx_r_galaxy_plane_id, _dyn_params_id, fit_opt_id, _params_id, ij_aring_id, nrings_reliable_id, _xpos_id, _ypos_id, _nxy_id) for _nxy_id in range(0, n_cpus)]
    while len(results_ids):
        done_ids, results_ids = ray.wait(results_ids)
        if done_ids:
            # _xs, _xe, _ys, _ye : variables inside the loop
            #_r_galaxy_plane_i = ray.get(done_ids)[0]
            #_nxy_record = ray.get(done_ids)[1]
            _r_galaxy_plane_i =  ray.get(done_ids[0])[0]
            _nxy_record = ray.get(done_ids[0])[1]
            if _nxy_record == n_cpus - 1:
                _is = _nxy_record*int(ij_aring.shape[0]/n_cpus)
                _ie = int(ij_aring.shape[0])
                r_galaxy_plane[_is:_ie] = _r_galaxy_plane_i
            else:
                _is = _nxy_record*int(ij_aring.shape[0]/n_cpus)
                _ie = (_nxy_record+1)*int(ij_aring.shape[0]/n_cpus)
                r_galaxy_plane[_is:_ie] = _r_galaxy_plane_i

            #print(_nxy_record, r_galaxy_plane[_nxy_record])

    r_galaxy_plane_within = np.where(r_galaxy_plane > _params['r_galaxy_plane_e'], _params['r_galaxy_plane_e'], r_galaxy_plane)

    #print(ray.get(results_ids)[0])

    #results_compile = ray.get(results_ids)
    #print(results_compile)
#    ray.shutdown()A
    #print(results_ids[0])
    # ---------------------------------------
    # ---------------------------------------


    # ---------------------------------------
    # Derive r_galaxy_plane from fx_r_galaxy_plane, a non-linear equation using newton method
    #_r_galaxy_plane_init = (del_x**2 + del_y**2)**0.5 # actually minimum r_galaxy_plane given PA / INCL

    #r_galaxy_plane = optimize.newton(fx_r_galaxy_plane, x0=0.1*_r_galaxy_plane_init, x1=5*_r_galaxy_plane_init, args=(_dyn_params, fit_opt, _params, ij_aring[:, 0], ij_aring[:, 1], nrings_reliable), tol=1E-2, maxiter=1000, disp=True)
    #r_galaxy_plane = optimize.newton(fx_r_galaxy_plane, x0=_r_galaxy_plane_init, args=(_dyn_params, fit_opt, _params, ij_aring[:, 0], ij_aring[:, 1], nrings_reliable), tol=4, maxiter=10000, disp=True)
    #r_galaxy_plane = fsolve(fx_r_galaxy_plane, x0=_r_galaxy_plane_init, args=(_dyn_params, fit_opt, _params, ij_aring[:, 0], ij_aring[:, 1], nrings_reliable), tol=1E-2, maxiter=1000, disp=True)
    #r_galaxy_plane = fsolve(fx_r_galaxy_plane, x0=_r_galaxy_plane_init, args=(_dyn_params, fit_opt, _params, ij_aring[:, 0], ij_aring[:, 1], nrings_reliable))
    #r_galaxy_plane = fsolve(fx_r_galaxy_plane, x0=_r_galaxy_plane_init, xtol=10, maxfev=10, factor=0.1, args=(_dyn_params, fit_opt, _params, ij_aring[:, 0], ij_aring[:, 1], nrings_reliable))



    #print(np.where(r_galaxy_plane - _r_galaxy_plane_init < 0))
    #print(r_galaxy_plane - _r_galaxy_plane_init)
    #sys.exit()


    #r_galaxy_plane = optimize.newton(fx_r_galaxy_plane, x0=0.5*_r_galaxy_plane_init, x1=5*_r_galaxy_plane_init, args=(_dyn_params, fit_opt, _params, ij_aring[:, 0], ij_aring[:, 1], nrings_reliable), tol=1E-1, maxiter=10000, disp=True)
    #roots = optimize.newton(fx_r_galaxy_plane, _r_galaxy_plane_init, args=(_dyn_params, fit_opt, _params, ij_aring[:, 0], ij_aring[:, 1], nrings_reliable), rtol=1E0, tol=0.5E0, maxiter=10000, disp=True)
    #root, count = Counter(roots).most_common(1)[0]
    #r_galaxy_plane = root


    #r_galaxy_plane = optimize.newton(fx_r_galaxy_plane, _r_galaxy_plane_init, args=(_dyn_params, fit_opt, _params, ij_aring[:, 0], ij_aring[:, 1], nrings_reliable), tol=1E-1, maxiter=10000, disp=True) 

    #roots = optimize.newton(fx_r_galaxy_plane, np.arange(0, 2*_r_galaxy_plane_init, (_r_galaxy_plane_init)), args=(_dyn_params, fit_opt, _params, ij_aring[:, 0], ij_aring[:, 1], nrings_reliable), rtol=1E-1, tol=0.5E-1, maxiter=10000, disp=True)
    #root, count = Counter(roots).most_common(1)[0]
    #r_galaxy_plane = root
    #print(root, count)

    #print(r_galaxy_plane[0]- r_galaxy_plane_newton[0], r_galaxy_plane[300]- r_galaxy_plane_newton[300], r_galaxy_plane[699]-r_galaxy_plane_newton[699])
    #r_galaxy_plane = (del_x**2 + del_y**2)**0.5 # actually minimum r_galaxy_plane given PA / INCL
    #print(_r_galaxy_plane_init.shape)
    # ---------------------------------------

    # ---------------------------------------
    # PA - bspline
    if n_coeffs_pa_bs != 0: # not constant
        for _nbs in range(0, n_coeffs_pa_bs):
            tck_pa_bs[1][_nbs] = _dyn_params[n_ring_params_free + n_coeffs_vrot_bs + _nbs]

        _pa_bs = BSpline(*tck_pa_bs, extrapolate=True)(r_galaxy_plane_within) # <-- tck_pa_bs[1] coeffs (0 ~ 360)
        #_pa_bs = _pa_bs*(_params['pa_bs_max'] - _params['pa_bs_min']) + _params['pa_bs_min'] # going back to PA unit (0~360)

        #print("pa", _pa_bs)
    else:
        _pa_bs = _pa

    # ---------------------------------------
    # INCL - bspline
    if n_coeffs_incl_bs != 0: # not constant
        for _nbs in range(0, n_coeffs_incl_bs):
            tck_incl_bs[1][_nbs] = _dyn_params[n_ring_params_free + n_coeffs_vrot_bs + n_coeffs_pa_bs + _nbs]

        _incl_bs = BSpline(*tck_incl_bs, extrapolate=True)(r_galaxy_plane_within) # <-- tck_incl_bs[1] coeffs (0 ~ 90)
        #_incl_bs = _incl_bs*(_params['incl_bs_max'] - _params['incl_bs_min']) + _params['incl_bs_min'] # going back to INCL unit (0~360)
        #print("incl", _incl_bs)
    else:
        _incl_bs = _incl

    # -------------------------------------
    # Derive cosp1, sinp1, cosi1, sini1, x_galaxy_plane, y_galaxy_plane, cost1, sint1 usint the derived _pa_bs and _incl
    # These are used for computing v_LOS below
    cosp1 = np.cos(deg_to_rad*_pa_bs)
    sinp1 = np.sin(deg_to_rad*_pa_bs)
    cosi1 = np.cos(deg_to_rad*_incl_bs) #cosine
    sini1 = np.sin(deg_to_rad*_incl_bs) # sine

    x_galaxy_plane = (-del_x * sinp1 + del_y * cosp1) # x in plane of galaxy
    y_galaxy_plane = (-del_x * cosp1 - del_y * sinp1) / cosi1 # y in plane of galaxy

    cost1 = x_galaxy_plane / r_galaxy_plane # cosine of angle in plane of galaxy 
    sint1 = y_galaxy_plane / r_galaxy_plane # sine of angle in plane of galaxy

    # -------------------------------------
    # _vrot_bs
    # derive _vrot_bs via BSpline using the generated coefficients from dynesty
    if n_coeffs_vrot_bs != 0: # not constant
        for _nbs in range(0, n_coeffs_vrot_bs):
            tck_vrot_bs[1][_nbs] = _dyn_params[n_ring_params_free+_nbs]
        _vrot_bs = BSpline(*tck_vrot_bs, extrapolate=True)(r_galaxy_plane_within)
        #_vrot_bs = np.where(r_galaxy_plane > _params['r_galaxy_plane_e'], _vrot_bs_within*(r_galaxy_plane - r_galaxy_plane_within), _vrot_bs_within)
    else:
        _vrot_bs = _vrot

    #ij_aring[:,2] = _vsys + (_vrot_bs * cost1 + _vrad * sint1) * sini1 # model LOS velocity

    _tr_model_vf[:, :] = np.nan # initialize with a blank value
    #_tr_model_vf[ij_aring[:, 1].astype(int), ij_aring[:, 0].astype(int)] = ij_aring[:, 2]
    _tr_model_vf[ij_aring[:, 1].astype(int), ij_aring[:, 0].astype(int)] = \
        _vsys + (_vrot_bs * cost1 + _vrad * sint1) * sini1 # model LOS velocity

    return _tr_model_vf

#-- END OF SUB-ROUTINE____________________________________________________________#




#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def extract_tr2dfit_params(_tr2dfit_results, _params, fit_opt_2d, ring):

    # args[0] (_dyn_params): 2dbat tr parameters (passed by dynesty)
    # args[4] (_params) : 2dbat tr parameters fitting option (passed by user)
    # args[5] (ring) : (passed by user)
    # return _tr_model_vf : vlos updated

    # _________________________________________
    # arguments
    # _________________________________________
    # 1) _dyn_params = args[0] : dynesty params passed
    #   ---------------------
    #   2dbat tr parameters
    #   ---------------------
    #   _dyn_params[0] : sigma
    #   _dyn_params[1] : xpos
    #   _dyn_params[2] : ypos
    #   _dyn_params[3] : vsys
    #   _dyn_params[4] : pa
    #   _dyn_params[5] : incl
    #   _dyn_params[6] : vrot
    #   _dyn_params[7] : vrad

    #   _dyn_params[8] : vrot_c0_bs
    #   _dyn_params[9] : vrot_c1_bs
    #   _dyn_params[10] : vrot_c2_bs

    # 
    # 2) _tr_model_vf : blank array
    # 3) i, j : coordinate x, y
    # 4) fit_opt : trfit2d fitting option
    # _________________________________________

    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    deg_to_rad = np.pi / 180.


    # ------------------------------------------------------
    # PA - bspline
    n_coeffs_pa_bs, tck_pa_bs = bspline_ncoeffs_tck(_params, 'pa', _params['nrings_intp'])
    n_coeffs_pa_bs, tck_pa_bs_e = bspline_ncoeffs_tck(_params, 'pa', _params['nrings_intp'])
    # ------------------------------------------------------
    # INCL - bspline
    n_coeffs_incl_bs, tck_incl_bs = bspline_ncoeffs_tck(_params, 'incl', _params['nrings_intp'])
    n_coeffs_incl_bs, tck_incl_bs_e = bspline_ncoeffs_tck(_params, 'incl', _params['nrings_intp'])
    # ------------------------------------------------------
    # VROT - bspline
    n_coeffs_vrot_bs, tck_vrot_bs = bspline_ncoeffs_tck(_params, 'vrot', _params['nrings_intp'])
    n_coeffs_vrot_bs, tck_vrot_bs_e = bspline_ncoeffs_tck(_params, 'vrot', _params['nrings_intp'])
    # ------------------------------------------------------
    # VRAD - bspline
    n_coeffs_vrad_bs, tck_vrad_bs = bspline_ncoeffs_tck(_params, 'vrad', _params['nrings_intp'])
    n_coeffs_vrad_bs, tck_vrad_bs_e = bspline_ncoeffs_tck(_params, 'vrad', _params['nrings_intp'])


    # Number of ring parameters except for the bs ones
    n_ring_params_free = 0 # starting from sigma
    # ------------------------------------------------------
    # sigma
    if fit_opt_2d[0] == 1: # sigma fixed:0 free:1
        _sigma = _tr2dfit_results[n_ring_params_free]
        _params['_sigma_init'] = _sigma # update _params
        n_ring_params_free += 1
    else:
        _sigma = _params['_sigma_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # ------------------------------------------------------
    # XPOS
    if fit_opt_2d[1] == 1: # xpos fixed:0 free:1
        _xpos = _tr2dfit_results[n_ring_params_free]
        _params['_xpos_init'] = _xpos # update _params
        n_ring_params_free += 1
    else:
        _xpos = _params['_xpos_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # YPOS
    if fit_opt_2d[2] == 1: # ypos fixed:0 free:1
        _ypos = _tr2dfit_results[n_ring_params_free]
        _params['_ypos_init'] = _ypos # update _params
        n_ring_params_free += 1
    else:
        _ypos = _params['_ypos_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # VSYS
    if fit_opt_2d[3] == 1: # vsys fixed:0 free:1
        _vsys = _tr2dfit_results[n_ring_params_free]
        _params['_vsys_init'] = _vsys # update _params
        n_ring_params_free += 1
    else:
        _vsys = _params['_vsys_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # PA
    if fit_opt_2d[4] == 1 and n_coeffs_pa_bs == 0: # pa fixed:0 free:1
        _pa = _tr2dfit_results[n_ring_params_free]
        _params['_pa_init'] = _pa # update _params
        n_ring_params_free += 1
    else:
        _pa = _params['_pa_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # INCL
    if fit_opt_2d[5] == 1 and n_coeffs_incl_bs == 0 : # incl fixed:0 free:1
        _incl = _tr2dfit_results[n_ring_params_free]
        _params['_incl_init'] = _incl # update _params
        n_ring_params_free += 1
    else:
        _incl = _params['_incl_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # VROT
    if fit_opt_2d[6] == 1 and n_coeffs_vrot_bs == 0: # vrot fixed:0 free:1
        _vrot = _tr2dfit_results[n_ring_params_free]
        _params['_vrot_init'] = _vrot # update _params
        n_ring_params_free += 1
    else:
        _vrot = _params['_vrot_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # VRAD
    if fit_opt_2d[7] == 1: # vrad fixed:0 free:1
        _vrad = _tr2dfit_results[n_ring_params_free]
        _params['_vrad_init'] = _vrad # update _params
        n_ring_params_free += 1
    else:
        _vrad = _params['_vrad_init'] # fixed to the previous fitting result



    # ---------------------------------------
    # Derive the minimum r_galaxy_plane given (xpos, ypos)
    # This is used for initial values in the newton method below

    # ---------------------------------------
    # Total number of parameters being fitted
    ndim_total = n_ring_params_free + n_coeffs_pa_bs + n_coeffs_incl_bs + n_coeffs_vrot_bs + n_coeffs_vrad_bs

    # Number of ring parameters except for the bs ones
    # Reset to zero for error params
    n_ring_params_free = 0 # starting from sigma-e
    # ------------------------------------------------------
    # sigma-e
    if fit_opt_2d[0] == 1: # sigma fixed:0 free:1
        _sigma_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _sigma_e = 999
    # ------------------------------------------------------
    # ------------------------------------------------------
    # XPOS-e
    if fit_opt_2d[1] == 1: # xpos fixed:0 free:1
        _xpos_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _xpos_e = 999
    # ------------------------------------------------------
    # YPOS-e
    if fit_opt_2d[2] == 1: # ypos fixed:0 free:1
        _ypos_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _ypos_e = 999
    # ------------------------------------------------------
    # VSYS-e
    if fit_opt_2d[3] == 1: # vsys fixed:0 free:1
        _vsys_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _vsys_e = 999
    # ------------------------------------------------------
    # PA-e
    if fit_opt_2d[4] == 1 and n_coeffs_pa_bs == 0: # pa fixed:0 free:1
        _pa_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _pa_e = 999
    # ------------------------------------------------------
    # INCL-e
    if fit_opt_2d[5] == 1 and n_coeffs_incl_bs == 0 : # incl fixed:0 free:1
        _incl_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _incl_e = 999
    # ------------------------------------------------------
    # VROT-e
    if fit_opt_2d[6] == 1 and n_coeffs_vrot_bs == 0: # vrot fixed:0 free:1
        _vrot_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _vrot_e = 999
    # ------------------------------------------------------
    # VRAD-e
    if fit_opt_2d[7] == 1: # vrad fixed:0 free:1
        _vrad_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _vrad_e = 999



    # -------------------------------------
    # -------------------------------------
    # B-spline parameters: PA, INCL, VROT, VRAD
    # -------------------------------------
    # -------------------------------------

    # -------------------------------------
    # _pa_bs
    # derive _pa_bs via BSpline using the generated coefficients from dynesty
    if n_coeffs_pa_bs != 0:
        _is = n_ring_params_free
        _ie = _is + n_coeffs_pa_bs 
        tck_pa_bs[1][:n_coeffs_pa_bs] = _tr2dfit_results[_is:_ie]
        tck_pa_bs_e[1][:n_coeffs_pa_bs] = _tr2dfit_results[_is+ndim_total:_ie+ndim_total]
    else:
        _pa_bs = _pa

    # -------------------------------------
    # _incl_bs
    # derive _incl_bs via BSpline using the generated coefficients from dynesty
    if n_coeffs_incl_bs != 0:
        _is = n_ring_params_free + n_coeffs_pa_bs
        _ie = _is + n_coeffs_incl_bs 
        tck_incl_bs[1][:n_coeffs_incl_bs] = _tr2dfit_results[_is:_ie]
        tck_incl_bs_e[1][:n_coeffs_incl_bs] = _tr2dfit_results[_is+ndim_total:_ie+ndim_total]
    else:
        _incl_bs = _incl

    # -------------------------------------
    # _vrot_bs
    # derive _vrot_bs via BSpline using the generated coefficients from dynesty
    if n_coeffs_vrot_bs != 0: # not constant
        _is = n_ring_params_free + n_coeffs_pa_bs + n_coeffs_incl_bs 
        _ie = _is + n_coeffs_vrot_bs 
        tck_vrot_bs[1][:n_coeffs_vrot_bs] = _tr2dfit_results[_is:_ie]
        tck_vrot_bs_e[1][:n_coeffs_vrot_bs] = _tr2dfit_results[_is+ndim_total:_ie+ndim_total]
    else:
        _vrot_bs = _vrot

    # -------------------------------------
    # _vrad_bs
    # derive _vrad_bs via BSpline using the generated coefficients from dynesty
    if n_coeffs_vrad_bs != 0: # not constant
        _is = n_ring_params_free + n_coeffs_pa_bs + n_coeffs_incl_bs + n_coeffs_vrot_bs
        _ie = _is + n_coeffs_vrad_bs 
        tck_vrad_bs[1][:n_coeffs_vrad_bs] = _tr2dfit_results[_is:_ie]
        tck_vrad_bs_e[1][:n_coeffs_vrad_bs] = _tr2dfit_results[_is+ndim_total:_ie+ndim_total]
    else:
        _vrad_bs = _vrad




    # ring ...
    # ---------------------------------------
    # ---------------------------------------
    # OUTER RINGS CONSTRAINTS
    # ---------------------------------------
    # ---------------------------------------
    if ring > _params['r_galaxy_plane_e']:
        ring = _params['r_galaxy_plane_e'] 

    # ---------------------------------------
    # PA - bspline
    if n_coeffs_pa_bs != 0: # not constant
        _pa_bs = BSpline(*tck_pa_bs, extrapolate=True)(ring) # <-- tck_pa_bs[1] coeffs (0 ~ 360)
        _pa_bs_e = BSpline(*tck_pa_bs_e, extrapolate=True)(ring) # <-- tck_pa_bs[1] coeffs (0 ~ 360)
        _params['_pa_init'] = _pa_bs # update _params
        #print(_pa_bs_e)
    else:
        _pa_bs_e = _pa_e

    # ---------------------------------------
    # INCL - bspline
    if n_coeffs_incl_bs != 0: # not constant
        _incl_bs = BSpline(*tck_incl_bs, extrapolate=True)(ring) # <-- tck_incl_bs[1] coeffs (0 ~ 90)
        _incl_bs_e = BSpline(*tck_incl_bs_e, extrapolate=True)(ring) # <-- tck_incl_bs[1] coeffs (0 ~ 90)
        _params['_incl_init'] = _incl_bs # update _params
    else:
        _incl_bs_e = _incl_e

    # -------------------------------------
    # _vrot_bs
    # derive _vrot_bs via BSpline using the generated coefficients from dynesty
    if n_coeffs_vrot_bs != 0: # not constant
        _vrot_bs = BSpline(*tck_vrot_bs, extrapolate=True)(ring)
        _vrot_bs_e = BSpline(*tck_vrot_bs_e, extrapolate=True)(ring)
        _params['_vrot_init'] = _vrot_bs # update _params
    else:
        _vrot_bs_e = _vrot_e

    # -------------------------------------
    # _vrad_bs
    # derive _vrad_bs via BSpline using the generated coefficients from dynesty
    if n_coeffs_vrad_bs != 0: # not constant
        _vrad_bs = BSpline(*tck_vrad_bs, extrapolate=True)(ring)
        _vrad_bs_e = BSpline(*tck_vrad_bs_e, extrapolate=True)(ring)
        _params['_vrad_init'] = _vrad_bs # update _params
    else:
        _vrad_bs_e = _vrad_e

    return _sigma, _sigma_e, _xpos, _xpos_e, _ypos, _ypos_e, _vsys, _vsys_e, _pa_bs, _pa_bs_e, _incl_bs, _incl_bs_e, _vrot_bs, _vrot_bs_e, _vrad_bs, _vrad_bs_e

#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def extract_tr2dfit_params_part_for_make_model_vf(_tr2dfit_results, _params, fit_opt_2d):

    # args[0] (_dyn_params): 2dbat tr parameters (passed by dynesty)
    # args[4] (_params) : 2dbat tr parameters fitting option (passed by user)
    # args[5] (ring) : (passed by user)
    # return _tr_model_vf : vlos updated

    # _________________________________________
    # arguments
    # _________________________________________
    # 1) _dyn_params = args[0] : dynesty params passed
    #   ---------------------
    #   2dbat tr parameters
    #   ---------------------
    #   _dyn_params[0] : sigma
    #   _dyn_params[1] : xpos
    #   _dyn_params[2] : ypos
    #   _dyn_params[3] : vsys
    #   _dyn_params[4] : pa
    #   _dyn_params[5] : incl
    #   _dyn_params[6] : vrot
    #   _dyn_params[7] : vrad

    #   _dyn_params[8] : vrot_c0_bs
    #   _dyn_params[9] : vrot_c1_bs
    #   _dyn_params[10] : vrot_c2_bs

    # 
    # 2) _tr_model_vf : blank array
    # 3) i, j : coordinate x, y
    # 4) fit_opt : trfit2d fitting option
    # _________________________________________

    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    deg_to_rad = np.pi / 180.

    # ------------------------------------------------------
    # PA - bspline
    n_coeffs_pa_bs, tck_pa_bs = bspline_ncoeffs_tck(_params, 'pa', _params['nrings_intp'])
    n_coeffs_pa_bs, tck_pa_bs_e = bspline_ncoeffs_tck(_params, 'pa', _params['nrings_intp'])
    # ------------------------------------------------------
    # INCL - bspline
    n_coeffs_incl_bs, tck_incl_bs = bspline_ncoeffs_tck(_params, 'incl', _params['nrings_intp'])
    n_coeffs_incl_bs, tck_incl_bs_e = bspline_ncoeffs_tck(_params, 'incl', _params['nrings_intp'])
    # ------------------------------------------------------
    # VROT - bspline
    n_coeffs_vrot_bs, tck_vrot_bs = bspline_ncoeffs_tck(_params, 'vrot', _params['nrings_intp'])
    n_coeffs_vrot_bs, tck_vrot_bs_e = bspline_ncoeffs_tck(_params, 'vrot', _params['nrings_intp'])
    # ------------------------------------------------------
    # VRAD - bspline
    n_coeffs_vrad_bs, tck_vrad_bs = bspline_ncoeffs_tck(_params, 'vrad', _params['nrings_intp'])
    n_coeffs_vrad_bs, tck_vrad_bs_e = bspline_ncoeffs_tck(_params, 'vrad', _params['nrings_intp'])


    # Number of ring parameters except for the bs ones
    n_ring_params_free = 0 # starting from sigma
    # ------------------------------------------------------
    # sigma
    if fit_opt_2d[0] == 1: # sigma fixed:0 free:1
        _sigma = _tr2dfit_results[n_ring_params_free]
        _params['_sigma_init'] = _sigma # update params
        n_ring_params_free += 1
    else:
        _sigma = _params['_sigma_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # ------------------------------------------------------
    # XPOS
    if fit_opt_2d[1] == 1: # xpos fixed:0 free:1
        _xpos = _tr2dfit_results[n_ring_params_free]
        _params['_xpos_init'] = _xpos # update params
        n_ring_params_free += 1
    else:
        _xpos = _params['_xpos_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # YPOS
    if fit_opt_2d[2] == 1: # ypos fixed:0 free:1
        _ypos = _tr2dfit_results[n_ring_params_free]
        _params['_ypos_init'] = _ypos # update params
        n_ring_params_free += 1
    else:
        _ypos = _params['_ypos_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # VSYS
    if fit_opt_2d[3] == 1: # vsys fixed:0 free:1
        _vsys = _tr2dfit_results[n_ring_params_free]
        _params['_vsys_init'] = _vsys # update params
        n_ring_params_free += 1
    else:
        _vsys = _params['_vsys_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # PA
    if fit_opt_2d[4] == 1 and n_coeffs_pa_bs == 0: # pa fixed:0 free:1
        _pa = _tr2dfit_results[n_ring_params_free]
        _params['_pa_init'] = _pa # update params
        n_ring_params_free += 1
    else:
        _pa = _params['_pa_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # INCL
    if fit_opt_2d[5] == 1 and n_coeffs_incl_bs == 0 : # incl fixed:0 free:1
        _incl = _tr2dfit_results[n_ring_params_free]
        _params['_incl_init'] = _incl # update params
        n_ring_params_free += 1
    else:
        _incl = _params['_incl_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # VROT
    if fit_opt_2d[6] == 1 and n_coeffs_vrot_bs == 0: # vrot fixed:0 free:1
        _vrot = _tr2dfit_results[n_ring_params_free]
        _params['_vrot_init'] = _vrot # update params
        n_ring_params_free += 1
    else:
        _vrot = _params['_vrot_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # VRAD
    if fit_opt_2d[7] == 1 and n_coeffs_vrad_bs == 0: # vrad fixed:0 free:1
        _vrad = _tr2dfit_results[n_ring_params_free]
        _params['_vrad_init'] = _vrad # update params
        n_ring_params_free += 1
    else:
        _vrad = _params['_vrad_init'] # fixed to the previous fitting result


    # ---------------------------------------
    # Derive the minimum r_galaxy_plane given (xpos, ypos)
    # This is used for initial values in the newton method below

    # ---------------------------------------
    # Total number of parameters being fitted
    ndim_total = n_ring_params_free + n_coeffs_pa_bs + n_coeffs_incl_bs + n_coeffs_vrot_bs + n_coeffs_vrad_bs

    # Number of ring parameters except for the bs ones
    # Reset to zero for error params
    n_ring_params_free = 0 # starting from sigma-e
    # ------------------------------------------------------
    # sigma-e
    if fit_opt_2d[0] == 1: # sigma fixed:0 free:1
        _sigma_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _sigma_e = 999
    # ------------------------------------------------------
    # ------------------------------------------------------
    # XPOS-e
    if fit_opt_2d[1] == 1: # xpos fixed:0 free:1
        _xpos_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _xpos_e = 999
    # ------------------------------------------------------
    # YPOS-e
    if fit_opt_2d[2] == 1: # ypos fixed:0 free:1
        _ypos_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _ypos_e = 999
    # ------------------------------------------------------
    # VSYS-e
    if fit_opt_2d[3] == 1: # vsys fixed:0 free:1
        _vsys_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _vsys_e = 999
    # ------------------------------------------------------
    # PA-e
    if fit_opt_2d[4] == 1 and n_coeffs_pa_bs == 0: # pa fixed:0 free:1
        _pa_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _pa_e = 999
    # ------------------------------------------------------
    # INCL-e
    if fit_opt_2d[5] == 1 and n_coeffs_incl_bs == 0 : # incl fixed:0 free:1
        _incl_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _incl_e = 999
    # ------------------------------------------------------
    # VROT-e
    if fit_opt_2d[6] == 1 and n_coeffs_vrot_bs == 0: # vrot fixed:0 free:1
        _vrot_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _vrot_e = 999
    # ------------------------------------------------------
    # VRAD-e
    if fit_opt_2d[7] == 1 and n_coeffs_vrad_bs == 0: # vrot fixed:0 free:1: # vrad fixed:0 free:1ssssss
        _vrad_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _vrad_e = 999





    # -------------------------------------
    # -------------------------------------
    # B-spline params: PA, INCL, VROT, VRAD

    # -------------------------------------
    # _pa_bs
    # derive _pa_bs via BSpline using the generated coefficients from dynesty
    if n_coeffs_pa_bs != 0:
        _is = n_ring_params_free
        _ie = _is + n_coeffs_pa_bs 
        tck_pa_bs[1][:n_coeffs_pa_bs] = _tr2dfit_results[_is:_ie]
        tck_pa_bs_e[1][:n_coeffs_pa_bs] = _tr2dfit_results[_is+ndim_total:_ie+ndim_total]

    # -------------------------------------
    # _incl_bs
    # derive _incl_bs via BSpline using the generated coefficients from dynesty
    if n_coeffs_incl_bs != 0:
        _is = n_ring_params_free + n_coeffs_pa_bs
        _ie = _is + n_coeffs_incl_bs 
        tck_incl_bs[1][:n_coeffs_incl_bs] = _tr2dfit_results[_is:_ie]
        tck_incl_bs_e[1][:n_coeffs_incl_bs] = _tr2dfit_results[_is+ndim_total:_ie+ndim_total]

    # -------------------------------------
    # _vrot_bs
    # derive _vrot_bs via BSpline using the generated coefficients from dynesty
    if n_coeffs_vrot_bs != 0: # not constant
        _is = n_ring_params_free + n_coeffs_pa_bs + n_coeffs_incl_bs 
        _ie = _is + n_coeffs_vrot_bs 
        tck_vrot_bs[1][:n_coeffs_vrot_bs] = _tr2dfit_results[_is:_ie]
        tck_vrot_bs_e[1][:n_coeffs_vrot_bs] = _tr2dfit_results[_is+ndim_total:_ie+ndim_total]


    # -------------------------------------
    # _vrad_bs
    # derive _vrad_bs via BSpline using the generated coefficients from dynesty
    if n_coeffs_vrad_bs != 0: # not constant
        _is = n_ring_params_free + n_coeffs_pa_bs + n_coeffs_incl_bs + n_coeffs_vrot_bs
        _ie = _is + n_coeffs_vrad_bs 
        tck_vrad_bs[1][:n_coeffs_vrad_bs] = _tr2dfit_results[_is:_ie]
        tck_vrad_bs_e[1][:n_coeffs_vrad_bs] = _tr2dfit_results[_is+ndim_total:_ie+ndim_total]

    return _sigma, _sigma_e, _xpos, _xpos_e, _ypos, _ypos_e, _vsys, _vsys_e, \
        _pa, _pa_e, _incl, _incl_e, _vrot, _vrot_e, _vrad, _vrad_e, \
        n_coeffs_pa_bs, tck_pa_bs, tck_pa_bs_e, \
        n_coeffs_incl_bs, tck_incl_bs, tck_incl_bs_e, \
        n_coeffs_vrot_bs, tck_vrot_bs, tck_vrot_bs_e, \
        n_coeffs_vrad_bs, tck_vrad_bs, tck_vrad_bs_e

#-- END OF SUB-ROUTINE____________________________________________________________#























#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def derive_vlos_model_2d_sc(_dyn_params, _tr_model_vf, _wi_2d, ij_aring, fit_opt, _params, nrings_reliable, r_galaxy_plane, n_coeffs_pa_bs, tck_pa_bs, n_coeffs_incl_bs, tck_incl_bs, n_coeffs_vrot_bs, tck_vrot_bs):

    # nuree4
    # args[0] (_dyn_params): 2dbat tr parameters (passed by dynesty)
    # args[1] (_tr_model_vf): blank array which will be filled up below (passed by user)
    # args[2] (ij_aring): (i, j coordinates of the fitting area) (passed by user)
    # args[3] (fit_opt) : 2dbat tr parameters fitting option (passed by user)
    # args[4] (_params) : 2dbat tr parameters fitting option (passed by user)
    # args[5] (_r_galaxy_plane_init_guess) : (passed by user)
    # return _tr_model_vf : vlos updated

    # _________________________________________
    # arguments
    # _________________________________________
    # 1) _dyn_params = args[0] : dynesty params passed
    #   ---------------------
    #   2dbat tr parameters
    #   ---------------------
    #   _dyn_params[0] : sigma
    #   _dyn_params[1] : xpos
    #   _dyn_params[2] : ypos
    #   _dyn_params[3] : vsys
    #   _dyn_params[4] : pa
    #   _dyn_params[5] : incl
    #   _dyn_params[6] : vrot
    #   _dyn_params[7] : vrad

    #   _dyn_params[8] : vrot_c0_bs
    #   _dyn_params[9] : vrot_c1_bs
    #   _dyn_params[10] : vrot_c2_bs

    # 
    # 2) _tr_model_vf : blank array
    # 3) i, j : coordinate x, y
    # 4) fit_opt : trfit fitting option
    # _________________________________________

    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    deg_to_rad = np.pi / 180.


    # ------------------------------------------------------
    # PA - bspline
    #n_coeffs_pa_bs, tck_pa_bs = bspline_ncoeffs_tck(_params, 'pa', _params['nrings_intp'])
    # ------------------------------------------------------
    # INCL - bspline
    #n_coeffs_incl_bs, tck_incl_bs = bspline_ncoeffs_tck(_params, 'incl', _params['nrings_intp'])
    # ------------------------------------------------------
    # VROT - bspline
    #n_coeffs_vrot_bs, tck_vrot_bs = bspline_ncoeffs_tck(_params, 'vrot', _params['nrings_intp'])


    n_ring_params_free = 0 # starting from sigma
    # ------------------------------------------------------
    # sigma
    if fit_opt[0] == 1: # sigma fixed:0 free:1
        _sigma = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _sigma = _params['_sigma_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # ------------------------------------------------------
    # XPOS
    if fit_opt[1] == 1: # xpos fixed:0 free:1
        _xpos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _xpos = _params['_xpos_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # YPOS
    if fit_opt[2] == 1: # ypos fixed:0 free:1
        _ypos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _ypos = _params['_ypos_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # VSYS
    if fit_opt[3] == 1: # vsys fixed:0 free:1
        _vsys = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vsys = _params['_vsys_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # PA
    if fit_opt[4] == 1 and n_coeffs_pa_bs == 0: # pa fixed:0 free:1
        _pa = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _pa = _params['_pa_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # INCL
    if fit_opt[5] == 1 and n_coeffs_incl_bs == 0 : # incl fixed:0 free:1
        _incl = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _incl = _params['_incl_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # VROT
    if fit_opt[6] == 1 and n_coeffs_vrot_bs == 0: # vrot fixed:0 free:1
        _vrot = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrot = _params['_vrot_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # VRAD
    if fit_opt[7] == 1: # vrad fixed:0 free:1
        _vrad = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrad = _params['_vrad_init'] # fixed to the previous fitting result



    # ---------------------------------------
    # Derive the minimum r_galaxy_plane given (xpos, ypos)
    # This is used for initial values in the newton method below
    cdelt1 = 1
    cdelt2 = 1
    del_x = (ij_aring[:, 0] - _xpos)*cdelt1 # calculate x
    del_y = (ij_aring[:, 1] - _ypos)*cdelt2 # calculate y
    #_r_galaxy_plane_init = (del_x**2 + del_y**2)**0.5 # actually minimum r_galaxy_plane given PA / INCL
    #r_galaxy_plane = (del_x**2 + del_y**2)**0.5 # actually minimum r_galaxy_plane given PA / INCL
    # ---------------------------------------

    # -------------------------------------
    # _pa_bs
    # derive _pa_bs via BSpline using the generated coefficients from dynesty
    if n_coeffs_pa_bs != 0:
        #for _nbs in range(0, n_coeffs_pa_bs):
        #    tck_pa_bs[1][_nbs] = _dyn_params[n_ring_params_free + _nbs]
        _is = n_ring_params_free
        _ie = _is +n_coeffs_pa_bs 
        tck_pa_bs[1][:n_coeffs_pa_bs] = _dyn_params[_is:_ie]
    else:
        _pa_bs = _pa

    # -------------------------------------
    # _incl_bs
    # derive _incl_bs via BSpline using the generated coefficients from dynesty
    if n_coeffs_incl_bs != 0:
        #for _nbs in range(0, n_coeffs_incl_bs):
        #    tck_incl_bs[1][_nbs] = _dyn_params[n_ring_params_free + n_coeffs_pa_bs + _nbs]
        _is = n_ring_params_free + n_coeffs_pa_bs
        _ie = _is +n_coeffs_incl_bs 
        tck_incl_bs[1][:n_coeffs_incl_bs] = _dyn_params[_is:_ie]
    else:
        _incl_bs = _incl

    # -------------------------------------
    # _vrot_bs
    # derive _vrot_bs via BSpline using the generated coefficients from dynesty
    if n_coeffs_vrot_bs != 0: # not constant
        _is = n_ring_params_free + n_coeffs_pa_bs + n_coeffs_incl_bs 
        _ie = _is +n_coeffs_vrot_bs 
        tck_vrot_bs[1][:n_coeffs_vrot_bs] = _dyn_params[_is:_ie]
#        for _nbs in range(0, n_coeffs_vrot_bs):
#            tck_vrot_bs[1][_nbs] = _dyn_params[n_ring_params_free + n_coeffs_pa_bs + n_coeffs_incl_bs + _nbs]
    else:
        _vrot_bs = _vrot


    # ---------------------------------------
    # ---------------------------------------

    r_galaxy_plane, _nxy = solve_r_galaxy_plane_newton_sc(fit_opt, _params, ij_aring, _xpos, _ypos, _pa, _incl, 0, \
                                                          n_coeffs_pa_bs, tck_pa_bs, n_coeffs_incl_bs, tck_incl_bs)


    # ---------------------------------------
    # ---------------------------------------
    # OUTER RINGS CONSTRAINTS
    # ---------------------------------------
    # ---------------------------------------
    #r_galaxy_plane = np.where(r_galaxy_plane < _params['r_galaxy_plane_e'], r_galaxy_plane, _params['r_galaxy_plane_e'])

#    if np.any(np.isnan(r_galaxy_plane)):
#        _tr_model_vf[0] = np.nan
#        return _tr_model_vf

    #r_galaxy_plane = (del_x**2 + del_y**2)**0.5 # actually minimum r_galaxy_plane given PA / INCL

    # ---------------------------------------
    # ---------------------------------------

    # ---------------------------------------
    # PA - bspline
    if n_coeffs_pa_bs != 0: # not constant
        _pa_bs = BSpline(*tck_pa_bs, extrapolate=True)(r_galaxy_plane) # <-- tck_pa_bs[1] coeffs (0 ~ 360)

    # ---------------------------------------
    # INCL - bspline
    if n_coeffs_incl_bs != 0: # not constant
        _incl_bs = BSpline(*tck_incl_bs, extrapolate=True)(r_galaxy_plane) # <-- tck_incl_bs[1] coeffs (0 ~ 90)

    # -------------------------------------
    # _vrot_bs
    # derive _vrot_bs via BSpline using the generated coefficients from dynesty
    if n_coeffs_vrot_bs != 0: # not constant
        _vrot_bs = BSpline(*tck_vrot_bs, extrapolate=True)(r_galaxy_plane)

    # -------------------------------------
    # Derive cosp1, sinp1, cosi1, sini1, x_galaxy_plane, y_galaxy_plane, cost1, sint1 usint the derived _pa_bs and _incl
    # These are used for computing v_LOS below
    cosp1 = np.cos(deg_to_rad*_pa_bs)
    sinp1 = np.sin(deg_to_rad*_pa_bs)
    cosi1 = np.cos(deg_to_rad*_incl_bs) #cosine
    sini1 = np.sin(deg_to_rad*_incl_bs) # sine

    x_galaxy_plane = (-del_x * sinp1 + del_y * cosp1) # x in plane of galaxy
    y_galaxy_plane = (-del_x * cosp1 - del_y * sinp1) / cosi1 # y in plane of galaxy

    cost1 = x_galaxy_plane / r_galaxy_plane # cosine of angle in plane of galaxy 
    sint1 = y_galaxy_plane / r_galaxy_plane # sine of angle in plane of galaxy

    # --------------------------------------
    # calculate cosine-weights
    #_theta_free = _params['free_angle']
    #_ri = _params['ri']
    #_ro = _params['ro']
    #_wpow = _params['weight_power']

    #_theta = np.where(r_galaxy_plane < 0.1, 0.0, np.arctan2(y_galaxy_plane, x_galaxy_plane) / deg_to_rad)
    #costh = np.abs(np.cos(_theta*deg_to_rad))
    #sin_theta_free = np.abs(np.sin(_theta_free*deg_to_rad))
    #wi = np.where((r_galaxy_plane > _ri) & (r_galaxy_plane < _ro) & (costh > sin_theta_free), costh**_wpow, 0.0) # weights of the points

#    # --------------------------------------
#    # calculate ellipse-perimeter-weights
#    #_theta_free = _params['free_angle']
#    #_ri = _params['r_galaxy_plane_s']
#    _ri = 0
#    _ro = _params['ro']
#    #_wpow = _params['weight_power']
#
#    _a = r_galaxy_plane # along major axis
#    _incl_bs_limit1 = np.where(_incl_bs > 89, 89, _incl_bs)
#    _incl_bs_limit2 = np.where(_incl_bs_limit1 < 1, 1, _incl_bs_limit1)
#    _b = _a * np.cos(_incl_bs_limit2*deg_to_rad) # along minor axis
#    _l = np.pi * (3.0*(_a + _b) - ((3*_a + _b) * (_a + 3*_b))**0.5) # ellipse perimeter approximation
#
#    _rmin_ = 0
#    _rmax_ = r_galaxy_plane.max()
#    npoints_in_1pixelwidth, bin_edges = np.histogram(r_galaxy_plane, bins=int(_rmax_)-int(_rmin_)+1, range=(int(_rmin_), int(_rmax_)+1))
#
#    #print(_rmin_, _rmax_, npoints_in_1pixelwidth.shape)
#    #npoints_for_each_r_galaxy_plane = r_galaxy_plane
#    #print(npoints_in_1pixelwidth.shape, npoints_in_1pixelwidth[25])
#    #print(r_galaxy_plane.astype(int).min())
#    npoints_for_each_r_galaxy_plane = npoints_in_1pixelwidth[r_galaxy_plane.astype(int)]  
#    #print(npoints_for_each_r_galaxy_plane)
#    #npoints_for_each_r_galaxy_plane = np.where(npoints_for_each_r_galaxy_plane == 0, 999, npoints_for_each_r_galaxy_plane)
#
#    #filling_factor = npoints_for_each_r_galaxy_plane / _l
#
#    #print("seheon", _incl_bs, npoints_for_each_r_galaxy_plane)
#    #print("seheon", _incl_bs, npoints_in_1pixelwidth)
#    #sys.exit()
#
#    #print(npoints_in_1pixelwidth[int(_rmax_)-2])
#    #print(npoints_for_each_r_galaxy_plane)
#
#    #print(r_galaxy_plane_within.astype(int))
#
#    #print(npoints_in_1pixelwidth[r_galaxy_plane_within.astype(int)])
#    #print(npoints_in_1pixelwidth.sum())
#    #wi = np.where( r_galaxy_plane)
#    #print(r_galaxy_plane_within.max(), r_galaxy_plane_within.min())
#
#    #_ai = 4 # along major axis
#    #_bi = _ai * np.cos(_incl_bs*deg_to_rad) # along minor axis
#    #_li = np.pi * (3.0*(_ai + _bi) - ((3*_ai + _bi) * (_ai + 3*_bi))**0.5) # ellipse perimeter approximation
#
#    #print(npoints_for_each_r_galaxy_plane)
#    #wi = np.where((r_galaxy_plane > _ri), 1. / _l, 0.0) # weights of the points

# -------------------------------
# -------------------------------
#    yi = a + b*xi + e
#    yi = a + b*xi + w*e
#    --> e = (yi - (a + b*xi) ) / w <-- if es are independant normal random variables with mean 0 and unknown variance sigma**2
#    --> The likelihood function  -(2/n)*ln(2pi*sigma**2)  - (1/[2sigma**2]) * Sigma(1, n) e**2
#    --> The likelihood function  -(2/n)*ln(2pi*sigma**2)  - (1/[2sigma**2]) * Sigma(1, n) (1/w**2) * (yi -  (a + b*xi) )**2
#    --> where W = (1 / w**2) 
#    --> If we choose a weight of w with (1 / length of arc), the actual W in the log-likelihood function is (1 / length of arc)**2

#    --> W affects the variance of e, i.e., sigma parameter.
#    --> So adjust the prior range of sigma or normalise W (0 ~ 1) if you'd like to use the the prior range of sigma with no weightes
# -------------------------------
# -------------------------------

#    #wi = np.where((r_galaxy_plane > _ri) & (npoints_for_each_r_galaxy_plane > 1), (1. / npoints_for_each_r_galaxy_plane)**2, 0.000000000001) # weights of the points
#    wi = np.where((r_galaxy_plane > _ri) & (npoints_for_each_r_galaxy_plane > 0), 1, 0.000000000001) # weights of the points
#    #wi = np.where(npoints_for_each_r_galaxy_plane == 999, 0.0, wi) # weights of the points
#    #wi = np.where((r_galaxy_plane < 26), wi, 0.0) # weights of the points
#    #wi = np.where(filling_factor > 0.9, wi, 0.0)
#    #wi = np.where(r_galaxy_plane > 20, wi, 0.0)
#    #print(wi)
#    #wi = np.where((r_galaxy_plane < _ri), 1. / (_li / 700.), wi) # weights of the points
#    #wi = 1. / (_l / 700.)
#    wi = wi / np.max(wi)
#
#    #_theta = np.where(r_galaxy_plane < 0.1, 0.0, np.arctan2(y_galaxy_plane, x_galaxy_plane) / deg_to_rad)
#    #costh = np.abs(np.cos(_theta*deg_to_rad))
#    #sin_theta_free = np.abs(np.sin(_theta_free*deg_to_rad))
#    #wi = np.where((r_galaxy_plane > _ri) & (r_galaxy_plane < _ro) & (costh > sin_theta_free), costh**_wpow, 0.0) # weights of the points


    # ---------------
#    print(BSpline(*tck_vrot_bs_init_from_trfit, extrapolate=True)(400)       )
#    _ring_t1 = range(0, 100, 5)
#    print(r_galaxy_plane.max())
#    temp1 = BSpline(*tck_vrot_bs, extrapolate=True)(r_galaxy_plane)       
#    fig = plt.figure()
#    ax = fig.add_subplot()
#    plt.xlim(0, 100)
#    plt.ylim(0, 1000)
#    plt.plot(r_galaxy_plane, temp1, c='r')	
#    plt.scatter(r_galaxy_plane, temp1, c='g')	
#    plt.show()
#    sys.exit()
    # ---------------

    #ij_aring[:,2] = _vsys + (_vrot_bs * cost1 + _vrad * sint1) * sini1 # model LOS velocity

#    _tr_model_vf[:, :] = -1E10 # initialize with a blank value
    #_tr_model_vf[ij_aring[:, 1].astype(int), ij_aring[:, 0].astype(int)] = ij_aring[:, 2]
    _tr_model_vf[ij_aring[:, 1].astype(int), ij_aring[:, 0].astype(int)] = \
        _vsys + (_vrot_bs * cost1 + _vrad * sint1) * sini1 # model LOS velocity

#    _wi_2d[ij_aring[:, 1].astype(int), ij_aring[:, 0].astype(int)] = wi

    #print(np.min(r_galaxy_plane), np.max(r_galaxy_plane), np.min(_pa_bs), np.max(_pa_bs), np.min(_incl_bs), np.max(_incl_bs), np.min(_vrot_bs), np.max(_vrot_bs))
    #return _tr_model_vf, _wi_2d, r_galaxy_plane, _pa_bs, _incl_bs, _vrot_bs
    return _tr_model_vf

#-- END OF SUB-ROUTINE____________________________________________________________#






#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def write_fits_images(_params, _tr_model_vf, _2dbat_run_i, fitsfile_name):
    _dir_2dbat_PI_output = _params['wdir'] + '/' + _params['_2dbatdir'] + ".%d" % _2dbat_run_i
    # If not present
    if not os.path.exists("%s" % _dir_2dbat_PI_output):
        make_dirs("%s" % _dir_2dbat_PI_output)

    #_input_vf_vlos_LU_masked_nogrid = np.where((_tr_model_vf > _params['_vlos_lower']), _tr_model_vf, np.nan)
    #_input_vf_vlos_LU_masked_nogrid = np.where((_tr_model_vf < _params['_vlos_upper']), _input_vf_vlos_LU_masked_nogrid, np.nan)


    # Read header from input FITS file
    with fits.open(_params['wdir'] + '/' + _params['input_vf'], 'update') as ref_fits:
        input_header = ref_fits[0].header

    # Create a new FITS file with data and copy the header from input FITS
    #hdu = fits.PrimaryHDU(_input_vf_vlos_LU_masked_nogrid, header=input_header)
    hdu = fits.PrimaryHDU(_tr_model_vf, header=input_header)

    # Save to a new FITS file
    output_filename = os.path.join(_dir_2dbat_PI_output, fitsfile_name)  # Use a FITS file extension
    hdu.writeto(output_filename, overwrite=True)
#-- END OF SUB-ROUTINE____________________________________________________________#




#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def make_vlos_model_vf_given_dyn_params(_input_vf_nogrid, _input_vf_tofit_2d, _wi_2d, _tr2dfit_results, _params, fit_opt_2d, _bsfit_vf, _2dbat_run_i):

    # nuree4
    # args[0] (_dyn_params): 2dbat tr parameters (passed by dynesty)
    # args[1] (_tr_model_vf): blank array which will be filled up below (passed by user)
    # args[2] (ij_aring): (i, j coordinates of the fitting area) (passed by user)
    # args[3] (fit_opt) : 2dbat tr parameters fitting option (passed by user)
    # args[4] (_params) : 2dbat tr parameters fitting option (passed by user)
    # args[5] (_r_galaxy_plane_init_guess) : (passed by user)
    # return _tr_model_vf : vlos updated

    # _________________________________________
    # arguments
    # _________________________________________
    # 1) _dyn_params = args[0] : dynesty params passed
    #   ---------------------
    #   2dbat tr parameters
    #   ---------------------
    #   _dyn_params[0] : sigma
    #   _dyn_params[1] : xpos
    #   _dyn_params[2] : ypos
    #   _dyn_params[3] : vsys
    #   _dyn_params[4] : pa
    #   _dyn_params[5] : incl
    #   _dyn_params[6] : vrot
    #   _dyn_params[7] : vrad

    #   _dyn_params[8] : vrot_c0_bs
    #   _dyn_params[9] : vrot_c1_bs
    #   _dyn_params[10] : vrot_c2_bs

    # 
    # 2) _tr_model_vf : blank array
    # 3) i, j : coordinate x, y
    # 4) fit_opt : trfit fitting option
    # _________________________________________

    _naxis1 = _params['naxis1']
    _naxis2 = _params['naxis2']
    _input_nogrid_m_trfit_vf = np.full((_naxis2, _naxis1), fill_value=np.nan, dtype=np.float64)
    _input_m_bsfit_vf_grid = np.full((_naxis2, _naxis1), fill_value=np.nan, dtype=np.float64)

    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    deg_to_rad = np.pi / 180.

    _sigma, _sigma_e, _xpos, _xpos_e, _ypos, _ypos_e, _vsys, _vsys_e, \
        _pa, _pa_e, _incl, _incl_e, _vrot, _vrot_e, _vrad, _vrad_e, \
        n_coeffs_pa_bs, tck_pa_bs, tck_pa_bs_e, \
        n_coeffs_incl_bs, tck_incl_bs, tck_incl_bs_e, \
        n_coeffs_vrot_bs, tck_vrot_bs, tck_vrot_bs_e, \
        n_coeffs_vrad_bs, tck_vrad_bs, tck_vrad_bs_e = extract_tr2dfit_params_part_for_make_model_vf(_tr2dfit_results, _params, fit_opt_2d)
    # ---------------------------------------

    #npoints_in_a_ring_total_including_blanks_t, npoints_in_ring_t, _ij_aring_nogrid_available = define_tilted_ring(_input_vf_nogrid, _xpos, _ypos, _pa, _incl, 0, 5*_params['r_galaxy_plane_e'], 0, _params)
    npoints_in_a_ring_total_including_blanks_t, npoints_in_ring_t, _ij_aring_nogrid_available, _wt_2d = define_tilted_ring(_input_vf_nogrid, _xpos, _ypos, 0, 1, 0, 5*_params['r_galaxy_plane_e'], 0, _params)

    # _ij_aring_nogrid_available --> r_galaxy_plane
    r_galaxy_plane, _nxy = solve_r_galaxy_plane_newton_sc(fit_opt_2d, _params, _ij_aring_nogrid_available, _xpos, _ypos, _pa, _incl, 0, \
                                                          n_coeffs_pa_bs, tck_pa_bs, n_coeffs_incl_bs, tck_incl_bs)

    # ---------------------------------------
    # Derive the minimum r_galaxy_plane given (xpos, ypos)
    # This is used for initial values in the newton method below
    cdelt1 = 1
    cdelt2 = 1
    del_x = (_ij_aring_nogrid_available[:, 0] - _xpos)*cdelt1 # calculate x
    del_y = (_ij_aring_nogrid_available[:, 1] - _ypos)*cdelt2 # calculate y
    #_r_galaxy_plane_init = (del_x**2 + del_y**2)**0.5 # actually minimum r_galaxy_plane given PA / INCL
    #r_galaxy_plane = (del_x**2 + del_y**2)**0.5 # actually minimum r_galaxy_plane given PA / INCL
    # ---------------------------------------



    # ---------------------------------------
    # ---------------------------------------
    # OUTER RINGS CONSTRAINTS
    # ---------------------------------------
    # ---------------------------------------
    #r_galaxy_plane = np.where(r_galaxy_plane < _params['r_galaxy_plane_e'], r_galaxy_plane, _params['r_galaxy_plane_e'])
    # ---------------------------------------
    # PA - bspline
    if n_coeffs_pa_bs != 0: # not constant
        _pa_bs = BSpline(*tck_pa_bs, extrapolate=True)(r_galaxy_plane) # <-- tck_pa_bs[1] coeffs (0 ~ 360)
    else:
        _pa_bs = _pa

    # ---------------------------------------
    # INCL - bspline
    if n_coeffs_incl_bs != 0: # not constant
        _incl_bs = BSpline(*tck_incl_bs, extrapolate=True)(r_galaxy_plane) # <-- tck_incl_bs[1] coeffs (0 ~ 90)
    else:
        _incl_bs = _incl

    # -------------------------------------
    # _vrot_bs
    # derive _vrot_bs via BSpline using the generated coefficients from dynesty
    if n_coeffs_vrot_bs != 0: # not constant
        _vrot_bs = BSpline(*tck_vrot_bs, extrapolate=True)(r_galaxy_plane)
    else:
        _vrot_bs = _vrot

    # -------------------------------------
    # _vrad_bs
    # derive _vrad_bs via BSpline using the generated coefficients from dynesty
    if n_coeffs_vrad_bs != 0: # not constant
        _vrad_bs = BSpline(*tck_vrad_bs, extrapolate=True)(r_galaxy_plane)
    else:
        _vrad_bs = _vrad

    # -------------------------------------
    # Derive cosp1, sinp1, cosi1, sini1, x_galaxy_plane, y_galaxy_plane, cost1, sint1 usint the derived _pa_bs and _incl
    # These are used for computing v_LOS below
    cosp1 = np.cos(deg_to_rad*_pa_bs)
    sinp1 = np.sin(deg_to_rad*_pa_bs)
    cosi1 = np.cos(deg_to_rad*_incl_bs) #cosine
    sini1 = np.sin(deg_to_rad*_incl_bs) # sine

    x_galaxy_plane = (-del_x * sinp1 + del_y * cosp1) # x in plane of galaxy
    y_galaxy_plane = (-del_x * cosp1 - del_y * sinp1) / cosi1 # y in plane of galaxy

    cost1 = x_galaxy_plane / r_galaxy_plane # cosine of angle in plane of galaxy 
    sint1 = y_galaxy_plane / r_galaxy_plane # sine of angle in plane of galaxy

    # --------------------------------------
    # calculate cosine-weights
    #_theta_free = _params['free_angle']
    #_ri = _params['ri']
    #_ro = _params['ro']
    #_wpow = _params['weight_power']

    #_theta = np.where(r_galaxy_plane < 0.1, 0.0, np.arctan2(y_galaxy_plane, x_galaxy_plane) / deg_to_rad)
    #costh = np.abs(np.cos(_theta*deg_to_rad))
    #sin_theta_free = np.abs(np.sin(_theta_free*deg_to_rad))
    #wi = np.where((r_galaxy_plane > _ri) & (r_galaxy_plane < _ro) & (costh > sin_theta_free), costh**_wpow, 0.0) # weights of the points

#    # --------------------------------------
#    # calculate ellipse-perimeter-weights
#    #_theta_free = _params['free_angle']
#    #_ri = _params['r_galaxy_plane_s']
#    _ri = 0
#    _ro = _params['ro']
#    #_wpow = _params['weight_power']
#
#    _a = r_galaxy_plane # along major axis
#    _incl_bs_limit1 = np.where(_incl_bs > 89, 89, _incl_bs)
#    _incl_bs_limit2 = np.where(_incl_bs_limit1 < 1, 1, _incl_bs_limit1)
#    _b = _a * np.cos(_incl_bs_limit2*deg_to_rad) # along minor axis
#    _l = np.pi * (3.0*(_a + _b) - ((3*_a + _b) * (_a + 3*_b))**0.5) # ellipse perimeter approximation
#
#    _rmin_ = 0
#    _rmax_ = r_galaxy_plane.max()
#    npoints_in_1pixelwidth, bin_edges = np.histogram(r_galaxy_plane, bins=int(_rmax_)-int(_rmin_)+1, range=(int(_rmin_), int(_rmax_)+1))
#
#    #print(_rmin_, _rmax_, npoints_in_1pixelwidth.shape)
#    #npoints_for_each_r_galaxy_plane = r_galaxy_plane
#    #print(npoints_in_1pixelwidth.shape, npoints_in_1pixelwidth[25])
#    #print(r_galaxy_plane.astype(int).min())
#    npoints_for_each_r_galaxy_plane = npoints_in_1pixelwidth[r_galaxy_plane.astype(int)]  
#    #print(npoints_for_each_r_galaxy_plane)
#    #npoints_for_each_r_galaxy_plane = np.where(npoints_for_each_r_galaxy_plane == 0, 999, npoints_for_each_r_galaxy_plane)
#
#    #filling_factor = npoints_for_each_r_galaxy_plane / _l
#
#    #print("seheon", _incl_bs, npoints_for_each_r_galaxy_plane)
#    #print("seheon", _incl_bs, npoints_in_1pixelwidth)
#    #sys.exit()
#
#    #print(npoints_in_1pixelwidth[int(_rmax_)-2])
#    #print(npoints_for_each_r_galaxy_plane)
#
#    #print(r_galaxy_plane_within.astype(int))
#
#    #print(npoints_in_1pixelwidth[r_galaxy_plane_within.astype(int)])
#    #print(npoints_in_1pixelwidth.sum())
#    #wi = np.where( r_galaxy_plane)
#    #print(r_galaxy_plane_within.max(), r_galaxy_plane_within.min())
#
#    #_ai = 4 # along major axis
#    #_bi = _ai * np.cos(_incl_bs*deg_to_rad) # along minor axis
#    #_li = np.pi * (3.0*(_ai + _bi) - ((3*_ai + _bi) * (_ai + 3*_bi))**0.5) # ellipse perimeter approximation
#
#    #print(npoints_for_each_r_galaxy_plane)
#    #wi = np.where((r_galaxy_plane > _ri), 1. / _l, 0.0) # weights of the points

# -------------------------------
# -------------------------------
#    yi = a + b*xi + e
#    yi = a + b*xi + w*e
#    --> e = (yi - (a + b*xi) ) / w <-- if es are independant normal random variables with mean 0 and unknown variance sigma**2
#    --> The likelihood function  -(2/n)*ln(2pi*sigma**2)  - (1/[2sigma**2]) * Sigma(1, n) e**2
#    --> The likelihood function  -(2/n)*ln(2pi*sigma**2)  - (1/[2sigma**2]) * Sigma(1, n) (1/w**2) * (yi -  (a + b*xi) )**2
#    --> where W = (1 / w**2) 
#    --> If we choose a weight of w with (1 / length of arc), the actual W in the log-likelihood function is (1 / length of arc)**2

#    --> W affects the variance of e, i.e., sigma parameter.
#    --> So adjust the prior range of sigma or normalise W (0 ~ 1) if you'd like to use the the prior range of sigma with no weightes
# -------------------------------
# -------------------------------

#    #wi = np.where((r_galaxy_plane > _ri) & (npoints_for_each_r_galaxy_plane > 1), (1. / npoints_for_each_r_galaxy_plane)**2, 0.000000000001) # weights of the points
#    wi = np.where((r_galaxy_plane > _ri) & (npoints_for_each_r_galaxy_plane > 0), 1, 0.000000000001) # weights of the points
#    #wi = np.where(npoints_for_each_r_galaxy_plane == 999, 0.0, wi) # weights of the points
#    #wi = np.where((r_galaxy_plane < 26), wi, 0.0) # weights of the points
#    #wi = np.where(filling_factor > 0.9, wi, 0.0)
#    #wi = np.where(r_galaxy_plane > 20, wi, 0.0)
#    #print(wi)
#    #wi = np.where((r_galaxy_plane < _ri), 1. / (_li / 700.), wi) # weights of the points
#    #wi = 1. / (_l / 700.)
#    wi = wi / np.max(wi)
#
#    #_theta = np.where(r_galaxy_plane < 0.1, 0.0, np.arctan2(y_galaxy_plane, x_galaxy_plane) / deg_to_rad)
#    #costh = np.abs(np.cos(_theta*deg_to_rad))
#    #sin_theta_free = np.abs(np.sin(_theta_free*deg_to_rad))
#    #wi = np.where((r_galaxy_plane > _ri) & (r_galaxy_plane < _ro) & (costh > sin_theta_free), costh**_wpow, 0.0) # weights of the points


    # ---------------
#    print(BSpline(*tck_vrot_bs_init_from_trfit, extrapolate=True)(400)       )
#    _ring_t1 = range(0, 100, 5)
#    print(r_galaxy_plane.max())
#    temp1 = BSpline(*tck_vrot_bs, extrapolate=True)(r_galaxy_plane)       
#    fig = plt.figure()
#    ax = fig.add_subplot()
#    plt.xlim(0, 100)
#    plt.ylim(0, 1000)
#    plt.plot(r_galaxy_plane, temp1, c='r')	
#    plt.scatter(r_galaxy_plane, temp1, c='g')	
#    plt.show()
#    sys.exit()
    # ---------------

    #ij_aring[:,2] = _vsys + (_vrot_bs * cost1 + _vrad * sint1) * sini1 # model LOS velocity


    _bsfit_vf[_ij_aring_nogrid_available[:, 1].astype(int), _ij_aring_nogrid_available[:, 0].astype(int)] = \
        _vsys + (_vrot_bs * cost1 + _vrad_bs * sint1) * sini1 # model LOS velocity

    _input_nogrid_m_trfit_vf[:] = _input_vf_nogrid[:] - _bsfit_vf[:]
    _input_m_bsfit_vf_grid[:] = _input_vf_tofit_2d[:] - _bsfit_vf[:]

    write_fits_images(_params, _bsfit_vf, _2dbat_run_i, 'bsfit_vf.fits')
    write_fits_images(_params, _input_m_bsfit_vf_grid, _2dbat_run_i, 'input_m_bsfit_vf_grid_x%dy%d.fits' % (_params['x_grid_tr'], _params['y_grid_tr']))
    write_fits_images(_params, _input_nogrid_m_trfit_vf, _2dbat_run_i, 'input_m_bsfit_vf_grid_x1y1.fits')

    return _bsfit_vf

#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def extract_vrot_bs_tr_rings_given_dyn_params(_input_vf_nogrid, _input_vf_tofit_2d, _tr2dfit_results, _params, fit_opt_2d, _bsfit_vf, _2dbat_run_i):

    # nuree4
    # args[0] (_dyn_params): 2dbat tr parameters (passed by dynesty)
    # args[1] (_tr_model_vf): blank array which will be filled up below (passed by user)
    # args[2] (ij_aring): (i, j coordinates of the fitting area) (passed by user)
    # args[3] (fit_opt) : 2dbat tr parameters fitting option (passed by user)
    # args[4] (_params) : 2dbat tr parameters fitting option (passed by user)
    # args[5] (_r_galaxy_plane_init_guess) : (passed by user)
    # return _tr_model_vf : vlos updated

    # _________________________________________
    # arguments
    # _________________________________________
    # 1) _dyn_params = args[0] : dynesty params passed
    #   ---------------------
    #   2dbat tr parameters
    #   ---------------------
    #   _dyn_params[0] : sigma
    #   _dyn_params[1] : xpos
    #   _dyn_params[2] : ypos
    #   _dyn_params[3] : vsys
    #   _dyn_params[4] : pa
    #   _dyn_params[5] : incl
    #   _dyn_params[6] : vrot
    #   _dyn_params[7] : vrad

    #   _dyn_params[8] : vrot_c0_bs
    #   _dyn_params[9] : vrot_c1_bs
    #   _dyn_params[10] : vrot_c2_bs

    # 
    # 2) _tr_model_vf : blank array
    # 3) i, j : coordinate x, y
    # 4) fit_opt : trfit fitting option
    # _________________________________________

    _naxis1 = _params['naxis1']
    _naxis2 = _params['naxis2']
    _input_nogrid_m_trfit_vf = np.full((_naxis2, _naxis1), fill_value=np.nan, dtype=np.float64)
    _input_m_bsfit_vf_grid = np.full((_naxis2, _naxis1), fill_value=np.nan, dtype=np.float64)

    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    deg_to_rad = np.pi / 180.

    _sigma, _sigma_e, _xpos, _xpos_e, _ypos, _ypos_e, _vsys, _vsys_e, \
        _pa, _pa_e, _incl, _incl_e, _vrot, _vrot_e, _vrad, _vrad_e, \
        n_coeffs_pa_bs, tck_pa_bs, tck_pa_bs_e, \
        n_coeffs_incl_bs, tck_incl_bs, tck_incl_bs_e, \
        n_coeffs_vrot_bs, tck_vrot_bs, tck_vrot_bs_e, \
        n_coeffs_vrad_bs, tck_vrad_bs, tck_vrad_bs_e = extract_tr2dfit_params_part_for_make_model_vf(_tr2dfit_results, _params, fit_opt_2d)
    # ---------------------------------------

    # ---------------------------------------
    # Derive the minimum r_galaxy_plane given (xpos, ypos)
    # This is used for initial values in the newton method below

    nrings_reliable = 2*_params['nrings_reliable']
    ring_w = _params['ring_w']
    _tr_rings = np.arange(0, nrings_reliable*ring_w, ring_w, dtype=np.float)
    _vrot_bs = np.zeros(nrings_reliable, dtype=np.float)
    _vrot_bs_e = np.zeros(nrings_reliable, dtype=np.float)

    # ---------------------------------------
    # ---------------------------------------
    # OUTER RINGS CONSTRAINTS
    # ---------------------------------------
    # ---------------------------------------
    _tr_rings = np.where(_tr_rings < _params['r_galaxy_plane_e'], _tr_rings, _params['r_galaxy_plane_e'])


    # -------------------------------------
    # _vrot_bs
    # derive _vrot_bs via BSpline using the generated coefficients from dynesty
    if n_coeffs_vrot_bs != 0: # not constant
        _vrot_bs[:] = BSpline(*tck_vrot_bs, extrapolate=True)(_tr_rings)
        _vrot_bs_e[:] = BSpline(*tck_vrot_bs_e, extrapolate=True)(_tr_rings)
    else:
        _vrot_bs[:] = _vrot
        _vrot_bs_e[:] = _vrot_e

    return _vrot_bs, _vrot_bs_e

#-- END OF SUB-ROUTINE____________________________________________________________#


def radius_galaxy_plane(radius, x, y, pa, incl):
    deg_to_rad = np.pi / 180.

    # Broadcast radius to the shape of x and y
    #broadcasted_radius = np.full_like(x, radius)

    xr = (-x * np.sin(pa * deg_to_rad) + y * np.cos(pa * deg_to_rad))
    yr = (-x * np.cos(pa * deg_to_rad) - y * np.sin(pa * deg_to_rad)) / np.cos(incl * deg_to_rad)

    #return broadcasted_radius - np.sqrt(xr * xr + yr * yr)
    return radius - np.sqrt(xr * xr + yr * yr)

#    make_vlos_model_vf_given_dyn_params_trfit_final_vrot(_tr2dfit_results, _input_vf_tofit_grid1, _input_vf_tofit_grid1, \
#                                                         _params, fit_opt_2d, _tr_model_vf, _2dbat_run_i)

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
#def make_vlos_model_vf_given_dyn_params_trfit_final_vrot(_tr2dfit_results, _input_vf_nogrid, _input_vf_tofit_grid1, \
#                                                         _params, fit_opt_2d, _tr_model_vf, _2dbat_run_i):
def make_vlos_model_vf_given_dyn_params_trfit_final_vrot(_input_vf_nogrid, _input_vf_tofit_2d, _wi_2d, _tr2dfit_results, \
                                                         _xpos_f_b, _ypos_f_b, _vsys_f_b, _pa_f_b, _incl_f_b, _vrot_f_b, _vrad_f_b, \
                                                         _params, fit_opt_2d, _tr_model_vf, _2dbat_run_i):
    # nuree4
    # args[0] (_dyn_params): 2dbat tr parameters (passed by dynesty)
    # args[1] (_tr_model_vf): blank array which will be filled up below (passed by user)
    # args[2] (ij_aring): (i, j coordinates of the fitting area) (passed by user)
    # args[3] (fit_opt) : 2dbat tr parameters fitting option (passed by user)
    # args[4] (_params) : 2dbat tr parameters fitting option (passed by user)
    # args[5] (_r_galaxy_plane_init_guess) : (passed by user)
    # return _tr_model_vf : vlos updated

    # _________________________________________
    # arguments
    # _________________________________________
    # 1) _dyn_params = args[0] : dynesty params passed
    #   ---------------------
    #   2dbat tr parameters
    #   ---------------------
    #   _dyn_params[0] : sigma
    #   _dyn_params[1] : xpos
    #   _dyn_params[2] : ypos
    #   _dyn_params[3] : vsys
    #   _dyn_params[4] : pa
    #   _dyn_params[5] : incl
    #   _dyn_params[6] : vrot
    #   _dyn_params[7] : vrad

    #   _dyn_params[8] : vrot_c0_bs
    #   _dyn_params[9] : vrot_c1_bs
    #   _dyn_params[10] : vrot_c2_bs

    # 
    # 2) _tr_model_vf : blank array
    # 3) i, j : coordinate x, y
    # 4) fit_opt : trfit fitting option
    # _________________________________________

    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    # DEG TO RAD
    deg_to_rad = np.pi / 180.

    _sigma, _sigma_e, _xpos, _xpos_e, _ypos, _ypos_e, _vsys, _vsys_e, \
        _pa, _pa_e, _incl, _incl_e, _vrot, _vrot_e, _vrad, _vrad_e, \
        n_coeffs_pa_bs, tck_pa_bs, tck_pa_bs_e, \
        n_coeffs_incl_bs, tck_incl_bs, tck_incl_bs_e, \
        n_coeffs_vrot_bs, tck_vrot_bs, tck_vrot_bs_e, \
        n_coeffs_vrad_bs, tck_vrad_bs, tck_vrad_bs_e = extract_tr2dfit_params_part_for_make_model_vf(_tr2dfit_results, _params, fit_opt_2d)
    # ---------------------------------------

    #npoints_in_a_ring_total_including_blanks_t, npoints_in_ring_t, _ij_aring_nogrid_available = define_tilted_ring(_input_vf_nogrid, _xpos, _ypos, _pa, _incl, 0, 5*_params['r_galaxy_plane_e'], 0, _params)
    npoints_in_a_ring_total_including_blanks_t, npoints_in_ring_t, _ij_aring_nogrid_available, _wt_2d, = define_tilted_ring(_input_vf_nogrid, _xpos, _ypos, 0, 1, 0, 5*_params['r_galaxy_plane_e'], 0, _params)

    #r_galaxy_plane, _nxy = solve_r_galaxy_plane_newton_sc(fit_opt_2d, _params, _ij_aring_nogrid_available, _xpos, _ypos, _pa, _incl, 0, \
    #                                                      n_coeffs_pa_bs, tck_pa_bs, n_coeffs_incl_bs, tck_incl_bs)

    # ---------------------------------------
    # Derive the minimum r_galaxy_plane given (xpos, ypos)
    # This is used for initial values in the newton method below
    cdelt1 = 1
    cdelt2 = 1
    del_x = (_ij_aring_nogrid_available[:, 0] - _xpos)*cdelt1 # calculate x
    del_y = (_ij_aring_nogrid_available[:, 1] - _ypos)*cdelt2 # calculate y
    #_r_galaxy_plane_init = (del_x**2 + del_y**2)**0.5 # actually minimum r_galaxy_plane given PA / INCL
    #r_galaxy_plane = (del_x**2 + del_y**2)**0.5 # actually minimum r_galaxy_plane given PA / INCL
    # ---------------------------------------

    _naxis1 = _params['naxis1']
    _naxis2 = _params['naxis2']
    nrings_reliable = _params['nrings_reliable']
    #ring_s = _params['ring_w']
    ring_s = 0
    ring_w = _params['ring_w']
    flt_epsilon = 0.00001
    _trfit_final_model_vf = np.full((_naxis2, _naxis1), fill_value=np.nan, dtype=np.float64)
    _trfit_final_model_vf_full = np.full((_naxis2, _naxis1), fill_value=np.nan, dtype=np.float64)
    _input_m_trfit_vf = np.full((_naxis2, _naxis1), fill_value=np.nan, dtype=np.float64)

    _pa_f = np.full(nrings_reliable*10, fill_value=np.nan, dtype=np.float64)
    _incl_f = np.full(nrings_reliable*10, fill_value=np.nan, dtype=np.float64)
    _vsys_f = np.full(nrings_reliable*10, fill_value=np.nan, dtype=np.float64)
    _vrot_f = np.full(nrings_reliable*10, fill_value=np.nan, dtype=np.float64)
    _vrad_f = np.full(nrings_reliable*10, fill_value=np.nan, dtype=np.float64)

    # copy the final fit results
    _pa_f[:nrings_reliable] = _pa_f_b[:nrings_reliable]
    _incl_f[:nrings_reliable] = _incl_f_b[:nrings_reliable]
    _vrad_f[:nrings_reliable] = _vrad_f_b[:nrings_reliable]
    _vsys_f[:nrings_reliable] = _vsys_f_b[:nrings_reliable]
    _vrot_f[:nrings_reliable] = _vrot_f_b[:nrings_reliable]


    # Assuming necessary functions like radius_galaxy_plane, sind, cosd, isinf, isnan are defined elsewhere
    for i in range(_naxis1):
        for j in range(_naxis2):
            x_pixel_from_xpos = i - _xpos
            y_pixel_from_ypos = j - _ypos
            r_pixel_from_centre = np.sqrt(x_pixel_from_xpos**2 + y_pixel_from_ypos**2)
            
            if r_pixel_from_centre > 10 * _params['r_galaxy_plane_e']:  # if radius is outside the outermost ring
                _trfit_final_model_vf[j, i] = np.nan # blank
            else:
                d1 = radius_galaxy_plane(0, x_pixel_from_xpos, y_pixel_from_ypos, _pa_f_b[0], _incl_f_b[0])
                d2 = radius_galaxy_plane(1 * _params['r_galaxy_plane_e'], x_pixel_from_xpos, y_pixel_from_ypos, _pa_f_b[nrings_reliable-1], _incl_f_b[nrings_reliable-1])
                
                if d1 * d2 > 0.0:
                   _trfit_final_model_vf[j, i] = np.nan # blank
                else:
                    for n in range(1, 3 * nrings_reliable):
                        if n >= nrings_reliable:
                            _pa_f[n] = _pa_f_b[nrings_reliable-1]
                            _incl_f[n] = _incl_f_b[nrings_reliable-1]
                            _vrad_f[n] = _vrad_f_b[nrings_reliable-1]
                            _vsys_f[n] = _vsys_f_b[nrings_reliable-1]
                            _vrot_f[n] = _vrot_f_b[nrings_reliable-1]

                        ri = ring_s + (n - 0) * ring_w - 0.5 * ring_w
                        ro = ring_s + (n - 0) * ring_w + 0.5 * ring_w
                        if ri < 0:
                            ri = 0
                        d2 = radius_galaxy_plane((ri + ro) / 2.0, x_pixel_from_xpos, y_pixel_from_ypos, _pa_f[n], _incl_f[n])
                        if d1 * d2 > 0.0:
                            d1 = d2
                        else:
                            break

                    ri = ring_s + (n - 1) * ring_w - 0.5 * ring_w
                    ro = ring_s + (n - 1) * ring_w + 0.5 * ring_w
                    if ri < 0:
                        ri = 0
                    ring00 = (ri + ro) / 2.0

                    ri = ring_s + (n - 0) * ring_w - 0.5 * ring_w
                    ro = ring_s + (n - 0) * ring_w + 0.5 * ring_w
                    if ri < 0:
                        ri = 0
                    ring01 = (ri + ro) / 2.0

                    # Set interpolation weight
                    rads01 = (ring01 * d1 - ring00 * d2) / (d1 - d2)
                    w1 = (ring00 - rads01) / (ring00 - ring01)
                    w2 = 1.0 - w1


                    if n < 100 * nrings_reliable:
                        posa01 = w1 * _pa_f[n] + w2 * _pa_f[n-1]
                        incl01 = w1 * _incl_f[n] + w2 * _incl_f[n-1]
                        vrot01 = w1 * _vrot_f[n] + w2 * _vrot_f[n-1]
                        vrad01 = w1 * _vrad_f[n] + w2 * _vrad_f[n-1]
                        vsys01 = w1 * _vsys_f[n] + w2 * _vsys_f[n-1]

                    if rads01 > flt_epsilon:
                        cost = (-x_pixel_from_xpos * np.sin(posa01*deg_to_rad) + y_pixel_from_ypos * np.cos(posa01*deg_to_rad)) / rads01
                        sint = (-x_pixel_from_xpos * np.cos(posa01*deg_to_rad) - y_pixel_from_ypos * np.sin(posa01*deg_to_rad)) / rads01 / np.cos(incl01*deg_to_rad)
                    else:
                        cost = 1.0
                        sint = 0.0

                    # full area
                    _trfit_final_model_vf_full[j, i] = vsys01 + (vrot01 * cost + vrad01 * sint) * np.sin(incl01*deg_to_rad)

                    # available area
                    if not math.isinf(_input_vf_nogrid[j, i]) and not math.isnan(_input_vf_nogrid[j, i]):  # no blank
                        _trfit_final_model_vf[j, i] = vsys01 + (vrot01 * cost + vrad01 * sint) * np.sin(incl01*deg_to_rad) 
                    else:
                        _trfit_final_model_vf[j, i] = np.nan
            
            _input_m_trfit_vf[j, i] = _input_vf_nogrid[j, i] - _trfit_final_model_vf[j, i]

    
    write_fits_images(_params, _input_vf_nogrid, _2dbat_run_i, 'input_vf_grid_x1y1.fits')
    write_fits_images(_params, _input_vf_tofit_2d, _2dbat_run_i, 'input_vf_grid_x%dy%d.fits' % (_params['x_grid_2d'], _params['y_grid_2d']))

    write_fits_images(_params, _trfit_final_model_vf, _2dbat_run_i, 'trfit_vf.fits')
    write_fits_images(_params, _trfit_final_model_vf_full, _2dbat_run_i, 'trfit_vf_full.fits')

    write_fits_images(_params, _input_m_trfit_vf, _2dbat_run_i, 'input_m_trfit_vf.fits')

#    _tr_model_vf[:, :] = -1E10 # initialize with a blank value
    #_tr_model_vf[ij_aring[:, 1].astype(int), ij_aring[:, 0].astype(int)] = ij_aring[:, 2]
    #_tr_model_vf[_ij_aring_nogrid_available[:, 1].astype(int), _ij_aring_nogrid_available[:, 0].astype(int)] = \
    #    _vsys + (_vrot_bs * cost1 + _vrad_bs * sint1) * sini1 # model LOS velocity


    return _trfit_final_model_vf

#-- END OF SUB-ROUTINE____________________________________________________________#


















#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def derive_vlos_model_2d_sc_checkplot(_dyn_params, _tr_model_vf, _wi_2d, ij_aring, fit_opt, _params, nrings_reliable, r_galaxy_plane):

    # nuree4
    # args[0] (_dyn_params): 2dbat tr parameters (passed by dynesty)
    # args[1] (_tr_model_vf): blank array which will be filled up below (passed by user)
    # args[2] (ij_aring): (i, j coordinates of the fitting area) (passed by user)
    # args[3] (fit_opt) : 2dbat tr parameters fitting option (passed by user)
    # args[4] (_params) : 2dbat tr parameters fitting option (passed by user)
    # args[5] (_r_galaxy_plane_init_guess) : (passed by user)
    # return _tr_model_vf : vlos updated

    # _________________________________________
    # arguments
    # _________________________________________
    # 1) _dyn_params = args[0] : dynesty params passed
    #   ---------------------
    #   2dbat tr parameters
    #   ---------------------
    #   _dyn_params[0] : sigma
    #   _dyn_params[1] : xpos
    #   _dyn_params[2] : ypos
    #   _dyn_params[3] : vsys
    #   _dyn_params[4] : pa
    #   _dyn_params[5] : incl
    #   _dyn_params[6] : vrot
    #   _dyn_params[7] : vrad

    #   _dyn_params[8] : vrot_c0_bs
    #   _dyn_params[9] : vrot_c1_bs
    #   _dyn_params[10] : vrot_c2_bs

    # 
    # 2) _tr_model_vf : blank array
    # 3) i, j : coordinate x, y
    # 4) fit_opt : trfit fitting option
    # _________________________________________

    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    deg_to_rad = np.pi / 180.


    # ------------------------------------------------------
    # VROT - bspline
    n_coeffs_vrot_bs, tck_vrot_bs = bspline_ncoeffs_tck(_params, 'vrot', _params['nrings_intp'])
    # ------------------------------------------------------
    # PA - bspline
    n_coeffs_pa_bs, tck_pa_bs = bspline_ncoeffs_tck(_params, 'pa', _params['nrings_intp'])
    # ------------------------------------------------------
    # INCL - bspline
    n_coeffs_incl_bs, tck_incl_bs = bspline_ncoeffs_tck(_params, 'incl', _params['nrings_intp'])


    n_ring_params_free = 0 # starting from sigma
    # ------------------------------------------------------
    # sigma
    if fit_opt[0] == 1: # sigma fixed:0 free:1
        _sigma = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _sigma = _params['_sigma_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # ------------------------------------------------------
    # XPOS
    if fit_opt[1] == 1: # xpos fixed:0 free:1
        _xpos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _xpos = _params['_xpos_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # YPOS
    if fit_opt[2] == 1: # ypos fixed:0 free:1
        _ypos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _ypos = _params['_ypos_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # VSYS
    if fit_opt[3] == 1: # vsys fixed:0 free:1
        _vsys = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vsys = _params['_vsys_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # PA
    if fit_opt[4] == 1 and n_coeffs_pa_bs == 0: # pa fixed:0 free:1
        _pa = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _pa = _params['_pa_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # INCL
    if fit_opt[5] == 1 and n_coeffs_incl_bs == 0 : # incl fixed:0 free:1
        _incl = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _incl = _params['_incl_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # VROT
    if fit_opt[6] == 1 and n_coeffs_vrot_bs == 0: # vrot fixed:0 free:1
        _vrot = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrot = _params['_vrot_init'] # fixed to the previous fitting result
    # ------------------------------------------------------
    # VRAD
    if fit_opt[7] == 1: # vrad fixed:0 free:1
        _vrad = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrad = _params['_vrad_init'] # fixed to the previous fitting result



    # ---------------------------------------
    # Derive the minimum r_galaxy_plane given (xpos, ypos)
    # This is used for initial values in the newton method below
    cdelt1 = 1
    cdelt2 = 1
    del_x = (ij_aring[:, 0] - _xpos)*cdelt1 # calculate x
    del_y = (ij_aring[:, 1] - _ypos)*cdelt2 # calculate y
    #_r_galaxy_plane_init = (del_x**2 + del_y**2)**0.5 # actually minimum r_galaxy_plane given PA / INCL
    #r_galaxy_plane = (del_x**2 + del_y**2)**0.5 # actually minimum r_galaxy_plane given PA / INCL
    # ---------------------------------------

    # ---------------------------------------
    # ---------------------------------------
    r_galaxy_plane, _nxy = solve_r_galaxy_plane_newton_sc(fit_opt, _params, ij_aring, _xpos, _ypos, _pa, _incl, 0)
    r_galaxy_plane = np.where(r_galaxy_plane <= 0, 0.1, r_galaxy_plane)
    #r_galaxy_plane = np.where(r_galaxy_plane > _params['r_galaxy_plane_e'], _params['r_galaxy_plane_e'], r_galaxy_plane)
    #r_galaxy_plane_within = np.where(r_galaxy_plane < _params['r_galaxy_plane_s'], _params['r_galaxy_plane_s'], r_galaxy_plane_within)

    # ---------------------------------------
    # ---------------------------------------

    # ---------------------------------------
    # PA - bspline
    if n_coeffs_pa_bs != 0: # not constant
        for _nbs in range(0, n_coeffs_pa_bs):
            tck_pa_bs[1][_nbs] = _dyn_params[n_ring_params_free + _nbs]

        _pa_bs = BSpline(*tck_pa_bs, extrapolate=True)(r_galaxy_plane) # <-- tck_pa_bs[1] coeffs (0 ~ 360)
        #_pa_bs = np.where(r_galaxy_plane > _params['r_galaxy_plane_e'], BSpline(*tck_pa_bs, extrapolate=True)(_params['r_galaxy_plane_e']), _pa_bs) 

        #_pa_bs = _pa_bs*(_params['pa_bs_max'] - _params['pa_bs_min']) + _params['pa_bs_min'] # going back to PA unit (0~360)
        #print("pa", _pa_bs)
    else:
        _pa_bs = _pa

    #_pa_bs = np.where(_pa_bs < 0, _pa_bs + 360, _pa_bs)
    #_pa_bs = np.where(_pa_bs > 360, _pa_bs - 360, _pa_bs)

    # ---------------
#    print(BSpline(*tck_vrot_bs_init_from_trfit, extrapolate=True)(400)       )
#    _ring_t1 = range(0, 100, 5)
#    print(r_galaxy_plane.max())
#    temp1 = BSpline(*tck_pa_bs, extrapolate=True)(r_galaxy_plane)       
#    temp1 = np.where(temp1 < 0, temp1 + 360, temp1)
#    temp1 = np.where(temp1 > 360, temp1 - 360, temp1)
#    fig = plt.figure()
#    ax = fig.add_subplot()
#    plt.xlim(0, 1000)
#    plt.ylim(-100, 360)
#    plt.plot(r_galaxy_plane, temp1, c='r')	
#    plt.scatter(r_galaxy_plane, temp1, c='g')	
#    plt.show()
#    sys.exit()
    # ---------------

    # ---------------------------------------
    # INCL - bspline
    if n_coeffs_incl_bs != 0: # not constant
        for _nbs in range(0, n_coeffs_incl_bs):
            tck_incl_bs[1][_nbs] = _dyn_params[n_ring_params_free + n_coeffs_pa_bs + _nbs]

        _incl_bs = BSpline(*tck_incl_bs, extrapolate=True)(r_galaxy_plane) # <-- tck_incl_bs[1] coeffs (0 ~ 90)
        #_incl_bs = np.where(r_galaxy_plane > _params['r_galaxy_plane_e'], BSpline(*tck_incl_bs, extrapolate=True)(_params['r_galaxy_plane_e']), _incl_bs) 
        #_incl_bs = _incl_bs*(_params['incl_bs_max'] - _params['incl_bs_min']) + _params['incl_bs_min'] # going back to INCL unit (0~360)
        #print("incl", _incl_bs)
    else:
        _incl_bs = _incl

    # -------------------------------------
    # Derive cosp1, sinp1, cosi1, sini1, x_galaxy_plane, y_galaxy_plane, cost1, sint1 usint the derived _pa_bs and _incl
    # These are used for computing v_LOS below
    cosp1 = np.cos(deg_to_rad*_pa_bs)
    sinp1 = np.sin(deg_to_rad*_pa_bs)
    cosi1 = np.cos(deg_to_rad*_incl_bs) #cosine
    sini1 = np.sin(deg_to_rad*_incl_bs) # sine

    x_galaxy_plane = (-del_x * sinp1 + del_y * cosp1) # x in plane of galaxy
    y_galaxy_plane = (-del_x * cosp1 - del_y * sinp1) / cosi1 # y in plane of galaxy

    cost1 = x_galaxy_plane / r_galaxy_plane # cosine of angle in plane of galaxy 
    sint1 = y_galaxy_plane / r_galaxy_plane # sine of angle in plane of galaxy

    # --------------------------------------
    # calculate cosine-weights
    #_theta_free = _params['free_angle']
    #_ri = _params['ri']
    #_ro = _params['ro']
    #_wpow = _params['weight_power']

    #_theta = np.where(r_galaxy_plane < 0.1, 0.0, np.arctan2(y_galaxy_plane, x_galaxy_plane) / deg_to_rad)
    #costh = np.abs(np.cos(_theta*deg_to_rad))
    #sin_theta_free = np.abs(np.sin(_theta_free*deg_to_rad))
    #wi = np.where((r_galaxy_plane > _ri) & (r_galaxy_plane < _ro) & (costh > sin_theta_free), costh**_wpow, 0.0) # weights of the points

#    # --------------------------------------
#    # calculate ellipse-perimeter-weights
#    #_theta_free = _params['free_angle']
#    #_ri = _params['r_galaxy_plane_s']
#    _ri = 0
#    _ro = _params['ro']
#    #_wpow = _params['weight_power']
#
#    _a = r_galaxy_plane # along major axis
#    _incl_bs_limit1 = np.where(_incl_bs > 89, 89, _incl_bs)
#    _incl_bs_limit2 = np.where(_incl_bs_limit1 < 1, 1, _incl_bs_limit1)
#    _b = _a * np.cos(_incl_bs_limit2*deg_to_rad) # along minor axis
#    _l = np.pi * (3.0*(_a + _b) - ((3*_a + _b) * (_a + 3*_b))**0.5) # ellipse perimeter approximation
#
#    _rmin_ = 0
#    _rmax_ = r_galaxy_plane.max()
#    npoints_in_1pixelwidth, bin_edges = np.histogram(r_galaxy_plane, bins=int(_rmax_)-int(_rmin_)+1, range=(int(_rmin_), int(_rmax_)+1))
#
#    #print(_rmin_, _rmax_, npoints_in_1pixelwidth.shape)
#    #npoints_for_each_r_galaxy_plane = r_galaxy_plane
#    #print(npoints_in_1pixelwidth.shape, npoints_in_1pixelwidth[25])
#    #print(r_galaxy_plane.astype(int).min())
#    npoints_for_each_r_galaxy_plane = npoints_in_1pixelwidth[r_galaxy_plane.astype(int)]  
#    #print(npoints_for_each_r_galaxy_plane)
#    #npoints_for_each_r_galaxy_plane = np.where(npoints_for_each_r_galaxy_plane == 0, 999, npoints_for_each_r_galaxy_plane)
#
#    #filling_factor = npoints_for_each_r_galaxy_plane / _l
#
#    #print("seheon", _incl_bs, npoints_for_each_r_galaxy_plane)
#    #print("seheon", _incl_bs, npoints_in_1pixelwidth)
#    #sys.exit()
#
#    #print(npoints_in_1pixelwidth[int(_rmax_)-2])
#    #print(npoints_for_each_r_galaxy_plane)
#
#    #print(r_galaxy_plane_within.astype(int))
#
#    #print(npoints_in_1pixelwidth[r_galaxy_plane_within.astype(int)])
#    #print(npoints_in_1pixelwidth.sum())
#    #wi = np.where( r_galaxy_plane)
#    #print(r_galaxy_plane_within.max(), r_galaxy_plane_within.min())
#
#    #_ai = 4 # along major axis
#    #_bi = _ai * np.cos(_incl_bs*deg_to_rad) # along minor axis
#    #_li = np.pi * (3.0*(_ai + _bi) - ((3*_ai + _bi) * (_ai + 3*_bi))**0.5) # ellipse perimeter approximation
#
#    #print(npoints_for_each_r_galaxy_plane)
#    #wi = np.where((r_galaxy_plane > _ri), 1. / _l, 0.0) # weights of the points

# -------------------------------
# -------------------------------
#    yi = a + b*xi + e
#    yi = a + b*xi + w*e
#    --> e = (yi - (a + b*xi) ) / w <-- if es are independant normal random variables with mean 0 and unknown variance sigma**2
#    --> The likelihood function  -(2/n)*ln(2pi*sigma**2)  - (1/[2sigma**2]) * Sigma(1, n) e**2
#    --> The likelihood function  -(2/n)*ln(2pi*sigma**2)  - (1/[2sigma**2]) * Sigma(1, n) (1/w**2) * (yi -  (a + b*xi) )**2
#    --> where W = (1 / w**2) 
#    --> If we choose a weight of w with (1 / length of arc), the actual W in the log-likelihood function is (1 / length of arc)**2

#    --> W affects the variance of e, i.e., sigma parameter.
#    --> So adjust the prior range of sigma or normalise W (0 ~ 1) if you'd like to use the the prior range of sigma with no weightes
# -------------------------------
# -------------------------------

#    #wi = np.where((r_galaxy_plane > _ri) & (npoints_for_each_r_galaxy_plane > 1), (1. / npoints_for_each_r_galaxy_plane)**2, 0.000000000001) # weights of the points
#    wi = np.where((r_galaxy_plane > _ri) & (npoints_for_each_r_galaxy_plane > 0), 1, 0.000000000001) # weights of the points
#    #wi = np.where(npoints_for_each_r_galaxy_plane == 999, 0.0, wi) # weights of the points
#    #wi = np.where((r_galaxy_plane < 26), wi, 0.0) # weights of the points
#    #wi = np.where(filling_factor > 0.9, wi, 0.0)
#    #wi = np.where(r_galaxy_plane > 20, wi, 0.0)
#    #print(wi)
#    #wi = np.where((r_galaxy_plane < _ri), 1. / (_li / 700.), wi) # weights of the points
#    #wi = 1. / (_l / 700.)
#    wi = wi / np.max(wi)
#
#    #_theta = np.where(r_galaxy_plane < 0.1, 0.0, np.arctan2(y_galaxy_plane, x_galaxy_plane) / deg_to_rad)
#    #costh = np.abs(np.cos(_theta*deg_to_rad))
#    #sin_theta_free = np.abs(np.sin(_theta_free*deg_to_rad))
#    #wi = np.where((r_galaxy_plane > _ri) & (r_galaxy_plane < _ro) & (costh > sin_theta_free), costh**_wpow, 0.0) # weights of the points

    # -------------------------------------
    # _vrot_bs
    # derive _vrot_bs via BSpline using the generated coefficients from dynesty
    if n_coeffs_vrot_bs != 0: # not constant
        for _nbs in range(0, n_coeffs_vrot_bs):
            tck_vrot_bs[1][_nbs] = _dyn_params[n_ring_params_free + n_coeffs_pa_bs + n_coeffs_incl_bs + _nbs]
        _vrot_bs = BSpline(*tck_vrot_bs, extrapolate=True)(r_galaxy_plane)
        #_vrot_bs = np.where(r_galaxy_plane > _params['r_galaxy_plane_e'], BSpline(*tck_vrot_bs, extrapolate=True)(_params['r_galaxy_plane_e']), _vrot_bs) 
        #_vrot_bs = np.where(r_galaxy_plane > _params['r_galaxy_plane_e'], _vrot_bs_within*(r_galaxy_plane - r_galaxy_plane_within), _vrot_bs_within)
    else:
        _vrot_bs = _vrot
    # ---------------
#    print(BSpline(*tck_vrot_bs_init_from_trfit, extrapolate=True)(400)       )
#    _ring_t1 = range(0, 100, 5)
#    print(r_galaxy_plane.max())
#    temp1 = BSpline(*tck_vrot_bs, extrapolate=True)(r_galaxy_plane)       
#    fig = plt.figure()
#    ax = fig.add_subplot()
#    plt.xlim(0, 100)
#    plt.ylim(0, 1000)
#    plt.plot(r_galaxy_plane, temp1, c='r')	
#    plt.scatter(r_galaxy_plane, temp1, c='g')	
#    plt.show()
#    sys.exit()
    # ---------------

    #ij_aring[:,2] = _vsys + (_vrot_bs * cost1 + _vrad * sint1) * sini1 # model LOS velocity

    _tr_model_vf[:, :] = np.nan # initialize with a blank value
    #_tr_model_vf[ij_aring[:, 1].astype(int), ij_aring[:, 0].astype(int)] = ij_aring[:, 2]
    _tr_model_vf[ij_aring[:, 1].astype(int), ij_aring[:, 0].astype(int)] = \
        _vsys + (_vrot_bs * cost1 + _vrad * sint1) * sini1 # model LOS velocity

    return r_galaxy_plane, _vrot_bs
#-- END OF SUB-ROUTINE_____________________________5______________________________#








#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def derive_vlos_model_2d_confirmed(_dyn_params, _tr_model_vf, ij_aring, fit_opt, _params, nrings_reliable):
    # nuree4
    # args[0] (_dyn_params): 2dbat tr parameters (passed by dynesty)
    # args[1] (_tr_model_vf): blank array which will be filled up below (passed by user)
    # args[2] (ij_aring): (i, j coordinates of the fitting area) (passed by user)
    # args[3] (fit_opt) : 2dbat tr parameters fitting option (passed by user)
    # args[4] (_params) : 2dbat tr parameters fitting option (passed by user)
    # args[5] (_r_galaxy_plane_init_guess) : (passed by user)
    # return _tr_model_vf : vlos updated

    # _________________________________________
    # arguments
    # _________________________________________
    # 1) _dyn_params = args[0] : dynesty params passed
    #   ---------------------
    #   2dbat tr parameters
    #   ---------------------
    #   _dyn_params[0] : sigma
    #   _dyn_params[1] : xpos
    #   _dyn_params[2] : ypos
    #   _dyn_params[3] : vsys
    #   _dyn_params[4] : pa
    #   _dyn_params[5] : incl
    #   _dyn_params[6] : vrot
    #   _dyn_params[7] : vrad

    #   _dyn_params[8] : vrot_c0_bs
    #   _dyn_params[9] : vrot_c1_bs
    #   _dyn_params[10] : vrot_c2_bs

    # 
    # 2) _tr_model_vf : blank array
    # 3) i, j : coordinate x, y
    # 4) fit_opt : trfit fitting option
    # _________________________________________


    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    deg_to_rad = np.pi / 180.

    n_ring_params_free = 0 # starting from sigma
    # ------------------------------------------------------
    # sigma
    if fit_opt[0] == 1: # sigma fixed:0 free:1
        _sigma = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _sigma = 10 # fixed to the previous fitting result
    # ------------------------------------------------------
    # ------------------------------------------------------
    # XPOS
    if fit_opt[1] == 1: # xpos fixed:0 free:1
        _xpos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _xpos = 20 # fixed to the previous fitting result
    # ------------------------------------------------------
    # YPOS
    if fit_opt[2] == 1: # ypos fixed:0 free:1
        _ypos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _ypos = 20 # fixed to the previous fitting result
    # ------------------------------------------------------
    # VSYS
    if fit_opt[3] == 1: # vsys fixed:0 free:1
        _vsys = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vsys = 1040 # fixed to the previous fitting result
    # ------------------------------------------------------
    # PA
    if fit_opt[4] == 1: # pa fixed:0 free:1
        _pa = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _pa = 45 # fixed to the previous fitting result
    # ------------------------------------------------------
    # INCL
    if fit_opt[5] == 1: # incl fixed:0 free:1
        _incl = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _incl = 60 # fixed to the previous fitting result
    # ------------------------------------------------------
    # VROT
    if fit_opt[6] == 1: # vrot fixed:0 free:1
        _vrot = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrot = 50 # fixed to the previous fitting result
    # ------------------------------------------------------
    # VRAD
    if fit_opt[7] == 1: # vrad fixed:0 free:1
        _vrad = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrad = 0 # fixed to the previous fitting result


    # ------------------------------------------------------
    # VROT - bspline
    n_coeffs_vrot_bs, tck_vrot_bs = bspline_ncoeffs_tck(_params, 'vrot', _params['nrings_intp'])
    # ------------------------------------------------------
    # PA - bspline
    n_coeffs_pa_bs, tck_pa_bs = bspline_ncoeffs_tck(_params, 'pa', _params['nrings_intp'])


    # ---------------------------------------
    # Derive the minimum r_galaxy_plane given (xpos, ypos)
    # This is used for initial values in the newton method below
    cdelt1 = 1
    cdelt2 = 1
    del_x = (ij_aring[:, 0] - _xpos)*cdelt1 # calculate x
    del_y = (ij_aring[:, 1] - _ypos)*cdelt2 # calculate y
    #_r_galaxy_plane_init = (del_x**2 + del_y**2)**0.5 # actually minimum r_galaxy_plane given PA / INCL
    # ---------------------------------------



    # ---------------------------------------
    # ---------------------------------------
    # ray.put : speed up
    fx_r_galaxy_plane_id = ray.put(fx_r_galaxy_plane)
    _dyn_params_id = ray.put(_dyn_params)
    fit_opt_id = ray.put(fit_opt)
    _params_id = ray.put(_params)
    ij_aring_id = ray.put(ij_aring)
    nrings_reliable_id = ray.put(nrings_reliable)
    _xpos_id = ray.put(_xpos)
    _ypos_id = ray.put(_ypos)
    #r_galaxy_plane = ray.get([solve_r_galaxy_plane_newton.remote(fx_r_galaxy_plane, _dyn_params_id, fit_opt_id, _params_id, ij_aring_id, nrings_reliable_id, _xpos_id, _ypos_id, _nxy) for _nxy in range(0, ij_aring.shape[0])])


    #r_galaxy_plane = del_x
    #results_ids = [ray.remote(optimize.newton(fx_r_galaxy_plane, _r_galaxy_plane_init, args=(_dyn_params, fit_opt, _params, ij_aring[_nxy, 0], ij_aring[_nxy, 1], nrings_reliable), tol=1E-1, maxiter=10000, disp=True)) for _nxy in range(0, ij_aring.shape[0])]
    #results_ids = [solve_r_galaxy_plane_newton.remote(fx_r_galaxy_plane, _dyn_params_id, fit_opt_id, _params_id, ij_aring_id, nrings_reliable_id, _xpos_id, _ypos_id, _nxy) for _nxy in range(0, ij_aring.shape[0])]
    #while len(results_ids):
    #    done_ids, results_ids = ray.wait(results_ids)
    #    if done_ids:
    #        # _xs, _xe, _ys, _ye : variables inside the loop
    #        #_r_galaxy_plane_i = ray.get(done_ids)[0]
    #        #_nxy_record = ray.get(done_ids)[1]
    #        _r_galaxy_plane_i =  ray.get(done_ids[0])[0]
    #        _nxy_record = ray.get(done_ids[0])[1]
    #        r_galaxy_plane[_nxy_record] = _r_galaxy_plane_i

#    results_compile = ray.get(results_ids)
#    print(results_compile)
#    ray.shutdown()A
    #print(results_ids[0])
    # ---------------------------------------
    # ---------------------------------------


    # ---------------------------------------
    # Derive r_galaxy_plane from fx_r_galaxy_plane, a non-linear equation using newton method
    #_r_galaxy_plane_init = (del_x**2 + del_y**2)**0.5 # actually minimum r_galaxy_plane given PA / INCL

    #roots = optimize.newton(fx_r_galaxy_plane, np.arange(0, _r_galaxy_plane_init, (_r_galaxy_plane_init/4.)), args=(_dyn_params, fit_opt, _params, ij_aring[:, 0], ij_aring[:, 1], nrings_reliable), rtol=1E-1, tol=0.5E-1, maxiter=10000, disp=True)
    #r_galaxy_plane = optimize.newton(fx_r_galaxy_plane, x0=0.1*_r_galaxy_plane_init, x1=5*_r_galaxy_plane_init, args=(_dyn_params, fit_opt, _params, ij_aring[:, 0], ij_aring[:, 1], nrings_reliable), tol=1E-2, maxiter=1000, disp=True)
    #r_galaxy_plane = optimize.newton(fx_r_galaxy_plane, x0=0.5*_r_galaxy_plane_init, x1=5*_r_galaxy_plane_init, args=(_dyn_params, fit_opt, _params, ij_aring[:, 0], ij_aring[:, 1], nrings_reliable), tol=1E-1, maxiter=10000, disp=True)
    #roots = optimize.newton(fx_r_galaxy_plane, _r_galaxy_plane_init, args=(_dyn_params, fit_opt, _params, ij_aring[:, 0], ij_aring[:, 1], nrings_reliable), rtol=1E0, tol=0.5E0, maxiter=10000, disp=True)
    #root, count = Counter(roots).most_common(1)[0]
    #r_galaxy_plane = root


    #r_galaxy_plane = optimize.newton(fx_r_galaxy_plane, _r_galaxy_plane_init, args=(_dyn_params, fit_opt, _params, ij_aring[:, 0], ij_aring[:, 1], nrings_reliable), tol=1E-1, maxiter=10000, disp=True) 

    #roots = optimize.newton(fx_r_galaxy_plane, np.arange(0, 2*_r_galaxy_plane_init, (_r_galaxy_plane_init)), args=(_dyn_params, fit_opt, _params, ij_aring[:, 0], ij_aring[:, 1], nrings_reliable), rtol=1E-1, tol=0.5E-1, maxiter=10000, disp=True)
    #root, count = Counter(roots).most_common(1)[0]
    #r_galaxy_plane = root
    #print(root, count)

    #print(r_galaxy_plane[0]- r_galaxy_plane_newton[0], r_galaxy_plane[300]- r_galaxy_plane_newton[300], r_galaxy_plane[699]-r_galaxy_plane_newton[699])
    #r_galaxy_plane = (del_x**2 + del_y**2)**0.5 # actually minimum r_galaxy_plane given PA / INCL
    #print(_r_galaxy_plane_init.shape)
    # ---------------------------------------

    # ---------------------------------------
    # PA - bspline
    for _nbs in range(0, n_coeffs_pa_bs):
        tck_pa_bs[1][_nbs] = _dyn_params[n_ring_params_free + n_coeffs_vrot_bs + _nbs]

    _pa_bs = BSpline(*tck_pa_bs, extrapolate=True)(r_galaxy_plane) # normalised PA_bs [0 ~ 1] <-- tck_pa_bs[1] coeffs (0 ~ 1)
    #_pa_bs = _pa_bs*(_params['pa_bs_max'] - _params['pa_bs_min']) + _params['pa_bs_min'] # going back to PA unit (0~360)

    # -------------------------------------
    # Derive cosp1, sinp1, cosi1, sini1, x_galaxy_plane, y_galaxy_plane, cost1, sint1 usint the derived _pa_bs and _incl
    # These are used for computing v_LOS below
    cosp1 = np.cos(deg_to_rad*_pa_bs)
    sinp1 = np.sin(deg_to_rad*_pa_bs)
    cosi1 = np.cos(deg_to_rad*_incl) #cosine
    sini1 = np.sin(deg_to_rad*_incl) # sine

    x_galaxy_plane = (-del_x * sinp1 + del_y * cosp1) # x in plane of galaxy
    y_galaxy_plane = (-del_x * cosp1 - del_y * sinp1) / cosi1 # y in plane of galaxy

    cost1 = x_galaxy_plane / r_galaxy_plane # cosine of angle in plane of galaxy 
    sint1 = y_galaxy_plane / r_galaxy_plane # sine of angle in plane of galaxy


    # -------------------------------------
    # _vrot_bs
    # derive _vrot_bs via BSpline using the generated coefficients from dynesty
    for _nbs in range(0, n_coeffs_vrot_bs):
        tck_vrot_bs[1][_nbs] = _dyn_params[n_ring_params_free+_nbs]
    _vrot_bs = BSpline(*tck_vrot_bs, extrapolate=True)(r_galaxy_plane)

    ij_aring[:,2] = _vsys + (_vrot_bs * cost1 + _vrad * sint1) * sini1 # model LOS velocity
    #ij_aring[:,2] = _vsys + (_vrot * cost1 + _vrad * sint1) * sini1 # model LOS velocity

    _tr_model_vf[:, :] = np.nan # initialize with a blank value
    _tr_model_vf[ij_aring[:, 1].astype(int), ij_aring[:, 0].astype(int)] = ij_aring[:, 2]

    return _tr_model_vf

#-- END OF SUB-ROUTINE____________________________________________________________#

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def derive_vlos_model_2d_single_core_old(_dyn_params, _tr_model_vf, ij_aring, fit_opt, _params, nrings_reliable):
    # nuree4
    # args[0] (_dyn_params): 2dbat tr parameters (passed by dynesty)
    # args[1] (_tr_model_vf): blank array which will be filled up below (passed by user)
    # args[2] (ij_aring): (i, j coordinates of the fitting area) (passed by user)
    # args[3] (fit_opt) : 2dbat tr parameters fitting option (passed by user)
    # args[4] (_params) : 2dbat tr parameters fitting option (passed by user)
    # args[5] (_r_galaxy_plane_init_guess) : (passed by user)
    # return _tr_model_vf : vlos updated

    # _________________________________________
    # arguments
    # _________________________________________
    # 1) _dyn_params = args[0] : dynesty params passed
    #   ---------------------
    #   2dbat tr parameters
    #   ---------------------
    #   _dyn_params[0] : sigma
    #   _dyn_params[1] : xpos
    #   _dyn_params[2] : ypos
    #   _dyn_params[3] : vsys
    #   _dyn_params[4] : pa
    #   _dyn_params[5] : incl
    #   _dyn_params[6] : vrot
    #   _dyn_params[7] : vrad

    #   _dyn_params[8] : vrot_c0_bs
    #   _dyn_params[9] : vrot_c1_bs
    #   _dyn_params[10] : vrot_c2_bs

    # 
    # 2) _tr_model_vf : blank array
    # 3) i, j : coordinate x, y
    # 4) fit_opt : trfit fitting option
    # _________________________________________

    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    deg_to_rad = np.pi / 180.

    n_ring_params_free = 0 # starting from sigma
    # ------------------------------------------------------
    # sigma
    if fit_opt[0] == 1: # sigma fixed:0 free:1
        _sigma = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _sigma = 10 # fixed to the previous fitting result
    # ------------------------------------------------------
    # ------------------------------------------------------
    # XPOS
    if fit_opt[1] == 1: # xpos fixed:0 free:1
        _xpos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _xpos = 20 # fixed to the previous fitting result
    # ------------------------------------------------------
    # YPOS
    if fit_opt[2] == 1: # ypos fixed:0 free:1
        _ypos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _ypos = 20 # fixed to the previous fitting result
    # ------------------------------------------------------
    # VSYS
    if fit_opt[3] == 1: # vsys fixed:0 free:1
        _vsys = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vsys = 1040 # fixed to the previous fitting result
    # ------------------------------------------------------
    # PA
    if fit_opt[4] == 1: # pa fixed:0 free:1
        _pa = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _pa = 45 # fixed to the previous fitting result
    
    # ------------------------------------------------------
    # INCL
    if fit_opt[5] == 1: # incl fixed:0 free:1
        _incl = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _incl = 60 # fixed to the previous fitting result
    # ------------------------------------------------------
    # VROT
    if fit_opt[6] == 1: # vrot fixed:0 free:1
        _vrot = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrot = 50 # fixed to the previous fitting result
    # ------------------------------------------------------
    # VRAD
    if fit_opt[7] == 1: # vrad fixed:0 free:1
        _vrad = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrad = 0 # fixed to the previous fitting result


    # ------------------------------------------------------
    # VROT - bspline
    n_coeffs_vrot_bs, tck_vrot_bs = bspline_ncoeffs_tck(_params, 'vrot', _params['nrings_intp'])
    # ------------------------------------------------------
    # PA - bspline
    n_coeffs_pa_bs, tck_pa_bs = bspline_ncoeffs_tck(_params, 'pa', _params['nrings_intp'])


    # ---------------------------------------
    # Derive the minimum r_galaxy_plane given (xpos, ypos)
    # This is used for initial values in the newton method below
    cdelt1 = 1
    cdelt2 = 1
    del_x = (ij_aring[:, 0] - _xpos)*cdelt1 # calculate x
    del_y = (ij_aring[:, 1] - _ypos)*cdelt2 # calculate y
    #_r_galaxy_plane_init = (del_x**2 + del_y**2)**0.5 # actually minimum r_galaxy_plane given PA / INCL
    # ---------------------------------------

    # ---------------------------------------
    # Derive r_galaxy_plane from fx_r_galaxy_plane, a non-linear equation using newton method
    _r_galaxy_plane_init = (del_x**2 + del_y**2)**0.5 # actually minimum r_galaxy_plane given PA / INCL
    r_galaxy_plane = optimize.newton(fx_r_galaxy_plane, _r_galaxy_plane_init, args=(_dyn_params, fit_opt, _params, ij_aring[:, 0], ij_aring[:, 1], nrings_reliable), tol=1E-1, maxiter=10000, disp=True) 
    #print(_r_galaxy_plane_init.shape)
    # ---------------------------------------

    # ---------------------------------------
    # PA - bspline
    for _nbs in range(0, n_coeffs_pa_bs):
        tck_pa_bs[1][_nbs] = _dyn_params[n_ring_params_free + n_coeffs_vrot_bs + _nbs]
    _pa_bs = BSpline(*tck_pa_bs, extrapolate=True)(r_galaxy_plane)

    _pa_bs = BSpline(*tck_pa_bs, extrapolate=True)(r_galaxy_plane) # normalised PA_bs [0 ~ 1] <-- tck_pa_bs[1] coeffs (0 ~ 1)
    #_pa_bs = _pa_bs*(_params['pa_bs_max'] - _params['pa_bs_min']) + _params['pa_bs_min'] # going back to PA unit (0~360)

    # -------------------------------------
    # Derive cosp1, sinp1, cosi1, sini1, x_galaxy_plane, y_galaxy_plane, cost1, sint1 usint the derived _pa_bs and _incl
    # These are used for computing v_LOS below
    cosp1 = np.cos(deg_to_rad*_pa_bs)
    sinp1 = np.sin(deg_to_rad*_pa_bs)
    cosi1 = np.cos(deg_to_rad*_incl) #cosine
    sini1 = np.sin(deg_to_rad*_incl) # sine

    x_galaxy_plane = (-del_x * sinp1 + del_y * cosp1) # x in plane of galaxy
    y_galaxy_plane = (-del_x * cosp1 - del_y * sinp1) / cosi1 # y in plane of galaxy

    cost1 = x_galaxy_plane / r_galaxy_plane # cosine of angle in plane of galaxy 
    sint1 = y_galaxy_plane / r_galaxy_plane # sine of angle in plane of galaxy


    # -------------------------------------
    # _vrot_bs
    # derive _vrot_bs via BSpline using the generated coefficients from dynesty
    for _nbs in range(0, n_coeffs_vrot_bs):
        tck_vrot_bs[1][_nbs] = _dyn_params[n_ring_params_free+_nbs]
    _vrot_bs = BSpline(*tck_vrot_bs, extrapolate=True)(r_galaxy_plane)

    ij_aring[:,2] = _vsys + (_vrot_bs * cost1 + _vrad * sint1) * sini1 # model LOS velocity
    #ij_aring[:,2] = _vsys + (_vrot * cost1 + _vrad * sint1) * sini1 # model LOS velocity

    _tr_model_vf[:, :] = np.nan # initialize with a blank value
    _tr_model_vf[ij_aring[:, 1].astype(int), ij_aring[:, 0].astype(int)] = ij_aring[:, 2]

    return _tr_model_vf

#-- END OF SUB-ROUTINE____________________________________________________________#

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def derive_vlos_model_2d_test(_dyn_params, _tr_model_vf, ij_aring, fit_opt, _params, nrings_reliable, r_galaxy_plane):
    # nuree4
    # args[0] (_dyn_params): 2dbat tr parameters (passed by dynesty)
    # args[1] (_tr_model_vf): blank array which will be filled up below (passed by user)
    # args[2] (ij_aring): (i, j coordinates of the fitting area) (passed by user)
    # args[3] (fit_opt) : 2dbat tr parameters fitting option (passed by user)
    # args[4] (_params) : 2dbat tr parameters fitting option (passed by user)
    # args[5] (_r_galaxy_plane_init_guess) : (passed by user)
    # return _tr_model_vf : vlos updated

    # _________________________________________
    # arguments
    # _________________________________________
    # 1) _dyn_params = args[0] : dynesty params passed
    #   ---------------------
    #   2dbat tr parameters
    #   ---------------------
    #   _dyn_params[0] : sigma
    #   _dyn_params[1] : xpos
    #   _dyn_params[2] : ypos
    #   _dyn_params[3] : vsys
    #   _dyn_params[4] : pa
    #   _dyn_params[5] : incl
    #   _dyn_params[6] : vrot
    #   _dyn_params[7] : vrad

    #   _dyn_params[8] : vrot_c0_bs
    #   _dyn_params[9] : vrot_c1_bs
    #   _dyn_params[10] : vrot_c2_bs

    # 
    # 2) _tr_model_vf : blank array
    # 3) i, j : coordinate x, y
    # 4) fit_opt : trfit fitting option
    # _________________________________________

    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    deg_to_rad = np.pi / 180.

    n_ring_params_free = 0 # starting from sigma
    # ------------------------------------------------------
    # sigma
    if fit_opt[0] == 1: # sigma fixed:0 free:1
        _sigma = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _sigma = 10 # fixed to the previous fitting result
    # ------------------------------------------------------
    # ------------------------------------------------------
    # XPOS
    if fit_opt[1] == 1: # xpos fixed:0 free:1
        _xpos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _xpos = 20 # fixed to the previous fitting result
    # ------------------------------------------------------
    # YPOS
    if fit_opt[2] == 1: # ypos fixed:0 free:1
        _ypos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _ypos = 20 # fixed to the previous fitting result
    # ------------------------------------------------------
    # VSYS
    if fit_opt[3] == 1: # vsys fixed:0 free:1
        _vsys = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vsys = 1040 # fixed to the previous fitting result
    # ------------------------------------------------------
    # PA
    if fit_opt[4] == 1: # pa fixed:0 free:1
        _pa = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _pa = 45 # fixed to the previous fitting result
    # ------------------------------------------------------
    # INCL
    if fit_opt[5] == 1: # incl fixed:0 free:1
        _incl = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _incl = 60 # fixed to the previous fitting result
    # ------------------------------------------------------
    # VROT
    if fit_opt[6] == 1: # vrot fixed:0 free:1
        _vrot = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrot = 50 # fixed to the previous fitting result
    # ------------------------------------------------------
    # VRAD
    if fit_opt[7] == 1: # vrad fixed:0 free:1
        _vrad = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrad = 0 # fixed to the previous fitting result


    # ------------------------------------------------------
    # PA - bspline
    n_coeffs_pa_bs, tck_pa_bs = bspline_ncoeffs_tck(_params, 'pa', _params['nrings_intp'])
    
    # ------------------------------------------------------
    # VROT - bspline
    n_coeffs_vrot_bs, tck_vrot_bs = bspline_ncoeffs_tck(_params, 'vrot', _params['nrings_intp'])


    # ---------------------------------------
    # Derive the minimum r_galaxy_plane given (xpos, ypos)
    # This is used for initial values in the newton method below
    cdelt1 = 1
    cdelt2 = 1
    del_x = (ij_aring[:, 0] - _xpos)*cdelt1 # calculate x
    del_y = (ij_aring[:, 1] - _ypos)*cdelt2 # calculate y
    _r_galaxy_plane_init = (del_x**2 + del_y**2)**0.5 # actually minimum r_galaxy_plane given PA / INCL
    # ---------------------------------------

    # ---------------------------------------
    # Derive r_galaxy_plane from fx_r_galaxy_plane, a non-linear equation using newton method
    r_galaxy_plane = (del_x**2 + del_y**2)**0.5 # actually minimum r_galaxy_plane given PA / INCL
    #r_galaxy_plane = optimize.newton(fx_r_galaxy_plane, _r_galaxy_plane_init, args=(_dyn_params, fit_opt, _params, ij_aring[:, 0], ij_aring[:, 1], nrings_reliable), tol=1E-1, maxiter=10000, disp=True) 
    # ---------------------------------------

    # ---------------------------------------
    # PA - bspline
    for _nbs in range(0, n_coeffs_pa_bs):
        tck_pa_bs[1][_nbs] = _dyn_params[n_ring_params_free + n_coeffs_vrot_bs + _nbs]

    _pa_bs = BSpline(*tck_pa_bs, extrapolate=True)(r_galaxy_plane) # normalised PA_bs [0 ~ 1] <-- tck_pa_bs[1] coeffs (0 ~ 1)
    #_pa_bs = _pa_bs*(_params['pa_bs_max'] - _params['pa_bs_min']) + _params['pa_bs_min'] # going back to PA unit (0~360)

    # -------------------------------------
    # Derive cosp1, sinp1, cosi1, sini1, x_galaxy_plane, y_galaxy_plane, cost1, sint1 usint the derived _pa_bs and _incl
    # These are used for computing v_LOS below
    cosp1 = np.cos(deg_to_rad*_pa_bs)
    sinp1 = np.sin(deg_to_rad*_pa_bs)
    cosi1 = np.cos(deg_to_rad*_incl) #cosine
    sini1 = np.sin(deg_to_rad*_incl) # sine

    x_galaxy_plane = (-del_x * sinp1 + del_y * cosp1) # x in plane of galaxy
    y_galaxy_plane = (-del_x * cosp1 - del_y * sinp1) / cosi1 # y in plane of galaxy

    cost1 = x_galaxy_plane / r_galaxy_plane # cosine of angle in plane of galaxy 
    sint1 = y_galaxy_plane / r_galaxy_plane # sine of angle in plane of galaxy


    # -------------------------------------
    # _vrot_bs
    # derive _vrot_bs via BSpline using the generated coefficients from dynesty
    for _nbs in range(0, n_coeffs_vrot_bs):
        tck_vrot_bs[1][_nbs] = _dyn_params[n_ring_params_free+_nbs]
    _vrot_bs = BSpline(*tck_vrot_bs, extrapolate=True)(r_galaxy_plane)

    ij_aring[:,2] = _vsys + (_vrot_bs * cost1 + _vrad * sint1) * sini1 # model LOS velocity
    #ij_aring[:,2] = _vsys + (_vrot * cost1 + _vrad * sint1) * sini1 # model LOS velocity

    _tr_model_vf[:, :] = np.nan # initialize with a blank value
    _tr_model_vf[ij_aring[:, 1].astype(int), ij_aring[:, 0].astype(int)] = ij_aring[:, 2]

    #return _tr_model_vf
    return r_galaxy_plane, _pa_bs

#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def derive_vlos_model_2d_vrot_bs_org(_dyn_params, _tr_model_vf, ij_aring, fit_opt, _params):
    # nuree4
    # args[0] (_dyn_params): 2dbat tr parameters (passed by dynesty)
    # args[1] (_tr_model_vf): blank array which will be filled up below (passed by user)
    # args[2] (ij_aring): (i, j coordinates of the fitting area) (passed by user)
    # args[3] (fit_opt) : 2dbat tr parameters fitting option (passed by user)
    # return _tr_model_vf : vlos updated

    # _________________________________________
    # arguments
    # _________________________________________
    # 1) _dyn_params = args[0] : dynesty params passed
    #   ---------------------
    #   2dbat tr parameters
    #   ---------------------
    #   _dyn_params[0] : sigma
    #   _dyn_params[1] : xpos
    #   _dyn_params[2] : ypos
    #   _dyn_params[3] : vsys
    #   _dyn_params[4] : pa
    #   _dyn_params[5] : incl
    #   _dyn_params[6] : vrot
    #   _dyn_params[7] : vrad

    #   _dyn_params[8] : vrot_c0_bs
    #   _dyn_params[9] : vrot_c1_bs
    #   _dyn_params[10] : vrot_c2_bs

    # 
    # 2) _tr_model_vf : blank array
    # 3) i, j : coordinate x, y
    # 4) fit_opt : trfit fitting option
    # _________________________________________

    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    deg_to_rad = np.pi / 180.

    n_ring_params_free = 0 # starting from sigma
    # ------------------------------------------------------
    # sigma
    if fit_opt[0] == 1: # sigma fixed:0 free:1
        _sigma = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _sigma = 10 # fixed to the previous fitting result
    # ------------------------------------------------------
    # ------------------------------------------------------
    # XPOS
    if fit_opt[1] == 1: # xpos fixed:0 free:1
        _xpos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _xpos = 20 # fixed to the previous fitting result
    # ------------------------------------------------------
    # YPOS
    if fit_opt[2] == 1: # ypos fixed:0 free:1
        _ypos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _ypos = 20 # fixed to the previous fitting result
    # ------------------------------------------------------
    # VSYS
    if fit_opt[3] == 1: # vsys fixed:0 free:1
        _vsys = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vsys = 173 # fixed to the previous fitting result
    # ------------------------------------------------------
    # PA
    if fit_opt[4] == 1: # pa fixed:0 free:1
        _pa = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _pa = 45 # fixed to the previous fitting result
    # ------------------------------------------------------
    # INCL
    if fit_opt[5] == 1: # incl fixed:0 free:1
        _incl = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _incl = 60 # fixed to the previous fitting result
    # ------------------------------------------------------
    # VROT
    if fit_opt[6] == 1: # vrot fixed:0 free:1
        _vrot = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrot = 50 # fixed to the previous fitting result
    # ------------------------------------------------------
    # VRAD
    if fit_opt[7] == 1: # vrad fixed:0 free:1
        _vrad = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrad = 0 # fixed to the previous fitting result


    # ------------------------------------------------------
    # VROT - bspline
    n_coeffs_vrot_bs, tck_vrot_bs = bspline_ncoeffs_tck(_params, 'vrot')


    # Derive radii on the galaxy plane in pixel, arcsec and pc units */
    cdelt1 = 1
    cdelt2 = 1

    cosp1 = np.cos(deg_to_rad*_pa)
    sinp1 = np.sin(deg_to_rad*_pa)

    cosi1 = np.cos(deg_to_rad*_incl) #cosine
    sini1 = np.sin(deg_to_rad*_incl) # sine

    del_x = (ij_aring[:, 0] - _xpos)*cdelt1 # calculate x
    del_y = (ij_aring[:, 1] - _ypos)*cdelt2 # calculate y

    x_galaxy_plane = (-del_x * sinp1 + del_y * cosp1) # x in plane of galaxy
    y_galaxy_plane = (-del_x * cosp1 - del_y * sinp1) / cosi1 # y in plane of galaxy

    r_galaxy_plane = (x_galaxy_plane**2 + y_galaxy_plane**2)**0.5
    cost1 = x_galaxy_plane / r_galaxy_plane # cosine of angle in plane of galaxy 
    sint1 = y_galaxy_plane / r_galaxy_plane # sine of angle in plane of galaxy

    # -------------------------------------
    # _vrot_bs
    # derive _vrot_bs via BSpline using the generated coefficients from dynesty
    for _nbs in range(0, n_coeffs_vrot_bs):
        tck_vrot_bs[1][_nbs] = _dyn_params[n_ring_params_free+_nbs]
    _vrot_bs = BSpline(*tck_vrot_bs, extrapolate=True)(r_galaxy_plane)

    ij_aring[:,2] = _vsys + (_vrot_bs * cost1 + _vrad * sint1) * sini1 # model LOS velocity
    #ij_aring[:,2] = _vsys + (_vrot * cost1 + _vrad * sint1) * sini1 # model LOS velocity

    _tr_model_vf[:, :] = np.nan 
    _tr_model_vf[ij_aring[:, 1].astype(int), ij_aring[:, 0].astype(int)] = ij_aring[:, 2]

    return _tr_model_vf

#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def derive_vlos_model_org(_dyn_params, i, j, fit_opt):
    # args[0] (_dyn_params): 2dbat tr parameters (passed by dynesty)
    # args[1] : (i coordinates of the fitting area) (passed by user)
    # args[2] : (j coordinates of the fitting area) (passed by user)
    # args[3] (fit_opt) : 2dbat tr parameters fitting option (passed by user)

    # _________________________________________
    # arguments
    # _________________________________________
    # 1) _dyn_params = args[0] : dynesty params passed
    #   ---------------------
    #   2dbat tr parameters
    #   ---------------------
    #   _dyn_params[0] : sigma
    #   _dyn_params[1] : xpos
    #   _dyn_params[2] : ypos
    #   _dyn_params[3] : vsys
    #   _dyn_params[4] : pa
    #   _dyn_params[5] : incl
    #   _dyn_params[6] : vrot
    #   _dyn_params[7] : vrad
    # 
    # 2) i : coordinate x
    # 3) j : coordinate y
    # 4) fit_opt : trfit fitting option
    # _________________________________________

    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    deg_to_rad = np.pi / 180.

    n_ring_params_free = 0 # starting from sigma
    # ------------------------------------------------------
    # sigma
    if fit_opt[0] == 1: # xpos fixed:0 free:1
        _sigma = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _sigma = 10 # fixed to the previous fitting result
    # ------------------------------------------------------
    # ------------------------------------------------------
    # XPOS
    if fit_opt[1] == 1: # xpos fixed:0 free:1
        _xpos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _xpos = 20 # fixed to the previous fitting result
    # ------------------------------------------------------
    # YPOS
    if fit_opt[2] == 1: # ypos fixed:0 free:1
        _ypos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _ypos = 20 # fixed to the previous fitting result
    # ------------------------------------------------------
    # VSYS
    if fit_opt[3] == 1: # vsys fixed:0 free:1
        _vsys = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vsys = 173 # fixed to the previous fitting result
    # ------------------------------------------------------
    # PA
    if fit_opt[4] == 1: # pa fixed:0 free:1
        _pa = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _pa = 45 # fixed to the previous fitting result
    # ------------------------------------------------------
    # INCL
    if fit_opt[5] == 1: # incl fixed:0 free:1
        _incl = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _incl = 60 # fixed to the previous fitting result
    # ------------------------------------------------------
    # VROT
    if fit_opt[6] == 1: # vrot fixed:0 free:1
        _vrot = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrot = 50 # fixed to the previous fitting result
    # ------------------------------------------------------
    # VRAD
    if fit_opt[7] == 1: # vrad fixed:0 free:1
        _vrad = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrad = 0 # fixed to the previous fitting result

    #print("seheon10", _xpos, _ypos, _vsys, _pa, _incl, _vrot, _vrad)
    # Derive radii on the galaxy plane in pixel, arcsec and pc units */
    cdelt1 = 1
    cdelt2 = 1

    cosp1 = np.cos(deg_to_rad*_pa) # cosine
    sinp1 = np.sin(deg_to_rad*_pa) # sine

    cosi1 = np.cos(deg_to_rad*_incl) #cosine
    sini1 = np.sin(deg_to_rad*_incl) # sine

    del_x = (i - _xpos)*cdelt1 # calculate x
    del_y = (j - _ypos)*cdelt2 # calculate y
    x_galaxy_plane = (-del_x * sinp1 + del_y * cosp1) # x in plane of galaxy
    y_galaxy_plane = (-del_x * cosp1 - del_y * sinp1) / cosi1 # y in plane of galaxy
    r_galaxy_plane = (x_galaxy_plane*x_galaxy_plane + y_galaxy_plane*y_galaxy_plane)**0.5
    cost1 = x_galaxy_plane / r_galaxy_plane # cosine of angle in plane of galaxy 
    sint1 = y_galaxy_plane / r_galaxy_plane # sine of angle in plane of galaxy

    vlos_model = _vsys + (_vrot * cost1 + _vrad * sint1 ) * sini1 # model LOS velocity

    return vlos_model

#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# run dynesty for each tilted ring
#@jit(nopython=True)
#@ray.remote(num_cpus=1)
def run_nested_sampler_trfit(_input_vf, _tr_model_vf, _wt_2d, _ij_aring, _params, fit_opt, ndim, tr_params_priors_init):

    # bspline options: pa (XX, XX), incl (XX, XX), vrot (XX, XX), vrad (XX, XX)
    bspline_opt = np.zeros(8, dtype=np.int32)

    _naxis1 = _params['naxis1']
    _naxis2 = _params['naxis2']
    _queue_size = int(_params['num_cpus_tr_dyn'])
    _tr_model_vf = np.full((_naxis2, _naxis1), fill_value=np.nan, dtype=np.float64)
    if _params['_dynesty_class_tr'] == 'static': # recommended calss option
        #---------------------------------------------------------
        # run dynesty 2.0.3

        # loglike_trift : log-likelihood
        # optimal_prior_trfit : set priors (from hypercube to real unit params)
        # ndim : dimensions

        # fit_opt : 2dbat tr parameters fitting option
        # bspline_opt : pa, incl, vrot, vrad b-spline order / knots
        # tr_params_priors_init : 2dbat tr parameters priors

        #rstate = np.random.default_rng(2)
        with mp.Pool(_queue_size) as pool:
            sampler = NestedSampler(loglike_trfit, optimal_prior_trfit, ndim,
                nlive=_params['nlive'],
                update_interval=_params['update_interval'],
                sample=_params['sample'],
                #sample='rwalk',
                pool=pool,
                queue_size=_queue_size,
                bound=_params['bound'],
                #rstate=rstate,
                facc=_params['facc'],
                fmove=_params['fmove'],
                max_move=_params['max_move'],
                logl_args=[_input_vf, _tr_model_vf, _wt_2d, _ij_aring, fit_opt, _params, bspline_opt], ptform_args=[fit_opt, bspline_opt, tr_params_priors_init])

            sampler.run_nested(dlogz=_params['dlogz_tr'], maxiter=_params['maxiter_tr'], maxcall=_params['maxcall'], print_progress=_params['print_progress_tr'])

    elif _params['_dynesty_class_tr'] == 'dynamic':

        #---------------------------------------------------------
        # run dynesty 2.0.3

        # loglike_trift : log-likelihood
        # optimal_prior_trfit : set priors (from hypercube to real unit params)
        # ndim : dimensions

        # fit_opt : 2dbat tr parameters fitting option
        # bspline_opt : pa, incl, vrot, vrad b-spline order / knots
        # tr_params_priors_init : 2dbat tr parameters priors
        sampler = DynamicNestedSampler(loglike_trfit, optimal_prior_trfit, ndim,
            nlive=_params['nlive'],
            update_interval=_params['update_interval'],
            sample=_params['sample'],
            bound=_params['bound'],
            facc=_params['facc'],
            fmove=_params['fmove'],
            max_move=_params['max_move'],
            logl_args=[_input_vf, _tr_model_vf, _wt_2d, _ij_aring, fit_opt, _params, bspline_opt], ptform_args=[fit_opt, bspline_opt, tr_params_priors_init])
        sampler.run_nested(dlogz_init=_params['dlogz_tr'], maxiter_init=_params['maxiter'], maxcall_init=_params['maxcall'], print_progress=_params['print_progress'])


    #---------------------------------------------------------
    _trfit_results_temp, _logz = get_dynesty_sampler_results(sampler)
    _trfit_results = np.zeros(2*ndim, dtype=np.float32)
    #---------------------------------------------------------
    # param1, param2, param3 ....param1-e, param2-e, param3-e
    # _trfit_results[0~2*ndim] = _trfit_results_temp[0~2*ndim]
    _trfit_results[:2*ndim] = _trfit_results_temp
    #---------------------------------------------------------

    return _trfit_results, ndim

    # Plot a summary of the run.
#    rfig, raxes = dyplot.runplot(sampler.results)
#    rfig.savefig("r.pdf")
    
    # Plot traces and 1-D marginalized posteriors.
#    tfig, taxes = dyplot.traceplot(sampler.results)
#    tfig.savefig("t.pdf")
    
    # Plot the 2-D marginalized posteriors.
    #cfig, caxes = dyplot.cornerplot(sampler.results)
    #cfig.savefig("c.pdf")
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# run dynesty for each tilted ring
#@jit(nopython=True)
#@ray.remote(num_cpus=1)
#@ray.remote
def run_nested_sampler_trfit_2d(_input_vf, _tr_model_vf, _wt_2d, _ij_aring, _params, tr_params_bounds, nrings_reliable, r_galaxy_plane, tck_vrot_bs_init_from_trfit, tck_pa_bs_init_from_trfit, tck_incl_bs_init_from_trfit, _2dbat_run_i):

# nuree1
    # number of (i, j) points in a ring
    n_ij_aring = _ij_aring.shape[0]

    # ring parameters fitting option: fixed:0 or free:1
    fit_opt_2d = np.zeros(8, dtype=np.int32)

    _ring_ns_t = np.zeros(_params['nrings']+2, dtype=np.float)
    _vrot_ns_t = np.zeros(_params['nrings']+2, dtype=np.float)

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
        fit_opt_2d[0] = 1
        ndim_t += 1
    if _params['xpos_fitting'] == 'free':
        fit_opt_2d[1] = 1
        ndim_t += 1
    if _params['ypos_fitting'] == 'free':
        fit_opt_2d[2] = 1
        ndim_t += 1
    if _params['vsys_fitting'] == 'free':
        fit_opt_2d[3] = 1
        ndim_t += 1
    if _params['pa_fitting'] == 'free' and n_coeffs_pa_bs == 0: # constant PA
        fit_opt_2d[4] = 1
        ndim_t += 1
    if _params['incl_fitting'] == 'free' and n_coeffs_incl_bs == 0: # constant INCL
        fit_opt_2d[5] = 1
        ndim_t += 1
    if _params['vrot_fitting'] == 'free' and n_coeffs_vrot_bs == 0: # constant VROT
        fit_opt_2d[6] = 1
        ndim_t += 1
    if _params['vrad_fitting'] == 'free':
        fit_opt_2d[7] = 1
        ndim_t += 1


    # add n_pa_c_bs
    ndim_t += n_coeffs_pa_bs
    # add n_pa_c_bs
    ndim_t += n_coeffs_incl_bs
    # add n_vrot_c_bs
    ndim_t += n_coeffs_vrot_bs

    ndim = ndim_t
    nparams = ndim_t

    # bspline options: pa (XX, XX), incl (XX, XX), vrot (XX, XX), vrad (XX, XX)
    bspline_opt = np.zeros(8, dtype=np.int32)

    if _params['_dynesty_class_2d'] == 'static': # recommended calss option
        #---------------------------------------------------------
        # run dynesty 2.0.3

        # loglike_trift : log-likelihood
        # optimal_prior_trfit : set priors (from hypercube to real unit params)
        # ndim : dimensions

        # fit_opt_2d : 2dbat tr parameters fitting option
        # bspline_opt : pa, incl, vrot, vrad b-spline order / knots
        # tr_params_priors_init : 2dbat tr parameters priors

        #LOGZ_TRUTH_GAU = 0
        _queue_size = int(_params['num_cpus_2d_dyn'])
        #rstate = np.random.default_rng(2)
        with mp.Pool(_queue_size) as pool:
            sampler = NestedSampler(loglike_trfit_2d, optimal_prior_trfit_2d, ndim,
                nlive=_params['nlive'],
                update_interval=_params['update_interval'],
                sample=_params['sample'],
                #sample='rwalk',
                pool=pool,
                queue_size=_queue_size,
                #enlarge=2,
                #rstate=rstate,
                first_update={
                    'min_eff': 10,
                    'min_ncall': 200},
                bound=_params['bound'],
                facc=_params['facc'],
                fmove=_params['fmove'],
                max_move=_params['max_move'],
                logl_args=[_input_vf, _tr_model_vf, _ij_aring, fit_opt_2d, bspline_opt, _params, nrings_reliable, r_galaxy_plane, _wt_2d, n_coeffs_pa_bs, tck_pa_bs, n_coeffs_incl_bs, tck_incl_bs, n_coeffs_vrot_bs, tck_vrot_bs], \
                ptform_args=[fit_opt_2d, bspline_opt, _params, tr_params_bounds, nrings_reliable, tck_vrot_bs_init_from_trfit, tck_pa_bs_init_from_trfit, tck_incl_bs_init_from_trfit, _input_vf, _ring_ns_t, _vrot_ns_t, \
                             n_coeffs_pa_bs, tck_pa_bs, n_coeffs_incl_bs, tck_incl_bs, n_coeffs_vrot_bs, tck_vrot_bs])

                #ptform_args=[fit_opt, bspline_opt, _params, tr_params_bounds, nrings_reliable, tck_vrot_bs_init_from_trfit, tck_pa_bs_init_from_trfit, tck_incl_bs_init_from_trfit, _input_vf, _ring_ns_t, _vrot_ns_t])

            sampler.run_nested(dlogz=_params['dlogz_2d'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=_params['print_progress_2d'])

            #assert (abs(LOGZ_TRUTH_GAU - sampler.results['logz'][-1]) <
            #    5. * sampler.results['logzerr'][-1])

            # save the dynesty sampling results in binary format
            _dir_2dbat_PI_output = _params['wdir'] + '/' + _params['_2dbatdir'] + ".%d" % _2dbat_run_i
            dyn_sampler_bin = 'dyn_sample.pkl'  # file name
            with open(_dir_2dbat_PI_output + '/' + dyn_sampler_bin, 'wb') as file:
                pickle.dump(sampler.results, file)


    elif _params['_dynesty_class_2d'] == 'dynamic':
        #---------------------------------------------------------
        # run dynesty 2.0.3

        # loglike_trift : log-likelihood
        # optimal_prior_trfit : set priors (from hypercube to real unit params)
        # ndim : dimensions

        # fit_opt : 2dbat tr parameters fitting option
        # bspline_opt : pa, incl, vrot, vrad b-spline order / knots
        # tr_params_priors_init : 2dbat tr parameters priors
        #dsampler = DynamicNestedSampler(loglike_trfit_2d, optimal_prior_trfit_2d, ndim,
        #    nlive=_params['nlive'],
        #    update_interval=_params['update_interval'],
        #    sample=_params['sample'],
        #    bound=_params['bound'],
        #    facc=_params['facc'],
        #    fmove=_params['fmove'],
        #    max_move=_params['max_move'],
        #    logl_args=[_input_vf, _tr_model_vf, _ij_aring, fit_opt_2d, bspline_opt, _params, nrings_reliable, r_galaxy_plane], ptform_args=[fit_opt_2d, bspline_opt, _params, tr_params_bounds, nrings_reliable])
        #dsampler.run_nested(dlogz_init=_params['dlogz_2d'], maxiter_init=_params['maxiter'], maxcall_init=_params['maxcall'], print_progress=True)


      #LOGZ_TRUTH_GAU = 0
        _queue_size = int(_params['num_cpus_2d_dyn'])
        rstate = np.random.default_rng(2)
        with mp.Pool(_queue_size) as pool:
            sampler = DynamicNestedSampler(loglike_trfit_2d, optimal_prior_trfit_2d, ndim,
                nlive=_params['nlive'],
                update_interval=_params['update_interval'],
                sample=_params['sample'],
                #sample='rwalk',
                pool=pool,
                queue_size=_queue_size,
                #enlarge=2,
                rstate=rstate,
                first_update={
                    'min_eff': 10,
                    'min_ncall': 200},
                bound=_params['bound'],
                facc=_params['facc'],
                fmove=_params['fmove'],
                max_move=_params['max_move'],
                logl_args=[_input_vf, _tr_model_vf, _ij_aring, fit_opt_2d, bspline_opt, _params, nrings_reliable, r_galaxy_plane, _wt_2d, n_coeffs_pa_bs, tck_pa_bs, n_coeffs_incl_bs, tck_incl_bs, n_coeffs_vrot_bs, tck_vrot_bs], \
                ptform_args=[fit_opt_2d, bspline_opt, _params, tr_params_bounds, nrings_reliable, tck_vrot_bs_init_from_trfit, tck_pa_bs_init_from_trfit, tck_incl_bs_init_from_trfit, _input_vf, _ring_ns_t, _vrot_ns_t, \
                             n_coeffs_pa_bs, tck_pa_bs, n_coeffs_incl_bs, tck_incl_bs, n_coeffs_vrot_bs, tck_vrot_bs])

                #ptform_args=[fit_opt, bspline_opt, _params, tr_params_bounds, nrings_reliable, tck_vrot_bs_init_from_trfit, tck_pa_bs_init_from_trfit, tck_incl_bs_init_from_trfit, _input_vf, _ring_ns_t, _vrot_ns_t])

            sampler.run_nested(dlogz_init=_params['dlogz_2d'], nlive_init=100, nlive_batch=50, n_effective = 20000, maxiter_init=_params['maxiter'], maxcall_init=_params['maxcall'], print_progress=_params['print_progress_2d'])
            #sampler.run_nested(dlogz_init=_params['dlogz_2d'], nlive_init=100, n_effective = 20000, maxiter_init=_params['maxiter'], maxcall_init=_params['maxcall'], print_progress=_params['print_progress_2d'])

            #assert (abs(LOGZ_TRUTH_GAU - sampler.results['logz'][-1]) <
            #    5. * sampler.results['logzerr'][-1])




    #---------------------------------------------------------
    res0 = sampler.results
    _trfit_results_temp, _logz = get_dynesty_sampler_results(sampler)

    # samples, relative weights
    samples, weights = sampler.results.samples, np.exp(sampler.results.logwt - sampler.results.logz[-1])

    res_total = []
    rstate = np.random.default_rng(2)
    for i in range(100):
        res_t = dyfunc.resample_run(res0, rstate=rstate)
        res_total.append(res_t)

    # 1st moment [0]
    # 2nd moment [1]
    x_arr = np.array([dyfunc.mean_and_cov(res_t.samples,
                      weights=np.exp(res_t.logwt - res_t.logz[-1]))[1]
                      for res_t in res_total])

    x_arr = [np.sqrt(np.diag(x)) for x in x_arr]

    std_resample_run = np.round(np.mean(x_arr, axis=0), 3)
    x_std = np.round(np.std(x_arr, axis=0), 3)
    #print('2nd.:     {0} +/- {1}'.format(x_mean, x_std))


    _trfit_results = np.zeros(2*ndim, dtype=np.float32)
    #---------------------------------------------------------
    # param1, param2, param3 ....param1-e, param2-e, param3-e
    # _trfit_results[0~2*ndim] = _trfit_results_temp[0~2*ndim]
    _trfit_results[:2*ndim] = _trfit_results_temp
    #print(_trfit_results)
    #---------------------------------------------------------

    #---------------------------------------------------------
    #---------------------------------------------------------
    # Save DYNESTY summary 
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        sampler.results.summary()
        # 버퍼에서 문자열을 가져와 저장
        dyn_summary = buf.getvalue()

    # set 2dbat PI output dir
    _dir_2dbat_PI_output = os.path.join(_params['wdir'], _params['_2dbatdir'] + ".%d" % _2dbat_run_i)

    #---------------------------------------------------------
    # if not present, creat dir
    if not os.path.exists(_dir_2dbat_PI_output):
        os.makedirs(_dir_2dbat_PI_output)

    #---------------------------------------------------------
    # save dynest summary in txt 
    summary_file_path = os.path.join(_dir_2dbat_PI_output, '2dbat_PI_dynesty.summary.txt')
    with open(summary_file_path, 'w') as file:
        file.write(dyn_summary)

    #---------------------------------------------------------
    # save plots
    #labels = ['a', 'b']
    _fig1, _axes = dyplot.cornerplot(res0, show_titles=True)
    _fig1.savefig('%s/dyn_posteriors_cornerplot.png' % _dir_2dbat_PI_output)

    #labels = ['a', 'b']
    _fig2, _axes = dyplot.runplot(res0)
    _fig2.savefig('%s/dyn_runplot.png' % _dir_2dbat_PI_output)

    #labels = ['a', 'b']
    _fig3, _axes = dyplot.traceplot(res0, show_titles=True)
    _fig3.savefig('%s/dyn_traceplot.png' % _dir_2dbat_PI_output)

    #labels = ['a', 'b']
    _fig4, _axes = dyplot.cornerpoints(res0)
    _fig4.savefig('%s/dyn_cornerpoints.png' % _dir_2dbat_PI_output)


    return _trfit_results, ndim, fit_opt_2d, std_resample_run

    # Plot a summary of the run.
#    rfig, raxes = dyplot.runplot(sampler.results)
#    rfig.savefig("r.pdf")
    
    # Plot traces and 1-D marginalized posteriors.
#    tfig, taxes = dyplot.traceplot(sampler.results)
#    tfig.savefig("t.pdf")
    
    # Plot the 2-D marginalized posteriors.
    #cfig, caxes = dyplot.cornerplot(sampler.results)
    #cfig.savefig("c.pdf")
#-- END OF SUB-ROUTINE____________________________________________________________#




























#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# derive rms of a profile via ngfit 
def derive_rms_npoints(_inputDataCube, _cube_mask_2d, _x, _params, ngauss):

    ndim = 3*ngauss + 2
    nparams = ndim

    naxis1 = int(_params['naxis1'])
    naxis2 = int(_params['naxis2'])

    naxis1_s0 = int(_params['naxis1_s0'])
    naxis1_e0 = int(_params['naxis1_e0'])
    naxis2_s0 = int(_params['naxis2_s0'])
    naxis2_e0 = int(_params['naxis2_e0'])

    naxis1_seg = naxis1_e0 - naxis1_s0
    naxis2_seg = naxis2_e0 - naxis2_s0

    nsteps_x = int(_params['nsteps_x_rms'])
    nsteps_y = int(_params['nsteps_y_rms'])

    _rms = np.zeros(nsteps_x*nsteps_y+1, dtype=np.float32)
    _bg = np.zeros(nsteps_x*nsteps_y+1, dtype=np.float32)
    # prior arrays for the single Gaussian fit
    gfit_priors_init = np.zeros(2*5, dtype=np.float32)
    _x_boundaries = np.full(2*ngauss, fill_value=-1E11, dtype=np.float32)
    #gfit_priors_init = [sig1, bg1, x1, std1, p1, sig2, bg2, x2, std2, p2]
    # for the first single gaussian fit: optimal priors will be updated later
    gfit_priors_init = [0.0, 0.0, 0.01, 0.01, 0.01, 0.5, 0.6, 0.99, 0.6, 1.01]

    k=0
    for x in range(0, nsteps_x):
        for y in range(0, nsteps_y):

            i = int(0.5*(naxis1_seg/nsteps_x) + x*(naxis1_seg/nsteps_x)) + naxis1_s0
            j = int(0.5*(naxis2_seg/nsteps_y) + y*(naxis2_seg/nsteps_y)) + naxis2_s0

            print("[--> measure background rms at (i:%d j:%d)...]" % (i, j))

            if(_cube_mask_2d[j, i] > 0 and not np.isnan(_inputDataCube[:, j, i]).any()): # if not masked: 

                _f_max = np.max(_inputDataCube[:, j, i]) # peak flux : being used for normalization
                _f_min = np.min(_inputDataCube[:, j, i]) # lowest flux : being used for normalization
    
                #---------------------------------------------------------
                if(ndim * (ndim + 1) // 2 > _params['nlive']):
                    _params['nlive'] = 1 + ndim * (ndim + 1) // 2 # optimal minimum nlive
    
                # run dynesty 1.1
                #sampler = NestedSampler(loglike_d, optimal_prior, ndim,
                #    vol_dec=_params['vol_dec'],
                #    vol_check=_params['vol_check'],
                #    facc=_params['facc'],
                #    nlive=_params['nlive'],
                #    sample=_params['sample'],
                #    bound=_params['bound'],
                #    #rwalk=_params['rwalk'],
                #    max_move=_params['max_move'],
                #    logl_args=[(_inputDataCube[:, j, i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])


                if _params['_dynesty_class_'] == 'static':
                    #---------------------------------------------------------
                    # run dynesty 2.0.3
                    sampler = NestedSampler(loglike_d, optimal_prior, ndim,
                        nlive=_params['nlive'],
                        update_interval=_params['update_interval'],
                        sample=_params['sample'],
                        #walks=_params['walks'],
                        bound=_params['bound'],
                        facc=_params['facc'],
                        fmove=_params['fmove'],
                        max_move=_params['max_move'],
                        logl_args=[(_inputDataCube[:, j, i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])

                    sampler.run_nested(dlogz=_params['dlogz_2d'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=True)
                    #_run_nested = jit(sampler.run_nested(dlogz=1000, maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=True), nopython=True, cache=True, nogil=True, fastmath=True)
                    #sampler.reset()
                    #_run_nested = jit(sampler.run_nested(dlogz=_params['dlogz'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=True), nopython=True, cache=True, nogil=True, fastmath=True)
                    #_run_nested = jit(sampler.run_nested(dlogz=_params['dlogz'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=True), nopython=True, cache=True, nogil=True, parallel=True)
                    #_run_nested()

                elif _params['_dynesty_class_'] == 'dynamic':
                    #---------------------------------------------------------
                    # run dynesty 2.0.3
                    dsampler = DynamicNestedSampler(loglike_d, optimal_prior, ndim,
                        nlive=_params['nlive'],
                        update_interval=_params['update_interval'],
                        sample=_params['sample'],
                        bound=_params['bound'],
                        facc=_params['facc'],
                        fmove=_params['fmove'],
                        max_move=_params['max_move'],
                        logl_args=[(_inputDataCube[:, j, i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])
                    dsampler.run_nested(dlogz_init=_params['dlogz_2d'], maxiter_init=_params['maxiter'], maxcall_init=_params['maxcall'], print_progress=True)

                    #---------------------------------------------------------
                _gfit_results_temp, _logz = get_dynesty_sampler_results(sampler)
    
    
                #---------------------------------------------------------
                # lower bounds : x1-3*std1, x2-3*std2, ...  | x:_gfit_results_temp[2, 5, 8, ...], std:_gfit_results_temp[3, 6, 9, ...]
                #_x_boundaries[0:ngauss] = _gfit_results_temp[2:nparams:3] - _params['nsigma_prior_range_gfit']*_gfit_results_temp[3:nparams:3] # x - 3*std
                _x_boundaries[0:ngauss] = _gfit_results_temp[2:nparams:3] - 5*_gfit_results_temp[3:nparams:3] # x - 3*std
                #print("g:", ngauss, "lower bounds:", _x_boundaries)
    
                #---------------------------------------------------------
                # upper bounds : x1+3*std1, x2+3*std2, ...  | x:_gfit_results_temp[2, 5, 8, ...], std:_gfit_results_temp[3, 6, 9, ...]
                #_x_boundaries[ngauss:2*ngauss] = _gfit_results_temp[2:nparams:3] + _params['nsigma_prior_range_gfit']*_gfit_results_temp[3:nparams:3] # x + 3*std
                _x_boundaries[ngauss:2*ngauss] = _gfit_results_temp[2:nparams:3] + 5*_gfit_results_temp[3:nparams:3] # x + 3*std
                #print("g:", ngauss, "upper bounds:", _x_boundaries)
    
                #---------------------------------------------------------
                # lower/upper bounds
                _x_boundaries_ft = np.delete(_x_boundaries, np.where(_x_boundaries < -1E3))
                _x_lower = np.sort(_x_boundaries_ft)[0]
                _x_upper = np.sort(_x_boundaries_ft)[-1]
                _x_lower = _x_lower if _x_lower > 0 else 0
                _x_upper = _x_upper if _x_upper < 1 else 1
                #print(_x_lower, _x_upper)
    
                #---------------------------------------------------------
                # derive the rms given the current ngfit
                _f_ngfit = f_gaussian_model(_x, _gfit_results_temp, ngauss)
                # residual : input_flux - ngfit_flux
                _res_spect = ((_inputDataCube[:, j, i]-_f_min)/(_f_max-_f_min) - _f_ngfit)
                # rms
                #print(np.where(_x < _x_lower or _x > _x_upper))
                _index_t = np.argwhere((_x < _x_lower) | (_x > _x_upper))
                #print(_index_t) <-- remove these elements
                _res_spect_ft = np.delete(_res_spect, _index_t)
    
                # rms
                _rms[k] = np.std(_res_spect_ft)*(_f_max - _f_min)
                # bg
                _bg[k] = _gfit_results_temp[1]*(_f_max - _f_min) + _f_min # bg
                print(i, j, _rms[k], _bg[k])
                k += 1

    # median values
    # first replace 0.0 (zero) to NAN value to use numpy nanmedian function instead of using numpy median
    zero_to_nan_rms = np.where(_rms == 0.0, np.nan, _rms)
    zero_to_nan_bg = np.where(_bg == 0.0, np.nan, _bg)

    _rms_med = np.nanmedian(zero_to_nan_rms)
    _bg_med = np.nanmedian(zero_to_nan_bg)
    # update _rms_med, _bg_med in _params
    _params['_rms_med'] = _rms_med
    _params['_bg_med'] = _bg_med
    print("rms_med:_", _rms_med)
    print("bg_med:_", _bg_med)
#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# derive rms of a profile using _gfit_results_temp derived from ngfit
def little_derive_rms_npoints(_inputDataCube, i, j, _x, _f_min, _f_max, ngauss, _gfit_results_temp):

    ndim = 3*ngauss + 2
    nparams = ndim

    _x_boundaries = np.full(2*ngauss, fill_value=-1E11, dtype=np.float32)
    #---------------------------------------------------------
    # lower bounds : x1-3*std1, x2-3*std2, ...  | x:_gfit_results_temp[2, 5, 8, ...], std:_gfit_results_temp[3, 6, 9, ...]
    #_x_boundaries[0:ngauss] = _gfit_results_temp[2:nparams:3] - _params['nsigma_prior_range_gfit']*_gfit_results_temp[3:nparams:3] # x - 3*std
    _x_boundaries[0:ngauss] = _gfit_results_temp[2:nparams:3] - 5*_gfit_results_temp[3:nparams:3] # x - 3*std
    #print("g:", ngauss, "lower bounds:", _x_boundaries)

    #---------------------------------------------------------
    # upper bounds : x1+3*std1, x2+3*std2, ...  | x:_gfit_results_temp[2, 5, 8, ...], std:_gfit_results_temp[3, 6, 9, ...]
    #_x_boundaries[ngauss:2*ngauss] = _gfit_results_temp[2:nparams:3] + _params['nsigma_prior_range_gfit']*_gfit_results_temp[3:nparams:3] # x + 3*std
    _x_boundaries[ngauss:2*ngauss] = _gfit_results_temp[2:nparams:3] + 5*_gfit_results_temp[3:nparams:3] # x + 3*std
    #print("g:", ngauss, "upper bounds:", _x_boundaries)

    #---------------------------------------------------------
    # lower/upper bounds
    _x_boundaries_ft = np.delete(_x_boundaries, np.where(_x_boundaries < -1E3))
    _x_lower = np.sort(_x_boundaries_ft)[0]
    _x_upper = np.sort(_x_boundaries_ft)[-1]
    _x_lower = _x_lower if _x_lower > 0 else 0
    _x_upper = _x_upper if _x_upper < 1 else 1
    #print(_x_lower, _x_upper)

    #---------------------------------------------------------
    # derive the rms given the current ngfit
    _f_ngfit = f_gaussian_model(_x, _gfit_results_temp, ngauss)
    # residual : input_flux - ngfit_flux
    _res_spect = ((_inputDataCube[:, j, i]-_f_min)/(_f_max-_f_min) - _f_ngfit)
    # rms
    #print(np.where(_x < _x_lower or _x > _x_upper))
    _index_t = np.argwhere((_x < _x_lower) | (_x > _x_upper))
    #print(_index_t) <-- remove these elements
    _res_spect_ft = np.delete(_res_spect, _index_t)

    # rms
    #_rms_ngfit = np.std(_res_spect_ft)*(_f_max - _f_min)
    _rms_ngfit = np.std(_res_spect_ft) # normalised
    # bg
    #_bg_ngfit = _gfit_results_temp[1]*(_f_max - _f_min) + _f_min # bg

    #if i == 531 and j == 531:
    #    #print(_x)
    #    print(_x_lower, _x_upper)
    #    #print(_f_max, _f_min)
    #    print(_res_spect_ft)
    #    print(_rms_ngfit*(_f_max-_f_min))
    #    #print((_inputDataCube[:, j, i]-_f_min)/(_f_max-_f_min))
    #    #print(_inputDataCube[:, j, i])

    del(_x_boundaries, _x_boundaries_ft, _index_t, _res_spect_ft)
    gc.collect()

    return _rms_ngfit # resturn normalised _rms
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# run dynesty for each line profile
#@jit(nopython=True)
#@ray.remote(num_cpus=1)
#@ray.remote
def run_dynesty_sampler_optimal_priors(_inputDataCube, _x, _peak_sn_map, _sn_int_map, _params, _is, _ie, i, _js, _je, _cube_mask_2d):

    _max_ngauss = _params['max_ngauss']
    _vel_min = _params['vel_min']
    _vel_max = _params['vel_max']
    _cdelt3 = _params['cdelt3']

    gfit_results = np.zeros(((_je-_js), _max_ngauss, 2*(2+3*_max_ngauss) + 7 + _max_ngauss), dtype=np.float32)
    _x_boundaries = np.full(2*_max_ngauss, fill_value=-1E11, dtype=np.float32)

    #print("CHECK S/N: %d %d | peak S/N: %.1f < %.1f | integrated S/N: %.1f < %.1f" \
    #    % (i, 0+_js, _peak_sn_map[0+_js, i], _params['peak_sn_limit'], _sn_int_map[0+_js, i], _params['int_sn_limit']))
    for j in range(0, _je -_js):

        _f_max = np.max(_inputDataCube[:,j+_js,i]) # peak flux : being used for normalization
        _f_min = np.min(_inputDataCube[:,j+_js,i]) # lowest flux : being used for normalization
        #print("f_max:", _f_max, "f_min:", _f_min)

        # prior arrays for the 1st single Gaussian fit
        gfit_priors_init = np.zeros(2*5, dtype=np.float32)
        #gfit_priors_init = [sig1, bg1, x1, std1, p1, sig2, bg2, x2, std2, p2]
        # for the first single gaussian fit: optimal priors will be updated later
        #gfit_priors_init = [0.0, 0.0, 0.01, 0.01, 0.01, 0.5, 0.6, 0.99, 0.6, 1.01]
        gfit_priors_init = [0.0, 0.0, 0.001, 0.001, 0.001, 0.5, 0.6, 0.999, 0.999, 1.01]

        if _cube_mask_2d[j+_js, i] <= 0 : # if masked, then skip : NOTE THE MASK VALUE SHOULD BE zero or negative.
            print("mask filtered: %d %d | peak S/N: %.1f :: S/N limit: %.1f | integrated S/N: %.1f :: S/N limit: %.1f :: f_min: %e :: f_max: %e" \
                % (i, j+_js, _peak_sn_map[j+_js, i], _params['peak_sn_limit'], _sn_int_map[j+_js, i], _params['int_sn_limit'], _f_min, _f_max))

            # save the current profile location
            for l in range(0, _max_ngauss):
                gfit_results[j][l][2*(3*_max_ngauss+2)+l] = _params['_rms_med'] # rms: the one derived from derive_rms_npoints_sgfit
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+0] = -1E11 # this is for sgfit: log-Z
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+1] = _is
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+2] = _ie
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+3] = _js
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+4] = _je
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+5] = i
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+6] = j + _js
            continue

        elif _sn_int_map[j+_js, i] < _params['int_sn_limit'] or _peak_sn_map[j+_js, i] < _params['peak_sn_limit'] \
            or np.isnan(_f_max) or np.isnan(_f_min) \
            or np.isinf(_f_min) or np.isinf(_f_min):

            print("low S/N: %d %d | peak S/N: %.1f :: S/N limit: %.1f | integrated S/N: %.1f :: S/N limit: %.1f :: f_min: %e :: f_max: %e" \
                % (i, j+_js, _peak_sn_map[j+_js, i], _params['peak_sn_limit'], _sn_int_map[j+_js, i], _params['int_sn_limit'], _f_min, _f_max))

            # save the current profile location
            for l in range(0, _max_ngauss):
                gfit_results[j][l][2*(3*_max_ngauss+2)+l] = _params['_rms_med'] # rms: the one derived from derive_rms_npoints_sgfit
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+0] = -1E11 # this is for sgfit: log-Z
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+1] = _is
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+2] = _ie
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+3] = _js
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+4] = _je
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+5] = i
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+6] = j + _js
            continue


        for k in range(0, _max_ngauss):
            ngauss = k+1  # set the number of gaussian
            ndim = 3*ngauss + 2
            nparams = ndim

            if(ndim * (ndim + 1) // 2 > _params['nlive']):
                _params['nlive'] = 1 + ndim * (ndim + 1) // 2 # optimal minimum nlive

            print("processing: %d %d | peak s/n: %.1f | integrated s/n: %.1f | gauss-%d" % (i, j+_js, _peak_sn_map[j+_js, i], _sn_int_map[j+_js, i], ngauss))

            #sampler = NestedSampler(loglike_d, uniform_prior_d, ndim, sample='unif',
            #    vol_dec = 0.2, vol_check = 2, facc=0.5, rwalk=1000, nlive=200, max_move=100,
            #    logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss])

            #---------------------------------------------------------
            # run dynesty 1.1
            #sampler = NestedSampler(loglike_d, optimal_prior, ndim,
            #    vol_dec=_params['vol_dec'],
            #    vol_check=_params['vol_check'],
            #    facc=_params['facc'],
            #    sample=_params['sample'],
            #    nlive=_params['nlive'],
            #    bound=_params['bound'],
            #    #rwalk=_params['rwalk'],
            #    max_move=_params['max_move'],
            #    logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])

            if _params['_dynesty_class_'] == 'static':
                #---------------------------------------------------------
                # run dynesty 2.0.3
                sampler = NestedSampler(loglike_d, optimal_prior, ndim,
                    nlive=_params['nlive'],
                    update_interval=_params['update_interval'],
                    sample=_params['sample'],
                    bound=_params['bound'],
                    facc=_params['facc'],
                    fmove=_params['fmove'],
                    max_move=_params['max_move'],
                    logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])
                sampler.run_nested(dlogz=_params['dlogz_2d'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=_params['print_progress'])
                #numba.jit(sampler.run_nested(dlogz=_params['dlogz'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=False), nopython=True, cache=True, nogil=True, parallel=True)
                #_run_nested = jit(sampler.run_nested(dlogz=_params['dlogz'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=False), nopython=True, cache=True, nogil=True, fastmath=True)

            elif _params['_dynesty_class_'] == 'dynamic':
                #---------------------------------------------------------
                # run dynesty 2.0.3
                dsampler = DynamicNestedSampler(loglike_d, optimal_prior, ndim,
                    nlive=_params['nlive'],
                    sample=_params['sample'],
                    #update_interval=_params['update_interval'],
                    bound=_params['bound'],
                    facc=_params['facc'],
                    fmove=_params['fmove'],
                    max_move=_params['max_move'],
                    logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])
                #sampler.reset()
                #numba.jit(sampler.run_nested(dlogz=0.1, maxiter=5000000, maxcall=50000000, print_progress=False), nopython=True, cache=True, nogil=True, parallel=True)
                dsampler.run_nested(dlogz_init=_params['dlogz_2d'], maxiter_init=_params['maxiter'], maxcall_init=_params['maxcall'], print_progress=_params['print_progress'])

            #---------------------------------------------------------
            _gfit_results_temp, _logz = get_dynesty_sampler_results(sampler)

            #---------------------------------------------------------
            # param1, param2, param3 ....param1-e, param2-e, param3-e
            #gfit_results[j][k][0~2*nparams] = _gfit_results_temp[0~2*nparams]
            gfit_results[j][k][:2*nparams] = _gfit_results_temp
            #---------------------------------------------------------

            #---------------------------------------------------------
            # derive rms of the profile given the current ngfit <---- || normalised (0~1) units ||
            _rms_ngfit = little_derive_rms_npoints(_inputDataCube, i, j+_js, _x, _f_min, _f_max, ngauss, _gfit_results_temp)
            #---------------------------------------------------------

            #---------------------------------------------------------
            if ngauss == 1: # check the peak s/n
                # load the normalised sgfit results : --> derive rms for s/n
                #_bg_sgfit = _gfit_results_temp[1]
                #_x_sgfit = _gfit_results_temp[2]
                #_std_sgfit = _gfit_results_temp[3]
                #_p_sgfit = _gfit_results_temp[4]
                # peak flux of the sgfit
                #_f_sgfit =_p_sgfit * exp( -0.5*((_x - _x_sgfit) / _std_sgfit)**2) + _bg_sgfit

                #---------------------------------------------------------
                # update gfit_priors_init
                nparams_n = 3*(ngauss+1) + 2
                gfit_priors_init = np.zeros(2*nparams_n, dtype=np.float32)
                # lower bound : the parameters for the current ngaussian components
                # nsigma_prior_range_gfit=3.0 (default)
                gfit_priors_init[:nparams] = _gfit_results_temp[:nparams] - _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]
                # upper bound : the parameters for the current ngaussian components
                gfit_priors_init[nparams+3:2*nparams+3] = _gfit_results_temp[:nparams] + _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]
                #---------------------------------------------------------


                # peak s/n : more accurate peak s/n from the first sgfit
                # <-- || normalised units (0~1)||
                _bg_sgfit = _gfit_results_temp[1]
                _p_sgfit = _gfit_results_temp[4] # bg already subtracted
                _peak_sn_sgfit = _p_sgfit/_rms_ngfit

                if _peak_sn_sgfit < _params['peak_sn_limit']: 
                    print("skip the rest of Gaussian fits: %d %d | rms:%.1f | bg:%.1f | peak:%.1f | peak_sgfit s/n: %.1f < %.1f" % (i, j+_js, _rms_ngfit, _bg_sgfit, _p_sgfit, _peak_sn_sgfit, _params['peak_sn_limit']))

                    # save the current profile location
                    for l in range(0, _max_ngauss):
                        if l == 0:
                        # for sgfit
                            gfit_results[j][l][2*(3*_max_ngauss+2)+l] = _rms_ngfit # this is for sgfit : rms
                            gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+0] = _logz # this is for sgfit: log-Z
                        else:
                            gfit_results[j][l][2*(3*_max_ngauss+2)+l] = 0 # put a blank value
                            gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+0] = -1E11 # put a blank value

                        gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+1] = _is
                        gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+2] = _ie
                        gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+3] = _js
                        gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+4] = _je
                        gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+5] = i
                        gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+6] = _js + j

                    #________________________________________________________________________________________|
                    #|---------------------------------------------------------------------------------------|
                    # unit conversion
                    # sigma-flux --> data cube units
                    gfit_results[j][k][0] = gfit_results[j][k][0]*(_f_max - _f_min) # sigma-flux
                    # background --> data cube units
                    gfit_results[j][k][1] = gfit_results[j][k][1]*(_f_max - _f_min) + _f_min # bg-flux
                    gfit_results[j][k][6 + 3*k] = gfit_results[j][k][6 + 3*k]*(_f_max - _f_min) # bg-flux-e
                    _bg_flux = gfit_results[j][k][1]
        
                    for m in range(0, k+1):
                        #________________________________________________________________________________________|
                        # UNIT CONVERSION
                        #________________________________________________________________________________________|
                        # velocity, velocity-dispersion --> km/s
                        if _cdelt3 > 0: # if velocity axis is with increasing order
                            gfit_results[j][k][2 + 3*m] = gfit_results[j][k][2 + 3*m]*(_vel_max - _vel_min) + _vel_min # velocity
                        elif _cdelt3 < 0: # if velocity axis is with decreasing order
                            gfit_results[j][k][2 + 3*m] = gfit_results[j][k][2 + 3*m]*(_vel_min - _vel_max) + _vel_max # velocity

                        gfit_results[j][k][3 + 3*m] = gfit_results[j][k][3 + 3*m]*(_vel_max - _vel_min) # velocity-dispersion

                        #________________________________________________________________________________________|
                        # peak flux --> data cube units
                        gfit_results[j][k][4 + 3*m] = gfit_results[j][k][4 + 3*m]*(_f_max - _f_min) # peak flux
        
                        #________________________________________________________________________________________|
                        # velocity-e, velocity-dispersion-e --> km/s
                        gfit_results[j][k][7 + 3*(m+k)] = gfit_results[j][k][7 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-e
                        gfit_results[j][k][8 + 3*(m+k)] = gfit_results[j][k][8 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-dispersion-e
                        #________________________________________________________________________________________|
                        gfit_results[j][k][9 + 3*(m+k)] = gfit_results[j][k][9 + 3*(m+k)]*(_f_max - _f_min) # flux-e

                    # lastly put rms 
                    gfit_results[j][k][2*(3*_max_ngauss+2)+k] = gfit_results[j][k][2*(3*_max_ngauss+2)+k]*(_f_max - _f_min) # rms-(k+1)gfit
                    #________________________________________________________________________________________|
                    #|---------------------------------------------------------------------------------------|
                    continue
            #---------------------------------------------------------


            # update optimal priors based on the current ngaussian fit results
            if ngauss < _max_ngauss:
                nparams_n = 3*(ngauss+1) + 2
                gfit_priors_init = np.zeros(2*nparams_n, dtype=np.float32)
                # lower bound : the parameters for the current ngaussian components
                # nsigma_prior_range_gfit=3.0 (default)
                gfit_priors_init[:nparams] = _gfit_results_temp[:nparams] - _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]
                # upper bound : the parameters for the current ngaussian components
                gfit_priors_init[nparams+3:2*nparams+3] = _gfit_results_temp[:nparams] + _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]
    
                # the parameters for the next gaussian component: based on the current ngaussians
                _x_min_t = _gfit_results_temp[2:nparams:3].min()
                _x_max_t = _gfit_results_temp[2:nparams:3].max()
                _std_min_t = _gfit_results_temp[3:nparams:3].min()
                _std_max_t = _gfit_results_temp[3:nparams:3].max()
                _p_min_t = _gfit_results_temp[4:nparams:3].min()
                _p_max_t = _gfit_results_temp[4:nparams:3].max()

                # sigma_prior_lowerbound_factor=0.2 (default), sigma_prior_upperbound_factor=2.0 (default)
                gfit_priors_init[0] = _params['sigma_prior_lowerbound_factor']*_gfit_results_temp[0]
                gfit_priors_init[nparams_n] = _params['sigma_prior_upperbound_factor']*_gfit_results_temp[0]

                # bg_prior_lowerbound_factor=0.2 (defaut), bg_prior_upperbound_factor=2.0 (default)
                gfit_priors_init[1] = _params['bg_prior_lowerbound_factor']*_gfit_results_temp[1]
                gfit_priors_init[nparams_n + 1] = _params['bg_prior_upperbound_factor']*_gfit_results_temp[1]

                #print("x:", _x_min_t, _x_max_t, "std:", _std_min_t, _std_max_t, "p:",_p_min_t, _p_max_t)

                #____________________________________________
                # x: lower bound
                if ngauss == 1:
                    # x_lowerbound_gfit=0.1 (default), x_upperbound_gfit=0.9 (default)
                    gfit_priors_init[nparams] = _params['x_lowerbound_gfit']
                    gfit_priors_init[2*nparams+3] = _params['x_upperbound_gfit']
                    #if gfit_priors_init[nparams] < 0 : gfit_priors_init[nparams] = 0
                else:
                    # x_prior_lowerbound_factor=5 (default), x_prior_upperbound_factor=5 (default)
                    gfit_priors_init[nparams] = _x_min_t - _params['x_prior_lowerbound_factor']*_std_max_t
                    gfit_priors_init[2*nparams+3] = _x_max_t + _params['x_prior_upperbound_factor']*_std_max_t
                    #if gfit_priors_init[2*nparams+3] > 1 : gfit_priors_init[2*nparams+3] = 1

                #____________________________________________
                # std: lower bound
                # std_prior_lowerbound_factor=0.1 (default)
                gfit_priors_init[nparams+1] = _params['std_prior_lowerbound_factor']*_std_min_t
                #gfit_priors_init[nparams+1] = 0.01
                #if gfit_priors_init[nparams+1] < 0 : gfit_priors_init[nparams+1] = 0
                # std: upper bound
                # std_prior_upperbound_factor=3.0 (default)
                gfit_priors_init[2*nparams+4] = _params['std_prior_upperbound_factor']*_std_max_t
                #gfit_priors_init[2*nparams+4] = 0.9
                #if gfit_priors_init[2*nparams+4] > 1 : gfit_priors_init[2*nparams+4] = 1
    
                #____________________________________________
                # p: lower bound
                # p_prior_lowerbound_factor=0.05 (default)
                gfit_priors_init[nparams+2] = _params['p_prior_lowerbound_factor']*_p_max_t # 5% of the maxium flux
                # p: upper bound
                # p_prior_upperbound_factor=1.0 (default)
                gfit_priors_init[2*nparams+5] = _params['p_prior_upperbound_factor']*_p_max_t

                gfit_priors_init = np.where(gfit_priors_init<0, 0, gfit_priors_init)
                gfit_priors_init = np.where(gfit_priors_init>1, 1, gfit_priors_init)


            gfit_results[j][k][2*(3*_max_ngauss+2)+k] = _rms_ngfit # rms_(k+1)gfit
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+0] = _logz
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+1] = _is
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+2] = _ie
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+3] = _js
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+4] = _je
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+5] = i
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+6] = _js + j
            #print(gfit_results[j][k])

            #|-----------------------------------------|
            #|-----------------------------------------|
            # example: 3 gaussians : bg + 3 * (x, std, peak) 
            #  ______________________________________  |
            # |_G1___________________________________| |
            # |_000000000000000000000000000000000000_| |
            # |_000000000000000000000000000000000000_| |
            # |_G1-rms : 0 : 0 : log-Z : xs-xe-ys-ye_| |
            #  ______________________________________  |
            # |_G1___________________________________| |
            # |_G2___________________________________| |
            # |_000000000000000000000000000000000000_| |
            # |_0 : G2-rms : 0 : log-Z : xs-xe-ys-ye_| |
            #  ______________________________________  |
            # |_G1___________________________________| |
            # |_G2___________________________________| |
            # |_G3___________________________________| |
            # |_0 : 0 : G3-rms : log-Z : xs-xe-ys-ye_| |

            #|-----------------------------------------|
            #gfit_results[j][k][0] : dist-sig
            #gfit_results[j][k][1] : bg
            #gfit_results[j][k][2] : g1-x --> *(vel_max-vel_min) + vel_min
            #gfit_results[j][k][3] : g1-s --> *(vel_max-vel_min)
            #gfit_results[j][k][4] : g1-p
            #gfit_results[j][k][5] : g2-x --> *(vel_max-vel_min) + vel_min
            #gfit_results[j][k][6] : g2-s --> *(vel_max-vel_min)
            #gfit_results[j][k][7] : g2-p
            #gfit_results[j][k][8] : g3-x --> *(vel_max-vel_min) + vel_min
            #gfit_results[j][k][9] : g3-s --> *(vel_max-vel_min)
            #gfit_results[j][k][10] : g3-p

            #gfit_results[j][k][11] : dist-sig-e
            #gfit_results[j][k][12] : bg-e
            #gfit_results[j][k][13] : g1-x-e --> *(vel_max-vel_min)
            #gfit_results[j][k][14] : g1-s-e --> *(vel_max-vel_min)
            #gfit_results[j][k][15] : g1-p-e
            #gfit_results[j][k][16] : g2-x-e --> *(vel_max-vel_min)
            #gfit_results[j][k][17] : g2-s-e --> *(vel_max-vel_min)
            #gfit_results[j][k][18] : g2-p-e
            #gfit_results[j][k][19] : g3-x-e --> *(vel_max-vel_min)
            #gfit_results[j][k][20] : g3-s-e --> *(vel_max-vel_min)
            #gfit_results[j][k][21] : g3-p-e --> *(f_max-bg_flux)

            #gfit_results[j][k][22] : g1-rms --> *(f_max-bg_flux) : the bg rms for the case with single gaussian fitting
            #gfit_results[j][k][23] : g2-rms --> *(f_max-bg_flux) : the bg rms for the case with double gaussian fitting
            #gfit_results[j][k][24] : g3-rms --> *(f_max-bg_flux) : the bg rms for the case with triple gaussian fitting

            #gfit_results[j][k][25] : log-Z : log-evidence : log-marginalization likelihood

            #gfit_results[j][k][26] : xs
            #gfit_results[j][k][27] : xe
            #gfit_results[j][k][28] : ys
            #gfit_results[j][k][29] : ye
            #gfit_results[j][k][30] : x
            #gfit_results[j][k][31] : y
            #|-----------------------------------------|

            #________________________________________________________________________________________|
            #|---------------------------------------------------------------------------------------|
            # UNIT CONVERSION
            # sigma-flux --> data cube units
            gfit_results[j][k][0] = gfit_results[j][k][0]*(_f_max - _f_min) # sigma-flux
            # background --> data cube units
            gfit_results[j][k][1] = gfit_results[j][k][1]*(_f_max - _f_min) + _f_min # bg-flux
            gfit_results[j][k][6 + 3*k] = gfit_results[j][k][6 + 3*k]*(_f_max - _f_min) # bg-flux-e
            _bg_flux = gfit_results[j][k][1]

            for m in range(0, k+1):
                #________________________________________________________________________________________|
                # UNIT CONVERSION

                #________________________________________________________________________________________|
                # velocity, velocity-dispersion --> km/s
                if _cdelt3 > 0: # if velocity axis is with increasing order
                    gfit_results[j][k][2 + 3*m] = gfit_results[j][k][2 + 3*m]*(_vel_max - _vel_min) + _vel_min # velocity
                elif _cdelt3 < 0: # if velocity axis is with decreasing order
                    gfit_results[j][k][2 + 3*m] = gfit_results[j][k][2 + 3*m]*(_vel_min - _vel_max) + _vel_max # velocity

                gfit_results[j][k][3 + 3*m] = gfit_results[j][k][3 + 3*m]*(_vel_max - _vel_min) # velocity-dispersion

                #________________________________________________________________________________________|
                # peak flux --> data cube units : (_f_max - _f_min) should be used for scaling 
                gfit_results[j][k][4 + 3*m] = gfit_results[j][k][4 + 3*m]*(_f_max - _f_min) # peak flux

                #________________________________________________________________________________________|
                # velocity-e, velocity-dispersion-e --> km/s
                gfit_results[j][k][7 + 3*(m+k)] = gfit_results[j][k][7 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-e
                gfit_results[j][k][8 + 3*(m+k)] = gfit_results[j][k][8 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-dispersion-e

                gfit_results[j][k][9 + 3*(m+k)] = gfit_results[j][k][9 + 3*(m+k)]*(_f_max - _f_min) # flux-e

            # lastly put rms 
            #________________________________________________________________________________________|
            gfit_results[j][k][2*(3*_max_ngauss+2)+k] = gfit_results[j][k][2*(3*_max_ngauss+2)+k]*(_f_max - _f_min) # rms-(k+1)gfit
            #________________________________________________________________________________________|
            #|---------------------------------------------------------------------------------------|


            print(gfit_results)

    #del(_gfit_results_temp, gfit_priors_init)
    #gc.collect()


    return gfit_results

    # Plot a summary of the run.
#    rfig, raxes = dyplot.runplot(sampler.results)
#    rfig.savefig("r.pdf")
    
    # Plot traces and 1-D marginalized posteriors.
#    tfig, taxes = dyplot.traceplot(sampler.results)
#    tfig.savefig("t.pdf")
    
    # Plot the 2-D marginalized posteriors.
    #cfig, caxes = dyplot.cornerplot(sampler.results)
    #cfig.savefig("c.pdf")
#-- END OF SUB-ROUTINE____________________________________________________________#

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# run dynesty for each line profile
#@jit(nopython=True)
#@ray.remote(num_cpus=1)
#@ray.remote
def run_dynesty_sampler_optimal_priors_org(_inputDataCube, _x, _peak_sn_map, _sn_int_map, _params, _is, _ie, i, _js, _je):

    _max_ngauss = _params['max_ngauss']
    _vel_min = _params['vel_min']
    _vel_max = _params['vel_max']
    gfit_results = np.zeros(((_je-_js), _max_ngauss, 2*(2+3*_max_ngauss) + 7 + _max_ngauss), dtype=np.float32)
    _x_boundaries = np.full(2*_max_ngauss, fill_value=-1E11, dtype=np.float32)

    #print("CHECK S/N: %d %d | peak S/N: %.1f < %.1f | integrated S/N: %.1f < %.1f" \
    #    % (i, 0+_js, _peak_sn_map[0+_js, i], _params['peak_sn_limit'], _sn_int_map[0+_js, i], _params['int_sn_limit']))
    for j in range(0, _je -_js):

        _f_max = np.max(_inputDataCube[:,j+_js,i]) # peak flux : being used for normalization
        _f_min = np.min(_inputDataCube[:,j+_js,i]) # lowest flux : being used for normalization
        #print(_f_max, _f_min)

        # prior arrays for the 1st single Gaussian fit
        gfit_priors_init = np.zeros(2*5, dtype=np.float32)
        #gfit_priors_init = [sig1, bg1, x1, std1, p1, sig2, bg2, x2, std2, p2]
        gfit_priors_init = [0.0, 0.0, 0.01, 0.01, 0.01, 0.5, 0.6, 0.99, 0.6, 1.01]

        if _sn_int_map[j+_js, i] < _params['int_sn_limit'] or _peak_sn_map[j+_js, i] < _params['peak_sn_limit'] \
            or np.isnan(_f_max) or np.isnan(_f_min) \
            or np.isinf(_f_min) or np.isinf(_f_min):

            print("low S/N: %d %d | peak S/N: %.1f :: S/N limit: %.1f | integrated S/N: %.1f :: S/N limit: %.1f :: f_min: %e :: f_max: %e" \
                % (i, j+_js, _peak_sn_map[j+_js, i], _params['peak_sn_limit'], _sn_int_map[j+_js, i], _params['int_sn_limit'], _f_min, _f_max))
            # save the current profile location
            for l in range(0, _max_ngauss):
                gfit_results[j][l][2*(3*_max_ngauss+2)+l] = _params['_rms_med'] # rms: the one derived from derive_rms_npoints_sgfit
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+0] = -1E11 # this is for sgfit: log-Z
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+1] = _is
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+2] = _ie
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+3] = _js
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+4] = _je
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+5] = i
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+6] = j + _js
            continue

        for k in range(0, _max_ngauss):
            ngauss = k+1  # set the number of gaussian
            ndim = 3*ngauss + 2
            nparams = ndim

            if(ndim * (ndim + 1) // 2 > _params['nlive']):
                _params['nlive'] = 1 + ndim * (ndim + 1) // 2 # optimal minimum nlive

            print("processing: %d %d | peak s/n: %.1f | integrated s/n: %.1f | gauss-%d" % (i, j+_js, _peak_sn_map[j+_js, i], _sn_int_map[j+_js, i], ngauss))

            #sampler = NestedSampler(loglike_d, uniform_prior_d, ndim, sample='unif',
            #    vol_dec = 0.2, vol_check = 2, facc=0.5, rwalk=1000, nlive=200, max_move=100,
            #    logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss])

            #---------------------------------------------------------
            # run dynesty 1.1
            #sampler = NestedSampler(loglike_d, optimal_prior, ndim,
            #    vol_dec=_params['vol_dec'],
            #    vol_check=_params['vol_check'],
            #    facc=_params['facc'],
            #    sample=_params['sample'],
            #    nlive=_params['nlive'],
            #    bound=_params['bound'],
            #    #rwalk=_params['rwalk'],
            #    max_move=_params['max_move'],
            #    logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])

            if _params['_dynesty_class_'] == 'static':
                #---------------------------------------------------------
                # run dynesty 2.0.3
                sampler = NestedSampler(loglike_d, optimal_prior, ndim,
                    nlive=_params['nlive'],
                    update_interval=_params['update_interval'],
                    sample=_params['sample'],
                    bound=_params['bound'],
                    facc=_params['facc'],
                    fmove=_params['fmove'],
                    max_move=_params['max_move'],
                    logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])
                sampler.run_nested(dlogz=_params['dlogz_2d'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=_params['print_progress'])
                #numba.jit(sampler.run_nested(dlogz=_params['dlogz'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=False), nopython=True, cache=True, nogil=True, parallel=True)
                #_run_nested = jit(sampler.run_nested(dlogz=_params['dlogz'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=False), nopython=True, cache=True, nogil=True, fastmath=True)

            elif _params['_dynesty_class_'] == 'dynamic':
                #---------------------------------------------------------
                # run dynesty 2.0.3
                dsampler = DynamicNestedSampler(loglike_d, optimal_prior, ndim,
                    nlive=_params['nlive'],
                    sample=_params['sample'],
                    update_interval=_params['update_interval'],
                    bound=_params['bound'],
                    facc=_params['facc'],
                    fmove=_params['fmove'],
                    max_move=_params['max_move'],
                    logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])
                dsampler.run_nested(dlogz_init=_params['dlogz_2d'], maxiter_init=_params['maxiter'], maxcall_init=_params['maxcall'], print_progress=_params['print_progress'])

                #sampler.reset()
                #numba.jit(sampler.run_nested(dlogz=0.1, maxiter=5000000, maxcall=50000000, print_progress=False), nopython=True, cache=True, nogil=True, parallel=True)

            #---------------------------------------------------------
            _gfit_results_temp, _logz = get_dynesty_sampler_results(sampler)

            #---------------------------------------------------------
            # param1, param2, param3 ....param1-e, param2-e, param3-e
            #gfit_results[j][k][0~2*nparams] = _gfit_results_temp[0~2*nparams]
            gfit_results[j][k][:2*nparams] = _gfit_results_temp
            #---------------------------------------------------------

            #---------------------------------------------------------
            # derive rms of the profile given the current ngfit
            _rms_ngfit = little_derive_rms_npoints(_inputDataCube, i, j+_js, _x, _f_min, _f_max, ngauss, _gfit_results_temp)
            #---------------------------------------------------------

            #---------------------------------------------------------
            if ngauss == 1: # check the peak s/n
                # load the normalised sgfit results : --> derive rms for s/n
                #_bg_sgfit = _gfit_results_temp[1]
                #_x_sgfit = _gfit_results_temp[2]
                #_std_sgfit = _gfit_results_temp[3]
                #_p_sgfit = _gfit_results_temp[4]
                # peak flux of the sgfit
                #_f_sgfit =_p_sgfit * exp( -0.5*((_x - _x_sgfit) / _std_sgfit)**2) + _bg_sgfit

                #---------------------------------------------------------
                # update gfit_priors_init
                nparams_n = 3*(ngauss+1) + 2
                gfit_priors_init = np.zeros(2*nparams_n, dtype=np.float32)
                # lower bound : the parameters for the current ngaussian components
                # nsigma_prior_range_gfit=3.0 (default)
                gfit_priors_init[:nparams] = _gfit_results_temp[:nparams] - _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]
                # upper bound : the parameters for the current ngaussian components
                gfit_priors_init[nparams+3:2*nparams+3] = _gfit_results_temp[:nparams] + _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]
                #---------------------------------------------------------


                # peak s/n : more accurate peak s/n from the first sgfit
                _bg_sgfit = _gfit_results_temp[1]
                _p_sgfit = _gfit_results_temp[4] # bg already subtracted
                _peak_sn_sgfit = _p_sgfit/_rms_ngfit

                if _peak_sn_sgfit < _params['peak_sn_limit']: 
                    print("skip the rest of Gaussian fits: %d %d | rms:%.1f | bg:%.1f | peak:%.1f | peak_sgfit s/n: %.1f < %.1f" % (i, j+_js, _rms_ngfit, _bg_sgfit, _p_sgfit, _peak_sn_sgfit, _params['peak_sn_limit']))

                    # save the current profile location
                    for l in range(0, _max_ngauss):
                        if l == 0:
                        # for sgfit
                            gfit_results[j][l][2*(3*_max_ngauss+2)+l] = _rms_ngfit # this is for sgfit : rms
                            gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+0] = _logz # this is for sgfit: log-Z
                        else:
                            gfit_results[j][l][2*(3*_max_ngauss+2)+l] = 0 # put a blank value
                            gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+0] = -1E11 # put a blank value

                        gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+1] = _is
                        gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+2] = _ie
                        gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+3] = _js
                        gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+4] = _je
                        gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+5] = i
                        gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+6] = _js + j

                    #________________________________________________________________________________________|
                    #|---------------------------------------------------------------------------------------|
                    # unit conversion
                    # sigma-flux --> data cube units
                    gfit_results[j][k][0] = gfit_results[j][k][0]*(_f_max - _f_min) # sigma-flux
                    # background --> data cube units
                    gfit_results[j][k][1] = gfit_results[j][k][1]*(_f_max - _f_min) + _f_min # bg-flux
                    gfit_results[j][k][6 + 3*k] = gfit_results[j][k][6 + 3*k]*(_f_max - _f_min) # bg-flux-e
                    _bg_flux = gfit_results[j][k][1]
        
                    for m in range(0, k+1):
                        # unit conversion
                        # peak flux --> data cube units
                        # velocity, velocity-dispersion --> km/s
                        gfit_results[j][k][2 + 3*m] = gfit_results[j][k][2 + 3*m]*(_vel_max - _vel_min) + _vel_min # velocity
                        gfit_results[j][k][3 + 3*m] = gfit_results[j][k][3 + 3*m]*(_vel_max - _vel_min) # velocity-dispersion
                        gfit_results[j][k][4 + 3*m] = gfit_results[j][k][4 + 3*m]*(_f_max - _f_min) # flux
        
                        gfit_results[j][k][7 + 3*(m+k)] = gfit_results[j][k][7 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-e
                        gfit_results[j][k][8 + 3*(m+k)] = gfit_results[j][k][8 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-dispersion-e
                        gfit_results[j][k][9 + 3*(m+k)] = gfit_results[j][k][9 + 3*(m+k)]*(_f_max - _f_min) # flux-e

                    # lastly put rms 
                    gfit_results[j][k][2*(3*_max_ngauss+2)+k] = gfit_results[j][k][2*(3*_max_ngauss+2)+k]*(_f_max - _f_min) # rms-(k+1)gfit
                    #________________________________________________________________________________________|
                    #|---------------------------------------------------------------------------------------|
                    continue
            #---------------------------------------------------------


            # update optimal priors based on the current ngaussian fit results
            if ngauss < _max_ngauss:
                nparams_n = 3*(ngauss+1) + 2
                gfit_priors_init = np.zeros(2*nparams_n, dtype=np.float32)
                # lower bound : the parameters for the current ngaussian components
                # nsigma_prior_range_gfit=3.0 (default)
                gfit_priors_init[:nparams] = _gfit_results_temp[:nparams] - _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]
                # upper bound : the parameters for the current ngaussian components
                gfit_priors_init[nparams+3:2*nparams+3] = _gfit_results_temp[:nparams] + _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]
    
                # the parameters for the next gaussian component: based on the current ngaussians
                _x_min_t = _gfit_results_temp[2:nparams:3].min()
                _x_max_t = _gfit_results_temp[2:nparams:3].max()
                _std_min_t = _gfit_results_temp[3:nparams:3].min()
                _std_max_t = _gfit_results_temp[3:nparams:3].max()
                _p_min_t = _gfit_results_temp[4:nparams:3].min()
                _p_max_t = _gfit_results_temp[4:nparams:3].max()

                # sigma_prior_lowerbound_factor=0.2 (default), sigma_prior_upperbound_factor=2.0 (default)
                gfit_priors_init[0] = _params['sigma_prior_lowerbound_factor']*_gfit_results_temp[0]
                gfit_priors_init[nparams_n] = _params['sigma_prior_upperbound_factor']*_gfit_results_temp[0]

                # bg_prior_lowerbound_factor=0.2 (defaut), bg_prior_upperbound_factor=2.0 (default)
                gfit_priors_init[1] = _params['bg_prior_lowerbound_factor']*_gfit_results_temp[1]
                gfit_priors_init[nparams_n + 1] = _params['bg_prior_upperbound_factor']*_gfit_results_temp[1]

                #print("x:", _x_min_t, _x_max_t, "std:", _std_min_t, _std_max_t, "p:",_p_min_t, _p_max_t)

                #____________________________________________
                # x: lower bound
                if ngauss == 1:
                    # x_lowerbound_gfit=0.1 (default), x_upperbound_gfit=0.9 (default)
                    gfit_priors_init[nparams] = _params['x_lowerbound_gfit']
                    gfit_priors_init[2*nparams+3] = _params['x_upperbound_gfit']
                    #if gfit_priors_init[nparams] < 0 : gfit_priors_init[nparams] = 0
                else:
                    # x_prior_lowerbound_factor=5 (default), x_prior_upperbound_factor=5 (default)
                    gfit_priors_init[nparams] = _x_min_t - _params['x_prior_lowerbound_factor']*_std_max_t
                    gfit_priors_init[2*nparams+3] = _x_max_t + _params['x_prior_upperbound_factor']*_std_max_t
                    #if gfit_priors_init[2*nparams+3] > 1 : gfit_priors_init[2*nparams+3] = 1

                #____________________________________________
                # std: lower bound
                # std_prior_lowerbound_factor=0.1 (default)
                gfit_priors_init[nparams+1] = _params['std_prior_lowerbound_factor']*_std_min_t
                #gfit_priors_init[nparams+1] = 0.01
                #if gfit_priors_init[nparams+1] < 0 : gfit_priors_init[nparams+1] = 0
                # std: upper bound
                # std_prior_upperbound_factor=3.0 (default)
                gfit_priors_init[2*nparams+4] = _params['std_prior_upperbound_factor']*_std_max_t
                #gfit_priors_init[2*nparams+4] = 0.9
                #if gfit_priors_init[2*nparams+4] > 1 : gfit_priors_init[2*nparams+4] = 1
    
                #____________________________________________
                # p: lower bound
                # p_prior_lowerbound_factor=0.05 (default)
                gfit_priors_init[nparams+2] = _params['p_prior_lowerbound_factor']*_p_max_t # 5% of the maxium flux
                # p: upper bound
                # p_prior_upperbound_factor=1.0 (default)
                gfit_priors_init[2*nparams+5] = _params['p_prior_upperbound_factor']*_p_max_t

                gfit_priors_init = np.where(gfit_priors_init<0, 0, gfit_priors_init)
                gfit_priors_init = np.where(gfit_priors_init>1, 1, gfit_priors_init)


            gfit_results[j][k][2*(3*_max_ngauss+2)+k] = _rms_ngfit # rms_(k+1)gfit
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+0] = _logz
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+1] = _is
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+2] = _ie
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+3] = _js
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+4] = _je
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+5] = i
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+6] = _js + j
            #print(gfit_results[j][k])

            #|-----------------------------------------|
            # example: 3 gaussians : bg + 3 * (x, std, peak) 
            #gfit_results[j][k][0] : dist-sig
            #gfit_results[j][k][1] : bg
            #gfit_results[j][k][2] : g1-x --> *(vel_max-vel_min) + vel_min
            #gfit_results[j][k][3] : g1-s --> *(vel_max-vel_min)
            #gfit_results[j][k][4] : g1-p
            #gfit_results[j][k][5] : g2-x --> *(vel_max-vel_min) + vel_min
            #gfit_results[j][k][6] : g2-s --> *(vel_max-vel_min)
            #gfit_results[j][k][7] : g2-p
            #gfit_results[j][k][8] : g3-x --> *(vel_max-vel_min) + vel_min
            #gfit_results[j][k][9] : g3-s --> *(vel_max-vel_min)
            #gfit_results[j][k][10] : g3-p

            #gfit_results[j][k][11] : dist-sig-e
            #gfit_results[j][k][12] : bg-e
            #gfit_results[j][k][13] : g1-x-e --> *(vel_max-vel_min)
            #gfit_results[j][k][14] : g1-s-e --> *(vel_max-vel_min)
            #gfit_results[j][k][15] : g1-p-e
            #gfit_results[j][k][16] : g2-x-e --> *(vel_max-vel_min)
            #gfit_results[j][k][17] : g2-s-e --> *(vel_max-vel_min)
            #gfit_results[j][k][18] : g2-p-e
            #gfit_results[j][k][19] : g3-x-e --> *(vel_max-vel_min)
            #gfit_results[j][k][20] : g3-s-e --> *(vel_max-vel_min)
            #gfit_results[j][k][21] : g3-p-e --> *(f_max-bg_flux)

            #gfit_results[j][k][22] : g1-rms --> *(f_max-bg_flux)
            #gfit_results[j][k][23] : g2-rms --> *(f_max-bg_flux)
            #gfit_results[j][k][24] : g3-rms --> *(f_max-bg_flux)

            #gfit_results[j][k][25] : log-Z : log-evidence : log-marginalization likelihood

            #gfit_results[j][k][26] : xs
            #gfit_results[j][k][27] : xe
            #gfit_results[j][k][28] : ys
            #gfit_results[j][k][29] : ye
            #gfit_results[j][k][30] : x
            #gfit_results[j][k][31] : y
            #|-----------------------------------------|

            #________________________________________________________________________________________|
            #|---------------------------------------------------------------------------------------|
            # unit conversion
            # sigma-flux --> data cube units
            gfit_results[j][k][0] = gfit_results[j][k][0]*(_f_max - _f_min) # sigma-flux
            # background --> data cube units
            gfit_results[j][k][1] = gfit_results[j][k][1]*(_f_max - _f_min) + _f_min # bg-flux
            gfit_results[j][k][6 + 3*k] = gfit_results[j][k][6 + 3*k]*(_f_max - _f_min) # bg-flux-e
            _bg_flux = gfit_results[j][k][1]

            for m in range(0, k+1):
                # unit conversion
                # peak flux --> data cube units
                # velocity, velocity-dispersion --> km/s
                gfit_results[j][k][2 + 3*m] = gfit_results[j][k][2 + 3*m]*(_vel_max - _vel_min) + _vel_min # velocity
                gfit_results[j][k][3 + 3*m] = gfit_results[j][k][3 + 3*m]*(_vel_max - _vel_min) # velocity-dispersion
                gfit_results[j][k][4 + 3*m] = gfit_results[j][k][4 + 3*m]*(_f_max - _f_min)  # peak flux

                gfit_results[j][k][7 + 3*(m+k)] = gfit_results[j][k][7 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-e
                gfit_results[j][k][8 + 3*(m+k)] = gfit_results[j][k][8 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-dispersion-e
                gfit_results[j][k][9 + 3*(m+k)] = gfit_results[j][k][9 + 3*(m+k)]*(_f_max - _f_min) # peak flux-e

            gfit_results[j][k][2*(3*_max_ngauss+2)+k] = gfit_results[j][k][2*(3*_max_ngauss+2)+k]*(_f_max - _f_min) # rms-(k+1)gfit
            #________________________________________________________________________________________|
            #|---------------------------------------------------------------------------------------|

    #del(_gfit_results_temp, gfit_priors_init)
    #gc.collect()

    return gfit_results

    # Plot a summary of the run.
#    rfig, raxes = dyplot.runplot(sampler.results)
#    rfig.savefig("r.pdf")
    
    # Plot traces and 1-D marginalized posteriors.
#    tfig, taxes = dyplot.traceplot(sampler.results)
#    tfig.savefig("t.pdf")
    
    # Plot the 2-D marginalized posteriors.
    #cfig, caxes = dyplot.cornerplot(sampler.results)
    #cfig.savefig("c.pdf")
#-- END OF SUB-ROUTINE____________________________________________________________#

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# run dynesty for each line profile
#@ray.remote(num_cpus=1)
#@ray.remote
def run_dynesty_sampler_uniform_priors(_x, _inputDataCube, _is, _ie, i, _js, _je, _max_ngauss, _vel_min, _vel_max):

    gfit_results = np.zeros(((_je-_js), _max_ngauss, 2*(2+3*_max_ngauss)+7), dtype=np.float32)
    for j in range(0, _je -_js):

        _f_max = np.max(_inputDataCube[:,j+_js,i]) # peak flux : being used for normalization
        _f_min = np.min(_inputDataCube[:,j+_js,i]) # lowest flux : being used for normalization

        #print(_f_max, _f_min)
        gfit_priors_init = np.zeros(2*5, dtype=np.float32)
        #gfit_priors_init = [sig1, bg1, x1, std1, p1, sig2, bg2, x2, std2, p2]
        gfit_priors_init = [0.0, 0.0, 0.01, 0.01, 0.01, 0.5, 0.5, 0.9, 0.6, 1.01]
        for k in range(0, _max_ngauss):
            ngauss = k+1  # set the number of gaussian
            ndim = 3*ngauss + 2
            nparams = ndim

#            if(ndim * (ndim + 1) // 2 > 100):
#                _nlive = 1 + ndim * (ndim + 1) // 2 # optimal minimum nlive
#            else:
#                _nlive = 100
    
            # run dynesty
            print("processing: %d %d gauss-%d" % (i, j+_js, ngauss))

            #sampler = NestedSampler(loglike_d, uniform_prior_d, ndim, sample='unif',
            #    vol_dec = 0.2, vol_check = 2, facc=0.5, rwalk=1000, nlive=200, max_move=100,
            #    logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss])

            #sampler = NestedSampler(loglike_d, uniform_prior, ndim,
            #    vol_dec = 0.2, vol_check = 2, facc=0.5, sample='hslice', nlive=100, bound='multi', rwalk=1000, max_move=100,
            #    logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])

            # run dynesty 1.1   
            #sampler = NestedSampler(loglike_d, uniform_prior, ndim,
            #    vol_dec = 0.2, vol_check = 2, facc=0.5, sample='auto', nlive=100, bound='multi', max_move=100,
            #    logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])
#
            if _params['_dynesty_class_'] == 'static':
                #---------------------------------------------------------
                # run dynesty 2.0.3
                sampler = NestedSampler(loglike_d, uniform_prior, ndim,
                    nlive=_params['nlive'],
                    update_interval=_params['update_interval'],
                    sample=_params['sample'],
                    bound=_params['bound'],
                    facc=_params['facc'],
                    fmove=_params['fmove'],
                    max_move=_params['max_move'],
                    logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])
                sampler.run_nested(dlogz=_params['dlogz_2d'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=_params['print_progress_tr'])
                #numba.jit(sampler.run_nested(dlogz=_params['dlogz'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=False), nopython=True, cache=True, nogil=True, parallel=True)
                #_run_nested = jit(sampler.run_nested(dlogz=_params['dlogz'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=False), nopython=True, cache=True, nogil=True, fastmath=True)

            elif _params['_dynesty_class_'] == 'dynamic':
                #---------------------------------------------------------
                # run dynesty 2.0.3
                dsampler = DynamicNestedSampler(loglike_d, uniform_prior, ndim,
                    nlive=_params['nlive'],
                    sample=_params['sample'],
                    update_interval=_params['update_interval'],
                    bound=_params['bound'],
                    facc=_params['facc'],
                    fmove=_params['fmove'],
                    max_move=_params['max_move'],
                    logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])
                dsampler.run_nested(dlogz_init=_params['dlogz_2d'], maxiter_init=_params['maxiter'], maxcall_init=_params['maxcall'], print_progress=_params['print_progress'])

                #sampler.reset()
                #numba.jit(sampler.run_nested(dlogz=0.1, maxiter=5000000, maxcall=50000000, print_progress=False), nopython=True, cache=True, nogil=True, parallel=True)
            #---------------------------------------------------------
            _gfit_results_temp, _logz = get_dynesty_sampler_results(sampler)

            # param1, param2, param3 ....param1-e, param2-e, param3-e
            #gfit_results[j][k][0~2*nparams] = _gfit_results_temp[0~2*nparams]
            gfit_results[j][k][:2*nparams] = _gfit_results_temp

            print(gfit_priors_init)

            gfit_results[j][k][2*(3*_max_ngauss+2)+0] = _logz
            gfit_results[j][k][2*(3*_max_ngauss+2)+1] = _is
            gfit_results[j][k][2*(3*_max_ngauss+2)+2] = _ie
            gfit_results[j][k][2*(3*_max_ngauss+2)+3] = _js
            gfit_results[j][k][2*(3*_max_ngauss+2)+4] = _je
            gfit_results[j][k][2*(3*_max_ngauss+2)+5] = i
            gfit_results[j][k][2*(3*_max_ngauss+2)+6] = _js + j
            #print(gfit_results[j][k])

            #|-----------------------------------------|
            # example: 3 gaussians : bg + 3 * (x, std, peak) 
            #gfit_results[j][k][0] : dist-sig
            #gfit_results[j][k][1] : bg
            #gfit_results[j][k][2] : g1-x1 --> *(vel_max-vel_min) + vel_min
            #gfit_results[j][k][3] : g1-s1 --> *(vel_max-vel_min)
            #gfit_results[j][k][4] : g1-p1
            #gfit_results[j][k][5] : g2-x2 --> *(vel_max-vel_min) + vel_min
            #gfit_results[j][k][6] : g2-s2 --> *(vel_max-vel_min)
            #gfit_results[j][k][7] : g2-p2
            #gfit_results[j][k][8] : g3-x3 --> *(vel_max-vel_min) + vel_min
            #gfit_results[j][k][9] : g3-s3 --> *(vel_max-vel_min)
            #gfit_results[j][k][10] : g3-p3

            #gfit_results[j][k][11] : dist-sig-e
            #gfit_results[j][k][12] : bg-e
            #gfit_results[j][k][13] : g1-x1-e --> *(vel_max-vel_min)
            #gfit_results[j][k][14] : g1-s1-e --> *(vel_max-vel_min)
            #gfit_results[j][k][15] : g1-p1-e
            #gfit_results[j][k][16] : g2-x2-e --> *(vel_max-vel_min)
            #gfit_results[j][k][17] : g2-s2-e --> *(vel_max-vel_min)
            #gfit_results[j][k][18] : g2-p2-e
            #gfit_results[j][k][19] : g3-x3-e --> *(vel_max-vel_min)

            #gfit_results[j][k][22] : log-Z : log-evidence : log-marginalization likelihood

            #gfit_results[j][k][23] : xs
            #gfit_results[j][k][24] : xe
            #gfit_results[j][k][25] : ys
            #gfit_results[j][k][26] : ye
            #gfit_results[j][k][27] : x
            #gfit_results[j][k][28] : y
            #|-----------------------------------------|

            # unit conversion
            # background --> data cube units
            gfit_results[j][k][1] = gfit_results[j][k][1]*(_f_max - _f_min) + _f_min # bg-flux
            gfit_results[j][k][6 + 3*k] = gfit_results[j][k][6 + 3*k]*(_f_max - _f_min) # bg-flux-e

            for m in range(0, k+1):
                # unit conversion
                # peak flux --> data cube units
                # velocity, velocity-dispersion --> km/s
                gfit_results[j][k][2 + 3*m] = gfit_results[j][k][2 + 3*m]*(_vel_max - _vel_min) + _vel_min # velocity
                gfit_results[j][k][3 + 3*m] = gfit_results[j][k][3 + 3*m]*(_vel_max - _vel_min) # velocity-dispersion
                gfit_results[j][k][4 + 3*m] = gfit_results[j][k][4 + 3*m]*(_f_max - _f_min) + _f_min # flux

                gfit_results[j][k][7 + 3*(m+k)] = gfit_results[j][k][7 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-e
                gfit_results[j][k][8 + 3*(m+k)] = gfit_results[j][k][8 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-dispersion-e
                gfit_results[j][k][9 + 3*(m+k)] = gfit_results[j][k][9 + 3*(m+k)]*(_f_max - _f_min) # flux-e

            #if(gfit_results[j][k][4] < 2E-3):
            #    break;
    
    del(ndim, nparams, ngauss, sampler)
    gc.collect()

    return gfit_results

    # Plot a summary of the run.
#    rfig, raxes = dyplot.runplot(sampler.results)
#    rfig.savefig("r.pdf")
    
    # Plot traces and 1-D marginalized posteriors.
#    tfig, taxes = dyplot.traceplot(sampler.results)
#    tfig.savefig("t.pdf")
    
    # Plot the 2-D marginalized posteriors.
    #cfig, caxes = dyplot.cornerplot(sampler.results)
    #cfig.savefig("c.pdf")
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def get_dynesty_sampler_results(_sampler):
    # Extract sampling results.
    samples = _sampler.results.samples  # samples
    weights = exp(_sampler.results.logwt - _sampler.results.logz[-1])  # normalized weights

    #print(_sampler.results.samples[-1, :]) 
    #print(_sampler.results.logwt.shape) 

    # Compute 10%-90% quantiles.
    quantiles = [dyfunc.quantile(samps, [0.1, 0.9], weights=weights)
                for samps in samples.T]
    
    # Compute weighted mean and covariance.
    mean, cov = dyfunc.mean_and_cov(samples, weights)
    std = np.sqrt(np.diag(cov)) 
    bestfit_results = _sampler.results.samples[-1, :]
    log_Z = _sampler.results.logz[-1]
    #log_Z = log(exp(_sampler.results.logz[-1]) - exp(_sampler.results.logz[-2]))
    #log_Z_cumulative = _sampler.results.logz[-1]
    #print("log-Z:", log_Z, "log-Z_cum:", log_Z_cumulative, _sampler.results.logz.shape, _sampler.results.summary())

    #print(bestfit_results, log_Z)
    #print(concatenate((bestfit_results, diag(cov)**0.5)))

    # Resample weighted samples.
    #samples_equal = dyfunc.resample_equal(samples, weights)
    #mean = np.mean(samples_equal, axis=0)
    #cov = np.cov(samples_equal, rowvar=False)
    #std = np.sqrt(np.diag(cov)) 

    # Generate a new set of results with statistical+sampling uncertainties.
    #results_sim = dyfunc.simulate_run(_sampler.results)

    #mean_std = np.concatenate((mean, diag(cov)**0.5))
    #return mean_std # meand + std of each parameter: std array is followed by the mean array
    del(samples, weights, quantiles)
    gc.collect()

    #return concatenate((bestfit_results, diag(cov)**0.5)), log_Z
    return concatenate((mean, std)), log_Z
    #return concatenate((bestfit_results, diag(cov)**0.5)), _sampler.results.logz[-1]
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def multi_gaussian_model_d(_x, _params, ngauss): # params: cube
    #_bg0 : _params[1]
    try:
        g = ((_params[3*i+4] * exp( -0.5*((_x - _params[3*i+2]) / _params[3*i+3])**2)) \
            for i in range(0, ngauss) \
            if _params[3*i+3] != 0 and not np.isnan(_params[3*i+3]) and not np.isinf(_params[3*i+3]))
    except:
        g = 1E9 * exp( -0.5*((_x - 0) / 1)**2)
        print(g)

    return sum(g, axis=1) + _params[1]
#-- END OF SUB-ROUTINE____________________________________________________________#

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def f_gaussian_model(_x, gfit_results, ngauss):
    #_bg0 : gfit_results[1]
    try:
        g = ((gfit_results[3*i+4] * exp( -0.5*((_x - gfit_results[3*i+2]) / gfit_results[3*i+3])**2)) \
            for i in range(0, ngauss) \
            if gfit_results[3*i+3] != 0 and not np.isnan(gfit_results[3*i+3]) and not np.isinf(gfit_results[3*i+3]))
    except:
        g = 1E9 * exp( -0.5*((_x - 0) / 1)**2)
        print(g)

    return sum(g, axis=1) + gfit_results[1]
#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def multi_gaussian_model_d_new(_x, _params, ngauss): # _x: global array, params: cube

    _gparam = _params[2:].reshape(ngauss, 3).T
    #_bg0 : _params[1]
    return (_gparam[2].reshape(ngauss, 1)*exp(-0.5*((_x-_gparam[0].reshape(ngauss, 1)) / _gparam[1].reshape(ngauss, 1))**2)).sum(axis=0) + _params[1]
#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def multi_gaussian_model_d_classic(_x, _params, ngauss): # params: cube
    _bg0 = _params[1]
    _y = np.zeros_like(_x, dtype=np.float32)
    for i in range(0, ngauss):
        _x0 = _params[3*i+2]
        _std0 = _params[3*i+3]
        _p0 = _params[3*i+4]

        _y += _p0 * exp( -0.5*((_x - _x0) / _std0)**2)
        #y += _p0 * (scipy.stats.norm.pdf(_x, loc=_x0, scale=_std0))
    _y += _bg0
    return _y
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def optimal_prior_trfit(*args):
    # nuree

    # args[0] : 2dbat tr parameters
    # args[1] (fit_opt) : 2dbat tr parameters fitting option
    # args[2] : pa, incl, vrot, vrad b-spline order / knots
    # args[3] : 2dbat tr parameters priors
    # <-- these arguments are passed by ptform_args=[ngauss, gfit_priors_init] in NestedSampler()

    #---------------------
    # args[0] : default arg passed by dynesty
    # 2dbat tr parameters
    #---------------------
    # args[0][0] : sigma
    # args[0][1] : xpos
    # args[0][2] : ypos
    # args[0][3] : vsys
    # args[0][4] : pa
    # args[0][5] : incl
    # args[0][6] : vrot
    # args[0][7] : vrad
    # ...
    #_____________________

    #---------------------
    # args[1] : fit_opt
    # 2dbat tr parameters fitting option
    #---------------------
    # args[1][0] : sigma free or fixed
    # args[1][1] : xpos free or fixed 
    # args[1][2] : ypos free or fixed
    # args[1][3] : vsys free or fixed
    # args[1][4] : pa free or fixed
    # args[1][5] : incl free or fixed
    # args[1][6] : vrot free or fixed
    # args[1][7] : vrad free or fixed
    #_____________________

    #---------------------
    # args[2] : bspline_opt
    # b-spline orders & knots
    #---------------------
    # args[2][0] : pa b-spline order ?
    # args[2][1] : pa b-spline knots ?
    # args[2][2] : incl b-spline order ?
    # args[2][3] : incl b-spline knots ?
    # args[2][4] : vrot b-spline order ?
    # args[2][5] : vrot b-spline knots ?
    # args[2][6] : vrad b-spline order ?
    # args[2][7] : vrad b-spline knots ?
    #_____________________

    #---------------------
    # args[3] : tr_param_priors
    # 2dbat tr parameters priors
    #---------------------
    # args[3][0] : _sigma0
    # args[3][1] : _xpos0
    # args[3][2] : _ypos0
    # args[3][3] : _vsys0
    # args[3][4] : _pa0
    # args[3][5] : _incl0
    # args[3][6] : _vrot0
    # args[3][7] : _vrad0
    #.....................
    # args[3][8] : _sigma1
    # args[3][9] : _xpos1
    # args[3][10] : _ypos1
    # args[3][11] : _vsys1
    # args[3][12] : _pa1
    # args[3][13] : _incl1
    # args[3][14] : _vrot1
    # args[3][15] : _vrad1
    #_____________________

    # nuree88

    #_____________________
    # sigma
    _sigma0 = args[3][0]
    _sigma1 = args[3][8]
    #_sigma1 = args[1][2+3*args[1]] # args[1]=bspline order knots
    #_____________________
    # xpos
    _xpos0 = args[3][1]
    _xpos1 = args[3][9]
    #_xpos1 = args[1][3+3*args[1]] # args[1]=ngauss
    #_____________________
    # ypos
    _ypos0 = args[3][2]
    _ypos1 = args[3][10]
    #_____________________
    # vsys
    _vsys0 = args[3][3]
    _vsys1 = args[3][11]
    #_____________________
    # pa
    _pa0 = args[3][4]
    _pa1 = args[3][12]
    #_____________________
    # incl
    _incl0 = args[3][5]
    _incl1 = args[3][13]
    #_____________________
    # vrot
    _vrot0 = args[3][6]
    _vrot1 = args[3][14]
    #_____________________
    # vrad
    _vrad0 = args[3][7]
    _vrad1 = args[3][15]


    n_ring_params_free = 0
    # sigma
    if args[1][0] == 1: # sigma fixed:0 free:1 
        args[0][n_ring_params_free] = _sigma0 + args[0][n_ring_params_free]*(_sigma1 - _sigma0)   # sigma: uniform prior
        n_ring_params_free += 1
    # xpos
    if args[1][1] == 1: # xpos fixed:0 free:1 
        args[0][n_ring_params_free] = _xpos0 + args[0][n_ring_params_free]*(_xpos1 - _xpos0)      # xpos: uniform prior
        n_ring_params_free += 1
    # ypos
    if args[1][2] == 1: # ypos fixed:0 free:1
        args[0][n_ring_params_free] = _ypos0 + args[0][n_ring_params_free]*(_ypos1 - _ypos0)      # ypos: uniform prior
        n_ring_params_free += 1
    # vsys
    if args[1][3] == 1: # vsys fixed:0 free:1
        args[0][n_ring_params_free] = _vsys0 + args[0][n_ring_params_free]*(_vsys1 - _vsys0)      # vsys: uniform prior
        n_ring_params_free += 1
    # pa
    if args[1][4] == 1: # pa fixed:0 free:1
        args[0][n_ring_params_free] = _pa0 + args[0][n_ring_params_free]*(_pa1 - _pa0)      # pa: uniform prior
        n_ring_params_free += 1
    # incl
    if args[1][5] == 1: # incl fixed:0 free:1
        args[0][n_ring_params_free] = _incl0 + args[0][n_ring_params_free]*(_incl1 - _incl0)      # incl: uniform prior
        n_ring_params_free += 1
    # vrot
    if args[1][6] == 1: # vrot fixed:0 free:1
        args[0][n_ring_params_free] = _vrot0 + args[0][n_ring_params_free]*(_vrot1 - _vrot0)      # vrot: uniform prior
        n_ring_params_free += 1

    # vrad
    if args[1][7] == 1: # vrad fixed:0 free:1
        args[0][n_ring_params_free] = _vrad0 + args[0][n_ring_params_free]*(_vrad1 - _vrad0)      # vrad: uniform prior
        n_ring_params_free += 1

    #print(args[0])
    return args[0]
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def optimal_prior_trfit_2d(*args):
    # nuree3

    # args[0] : 2dbat tr parameters
    # args[1] (fit_opt) : 2dbat tr parameters fitting option
    # args[2] : pa, incl, vrot, vrad b-spline order / knots
    # args[3] : 2dbat _params
    # args[4] : tr_params_bounds : xpos, ypos, vsys, incl, pa, vrot
    # args[5] : nrings_reliable
    # args[6] : tck_vrot_bs_init_from_trfit 
    # args[7] : tck_pa_bs_init_from_trfit
    # args[8] : tck_incl_bs_init_from_trfit
    # args[9] : _input_vf
    # args[10] : _ring_t
    # args[11] : _vrot_ns_t

    # args[12] : n_coeffs_pa_bs (passed by user)
    # args[13] : tck_pa_bs (passed by user)
    # args[14] : n_coeffs_incl_bs (passed by user)
    # args[15] : tck_incl_bs (passed by user)
    # args[16] : n_coeffs_vrot_bs (passed by user)
    # args[17] : tck_vrot_bs (passed by user)


    # <-- these arguments are passed by ptform_args=[ngauss, gfit_priors_init] in NestedSampler()

    #---------------------
    # args[0] : default arg passed by dynesty
    # 2dbat tr parameters
    #---------------------
    # args[0][0] : sigma
    # args[0][1] : xpos
    # args[0][2] : ypos
    # args[0][3] : vsys
    # args[0][4] : pa
    # args[0][5] : incl
    # args[0][6] : vrot
    # args[0][7] : vrad
    # ...
    # args[0][8] : vrot_c0_bs
    # args[0][9] : vrot_c1_bs
    # args[0][10] : vrot_c2_bs
    #_____________________

    #---------------------
    # args[1] : fit_opt
    # 2dbat tr parameters fitting option
    #---------------------
    # args[1][0] : sigma free or fixed
    # args[1][1] : xpos free or fixed 
    # args[1][2] : ypos free or fixed
    # args[1][3] : vsys free or fixed
    # args[1][4] : pa free or fixed
    # args[1][5] : incl free or fixed
    # args[1][6] : vrot free or fixed
    # args[1][7] : vrad free or fixed
    #_____________________

    #---------------------
    # args[2] : bspline_opt
    # b-spline orders & knots
    #---------------------
    # args[2][0] : pa b-spline order ?
    # args[2][1] : pa b-spline knots ?
    # args[2][2] : incl b-spline order ?
    # args[2][3] : incl b-spline knots ?
    # args[2][4] : vrot b-spline order ?
    # args[2][5] : vrot b-spline knots ?
    # args[2][6] : vrad b-spline order ?
    # args[2][7] : vrad b-spline knots ?
    #_____________________

    #---------------------
    # args[3] : tr_param_priors
    # 2dbat tr parameters priors
    #---------------------
    # args[3][0] : _sigma0
    # args[3][1] : _xpos0
    # args[3][2] : _ypos0
    # args[3][3] : _vsys0
    # args[3][4] : _pa0
    # args[3][5] : _incl0
    # args[3][6] : _vrot0
    # args[3][7] : _vrad0
    #.....................
    # args[3][8] : _sigma1
    # args[3][9] : _xpos1
    # args[3][10] : _ypos1
    # args[3][11] : _vsys1
    # args[3][12] : _pa1
    # args[3][13] : _incl1
    # args[3][14] : _vrot1
    # args[3][15] : _vrad1
    #_____________________

    #---------------------
    # args[4] : 2dbat _params
    # 2dbat _params
    #---------------------
    # args[4][0] : _sigma0
    # args[4][1] : _xpos0
    # args[4][2] : _ypos0
    # args[4][3] : _vsys0
    # args[4][4] : _pa0
    # args[4][5] : _incl0
    # args[4][6] : _vrot0
    # args[4][7] : _vrad0
    #.....................
    # args[4][8] : _sigma1
    # args[4][9] : _xpos1
    # args[4][10] : _ypos1
    # args[4][11] : _vsys1
    # args[4][12] : _pa1
    # args[4][13] : _incl1
    # args[4][14] : _vrot1
    # args[4][15] : _vrad1
    #_____________________


    #_____________________
    # sigma
    _sigma0 = args[4][0, 0]
    _sigma1 = args[4][0, 1]
    #_____________________
    # xpos
    _xpos0 = args[4][1, 0]
    _xpos1 = args[4][1, 1]
    #_____________________
    # ypos
    _ypos0 = args[4][2, 0]
    _ypos1 = args[4][2, 1]
    #_____________________
    # vsys
    _vsys0 = args[4][3, 0]
    _vsys1 = args[4][3, 1]
    #_____________________
    # pa
    _pa0 = args[4][4, 0]
    _pa1 = args[4][4, 1]
    #_____________________
    # incl
    _incl0 = args[4][5, 0]
    _incl1 = args[4][5, 1]
    #_____________________
    # vrot
    _vrot0 = args[4][6, 0]
    _vrot1 = args[4][6, 1]
    #_____________________
    # vrad
    _vrad0 = args[4][7, 0]
    _vrad1 = args[4][7, 1]


#    print("sigma", _sigma0, _sigma1)
#    print("xpos", _xpos0, _xpos1)
#    print("ypos", _ypos0, _ypos1)
#    print("vsys", _vsys0, _vsys1)
#    print("pa", _pa0, _pa1)
#    print("incl", _incl0, _incl1)
#    print("vrot", _vrot0, _vrot1)
#    print("vrad", _vrad0, _vrad1)
#    print("")
#    print("")

    # --------------------------------------------
    # VROT-BS coefficients
    #n_coeffs_vrot_bs = args[3]['n_coeffs_vrot_bs']
    #n_coeffs_vrot_bs, tck_vrot_bs = bspline_ncoeffs_tck(args[3], 'vrot', args[3]['nrings_intp'])
    # --------------------------------------------
    # PA-BS coefficients
    #n_coeffs_pa_bs = args[3]['n_coeffs_pa_bs']
    #n_coeffs_pa_bs, tck_pa_bs = bspline_ncoeffs_tck(args[3], 'pa', args[3]['nrings_intp'])
    # --------------------------------------------
    # INCL-BS coefficients
    #n_coeffs_incl_bs = args[3]['n_coeffs_incl_bs']
    #n_coeffs_incl_bs, tck_incl_bs = bspline_ncoeffs_tck(args[3], 'incl', args[3]['nrings_intp'])


    #n_coeffs_pa_bs, tck_pa_bs = args[12], args[13]
    #n_coeffs_incl_bs, tck_incl_bs = args[14], args[15]
    #n_coeffs_vrot_bs, tck_vrot_bs = args[16], args[17]

    n_coeffs_pa_bs = args[12]
    n_coeffs_incl_bs = args[14]
    n_coeffs_vrot_bs = args[16]


    n_ring_params_free = 0
    # sigma
    if args[1][0] == 1: # sigma fixed:0 free:1 
        args[0][n_ring_params_free] = _sigma0 + args[0][n_ring_params_free]*(_sigma1 - _sigma0)   # sigma: uniform prior
        n_ring_params_free += 1

    # xpos
    if args[1][1] == 1: # xpos fixed:0 free:1 
        args[0][n_ring_params_free] = _xpos0 + args[0][n_ring_params_free]*(_xpos1 - _xpos0)      # xpos: uniform prior
        #_xpos_ns = args[0][n_ring_params_free] 
        n_ring_params_free += 1

    # ypos
    if args[1][2] == 1: # ypos fixed:0 free:1
        args[0][n_ring_params_free] = _ypos0 + args[0][n_ring_params_free]*(_ypos1 - _ypos0)      # ypos: uniform prior
        #_ypos_ns = args[0][n_ring_params_free] 
        n_ring_params_free += 1

    # vsys
    if args[1][3] == 1: # vsys fixed:0 free:1
        args[0][n_ring_params_free] = _vsys0 + args[0][n_ring_params_free]*(_vsys1 - _vsys0)      # vsys: uniform prior
        #_vsys_ns = args[0][n_ring_params_free] 
        n_ring_params_free += 1

    # pa
    if args[1][4] == 1 and n_coeffs_pa_bs == 0: # constant PA : # pa fixed:0 free:1
        args[0][n_ring_params_free] = _pa0 + args[0][n_ring_params_free]*(_pa1 - _pa0)      # pa: uniform prior
        #_pa_ns = args[0][n_ring_params_free] 
        n_ring_params_free += 1

    # incl
    if args[1][5] == 1 and n_coeffs_incl_bs == 0: # constant INCL : # incl fixed:0 free:1
        args[0][n_ring_params_free] = _incl0 + args[0][n_ring_params_free]*(_incl1 - _incl0)      # incl: uniform prior
        #_incl_ns = args[0][n_ring_params_free] 
        n_ring_params_free += 1

    # vrot
    if args[1][6] == 1 and n_coeffs_vrot_bs == 0: # constant VROT : # vrot fixed:0 free:1
        args[0][n_ring_params_free] = _vrot0 + args[0][n_ring_params_free]*(_vrot1 - _vrot0)      # vrot: uniform prior
        #_vrot_ns = args[0][n_ring_params_free] 
        n_ring_params_free += 1

    # vrad
    if args[1][7] == 1: # vrad fixed:0 free:1
       args[0][n_ring_params_free] = _vrad0 + args[0][n_ring_params_free]*(_vrad1 - _vrad0)      # vrad: uniform prior
       #_vrad_ns = args[0][n_ring_params_free] 
       n_ring_params_free += 1


    # pa_c0_bs
    if n_coeffs_pa_bs != 0:
        _is = n_ring_params_free
        _ie = _is + n_coeffs_pa_bs 

        args[0][_is:_ie] = args[4][7+1:7+1+n_coeffs_pa_bs, 0] \
                         + args[0][_is:_ie] \
                         * (args[4][7+1:7+1+n_coeffs_pa_bs, 1] - args[4][7+1:7+1+n_coeffs_pa_bs, 0]) # pa_c_bs: uniform prior

#    for _nbs in range(0, n_coeffs_pa_bs):
#        args[0][n_ring_params_free] = args[4][7+1+_nbs, 0] \
#                                    + args[0][n_ring_params_free] \
#                                    * (args[4][7+1+_nbs, 1] - args[4][7+1+_nbs, 0])      # pa_c_bs: uniform prior
#        n_ring_params_free += 1


    # incl_c0_bs
    if n_coeffs_incl_bs != 0:
        _is = n_ring_params_free + n_coeffs_pa_bs
        _ie = _is + n_coeffs_incl_bs 

        args[0][_is:_ie] = args[4][7+1+n_coeffs_pa_bs:7+1+n_coeffs_pa_bs+n_coeffs_incl_bs, 0] \
                         + args[0][_is:_ie] \
                         * (args[4][7+1+n_coeffs_pa_bs:7+1+n_coeffs_pa_bs+n_coeffs_incl_bs, 1] - args[4][7+1+n_coeffs_pa_bs:7+1+n_coeffs_pa_bs+n_coeffs_incl_bs, 0]) # incl_c_bs: uniform prior

#    # incl_c0_bs
#    for _nbs in range(0, n_coeffs_incl_bs):
#        args[0][n_ring_params_free] = args[4][7+1+n_coeffs_pa_bs+_nbs, 0] \
#                                    + args[0][n_ring_params_free] \
#                                    * (args[4][7+1+n_coeffs_pa_bs+_nbs, 1] - args[4][7+1+n_coeffs_pa_bs+_nbs, 0])      # incl_c_bs: uniform prior
#        n_ring_params_free += 1


    # vrot_c0_bs
    if n_coeffs_vrot_bs != 0:
        _is = n_ring_params_free + n_coeffs_pa_bs + n_coeffs_incl_bs
        _ie = _is + n_coeffs_vrot_bs 

        args[0][_is:_ie] = args[4][7+1+n_coeffs_pa_bs+n_coeffs_incl_bs:7+1+n_coeffs_pa_bs+n_coeffs_incl_bs+n_coeffs_vrot_bs, 0] \
                         + args[0][_is:_ie] \
                         * (args[4][7+1+n_coeffs_pa_bs+n_coeffs_incl_bs:7+1+n_coeffs_pa_bs+n_coeffs_incl_bs+n_coeffs_vrot_bs, 1] - args[4][7+1+n_coeffs_pa_bs+n_coeffs_incl_bs:7+1+n_coeffs_pa_bs+n_coeffs_incl_bs+n_coeffs_vrot_bs, 0]) # incl_c_bs: uniform prior
        

#    # vrot_c0_bs
#    for _nbs in range(0, n_coeffs_vrot_bs):
#        args[0][n_ring_params_free] = args[4][7+1+n_coeffs_pa_bs+n_coeffs_incl_bs+_nbs, 0] \
#                                    + args[0][n_ring_params_free] \
#                                    * (args[4][7+1+n_coeffs_pa_bs+n_coeffs_incl_bs+_nbs, 1] - args[4][7+1+n_coeffs_pa_bs+n_coeffs_incl_bs+_nbs, 0])      # vrot_c_bs: uniform prior
#
#        n_ring_params_free += 1


    #print(args[0])
    return args[0]
#-- END OF SUB-ROUTINE____________________________________________________________#


def trfit_aring(_vrot_bs, _A, _B):
    return np.sum((_A - _vrot_bs * _B)**2)

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# parameters are sigma, bg, _x01, _std01, _p01, _x02, _std02, _p02...
def optimal_prior(*args):

    #---------------------
    # args[0][0] : sigma
    # args[0][1] : bg0
    # args[0][2] : x0
    # args[0][3] : std0
    # args[0][4] : p0
    # ...
    #_____________________
    #---------------------
    # args[1] : ngauss
    # e.g., if ngauss == 3
    #_____________________
    #---------------------
    # args[2][0] : _sigma0
    # args[2][1] : _bg0
    #.....................
    # args[2][2] : _x10
    # args[2][3] : _std10
    # args[2][4] : _p10
    #.....................
    # args[2][5] : _x20
    # args[2][6] : _std20
    # args[2][7] : _p20
    #.....................
    # args[2][8] : _x30
    # args[2][9] : _std30
    # args[2][10] : _p30
    #_____________________
    #---------------------
    # args[2][11] : _sigma1
    # args[2][12] : _bg1
    #.....................
    # args[2][13] : _x11
    # args[2][14] : _std11
    # args[2][15] : _p11
    #.....................
    # args[2][16] : _x21
    # args[2][17] : _std21
    # args[2][18] : _p21
    #.....................
    # args[2][19] : _x31
    # args[2][20] : _std31
    # args[2][21] : _p31
    #---------------------

    # sigma
    _sigma0 = args[2][0]
    _sigma1 = args[2][2+3*args[1]] # args[1]=ngauss
    # bg
    _bg0 = args[2][1]
    _bg1 = args[2][3+3*args[1]] # args[1]=ngauss

    # partial[2:] copy cube to params_t --> x, std, p ....
    #params_t = args[0][2:].reshape(args[1], 3).T
    params_t = args[0][2:].reshape(args[1], 3).T

    _xn_0 = np.zeros(args[1])
    _xn_1 = np.zeros(args[1])
    _stdn_0 = np.zeros(args[1])
    _stdn_1 = np.zeros(args[1])
    _pn_0 = np.zeros(args[1])
    _pn_1 = np.zeros(args[1])

    for i in range(0, args[1]):
        _xn_0[i] = args[2][2+3*i]
        _xn_1[i] = args[2][4+3*args[1]+3*i]

        _stdn_0[i] = args[2][3+3*i]
        _stdn_1[i] = args[2][5+3*args[1]+3*i]

        _pn_0[i] = args[2][4+3*i]
        _pn_1[i] = args[2][6+3*args[1]+3*i]


    # vectorization
    # sigma and bg
    args[0][0] = _sigma0 + args[0][0]*(_sigma1 - _sigma0)   # sigma: uniform prior between 0:1
    args[0][1] = _bg0 + args[0][1]*(_bg1 - _bg0)            # bg: uniform prior betwargs[1]een 0:1

    # n-gaussians
    # x
    params_t[0] = (_xn_0 + params_t[0].T*(_xn_1 - _xn_0)).T
    # std
    params_t[1] = (_stdn_0 + params_t[1].T*(_stdn_1 - _stdn_0)).T
    # p
    params_t[2] = (_pn_0 + params_t[2].T*(_pn_1 - _pn_0)).T

    params_t_conc = np.hstack((params_t[0].reshape(args[1], 1), params_t[1].reshape(args[1], 1), params_t[2].reshape(args[1], 1))).reshape(1, 3*args[1])
    #print(params_t_conc)
    args[0][2:] = params_t_conc

    #print(args[0])
    #del(_bg0, _bg1, _x0, _x1, _std0, _std1, _p0, _p1, _sigma0, _sigma1, params_t, params_t_conc)
    return args[0]
#-- END OF SUB-ROUTINE____________________________________________________________#

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# parameters are sigma, bg, _x01, _std01, _p01, _x02, _std02, _p02...
def uniform_prior(*args):

    # args[0][0] : sigma
    # args[0][1] : bg0
    # args[0][2] : x0
    # args[0][3] : std0
    # args[0][4] : p0
    # ...
    # args[1] : ngauss
    # args[2][0] : _sigma0
    # args[2][1] : _bg0
    # args[2][2] : _x0
    # args[2][3] : _std0
    # args[2][4] : _p0
    # args[2][5] : _sigma1
    # args[2][6] : _bg1
    # args[2][7] : _x1
    # args[2][8] : _std1
    # args[2][9] : _p1

    # sigma
    _sigma0 = args[2][0]
    _sigma1 = args[2][5]
    # bg
    _bg0 = args[2][1]
    _bg1 = args[2][6]
    # _x0
    _x0 = args[2][2]
    _x1 = args[2][7]
    # _std0
    _std0 = args[2][3]
    _std1 = args[2][8]
    # _p0
    _p0 = args[2][4]
    _p1 = args[2][9]

    # sigma
    #_sigma0 = 0
    #_sigma1 = 0.03 
    ## bg
    #_bg0 = -0.02
    #_bg1 = 0.02
    ## _x0
    #_x0 = 0
    #_x1 = 0.8
    ## _std0
    #_std0 = 0.0
    #_std1 = 0.5
    ## _p0
    #_p0 = 0.0
    #_p1 = 0.5

    # partial[2:] copy cube to params_t --> x, std, p ....
    #params_t = args[0][2:].reshape(args[1], 3).T
    params_t = args[0][2:].reshape(args[1], 3).T

    # vectorization
    # sigma and bg
    args[0][0] = _sigma0 + args[0][0]*(_sigma1 - _sigma0)            # bg: uniform prior between 0:1
    args[0][1] = _bg0 + args[0][1]*(_bg1 - _bg0)            # bg: uniform prior betwargs[1]een 0:1

    # n-gaussians
    # x
    params_t[0] = (_x0 + params_t[0].T*(_x1 - _x0)).T
    #params_t[0] = _x0 + params_t[0]*(_x1 - _x0)
    #print(params_t[0])
    # std
    params_t[1] = (_std0 + params_t[1].T*(_std1 - _std0)).T
    #params_t[1] = _std0 + params_t[1]*(_std1 - _std0)
    # p
    params_t[2] = (_p0 + params_t[2].T*(_p1 - _p0)).T
    #params_t[2] = _p0 + params_t[2]*(_p1 - _p0)

    #print(params_t[1].reshape(args[1], 1))

    params_t_conc = np.hstack((params_t[0].reshape(args[1], 1), params_t[1].reshape(args[1], 1), params_t[2].reshape(args[1], 1))).reshape(1, 3*args[1])
    #params_t_conc1 = np.concatenate((params_t[0], params_t[1], params_t[2]), axis=0)
    #print(params_t_conc)
    args[0][2:] = params_t_conc

    #print(args[0])
    #del(_bg0, _bg1, _x0, _x1, _std0, _std1, _p0, _p1, _sigma0, _sigma1, params_t, params_t_conc)
    return args[0]
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# parameters are sigma, bg, _x01, _std01, _p01, _x02, _std02, _p02...
def uniform_prior_d_pre(*args):

    # args[0][0] : sigma
    # args[0][1] : bg0
    # args[0][2] : x0
    # args[0][3] : std0
    # args[0][4] : p0
    # ...
    # args[1] : ngauss
    # args[2][0] : _sigma0
    # args[2][1] : _sigma1
    # args[2][2] : _bg0
    # args[2][3] : _bg1
    # args[2][4] : _x0
    # args[2][5] : _x1
    # args[2][6] : _std0
    # args[2][7] : _std1
    # args[2][8] : _p0
    # args[2][9] : _p1

    # sigma
    _sigma0 = args[2][0]
    _sigma1 = args[2][5]
    # bg
    _bg0 = args[2][1]
    _bg1 = args[2][6]
    # _x0
    _x0 = args[2][2]
    _x1 = args[2][7]
    # _std0
    _std0 = args[2][3]
    _std1 = args[2][8]
    # _p0
    _p0 = args[2][4]
    _p1 = args[2][9]

    # sigma
    #_sigma0 = 0
    #_sigma1 = 0.03 
    ## bg
    #_bg0 = -0.02
    #_bg1 = 0.02
    ## _x0
    #_x0 = 0
    #_x1 = 0.8
    ## _std0
    #_std0 = 0.0
    #_std1 = 0.5
    ## _p0
    #_p0 = 0.0
    #_p1 = 0.5

    # partial[2:] copy cube to params_t --> x, std, p ....
    #params_t = args[0][2:].reshape(args[1], 3).T
    params_t = args[0][2:].reshape(args[1], 3).T

    # vectorization
    # sigma and bg
    args[0][0] = _sigma0 + args[0][0]*(_sigma1 - _sigma0)            # bg: uniform prior between 0:1
    args[0][1] = _bg0 + args[0][1]*(_bg1 - _bg0)            # bg: uniform prior betwargs[1]een 0:1

    # n-gaussians
    # x
    params_t[0] = _x0 + params_t[0]*(_x1 - _x0)
    # std
    params_t[1] = _std0 + params_t[1]*(_std1 - _std0)
    # p
    params_t[2] = _p0 + params_t[2]*(_p1 - _p0)

    #print(params_t[1].reshape(args[1], 1))

    params_t_conc = np.hstack((params_t[0].reshape(args[1], 1), params_t[1].reshape(args[1], 1), params_t[2].reshape(args[1], 1))).reshape(1, 3*args[1])
    #params_t_conc1 = np.concatenate((params_t[0], params_t[1], params_t[2]), axis=0)
    #print(params_t_conc)
    args[0][2:] = params_t_conc

    #print(args[0])
    #del(_bg0, _bg1, _x0, _x1, _std0, _std1, _p0, _p1, _sigma0, _sigma1, params_t, params_t_conc)
    return args[0]
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def loglike_d(*args):
    # args[0] : params
    # args[1] : _spect : input velocity profile array [N channels] <-- normalized (F-f_max)/(f_max-f_min)
    # args[2] : _x
    # args[3] : ngauss
    # _bg, _x0, _std, _p0, .... = params[1], params[2], params[3], params[4]
    # sigma = params[0] # loglikelihoood sigma
    #print(args[1])

    npoints = args[2].size
    sigma = args[0][0] # loglikelihoood sigma

    gfit = multi_gaussian_model_d(args[2], args[0], args[3])

    log_n_sigma = -0.5*npoints*log(2.0*pi) - 1.0*npoints*log(sigma)
    chi2 = sum((-1.0 / (2*sigma**2)) * ((gfit - args[1])**2))

    return log_n_sigma + chi2


#-- END OF SUB-ROUTINE____________________________________________________________#

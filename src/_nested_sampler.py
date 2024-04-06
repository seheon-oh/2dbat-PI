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

import time, sys, os
import io
import contextlib


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
from scipy.special import gamma, loggamma



from collections import Counter

import matplotlib.pyplot as plt
import math

import numba
from numba import njit
from numba import jit
from numba import vectorize, float64, jit, cuda
import numba as nb
from numba import njit, prange

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

from joblib import Parallel, delayed


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


from _dirs_files import make_dirs

num_threads = 8  # Set this to the number of threads you want to use
os.environ["JULIA_NUM_THREADS"] = str(num_threads)

def max_val(x, y):
    if x > y:
        return x
    else:
        return y

def min_val(x, y):
    if x < y:
        return x
    else:
        return y

def nint_val(x):
    if x > 0.0:
        return (int)(x+0.5)
    else:
        return (int)(x-0.5)




def define_tilted_ring_w(_input_int, xpos, ypos, pa, incl, ri, ro, side, _params):

    naxis1 = _params['naxis1']
    naxis2 = _params['naxis2']
    _wt_2d = np.full((naxis2, naxis1), fill_value=0, dtype=np.float64)

    x0 = 0
    y0 = 0
    x1 = naxis1
    y1 = naxis2


    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    ri_deg = cdelt1*ri
    ro_deg = cdelt1*ro

    deg_to_rad = np.pi / 180.

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


    i0_lo = 0
    i0_up = naxis1
    j0_lo = 0
    j0_up = naxis2

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





    i0, j0 = np.meshgrid(range(i0_lo, i0_up), range(j0_lo, j0_up), indexing='ij')

    rx = cdelt1 * i0
    ry = cdelt2 * j0

    xr = (- (rx - cdelt1 * xpos) * sinp + (ry - cdelt2 * ypos) * cosp)
    yr = (- (rx - cdelt1 * xpos) * cosp - (ry - cdelt2 * ypos) * sinp) / cosi

    r = np.sqrt(xr**2 + yr**2)
    theta = np.where(r < 0.1, 0.0, np.arctan2(yr, xr) / deg_to_rad) # in degrees

    costh = np.abs(np.cos(deg_to_rad * theta)) # in radians


    if side == -1: # approaching side
        npoints_in_a_ring_total_including_blanks_app = np.sum((r > ri_deg) & (r < ro_deg) & (np.fabs(theta) >= 90.0))
    elif side == 1: # receding side
        npoints_in_a_ring_total_including_blanks_rec = np.sum((r > ri_deg) & (r < ro_deg) & (np.fabs(theta) <= 90.0))
    elif side == 0: # both sides
        npoints_in_a_ring_total_including_blanks_both = np.sum((r > ri_deg) & (r < ro_deg))
    elif side == 999: # both sides : trfit_2d
        npoints_in_a_ring_total_including_blanks_both = np.sum((r > ri_deg) & (r < ro_deg))


    x0 = 0
    y0 = 0
    x1 = naxis1
    y1 = naxis2
		


    i0, j0 = np.meshgrid(range(i0_lo, i0_up), range(j0_lo, j0_up), indexing='ij')

    valid_points = (_input_int[j0, i0] > _params['_int_lower']) & (_input_int[j0, i0] < _params['_int_upper'])

    rx = cdelt1 * i0
    ry = cdelt2 * j0

    xr = (- (rx - cdelt1 * xpos) * sinp + (ry - cdelt2 * ypos) * cosp)
    yr = (- (rx - cdelt1 * xpos) * cosp - (ry - cdelt2 * ypos) * sinp) / cosi

    r = np.sqrt(xr**2 + yr**2)
    theta = np.where(r < 0.1, 0.0, np.arctan2(yr, xr) / deg_to_rad) # in degrees

    costh = np.abs(np.cos(deg_to_rad * theta)) # in radians



    if side == -1: # approaching side
        weighted_pixels = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle) & (np.fabs(theta) >= 90.0)
        npoints_in_ring_t_app = np.sum(weighted_pixels)
    elif side == 1: # receding side
        weighted_pixels = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle) & (np.fabs(theta) < 90.0)
        npoints_in_ring_t_rec = np.sum(weighted_pixels)
    elif side == 0: # both sides
        weighted_pixels = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle)
        npoints_in_ring_t_both = np.sum(weighted_pixels)
    elif side == 999: # both sides : trfit_2d
        weighted_pixels = valid_points & (r > ri_deg) & (r < ro_deg)
        npoints_in_ring_t_both = np.sum(weighted_pixels)



    i0, j0 = np.meshgrid(range(i0_lo, i0_up), range(j0_lo, j0_up), indexing='ij')

    valid_points = (_input_int[j0, i0] > _params['_int_lower']) & (_input_int[j0, i0] < _params['_int_upper'])

    rx = cdelt1 * i0
    ry = cdelt2 * j0

    xr = (- (rx - cdelt1 * xpos) * sinp + (ry - cdelt2 * ypos) * cosp)
    yr = (- (rx - cdelt1 * xpos) * cosp - (ry - cdelt2 * ypos) * sinp) / cosi

    r = np.sqrt(xr**2 + yr**2)
    theta = np.where(r < 0.1, 0.0, np.arctan2(yr, xr) / deg_to_rad)  # in degrees

    costh = np.abs(np.cos(deg_to_rad * theta))  # in radians
    cosine_weight = costh ** _params['cosine_weight_power']


    selected_points = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle)

    if side == -1: # approaching side
        selected_points = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle) & (np.fabs(theta) >= 90.0)
    elif side == 1: # receding side
        selected_points = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle) & (np.fabs(theta) < 90.0)
    elif side == 0: # both sides
        selected_points = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle)
    elif side == 999: # both sides : trfit_2d
        selected_points = valid_points & (r > ri_deg) & (r < ro_deg)

    i0_selected = i0[selected_points]
    j0_selected = j0[selected_points]

    ij_tilted_ring = np.stack((i0_selected.flatten(), j0_selected.flatten(), np.full(i0_selected.size, np.nan)), axis=-1)

    _wt_2d[j0_selected, i0_selected] = cosine_weight[selected_points]

    if side == -1: # approaching
        return npoints_in_a_ring_total_including_blanks_app, npoints_in_ring_t_app, ij_tilted_ring, _wt_2d
    elif side == 1: # receding
        return npoints_in_a_ring_total_including_blanks_rec, npoints_in_ring_t_rec, ij_tilted_ring, _wt_2d
    elif side == 0: # both
        return npoints_in_a_ring_total_including_blanks_both, npoints_in_ring_t_both, ij_tilted_ring, _wt_2d
    elif side == 999: # both : trfit_2d
        return npoints_in_a_ring_total_including_blanks_both, npoints_in_ring_t_both, ij_tilted_ring, _wt_2d






def find_maximum_radius1(_input_vf, xpos, ypos, pa, incl, ri, max_major_radius, step_size, _params):

    naxis1 = _params['naxis1']
    naxis2 = _params['naxis2']
    _wt_2d = np.full((naxis2, naxis1), fill_value=0, dtype=np.float64)

    deg_to_rad = np.pi / 180.

    sinp = np.sin(deg_to_rad * pa)
    cosp = np.cos(deg_to_rad * pa)
    sini = np.sin(deg_to_rad * incl)
    cosi = np.cos(deg_to_rad * incl)

    i0_lo = 0
    i0_up = naxis1
    j0_lo = 0
    j0_up = naxis2

    current_radius = ri
    max_radius = ri

    npoints_in_a_ring_vaild_total_pre = 0
    while current_radius <= max_major_radius:

        current_radius += step_size
        ro = current_radius
        ri = current_radius - step_size

        npoints_in_a_ring_vaild_total_cur = 0
        for i0 in range(i0_lo, i0_up):
            for j0 in range(j0_lo, j0_up):
                rx = i0
                ry = j0

                xr = (-(rx - xpos) * sinp + (ry - ypos) * cosp)
                yr = (-(rx - xpos) * cosp - (ry - ypos) * sinp) / cosi
                r = np.sqrt(xr**2 + yr**2)

                if ri < r < ro and not np.isnan(_input_vf[j0, i0]):
                    npoints_in_a_ring_vaild_total_cur += 1


        _ai = ri # along major axis
        _bi = _ai * np.cos(incl*deg_to_rad) # along minor axis
        _li = np.pi * (3.0*(_ai + _bi) - ((3*_ai + _bi) * (_ai + 3*_bi))**0.5) # ellipse perimeter approximation


        if npoints_in_a_ring_vaild_total_cur < npoints_in_a_ring_vaild_total_pre:
            return max_radius-1
        else:
            max_radius = current_radius

        npoints_in_a_ring_vaild_total_pre = npoints_in_a_ring_vaild_total_cur





def find_maximum_radius2(_input_vf, xpos, ypos, pa, incl, ri, max_major_radius, step_size, _params, _2dbat_run_i):
    naxis1 = _params['naxis1']
    naxis2 = _params['naxis2']
    _wt_2d_geo = np.full((naxis2, naxis1), fill_value=0, dtype=np.float64)

    deg_to_rad = np.pi / 180.

    sinp = np.sin(deg_to_rad * pa)
    cosp = np.cos(deg_to_rad * pa)
    sini = np.sin(deg_to_rad * incl)
    cosi = np.cos(deg_to_rad * incl)

    i0_lo = 0
    i0_up = naxis1
    j0_lo = 0
    j0_up = naxis2

    current_radius = ri
    max_radius = ri

    npoints_in_a_ring_valid_total_pre = 0
    found_index = 0

    while current_radius <= max_major_radius:
        current_radius += step_size
        ro = current_radius
        ri = current_radius - step_size

        xs, ys = np.meshgrid(np.arange(naxis1) - xpos, np.arange(naxis2) - ypos)
        xr = -(xs * sinp + ys * cosp)
        yr = -(xs * cosp - ys * sinp) / cosi
        r = np.sqrt(xr**2 + yr**2)

        npoints_in_a_ring_total_cur = np.sum((ri <= r) & (r < ro))
        npoints_in_a_ring_valid_total_cur = np.sum((ri <= r) & (r < ro) & ~np.isnan(_input_vf))

        ratio_valid_to_total = npoints_in_a_ring_valid_total_cur / npoints_in_a_ring_total_cur if npoints_in_a_ring_total_cur != 0 else 0
        ratio_1pixel_to_total = 1 / npoints_in_a_ring_total_cur if npoints_in_a_ring_total_cur != 0 else 0

        valid_indices = np.where((ri <= r) & (r < ro) & ~np.isnan(_input_vf))
        _wt_2d_geo[valid_indices] = ratio_valid_to_total

        if found_index == 0 and ratio_valid_to_total < 1:
            found_index = 1
            _r_major_max = max_radius - 1
        else:
            max_radius = current_radius

        npoints_in_a_ring_valid_total_pre = npoints_in_a_ring_valid_total_cur

    write_fits_images(_params, _wt_2d_geo, _2dbat_run_i, '_wt_2d_geo.fits')

    return _r_major_max, _wt_2d_geo









def find_maximum_radius(_input_vf, xpos, ypos, pa, incl, ri, max_major_radius, step_size, _params, _2dbat_run_i):
    naxis1 = _params['naxis1']
    naxis2 = _params['naxis2']
    _wt_2d_geo = np.full((naxis2, naxis1), fill_value=0, dtype=np.float64)

    deg_to_rad = np.pi / 180.

    sinp = np.sin(deg_to_rad * pa)
    cosp = np.cos(deg_to_rad * pa)
    sini = np.sin(deg_to_rad * incl)
    cosi = np.cos(deg_to_rad * incl)

    i0_lo = 0
    i0_up = naxis1
    j0_lo = 0
    j0_up = naxis2

    current_radius = ri
    max_radius = ri

    npoints_in_a_ring_valid_total_pre = 0
    found_index = 0
    while current_radius <= max_major_radius:

        current_radius += step_size
        ro = current_radius
        ri = current_radius - step_size

        npoints_in_a_ring_valid_total_cur = 0
        npoints_in_a_ring_total_cur = 0
        for i0 in range(i0_lo, i0_up):
            for j0 in range(j0_lo, j0_up):
                rx = i0
                ry = j0

                xr = (-(rx - xpos) * sinp + (ry - ypos) * cosp)
                yr = (-(rx - xpos) * cosp - (ry - ypos) * sinp) / cosi
                r = np.sqrt(xr**2 + yr**2)

                if ri <= r < ro:
                    npoints_in_a_ring_total_cur += 1
                    if not np.isnan(_input_vf[j0, i0]):
                        npoints_in_a_ring_valid_total_cur += 1

        ratio_valid_to_total = npoints_in_a_ring_valid_total_cur / npoints_in_a_ring_total_cur if npoints_in_a_ring_total_cur != 0 else 0
        ratio_1pixel_to_total = 1 / npoints_in_a_ring_total_cur if npoints_in_a_ring_total_cur != 0 else 0

        for i0 in range(i0_lo, i0_up):
            for j0 in range(j0_lo, j0_up):
                rx = i0
                ry = j0

                xr = (-(rx - xpos) * sinp + (ry - ypos) * cosp)
                yr = (-(rx - xpos) * cosp - (ry - ypos) * sinp) / cosi
                r = np.sqrt(xr**2 + yr**2)

                if ri <= r < ro and not np.isnan(_input_vf[j0, i0]):
                    _wt_2d_geo[j0, i0] = ratio_valid_to_total


        if found_index == 0 and ratio_valid_to_total < _params['ratio_valid_to_total'] and current_radius > 0.5*_params['_r_max_el']:
            found_index = 1
            _r_major_max = max_radius-1
        else:
            max_radius = current_radius

        npoints_in_a_ring_valid_total_pre = npoints_in_a_ring_valid_total_cur

    write_fits_images(_params, _wt_2d_geo, _2dbat_run_i, '_wt_2d_geo.fits')

    return _r_major_max, _wt_2d_geo














def define_tilted_ring(_input_vf, xpos, ypos, pa, incl, ri, ro, side, _params):



    naxis1 = _params['naxis1']
    naxis2 = _params['naxis2']
    _wt_2d = np.full((naxis2, naxis1), fill_value=np.nan, dtype=np.float64)

    x0 = 0
    y0 = 0
    x1 = naxis1
    y1 = naxis2


    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    ri_deg = cdelt1*ri
    ro_deg = cdelt1*ro

    deg_to_rad = np.pi / 180.

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


    i0_lo = 0
    i0_up = naxis1
    j0_lo = 0
    j0_up = naxis2

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





    i0, j0 = np.meshgrid(range(i0_lo, i0_up), range(j0_lo, j0_up), indexing='ij')

    rx = cdelt1 * i0
    ry = cdelt2 * j0

    xr = (- (rx - cdelt1 * xpos) * sinp + (ry - cdelt2 * ypos) * cosp)
    yr = (- (rx - cdelt1 * xpos) * cosp - (ry - cdelt2 * ypos) * sinp) / cosi

    r = np.sqrt(xr**2 + yr**2)
    theta = np.where(r < 0.1, 0.0, np.arctan2(yr, xr) / deg_to_rad) # in degrees

    costh = np.abs(np.cos(deg_to_rad * theta)) # in radians


    if side == -1: # approaching side
        npoints_in_a_ring_total_including_blanks_app = np.sum((r > ri_deg) & (r < ro_deg) & (np.fabs(theta) >= 90.0))
    elif side == 1: # receding side
        npoints_in_a_ring_total_including_blanks_rec = np.sum((r > ri_deg) & (r < ro_deg) & (np.fabs(theta) <= 90.0))
    elif side == 0: # both sides
        npoints_in_a_ring_total_including_blanks_both = np.sum((r > ri_deg) & (r < ro_deg))
    elif side == 999: # both sides : trfit_2d
        npoints_in_a_ring_total_including_blanks_both = np.sum((r > ri_deg) & (r < ro_deg))


    x0 = 0
    y0 = 0
    x1 = naxis1
    y1 = naxis2
		

    i0, j0 = np.meshgrid(range(i0_lo, i0_up), range(j0_lo, j0_up), indexing='ij')

    valid_points = (_input_vf[j0, i0] > _params['_vlos_lower']) & (_input_vf[j0, i0] < _params['_vlos_upper'])

    rx = cdelt1 * i0
    ry = cdelt2 * j0

    xr = (- (rx - cdelt1 * xpos) * sinp + (ry - cdelt2 * ypos) * cosp)
    yr = (- (rx - cdelt1 * xpos) * cosp - (ry - cdelt2 * ypos) * sinp) / cosi

    r = np.sqrt(xr**2 + yr**2)
    theta = np.where(r < 0.1, 0.0, np.arctan2(yr, xr) / deg_to_rad) # in degrees

    costh = np.abs(np.cos(deg_to_rad * theta)) # in radians



    if side == -1: # approaching side
        weighted_pixels = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle) & (np.fabs(theta) >= 90.0)
        npoints_in_ring_t_app = np.sum(weighted_pixels)
    elif side == 1: # receding side
        weighted_pixels = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle) & (np.fabs(theta) < 90.0)
        npoints_in_ring_t_rec = np.sum(weighted_pixels)
    elif side == 0: # both sides
        weighted_pixels = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle)
        npoints_in_ring_t_both = np.sum(weighted_pixels)
    elif side == 999: # both sides : trfit_2d
        weighted_pixels = valid_points & (r > ri_deg) & (r < ro_deg)
        npoints_in_ring_t_both = np.sum(weighted_pixels)



    i0, j0 = np.meshgrid(range(i0_lo, i0_up), range(j0_lo, j0_up), indexing='ij')

    valid_points = (_input_vf[j0, i0] > _params['_vlos_lower']) & (_input_vf[j0, i0] < _params['_vlos_upper'])

    rx = cdelt1 * i0
    ry = cdelt2 * j0

    xr = (- (rx - cdelt1 * xpos) * sinp + (ry - cdelt2 * ypos) * cosp)
    yr = (- (rx - cdelt1 * xpos) * cosp - (ry - cdelt2 * ypos) * sinp) / cosi

    r = np.sqrt(xr**2 + yr**2)
    theta = np.where(r < 0.1, 0.0, np.arctan2(yr, xr) / deg_to_rad)  # in degrees

    costh = np.abs(np.cos(deg_to_rad * theta))  # in radians
    cosine_weight = costh ** _params['cosine_weight_power']


    selected_points = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle)

    if side == -1: # approaching side
        selected_points = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle) & (np.fabs(theta) >= 90.0)
    elif side == 1: # receding side
        selected_points = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle) & (np.fabs(theta) < 90.0)
    elif side == 0: # both sides
        selected_points = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle)
    elif side == 999: # both sides : trfit_2d
        selected_points = valid_points & (r > ri_deg) & (r < ro_deg)

    i0_selected = i0[selected_points]
    j0_selected = j0[selected_points]

    ij_tilted_ring = np.stack((i0_selected.flatten(), j0_selected.flatten(), np.full(i0_selected.size, np.nan)), axis=-1)

    _wt_2d[j0_selected, i0_selected] = cosine_weight[selected_points]



    if side == -1: # approaching
        return npoints_in_a_ring_total_including_blanks_app, npoints_in_ring_t_app, ij_tilted_ring, _wt_2d
    elif side == 1: # receding
        return npoints_in_a_ring_total_including_blanks_rec, npoints_in_ring_t_rec, ij_tilted_ring, _wt_2d
    elif side == 0: # both
        return npoints_in_a_ring_total_including_blanks_both, npoints_in_ring_t_both, ij_tilted_ring, _wt_2d
    elif side == 999: # both : trfit_2d
        return npoints_in_a_ring_total_including_blanks_both, npoints_in_ring_t_both, ij_tilted_ring, _wt_2d







def define_tilted_ring_full(_input_vf, xpos, ypos, pa, incl, ri, ro, side, _params):



    naxis1 = _params['naxis1']
    naxis2 = _params['naxis2']
    _wt_2d = np.full((naxis2, naxis1), fill_value=0, dtype=np.float64)

    x0 = 0
    y0 = 0
    x1 = naxis1
    y1 = naxis2


    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    ri_deg = cdelt1*ri
    ro_deg = cdelt1*ro

    deg_to_rad = np.pi / 180.

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


    i0_lo = 0
    i0_up = naxis1
    j0_lo = 0
    j0_up = naxis2

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





    i0, j0 = np.meshgrid(range(i0_lo, i0_up), range(j0_lo, j0_up), indexing='ij')

    rx = cdelt1 * i0
    ry = cdelt2 * j0

    xr = (- (rx - cdelt1 * xpos) * sinp + (ry - cdelt2 * ypos) * cosp)
    yr = (- (rx - cdelt1 * xpos) * cosp - (ry - cdelt2 * ypos) * sinp) / cosi

    r = np.sqrt(xr**2 + yr**2)
    theta = np.where(r < 0.1, 0.0, np.arctan2(yr, xr) / deg_to_rad) # in degrees

    costh = np.abs(np.cos(deg_to_rad * theta)) # in radians


    if side == -1: # approaching side
        npoints_in_a_ring_total_including_blanks_app = np.sum((r > ri_deg) & (r < ro_deg) & (np.fabs(theta) >= 90.0))
    elif side == 1: # receding side
        npoints_in_a_ring_total_including_blanks_rec = np.sum((r > ri_deg) & (r < ro_deg) & (np.fabs(theta) <= 90.0))
    elif side == 0: # both sides
        npoints_in_a_ring_total_including_blanks_both = np.sum((r > ri_deg) & (r < ro_deg))
    elif side == 999: # both sides : trfit_2d
        npoints_in_a_ring_total_including_blanks_both = np.sum((r > ri_deg) & (r < ro_deg))


    x0 = 0
    y0 = 0
    x1 = naxis1
    y1 = naxis2
		

    i0, j0 = np.meshgrid(range(i0_lo, i0_up), range(j0_lo, j0_up), indexing='ij')

    valid_points = (_input_vf[j0, i0] > _params['_vlos_lower']) & (_input_vf[j0, i0] < _params['_vlos_upper'])

    rx = cdelt1 * i0
    ry = cdelt2 * j0

    xr = (- (rx - cdelt1 * xpos) * sinp + (ry - cdelt2 * ypos) * cosp)
    yr = (- (rx - cdelt1 * xpos) * cosp - (ry - cdelt2 * ypos) * sinp) / cosi

    r = np.sqrt(xr**2 + yr**2)
    theta = np.where(r < 0.1, 0.0, np.arctan2(yr, xr) / deg_to_rad) # in degrees

    costh = np.abs(np.cos(deg_to_rad * theta)) # in radians



    if side == -1: # approaching side
        weighted_pixels = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle) & (np.fabs(theta) >= 90.0)
        npoints_in_ring_t_app = np.sum(weighted_pixels)
    elif side == 1: # receding side
        weighted_pixels = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle) & (np.fabs(theta) < 90.0)
        npoints_in_ring_t_rec = np.sum(weighted_pixels)
    elif side == 0: # both sides
        weighted_pixels = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle)
        npoints_in_ring_t_both = np.sum(weighted_pixels)
    elif side == 999: # both sides : trfit_2d
        weighted_pixels = valid_points & (r > ri_deg) & (r < ro_deg)
        npoints_in_ring_t_both = np.sum(weighted_pixels)



    i0, j0 = np.meshgrid(range(i0_lo, i0_up), range(j0_lo, j0_up), indexing='ij')

    valid_points = (_input_vf[j0, i0] > _params['_vlos_lower']) & (_input_vf[j0, i0] < _params['_vlos_upper'])

    rx = cdelt1 * i0
    ry = cdelt2 * j0

    xr = (- (rx - cdelt1 * xpos) * sinp + (ry - cdelt2 * ypos) * cosp)
    yr = (- (rx - cdelt1 * xpos) * cosp - (ry - cdelt2 * ypos) * sinp) / cosi

    r = np.sqrt(xr**2 + yr**2)
    theta = np.where(r < 0.1, 0.0, np.arctan2(yr, xr) / deg_to_rad)  # in degrees

    costh = np.abs(np.cos(deg_to_rad * theta))  # in radians
    cosine_weight = costh ** _params['cosine_weight_power']


    selected_points = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle)

    if side == -1: # approaching side
        selected_points = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle) & (np.fabs(theta) >= 90.0)
    elif side == 1: # receding side
        selected_points = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle) & (np.fabs(theta) < 90.0)
    elif side == 0: # both sides
        selected_points = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle)
    elif side == 999: # both sides : trfit_2d
        selected_points = valid_points & (r > ri_deg) & (r < ro_deg)

    i0_selected = i0[selected_points]
    j0_selected = j0[selected_points]

    ij_tilted_ring = np.stack((i0_selected.flatten(), j0_selected.flatten(), np.full(i0_selected.size, np.nan)), axis=-1)

    _wt_2d[j0_selected, i0_selected] = cosine_weight[selected_points]



    if side == -1: # approaching
        return npoints_in_a_ring_total_including_blanks_app, npoints_in_ring_t_app, ij_tilted_ring, _wt_2d
    elif side == 1: # receding
        return npoints_in_a_ring_total_including_blanks_rec, npoints_in_ring_t_rec, ij_tilted_ring, _wt_2d
    elif side == 0: # both
        return npoints_in_a_ring_total_including_blanks_both, npoints_in_ring_t_both, ij_tilted_ring, _wt_2d
    elif side == 999: # both : trfit_2d
        return npoints_in_a_ring_total_including_blanks_both, npoints_in_ring_t_both, ij_tilted_ring, _wt_2d










def define_tilted_ring_geo(_input_int, xpos, ypos, pa, incl, ri, ro, side, _params):



    naxis1 = _params['naxis1']
    naxis2 = _params['naxis2']
    _wt_2d = np.full((naxis2, naxis1), fill_value=0, dtype=np.float64)

    x0 = 0
    y0 = 0
    x1 = naxis1
    y1 = naxis2


    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    ri_deg = cdelt1*ri
    ro_deg = cdelt1*ro

    deg_to_rad = np.pi / 180.

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


    i0_lo = 0
    i0_up = naxis1
    j0_lo = 0
    j0_up = naxis2

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





    i0, j0 = np.meshgrid(range(i0_lo, i0_up), range(j0_lo, j0_up), indexing='ij')

    rx = cdelt1 * i0
    ry = cdelt2 * j0

    xr = (- (rx - cdelt1 * xpos) * sinp + (ry - cdelt2 * ypos) * cosp)
    yr = (- (rx - cdelt1 * xpos) * cosp - (ry - cdelt2 * ypos) * sinp) / cosi

    r = np.sqrt(xr**2 + yr**2)
    theta = np.where(r < 0.1, 0.0, np.arctan2(yr, xr) / deg_to_rad) # in degrees

    costh = np.abs(np.cos(deg_to_rad * theta)) # in radians


    if side == -1: # approaching side
        npoints_in_a_ring_total_including_blanks_app = np.sum((r > ri_deg) & (r < ro_deg) & (np.fabs(theta) >= 90.0))
    elif side == 1: # receding side
        npoints_in_a_ring_total_including_blanks_rec = np.sum((r > ri_deg) & (r < ro_deg) & (np.fabs(theta) <= 90.0))
    elif side == 0: # both sides
        npoints_in_a_ring_total_including_blanks_both = np.sum((r > ri_deg) & (r < ro_deg))
    elif side == 999: # both sides : trfit_2d
        npoints_in_a_ring_total_including_blanks_both = np.sum((r > ri_deg) & (r < ro_deg))


    x0 = 0
    y0 = 0
    x1 = naxis1
    y1 = naxis2
		

    i0, j0 = np.meshgrid(range(i0_lo, i0_up), range(j0_lo, j0_up), indexing='ij')

    valid_points = (_input_int[j0, i0] > _params['_int_lower']) & (_input_int[j0, i0] < _params['_int_upper'])

    rx = cdelt1 * i0
    ry = cdelt2 * j0

    xr = (- (rx - cdelt1 * xpos) * sinp + (ry - cdelt2 * ypos) * cosp)
    yr = (- (rx - cdelt1 * xpos) * cosp - (ry - cdelt2 * ypos) * sinp) / cosi

    r = np.sqrt(xr**2 + yr**2)
    theta = np.where(r < 0.1, 0.0, np.arctan2(yr, xr) / deg_to_rad) # in degrees

    costh = np.abs(np.cos(deg_to_rad * theta)) # in radians



    if side == -1: # approaching side
        weighted_pixels = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle) & (np.fabs(theta) >= 90.0)
        npoints_in_ring_t_app = np.sum(weighted_pixels)
    elif side == 1: # receding side
        weighted_pixels = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle) & (np.fabs(theta) < 90.0)
        npoints_in_ring_t_rec = np.sum(weighted_pixels)
    elif side == 0: # both sides
        weighted_pixels = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle)
        npoints_in_ring_t_both = np.sum(weighted_pixels)
    elif side == 999: # both sides : trfit_2d
        weighted_pixels = valid_points & (r > ri_deg) & (r < ro_deg)
        npoints_in_ring_t_both = np.sum(weighted_pixels)



    i0, j0 = np.meshgrid(range(i0_lo, i0_up), range(j0_lo, j0_up), indexing='ij')

    valid_points = (_input_int[j0, i0] > _params['_int_lower']) & (_input_int[j0, i0] < _params['_int_upper'])

    rx = cdelt1 * i0
    ry = cdelt2 * j0

    xr = (- (rx - cdelt1 * xpos) * sinp + (ry - cdelt2 * ypos) * cosp)
    yr = (- (rx - cdelt1 * xpos) * cosp - (ry - cdelt2 * ypos) * sinp) / cosi

    r = np.sqrt(xr**2 + yr**2)
    theta = np.where(r < 0.1, 0.0, np.arctan2(yr, xr) / deg_to_rad)  # in degrees

    costh = np.abs(np.cos(deg_to_rad * theta))  # in radians
    cosine_weight = costh ** _params['cosine_weight_power']


    selected_points = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle)

    if side == -1: # approaching side
        selected_points = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle) & (np.fabs(theta) >= 90.0)
    elif side == 1: # receding side
        selected_points = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle) & (np.fabs(theta) < 90.0)
    elif side == 0: # both sides
        selected_points = valid_points & (r > ri_deg) & (r < ro_deg) & (costh > sine_free_angle)
    elif side == 999: # both sides : trfit_2d
        selected_points = valid_points & (r > ri_deg) & (r < ro_deg)

    i0_selected = i0[selected_points]
    j0_selected = j0[selected_points]

    ij_tilted_ring = np.stack((i0_selected.flatten(), j0_selected.flatten(), np.full(i0_selected.size, np.nan)), axis=-1)

    _wt_2d[j0_selected, i0_selected] = _input_int[selected_points]
    _wt_2d = np.where(_wt_2d != 0, 1, 0)



    if side == -1: # approaching
        return npoints_in_a_ring_total_including_blanks_app, npoints_in_ring_t_app, ij_tilted_ring, _wt_2d
    elif side == 1: # receding
        return npoints_in_a_ring_total_including_blanks_rec, npoints_in_ring_t_rec, ij_tilted_ring, _wt_2d
    elif side == 0: # both
        return npoints_in_a_ring_total_including_blanks_both, npoints_in_ring_t_both, ij_tilted_ring, _wt_2d
    elif side == 999: # both : trfit_2d
        return npoints_in_a_ring_total_including_blanks_both, npoints_in_ring_t_both, ij_tilted_ring, _wt_2d



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

    _P = (1./np.sin(_i_p1*deg_to_rad)) * (vrot1 * np.sin(i1*deg_to_rad) / _vrot_p1)
    _C = _P * (-(x1 - xpos) * np.sin(phi1*deg_to_rad) + (y1 - ypos) * np.cos(phi1*deg_to_rad))

    result = minimize(of1, _phi_p, args=(_A, _B, _C))

    _phi_pp[0] = result.x[0]

    _P = (1./np.sin(_i_p1*deg_to_rad)) * (vrot1 * np.sin(i1*deg_to_rad) / _vrot_p2)
    _C = _P * (-(x1 - xpos) * np.sin(phi1*deg_to_rad) + (y1 - ypos) * np.cos(phi1*deg_to_rad))

    result = minimize(of1, _phi_p, args=(_A, _B, _C))

    _phi_pp[1] = result.x[0]

    _P = (1./np.sin(_i_p2*deg_to_rad)) * (vrot1 * np.sin(i1*deg_to_rad) / _vrot_p1)
    _C = _P * (-(x1 - xpos) * np.sin(phi1*deg_to_rad) + (y1 - ypos) * np.cos(phi1*deg_to_rad))

    result = minimize(of1, _phi_p, args=(_A, _B, _C))

    _phi_pp[2] = result.x[0]

    _P = (1./np.sin(_i_p2*deg_to_rad)) * (vrot1 * np.sin(i1*deg_to_rad) / _vrot_p2)
    _C = _P * (-(x1 - xpos) * np.sin(phi1*deg_to_rad) + (y1 - ypos) * np.cos(phi1*deg_to_rad))

    result = minimize(of1, _phi_p, args=(_A, _B, _C))

    _phi_pp[3] = result.x[0]


    return _phi_pp.min(), _phi_pp.max()




def of1(_phi_p, _A, _B, _C):
    deg_to_rad = np.pi / 180.
    return np.sum((_C - (_A * np.cos(_phi_p*deg_to_rad) + _B * np.sin(_phi_p*deg_to_rad)))**2)





def solve_r_galaxy_plane_newton_org(fx_r_galaxy_plane, _dyn_params, fit_opt, _params, ij_aring, nrings_reliable, _xpos, _ypos, _nxy):

    cdelt1 = 1
    cdelt2 = 1
    del_xi = (ij_aring[_nxy, 0] - _xpos)*cdelt1 # calculate x
    del_yi = (ij_aring[_nxy, 1] - _ypos)*cdelt2 # calculate y
    _r_galaxy_plane_i_init = 3*(del_xi**2 + del_yi**2)**0.5

    return optimize.newton(fx_r_galaxy_plane, _r_galaxy_plane_i_init, args=(_dyn_params, fit_opt, _params, ij_aring[_nxy, 0], ij_aring[_nxy, 1], nrings_reliable), tol=1E-1, maxiter=10000, disp=False)




@ray.remote(num_cpus=1)
def solve_r_galaxy_plane_newton(fx_r_galaxy_plane, _dyn_params, fit_opt, _params, ij_aring, nrings_reliable, _xpos, _ypos, _nxy):

    n_cpus = int(_params['num_cpus_2d_dyn'])
    cdelt1 = 1
    cdelt2 = 1

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


    return roots, _nxy




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

        _pa_el = _params['_pa_el']
        _incl_el = _params['_incl_el']

        cosp1 = np.cos(deg_to_rad*_pa_el)
        sinp1 = np.sin(deg_to_rad*_pa_el)
        cosi1 = np.cos(deg_to_rad*_incl_el) #cosine
        sini1 = np.sin(deg_to_rad*_incl_el) # sine

        x_galaxy_plane = (-del_xi * sinp1 + del_yi * cosp1) # x in plane of galaxy
        y_galaxy_plane = (-del_xi * cosp1 - del_yi * sinp1) / cosi1 # y in plane of galaxy
        _r_galaxy_plane_i_init = (x_galaxy_plane**2 + y_galaxy_plane**2)**0.5

        _x = ij_aring[_is:_ie, 0]
        _y = ij_aring[_is:_ie, 1]

        if fit_opt[4] == 1 and n_coeffs_pa_bs == 0 and fit_opt[5] == 1 and n_coeffs_incl_bs == 0:
            roots = solve_r_galaxy_plane_jit_pa_const_incl_const_test(_x, _y, _xpos, _ypos, _pa, _incl, _r_galaxy_plane_i_init, 1.0*_params['r_galaxy_plane_e'])
            return roots, _nxy

        if fit_opt[4] == 0 and n_coeffs_pa_bs != 0 and fit_opt[5] == 1 and n_coeffs_incl_bs == 0:
            roots = solve_r_galaxy_plane_jit_pa_bs_incl_const_test(_x, _y, _xpos, _ypos, tck_pa_bs, _incl, _r_galaxy_plane_i_init, 1.0*_params['r_galaxy_plane_e'])
            return roots, _nxy

        if fit_opt[4] == 1 and n_coeffs_pa_bs == 0 and fit_opt[5] == 0 and n_coeffs_incl_bs != 0:
            roots = solve_r_galaxy_plane_jit_pa_const_incl_bs_test(_x, _y, _xpos, _ypos, _pa, tck_incl_bs, _r_galaxy_plane_i_init, 1.0*_params['r_galaxy_plane_e'])
            return roots, _nxy

        if fit_opt[4] == 0 and n_coeffs_pa_bs != 0 and fit_opt[5] == 0 and n_coeffs_incl_bs != 0:
            roots = solve_r_galaxy_plane_jit_pa_bs_incl_bs_test(_x, _y, _xpos, _ypos, tck_pa_bs, tck_incl_bs, _r_galaxy_plane_i_init, 1.0*_params['r_galaxy_plane_e'])
            return roots, _nxy

    else:
        _is = _nxy*int(ij_aring.shape[0]/n_cpus)
        _ie = (_nxy+1)*int(ij_aring.shape[0]/n_cpus)

        del_xi = (ij_aring[_is:_ie, 0] - _xpos)*cdelt1 # calculate x
        del_yi = (ij_aring[_is:_ie, 1] - _ypos)*cdelt2 # calculate y
        print("CHECK FIT OPTIONS: BSPLINE PA | BSPLINE INCL | CONST PA | CONST INCL")




    return roots, _nxy




@jit(nopython=True)
def fx_r_galaxy_plane_jit_pa_const_incl_const(x, i, deg_to_rad, _pa, _incl, del_x, del_y):
    cosi1 = np.cos(deg_to_rad * _incl) # RADIAN
    _pa_x0 = _pa * deg_to_rad # RADIAN
    left_side = (-del_x[i] * np.sin(_pa_x0) + del_y[i] * np.cos(_pa_x0))**2 + \
                ((-del_x[i] * np.cos(_pa_x0) - del_y[i] * np.sin(_pa_x0)) / cosi1)**2
    return left_side - x**2

def find_root_pa_const_incl_const(i, deg_to_rad, _pa, _incl, del_x, del_y, _r_galaxy_plane_i_init, _r_max):
    _a = 0.8 * _r_galaxy_plane_i_init[i]
    _b = min(1.2 * _r_galaxy_plane_i_init[i], 1.5*_r_max)

    for _ in range(100):
        try:
            result = root_scalar(fx_r_galaxy_plane_jit_pa_const_incl_const, args=(i, deg_to_rad, _pa, _incl, del_x, del_y), bracket=(_a, _b), method='brentq')
            return result.root


        except ValueError:  # if not found, increase the bounds
            _a -= 0.1 * abs(_a)
            if _a < 0:
                _a = 0
            _b += 0.1 * abs(_b)
    return np.nan  # if not found


def solve_r_galaxy_plane_jit_pa_const_incl_const_test(_x, _y, _xpos, _ypos, _pa, _incl, _r_galaxy_plane_i_init, _r_max):
    deg_to_rad = np.pi / 180.0

    del_x = _x - _xpos
    del_y = _y - _ypos

    solutions = np.empty_like(del_x, dtype=np.float64)

    for i in range(del_x.shape[0]):
        solutions[i] = find_root_pa_const_incl_const(i, deg_to_rad, _pa, _incl, del_x, del_y, _r_galaxy_plane_i_init, _r_max)

    return solutions







def find_root_cpu(i, deg_to_rad, x_values, bspline_values_pa, _incl, del_x, del_y, _r_galaxy_plane_i_init, _r_max):
    _a = 0.5 * _r_galaxy_plane_i_init[i]
    _b = min(1.5 * _r_galaxy_plane_i_init[i], 1.5*_r_max)
    for _ in range(100):
        try:
            result = root_scalar(fx_r_galaxy_plane_jit_pa_bs_incl_const, args=(i, deg_to_rad, x_values, bspline_values_pa, _incl, del_x, del_y), bracket=(_a, _b), method='brentq')
            return result.root
        except ValueError:
            _a -= 0.1 * abs(_a)
            if _a < 0:
                _a = 0
            _b += 0.1 * abs(_b)
    return np.nan

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

        solutions[idx] = find_root_cpu(idx, deg_to_rad, x_values, bspline_values_pa, _incl, del_x, del_y, _r_galaxy_plane_i_init, _r_max)

def solve_r_galaxy_plane_jit_pa_bs_incl_const_test_gpu(_x, _y, _xpos, _ypos, tck_pa_bs, _incl, _r_galaxy_plane_i_init, _r_max):
    solutions = np.empty_like(_x, dtype=np.float64)
    threadsperblock = 12
    blockspergrid = (_x.size + (threadsperblock - 1)) // threadsperblock
    solve_r_galaxy_plane_gpu[blockspergrid, threadsperblock](_x, _y, _xpos, _ypos, tck_pa_bs, _incl, _r_galaxy_plane_i_init, _r_max, solutions)
    return solutions






@jit(nopython=True)
def fx_r_galaxy_plane_jit_pa_bs_incl_const(x, i, deg_to_rad, x_values, bspline_values_pa, _incl, del_x, del_y):
    cosi1 = np.cos(deg_to_rad * _incl) # RADIAN
    _pa_x0 = np.interp(x, x_values, bspline_values_pa) * deg_to_rad # RADIAN, Interpolate bspline values

    left_side = np.sqrt((-del_x[i] * np.sin(_pa_x0) + del_y[i] * np.cos(_pa_x0))**2 + \
                ((-del_x[i] * np.cos(_pa_x0) - del_y[i] * np.sin(_pa_x0)) / cosi1)**2)
    return left_side - x**1

def find_root_pa_bs_incl_const(i, deg_to_rad, x_values, bspline_values_pa, _incl, del_x, del_y, _r_galaxy_plane_i_init, _r_max):
    _a = 0.8 * _r_galaxy_plane_i_init[i]
    _b = min(1.2 * _r_galaxy_plane_i_init[i], 1.5*_r_max)

    for _ in range(100):
        try:
            result = root_scalar(fx_r_galaxy_plane_jit_pa_bs_incl_const, args=(i, deg_to_rad, x_values, bspline_values_pa, _incl, del_x, del_y), bracket=(_a, _b), method='brentq')
            return result.root
        except ValueError:  # if not found, increase the bounds
            _a -= 0.1 * abs(_a)
            if _a < 0:
                _a = 0
            _b += 0.1 * abs(_b)
    return np.nan  # 최대 반복 횟수 동안 근을 찾지 못한 경우

def solve_r_galaxy_plane_jit_pa_bs_incl_const_test(_x, _y, _xpos, _ypos, tck_pa_bs, _incl, _r_galaxy_plane_i_init, _r_max):
    deg_to_rad = np.pi / 180.0

    bspline = BSpline(*tck_pa_bs) # DEGREE

    x_values = np.linspace(0, int(_r_max)*5, int(_r_max)*5)  # Adjust the number of points as needed
    bspline_values_pa = bspline(x_values) # DEGREE

    del_x = _x - _xpos
    del_y = _y - _ypos

    solutions = np.empty_like(del_x, dtype=np.float64)

    for i in range(del_x.shape[0]):
        solutions[i] = find_root_pa_bs_incl_const(i, deg_to_rad, x_values, bspline_values_pa, _incl, del_x, del_y, _r_galaxy_plane_i_init, _r_max)

    return solutions




@jit(nopython=True)
def fx_r_galaxy_plane_jit_pa_const_incl_bs(x, i, deg_to_rad, x_values, _pa, bspline_values_incl, del_x, del_y):
    _pa_x0 = _pa * deg_to_rad # RADIAN
    _incl_x0 = np.interp(x, x_values, bspline_values_incl) * deg_to_rad # RADIAN, Interpolate bspline values
    cosi1 = np.cos(_incl_x0)

    left_side = (-del_x[i] * np.sin(_pa_x0) + del_y[i] * np.cos(_pa_x0))**2 + \
                ( (-del_x[i] * np.cos(_pa_x0) - del_y[i] * np.sin(_pa_x0)) / cosi1 )**2
    return left_side - x**2

def find_root_pa_const_incl_bs(i, deg_to_rad, x_values, _pa, bspline_values_incl, del_x, del_y, _r_galaxy_plane_i_init, _r_max):
    _a = 0.8 * _r_galaxy_plane_i_init[i]
    _b = min(1.2 * _r_galaxy_plane_i_init[i], 1.5*_r_max)

    for _ in range(100):
        try:
            result = root_scalar(fx_r_galaxy_plane_jit_pa_const_incl_bs, args=(i, deg_to_rad, x_values, _pa, bspline_values_incl, del_x, del_y), bracket=(_a, _b), method='brentq')
            return result.root

        except ValueError:  # if not found, increase the bounds
            _a -= 0.1 * abs(_a)
            if _a < 0:
                _a = 0
            _b += 0.1 * abs(_b)

    print("roots not found")
    return np.nan  # if not found

def solve_r_galaxy_plane_jit_pa_const_incl_bs_test(_x, _y, _xpos, _ypos, _pa, tck_incl_bs, _r_galaxy_plane_i_init, _r_max):
    deg_to_rad = np.pi / 180.0

    bspline_incl = BSpline(*tck_incl_bs, extrapolate=True) # DEGREE

    x_values = np.linspace(0, int(_r_max)*5, int(_r_max)*5)  # Adjust the number of points as needed
    bspline_values_incl = bspline_incl(x_values) # DEGREE

    bspline_values_incl = np.where((bspline_values_incl > 90), 89, bspline_values_incl )
    bspline_values_incl = np.where((bspline_values_incl < 0), 0, bspline_values_incl )

    del_x = _x - _xpos
    del_y = _y - _ypos

    solutions = np.empty_like(del_x, dtype=np.float64)

    for i in range(del_x.shape[0]):
        solutions[i] = find_root_pa_const_incl_bs(i, deg_to_rad, x_values, _pa, bspline_values_incl, del_x, del_y, _r_galaxy_plane_i_init, _r_max)

    return solutions



@jit(nopython=True)
def fx_r_galaxy_plane_jit_pa_bs_incl_bs(x, i, deg_to_rad, x_values, bspline_values_pa, bspline_values_incl, del_x, del_y):
    _pa_x0 = np.interp(x, x_values, bspline_values_pa) * deg_to_rad # RADIAN, Interpolate bspline values
    _incl_x0 = np.interp(x, x_values, bspline_values_incl) * deg_to_rad # RADIAN, Interpolate bspline values
    cosi1 = np.cos(_incl_x0)

    left_side = np.sqrt((-del_x[i] * np.sin(_pa_x0) + del_y[i] * np.cos(_pa_x0))**2 + \
                ((-del_x[i] * np.cos(_pa_x0) - del_y[i] * np.sin(_pa_x0)) / cosi1)**2)
    return left_side - x**1

def find_root_pa_bs_incl_bs(i, deg_to_rad, x_values, bspline_values_pa, bspline_values_incl, del_x, del_y, _r_galaxy_plane_i_init, _r_max):
    _a = 0.8 * _r_galaxy_plane_i_init[i]
    _b = min(1.2 * _r_galaxy_plane_i_init[i], 1.5*_r_max)

    for _ in range(100):
        try:
            result = root_scalar(fx_r_galaxy_plane_jit_pa_bs_incl_bs, args=(i, deg_to_rad, x_values, bspline_values_pa, bspline_values_incl, del_x, del_y), bracket=(_a, _b), method='brentq')
            return result.root
        except ValueError:  # if not found, increase the bounds
            _a -= 0.1 * abs(_a)
            if _a < 0:
                _a = 0
            _b += 0.1 * abs(_b)
    return np.nan  # if not found

def solve_r_galaxy_plane_jit_pa_bs_incl_bs_test(_x, _y, _xpos, _ypos, tck_pa_bs, tck_incl_bs, _r_galaxy_plane_i_init, _r_max):
    deg_to_rad = np.pi / 180.0

    bspline_pa = BSpline(*tck_pa_bs) # degree
    bspline_incl = BSpline(*tck_incl_bs) # degree

    x_values = np.linspace(0, int(_r_max)*5, int(_r_max)*5)  # Adjust the number of points as needed
    bspline_values_pa = bspline_pa(x_values) # DEGREE

    bspline_values_incl = bspline_incl(x_values) # DEGREE
    bspline_values_incl = np.where((bspline_values_incl > 90), 89, bspline_values_incl )
    bspline_values_incl = np.where((bspline_values_incl < 0), 0, bspline_values_incl )


    del_x = _x - _xpos
    del_y = _y - _ypos

    solutions = np.empty_like(del_x, dtype=np.float64)

    for i in range(del_x.shape[0]):
        solutions[i] = find_root_pa_bs_incl_bs(i, deg_to_rad, x_values, bspline_values_pa, bspline_values_incl, del_x, del_y, _r_galaxy_plane_i_init, _r_max)

    return solutions





@jit(nopython=True)
def fx_r_galaxy_plane_vectorized(r_galaxy_plane, x_galaxy_plane, y_galaxy_plane):
    fx = (x_galaxy_plane**2 + y_galaxy_plane**2)**0.5 - r_galaxy_plane
    return fx


@jit(nopython=True) # fast
def secant_vectorized_jitted(f, x0, x1, x_galaxy_plane, y_galaxy_plane):
    tol = 0.1
    max_iter = 1000
    x_new = x0
    for _ in range(max_iter):
        f_x0 = f(x0, x_galaxy_plane, y_galaxy_plane)
        f_x1 = f(x1, x_galaxy_plane, y_galaxy_plane)

        x_new = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)

        if np.all(np.abs(x_new - x1) < tol):
            return x_new

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

        new_x0s = x1s - f_x1 * (x1s - x0s) / (f_x1 - f_x0)

        if np.all(np.abs(new_x0s - x0s) < tol):
            return new_x0s

        x0s, x1s = new_x0s, x0s

    return x0s








@jit(nopython=True, fastmath=True)
def secant_vectorized_jitted_pa_const_incl_const_new(f, x0, x1, _x, _y, _xpos, _ypos, cosp1, sinp1, cosi1, sini1):
    tol = 0.1
    max_iter = 100  # Adjust this as needed for better convergence

    del_x = (_x - _xpos)
    del_y = (_y - _ypos)
    x_galaxy_plane = (-del_x * sinp1 + del_y * cosp1)
    y_galaxy_plane = (-del_x * cosp1 - del_y * sinp1) / cosi1

    f_x0 = f(x0, x_galaxy_plane, y_galaxy_plane)
    f_x1 = f(x1, x_galaxy_plane, y_galaxy_plane)
    
    for _ in range(max_iter):
        if f_x0 * f_x1 < 0.0:
            x_mid = (x0 + x1) / 2
            f_mid = f(x_mid, x_galaxy_plane, y_galaxy_plane)

            if np.abs(f_mid) < tol:
                return x_mid

            if f_mid * f_x0 > 0:
                x0, f_x0 = x_mid, f_mid
            else:
                x1, f_x1 = x_mid, f_mid
        else:
            df = f_x1 - f_x0
            if df == 0:
                return x1

            x_new = x1 - f_x1 * (x1 - x0) / df
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

def secant_vectorized_pa_const_incl_bs(f, x0, x1, _x, _y, _xpos, _ypos, _pa, tck_incl_bs):
    _incl = BSpline(*tck_incl_bs, extrapolate=True)(x1)
    return secant_vectorized_jitted(f, x0, x1, _x, _y, _xpos, _ypos, _pa, _incl)



def secant_vectorized_pa_bs_incl_bs(f, x0, x1, _x, _y, _xpos, _ypos, tck_pa_bs, tck_incl_bs):
    _pa = BSpline(*tck_pa_bs, extrapolate=True)(x1)
    _incl = BSpline(*tck_incl_bs, extrapolate=True)(x1)
    return secant_vectorized_jitted(f, x0, x1, _x, _y, _xpos, _ypos, _pa, _incl)





















@jit(nopython=True, parallel=True, cache=True)
def secant_method(f, x0, x1, args, tol=0.1, max_iter=1000):
    f_x0 = f(x0, *args)
    f_x1 = f(x1, *args)

    for _ in range(max_iter):
        if np.all(np.abs(f_x1) < tol):  # Check if all elements are within tolerance
            return x1

        x_next = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)

        x0, x1 = x1, x_next
        f_x0, f_x1 = f_x1, f(x1, *args)

    return x1

























@ray.remote(num_cpus=1)
def solve_r_galaxy_plane_fsolve(fx_r_galaxy_plane, _dyn_params, fit_opt, _params, ij_aring, nrings_reliable, _xpos, _ypos, _nxy):

    cdelt1 = 1
    cdelt2 = 1
    del_xi = (ij_aring[_nxy, 0] - _xpos)*cdelt1 # calculate x
    del_yi = (ij_aring[_nxy, 1] - _ypos)*cdelt2 # calculate y
    _r_galaxy_plane_i_init = 1*(del_xi**2 + del_yi**2)**0.5

    roots = fsolve(fx_r_galaxy_plane, x0=_r_galaxy_plane_i_init, xtol=10, maxfev=10, factor=0.1, args=(_dyn_params, fit_opt, _params, ij_aring[_nxy, 0], ij_aring[_nxy, 1], nrings_reliable))
    return roots

def r_galaxy_plane_min_given_bounds(x, _x, _y, _xpos, _ypos):

    deg_to_rad = np.pi / 180.

    cdelt1 = 1
    cdelt2 = 1
    del_x = (_x - _xpos)*cdelt1 # calculate x
    del_y = (_y - _ypos)*cdelt2 # calculate y

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

    cosp1 = np.cos(deg_to_rad*x[0])
    sinp1 = np.sin(deg_to_rad*x[0])

    cosi1 = np.cos(deg_to_rad*x[1]) #cosine
    sini1 = np.sin(deg_to_rad*x[1]) # sine

    x_galaxy_plane = (-del_x * sinp1 + del_y * cosp1) # x in plane of galaxy
    y_galaxy_plane = (-del_x * cosp1 - del_y * sinp1) / cosi1 # y in plane of galaxy

    r_galaxy_plane = (x_galaxy_plane**2 + y_galaxy_plane**2)**0.5
    return (1.0 / r_galaxy_plane)



def fx_r_galaxy_plane(r_galaxy_plane, _dyn_params, fit_opt, _params, _x, _y, nrings_reliable, _xpos, _ypos, tck_pa_bs, tck_incl_bs):




    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    deg_to_rad = np.pi / 180.


    _pa_bs = BSpline(*tck_pa_bs, extrapolate=True)(r_galaxy_plane) # <-- tck_pa_bs[1] coeffs (0 ~ 360)
    _incl_bs = splev(r_galaxy_plane, tck_incl_bs)

    

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

    n_coeffs_pa_bs, tck_pa_bs = bspline_ncoeffs_tck(_params, 'pa', _params['nrings_intp'])
    
    n_coeffs_incl_bs, tck_incl_bs = bspline_ncoeffs_tck(_params, 'incl', _params['nrings_intp'])

    n_coeffs_vrot_bs, tck_vrot_bs = bspline_ncoeffs_tck(_params, 'vrot', _params['nrings_intp'])

    _xpos = 10
    _ypos = 10

    n_ring_params_free = 0 # starting from sigma

    if n_coeffs_pa_bs != 0:
        for _nbs in range(0, n_coeffs_pa_bs):
            tck_pa_bs[1][_nbs] = _dyn_params[n_ring_params_free + _nbs]

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
    _incl_bs = splev(r_galaxy_plane, tck_incl_bs)

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




def evaluate_constant_spline_vectorized(x, tck_vrot_bs):
    intervals, avg_values, k_bs = tck_vrot_bs
    edges = intervals.flatten()
    indices = np.searchsorted(edges, x, side='right')
    indices = (indices - 1) // 2
    indices = np.clip(indices, 0, len(avg_values)-1)
    y_eval = avg_values[indices]

    return y_eval



def bspline_ncoeffs_tck(_params, tr_param, nrings_reliable):

    try: 
        xs_bs = _params['r_galaxy_plane_s']
        xe_bs = _params['r_galaxy_plane_e']

        nrings = nrings_reliable
        x_bs = np.linspace(xs_bs, xe_bs, nrings, endpoint=True)
        y_bs = np.linspace(xs_bs, xe_bs, nrings, endpoint=True) # dummy y_bs for generating dummy tck


    except:
        pass

    if tr_param == 'vrot':


        n_knots_inner = _params['n_vrot_bs_knots_inner'] # 0, 1, 2, ...
        vrot_bs_knots_inner = np.linspace(xs_bs, xe_bs, (2+n_knots_inner))[1:-1]


        k_bs = _params['k_vrot_bs'] # 1, 2, ...
        n_coeffs_vrot_bs = k_bs + 1 + n_knots_inner

        if(n_coeffs_vrot_bs == 1): # constant vrot
            n_coeffs_vrot_bs = 0
            tck_vrot_bs =  0
            return n_coeffs_vrot_bs, tck_vrot_bs

        else:
            tck_vrot_bs = splrep(x_bs, y_bs, t=vrot_bs_knots_inner, k=k_bs)
            return n_coeffs_vrot_bs, tck_vrot_bs
        



    
    elif tr_param == 'pa':


        n_knots_inner = _params['n_pa_bs_knots_inner'] # 0, 1, 2, ...
        pa_bs_knots_inner = np.linspace(xs_bs, xe_bs, (2+n_knots_inner))[1:-1]
        k_bs = _params['k_pa_bs'] # 1, 2, ...
        n_coeffs_pa_bs = k_bs + 1 + n_knots_inner

        if(n_coeffs_pa_bs == 1): # constant pa
            n_coeffs_pa_bs = 0
            tck_pa_bs =  0
            return n_coeffs_pa_bs, tck_pa_bs
        else:
            tck_pa_bs = splrep(x_bs, y_bs, t=pa_bs_knots_inner, k=k_bs)
            return n_coeffs_pa_bs, tck_pa_bs

    elif tr_param == 'incl':


        n_knots_inner = _params['n_incl_bs_knots_inner'] # 0, 1, 2, ...
        incl_bs_knots_inner = np.linspace(xs_bs, xe_bs, (2+n_knots_inner))[1:-1]



        k_bs = _params['k_incl_bs'] # 1, 2, ...
        n_coeffs_incl_bs = k_bs + 1 + n_knots_inner

        if(n_coeffs_incl_bs == 1): # constant incl
            n_coeffs_incl_bs = 0
            tck_incl_bs =  0
            return n_coeffs_incl_bs, tck_incl_bs
        else:
            tck_incl_bs = splrep(x_bs, y_bs, t=incl_bs_knots_inner, k=k_bs)
            return n_coeffs_incl_bs, tck_incl_bs

    elif tr_param == 'vrad':


        n_knots_inner = _params['n_vrad_bs_knots_inner'] # 0, 1, 2, ...
        vrad_bs_knots_inner = np.linspace(xs_bs, xe_bs, (2+n_knots_inner))[1:-1]
        k_bs = _params['k_vrad_bs'] # 1, 2, ...
        n_coeffs_vrad_bs = k_bs + 1 + n_knots_inner

        if(n_coeffs_vrad_bs == 1): # constant vrad 
            n_coeffs_vrad_bs = 0
            tck_vrad_bs =  0
            return n_coeffs_vrad_bs, tck_vrad_bs
        else:
            tck_vrad_bs = splrep(x_bs, y_bs, t=vrad_bs_knots_inner, k=k_bs)
            return n_coeffs_vrad_bs, tck_vrad_bs



def loglike_trfit_2d(*args):
    sigma = args[0][0]  # loglikelihoood sigma: dynesty default params[0]
    tr_model_vf = derive_vlos_model_2d_sc(args[0], args[2], args[9], args[3], \
                                            args[4], args[6], args[7], args[8], \
                                            args[10], args[11], args[12], args[13], \
                                            args[14], args[15], \
                                            args[16], args[1])


    res = (args[1] - tr_model_vf ) / sigma

    res1 = (res**2) / args[17]
    npoints = np.count_nonzero(~np.isnan(res1))
    loglike = -0.5 * (np.nansum(res1) + np.nansum(np.log(2 * np.pi * sigma**2)/args[17]))

    return loglike









def loglike_trfit(*args):









    sigma = args[0][0] # loglikelihoood sigma: dynesty default params[0]


    tr_model_vf = derive_vlos_model(args[0], args[2], args[4], args[5], args[6])





    res = (args[1] - tr_model_vf) / sigma

    res1 = res**2
    npoints = np.count_nonzero(~np.isnan(res1))
    loglike = -0.5 * (np.nansum(res1) + npoints* (np.log(2 * np.pi * sigma**2)))

    return loglike









def loglike_trfit_2d_t(*args):


    sigma = args[0][0]  # loglikelihood sigma: dynesty default params[0]

    tr_model_vf, _norm_vrot_sini_costh_2d, _vsys, _r_gal, _A, _vrot_bs_max, sini1_max = derive_vlos_model_2d_sc(args[0], args[2], args[9], args[3], args[4], args[6], args[7], args[8], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[1])

    _nu = 29.

    res = (args[1] - tr_model_vf) / sigma

    res1 = np.where((np.isnan(res)) | (np.isinf(res)), 0, res)

    npoints = np.count_nonzero(res1**2)  # within r_galaxy_plane_e_geo

    loglike = np.nansum( loggamma((_nu + 1) / 2) - loggamma(_nu / 2) - 0.5 * np.log(_nu * np.pi * sigma**2) - ((_nu + 1) / 2) * np.log(1 + (res1**2) / _nu) )

    return loglike







def loglike_trfit_2d_t(*args):

    sigma = args[0][0] # loglikelihoood sigma: dynesty default params[0]
    tr_model_vf, _norm_vrot_sini_costh_2d, _vsys, _r_gal, _A, _vrot_bs_max, sini1_max  = derive_vlos_model_2d_sc(args[0], args[2], args[9], args[3], args[4], args[6], args[7], args[8], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[1])





    res = (args[1] - tr_model_vf) / sigma
    res1 = np.where((np.isnan(res)) | (np.isinf(res)), 0, res)

    res1 = res1**2
    npoints = np.count_nonzero(res1) # within r_galaxy_plane_e_geo 

    _ll_sum = 0

    print(np.min(_r_gal), np.max(_r_gal))


    for _r in range(np.min(_r_gal), np, max(_r_gal), 1):
        _ri = _r
        _ro = _r + 1

        index = ((_r_gal > _ri) & (_r_gal < _ro))

        _ll_t = np.nanmedian(res1[index])
        _ll_sum += _ll_t

    loglike = -0.5 * (

        _ll_sum + npoints * np.log(2 * np.pi * sigma**2) )

    return loglike 


def loglike_trfit_2d_tt(*args):
    sigma = args[0][0]  # loglikelihoood sigma: dynesty default params[0]
    tr_model_vf, _norm_vrot_sini_costh_2d, _vsys, _r_gal, _A, _vrot_bs_max, sini1_max = derive_vlos_model_2d_sc(args[0], args[2], args[9], args[3], args[4], args[6], args[7], args[8], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[1])

    res = (args[1] - tr_model_vf) / sigma
    res1 = np.where((np.isnan(res)) | (np.isinf(res)), 0, res)

    res1 = res1**2
    npoints = np.count_nonzero(~np.isnan(args[1]))

    _ll_sum = 0



    _r_s = 0
    for _r in range(int(np.nanmin(_r_gal)), int(np.nanmax(_r_gal)), 2):
        _ri = _r_s
        _ro = _ri + 2
        _r_s = _ro

        index = ((_r_gal > _ri) & (_r_gal < _ro))

        _ll_t = np.nanmedian(res1[index])

        if np.isnan(_ll_t): continue
        _ll_sum += _ll_t


    loglike = -0.5 * (_ll_sum + npoints * np.log(2 * np.pi * sigma**2))

    return loglike







def loglike_trfit_2d_normal(*args):













    sigma = args[0][0] # loglikelihoood sigma: dynesty default params[0]


    tr_model_vf, _norm_vrot_sini_costh_2d, _vsys, _r_gal, _A, _vrot_bs_max, sini1_max = derive_vlos_model_2d_sc(args[0], args[2], args[9], args[3], args[4], args[6], args[7], args[8], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[1])













    










    _input_vdisp_norm = args[9]

































    



    















    vo_mean = np.nanmean(args[1])
    vo_std = np.nanstd(args[1])

    vo_min = np.nanmin(args[1])
    vo_max = np.nanmax(args[1])

    vm_mean = np.nanmean(tr_model_vf)
    vm_std = np.nanstd(tr_model_vf)



    res = (args[1] - tr_model_vf)

    res_mean = np.nanmean(res)
    res_rms = np.nanstd(res)


    res1 = np.where((np.isnan(res)), 0, res)
    res1 = np.where((np.isinf(res1)), 0, res1)

    npoints = np.count_nonzero(res1) # within r_galaxy_plane_e_geo 

    res1 = res1 / sigma
    res1 =  (res1**2)












    loglike = -0.5 * (

   

        












    


        np.nansum(res1) + npoints * np.log(2 * np.pi * sigma**2) )


    


    return loglike 









def	tr_vlos_model_org(_dyn_params, _tr_model_vf, ij_aring, fit_opt):

    npoints = ij_aring.shape[0]


    for n in range(npoints):
        i = ij_aring[n, 0]
        j = ij_aring[n, 1]

        tr_vlos_model = derive_vlos_model(_dyn_params, i, j, fit_opt)

        _tr_model_vf[j, i] = tr_vlos_model

    return _tr_model_vf




        
def derive_vlos_model(_dyn_params, _tr_model_vf, ij_aring, fit_opt, _params):


    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    deg_to_rad = np.pi / 180.

    n_ring_params_free = 0 # starting from sigma
    if fit_opt[0] == 1: # sigma fixed:0 free:1
        _sigma = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _sigma = _params['_sigma_init'] # fixed to the previous fitting result
    if fit_opt[1] == 1: # xpos fixed:0 free:1
        _xpos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _xpos = _params['_xpos_init'] # fixed to the previous fitting result
    if fit_opt[2] == 1: # ypos fixed:0 free:1
        _ypos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _ypos = _params['_ypos_init'] # fixed to the previous fitting result
    if fit_opt[3] == 1: # vsys fixed:0 free:1
        _vsys = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vsys = _params['_vsys_init'] # fixed to the previous fitting result
    if fit_opt[4] == 1: # pa fixed:0 free:1
        _pa = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _pa = _params['_pa_init'] # fixed to the previous fitting result
    if fit_opt[5] == 1: # incl fixed:0 free:1
        _incl = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _incl = _params['_incl_init'] # fixed to the previous fitting result
    if fit_opt[6] == 1: # vrot fixed:0 free:1
        _vrot = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrot = _params['_vrot_init'] # fixed to the previous fitting result
    if fit_opt[7] == 1: # vrad fixed:0 free:1
        _vrad = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrad = _params['_vrad_init'] # fixed to the previous fitting result

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


    _tr_model_vf[ij_aring[:, 1].astype(int), ij_aring[:, 0].astype(int)] = _vsys + (_vrot * cost1 + 0. * sint1) * sini1 # model LOS velocity

    return _tr_model_vf


def derive_vlos_model_2d(_dyn_params, _tr_model_vf, ij_aring, fit_opt, _params, nrings_reliable, r_galaxy_plane):




    n_cpus = int(_params['num_cpus_2d_dyn'])

    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    deg_to_rad = np.pi / 180.

    n_ring_params_free = 0 # starting from sigma
    if fit_opt[0] == 1: # sigma fixed:0 free:1
        _sigma = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _sigma = _params['_sigma_init'] # fixed to the previous fitting result
    if fit_opt[1] == 1: # xpos fixed:0 free:1
        _xpos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _xpos = _params['_xpos_init'] # fixed to the previous fitting result
    if fit_opt[2] == 1: # ypos fixed:0 free:1
        _ypos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _ypos = _params['_ypos_init'] # fixed to the previous fitting result
    if fit_opt[3] == 1: # vsys fixed:0 free:1
        _vsys = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vsys = _params['_vsys_init'] # fixed to the previous fitting result
    if fit_opt[4] == 1: # pa fixed:0 free:1
        _pa = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _pa = _params['_pa_init'] # fixed to the previous fitting result
    if fit_opt[5] == 1: # incl fixed:0 free:1
        _incl = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _incl = _params['_incl_init'] # fixed to the previous fitting result
    if fit_opt[6] == 1: # vrot fixed:0 free:1
        _vrot = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrot = _params['_vrot_init'] # fixed to the previous fitting result
    if fit_opt[7] == 1: # vrad fixed:0 free:1
        _vrad = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrad = 0 # fixed to the previous fitting result


    n_coeffs_vrot_bs, tck_vrot_bs = bspline_ncoeffs_tck(_params, 'vrot', _params['nrings_intp'])
    n_coeffs_pa_bs, tck_pa_bs = bspline_ncoeffs_tck(_params, 'pa', _params['nrings_intp'])
    n_coeffs_incl_bs, tck_incl_bs = bspline_ncoeffs_tck(_params, 'incl', _params['nrings_intp'])


    cdelt1 = 1
    cdelt2 = 1
    del_x = (ij_aring[:, 0] - _xpos)*cdelt1 # calculate x
    del_y = (ij_aring[:, 1] - _ypos)*cdelt2 # calculate y

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


    results_ids = [solve_r_galaxy_plane_newton.remote(fx_r_galaxy_plane_id, _dyn_params_id, fit_opt_id, _params_id, ij_aring_id, nrings_reliable_id, _xpos_id, _ypos_id, _nxy_id) for _nxy_id in range(0, n_cpus)]
    while len(results_ids):
        done_ids, results_ids = ray.wait(results_ids)
        if done_ids:
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


    r_galaxy_plane_within = np.where(r_galaxy_plane > _params['r_galaxy_plane_e'], _params['r_galaxy_plane_e'], r_galaxy_plane)















    if n_coeffs_pa_bs != 0: # not constant
        for _nbs in range(0, n_coeffs_pa_bs):
            tck_pa_bs[1][_nbs] = _dyn_params[n_ring_params_free + n_coeffs_vrot_bs + _nbs]

        _pa_bs = BSpline(*tck_pa_bs, extrapolate=True)(r_galaxy_plane_within) # <-- tck_pa_bs[1] coeffs (0 ~ 360)

    else:
        _pa_bs = _pa

    if n_coeffs_incl_bs != 0: # not constant
        for _nbs in range(0, n_coeffs_incl_bs):
            tck_incl_bs[1][_nbs] = _dyn_params[n_ring_params_free + n_coeffs_vrot_bs + n_coeffs_pa_bs + _nbs]

        _incl_bs = splev(r_galaxy_plane_within, tck_incl_bs)
    else:
        _incl_bs = _incl

    cosp1 = np.cos(deg_to_rad*_pa_bs)
    sinp1 = np.sin(deg_to_rad*_pa_bs)
    cosi1 = np.cos(deg_to_rad*_incl_bs) #cosine
    sini1 = np.sin(deg_to_rad*_incl_bs) # sine

    x_galaxy_plane = (-del_x * sinp1 + del_y * cosp1) # x in plane of galaxy
    y_galaxy_plane = (-del_x * cosp1 - del_y * sinp1) / cosi1 # y in plane of galaxy

    cost1 = x_galaxy_plane / r_galaxy_plane # cosine of angle in plane of galaxy 
    sint1 = y_galaxy_plane / r_galaxy_plane # sine of angle in plane of galaxy

    if n_coeffs_vrot_bs != 0: # not constant
        for _nbs in range(0, n_coeffs_vrot_bs):
            tck_vrot_bs[1][_nbs] = _dyn_params[n_ring_params_free+_nbs]
        _vrot_bs = splev(r_galaxy_plane_within, tck_vrot_bs)

    else:
        _vrot_bs = _vrot


    _tr_model_vf[:, :] = np.nan # initialize with a blank value
    _tr_model_vf[ij_aring[:, 1].astype(int), ij_aring[:, 0].astype(int)] = \
        _vsys + (_vrot_bs * cost1 + _vrad * sint1) * sini1 # model LOS velocity

    return _tr_model_vf





def extract_tr2dfit_params(_tr2dfit_results, _params, fit_opt_2d, ring, region):





    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    deg_to_rad = np.pi / 180.


    if region == 'entire':
        n_coeffs_pa_bs, tck_pa_bs = bspline_ncoeffs_tck(_params, 'pa', _params['nrings_intp'])
        n_coeffs_pa_bs, tck_pa_bs_e = bspline_ncoeffs_tck(_params, 'pa', _params['nrings_intp'])
        n_coeffs_incl_bs, tck_incl_bs = bspline_ncoeffs_tck(_params, 'incl', _params['nrings_intp'])
        n_coeffs_incl_bs, tck_incl_bs_e = bspline_ncoeffs_tck(_params, 'incl', _params['nrings_intp'])
        n_coeffs_vrot_bs, tck_vrot_bs = bspline_ncoeffs_tck(_params, 'vrot', _params['nrings_intp'])
        n_coeffs_vrot_bs, tck_vrot_bs_e = bspline_ncoeffs_tck(_params, 'vrot', _params['nrings_intp'])
        n_coeffs_vrad_bs, tck_vrad_bs = bspline_ncoeffs_tck(_params, 'vrad', _params['nrings_intp'])
        n_coeffs_vrad_bs, tck_vrad_bs_e = bspline_ncoeffs_tck(_params, 'vrad', _params['nrings_intp'])

    elif region == 'sector':
        n_coeffs_pa_bs = 0
        n_coeffs_incl_bs = 0
        n_coeffs_vrot_bs = 0
        n_coeffs_vrad_bs = 0


    n_ring_params_free = 0 # starting from sigma
    if fit_opt_2d[0] == 1: # sigma fixed:0 free:1
        _sigma = _tr2dfit_results[n_ring_params_free]
        _params['_sigma_init'] = _sigma # update _params
        n_ring_params_free += 1
    else:
        _sigma = _params['_sigma_init'] # fixed to the previous fitting result
    if fit_opt_2d[1] == 1: # xpos fixed:0 free:1
        _xpos = _tr2dfit_results[n_ring_params_free]
        _params['_xpos_init'] = _xpos # update _params
        n_ring_params_free += 1
    else:
        _xpos = _params['_xpos_init'] # fixed to the previous fitting result
    if fit_opt_2d[2] == 1: # ypos fixed:0 free:1
        _ypos = _tr2dfit_results[n_ring_params_free]
        _params['_ypos_init'] = _ypos # update _params
        n_ring_params_free += 1
    else:
        _ypos = _params['_ypos_init'] # fixed to the previous fitting result
    if fit_opt_2d[3] == 1: # vsys fixed:0 free:1
        _vsys = _tr2dfit_results[n_ring_params_free]
        _params['_vsys_init'] = _vsys # update _params
        n_ring_params_free += 1
    else:
        _vsys = _params['_vsys_init'] # fixed to the previous fitting result
    if fit_opt_2d[4] == 1 and n_coeffs_pa_bs == 0: # pa fixed:0 free:1
        _pa = _tr2dfit_results[n_ring_params_free]
        _params['_pa_init'] = _pa # update _params
        n_ring_params_free += 1
    else:
        _pa = _params['_pa_init'] # fixed to the previous fitting result
    if fit_opt_2d[5] == 1 and n_coeffs_incl_bs == 0 : # incl fixed:0 free:1
        _incl = _tr2dfit_results[n_ring_params_free]
        _params['_incl_init'] = _incl # update _params
        n_ring_params_free += 1
    else:
        _incl = _params['_incl_init'] # fixed to the previous fitting result
    if fit_opt_2d[6] == 1 and n_coeffs_vrot_bs == 0: # vrot fixed:0 free:1
        _vrot = _tr2dfit_results[n_ring_params_free]
        _params['_vrot_init'] = _vrot # update _params
        n_ring_params_free += 1
    else:
        _vrot = _params['_vrot_init'] # fixed to the previous fitting result
    if fit_opt_2d[7] == 1: # vrad fixed:0 free:1
        _vrad = _tr2dfit_results[n_ring_params_free]
        _params['_vrad_init'] = _vrad # update _params
        n_ring_params_free += 1
    else:
        _vrad = _params['_vrad_init'] # fixed to the previous fitting result




    ndim_total = n_ring_params_free + n_coeffs_pa_bs + n_coeffs_incl_bs + n_coeffs_vrot_bs + n_coeffs_vrad_bs

    n_ring_params_free = 0 # starting from sigma-e
    if fit_opt_2d[0] == 1: # sigma fixed:0 free:1
        _sigma_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _sigma_e = 999
    if fit_opt_2d[1] == 1: # xpos fixed:0 free:1
        _xpos_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _xpos_e = 999
    if fit_opt_2d[2] == 1: # ypos fixed:0 free:1
        _ypos_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _ypos_e = 999
    if fit_opt_2d[3] == 1: # vsys fixed:0 free:1
        _vsys_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _vsys_e = 999
    if fit_opt_2d[4] == 1 and n_coeffs_pa_bs == 0: # pa fixed:0 free:1
        _pa_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _pa_e = 999
    if fit_opt_2d[5] == 1 and n_coeffs_incl_bs == 0 : # incl fixed:0 free:1
        _incl_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _incl_e = 999
    if fit_opt_2d[6] == 1 and n_coeffs_vrot_bs == 0: # vrot fixed:0 free:1
        _vrot_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _vrot_e = 999
    if fit_opt_2d[7] == 1: # vrad fixed:0 free:1
        _vrad_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _vrad_e = 999




    if n_coeffs_pa_bs != 0:
        _is = n_ring_params_free
        _ie = _is + n_coeffs_pa_bs 
        tck_pa_bs[1][:n_coeffs_pa_bs] = _tr2dfit_results[_is:_ie]
        tck_pa_bs_e[1][:n_coeffs_pa_bs] = _tr2dfit_results[_is+ndim_total:_ie+ndim_total]
    else:
        _pa_bs = _pa

    if n_coeffs_incl_bs != 0:
        _is = n_ring_params_free + n_coeffs_pa_bs
        _ie = _is + n_coeffs_incl_bs 
        tck_incl_bs[1][:n_coeffs_incl_bs] = _tr2dfit_results[_is:_ie]
        tck_incl_bs_e[1][:n_coeffs_incl_bs] = _tr2dfit_results[_is+ndim_total:_ie+ndim_total]
    else:
        _incl_bs = _incl

    if n_coeffs_vrot_bs != 0: # not constant
        _is = n_ring_params_free + n_coeffs_pa_bs + n_coeffs_incl_bs 
        _ie = _is + n_coeffs_vrot_bs 
        tck_vrot_bs[1][:n_coeffs_vrot_bs] = _tr2dfit_results[_is:_ie]
        tck_vrot_bs_e[1][:n_coeffs_vrot_bs] = _tr2dfit_results[_is+ndim_total:_ie+ndim_total]
    else:
        _vrot_bs = _vrot

    if n_coeffs_vrad_bs != 0: # not constant
        _is = n_ring_params_free + n_coeffs_pa_bs + n_coeffs_incl_bs + n_coeffs_vrot_bs
        _ie = _is + n_coeffs_vrad_bs 
        tck_vrad_bs[1][:n_coeffs_vrad_bs] = _tr2dfit_results[_is:_ie]
        tck_vrad_bs_e[1][:n_coeffs_vrad_bs] = _tr2dfit_results[_is+ndim_total:_ie+ndim_total]
    else:
        _vrad_bs = _vrad




    if ring > _params['r_galaxy_plane_e']:
        ring = _params['r_galaxy_plane_e'] 

    if n_coeffs_pa_bs != 0: # not constant
        _pa_bs = BSpline(*tck_pa_bs, extrapolate=True)(ring) # <-- tck_pa_bs[1] coeffs (0 ~ 360)
        _pa_bs_e = BSpline(*tck_pa_bs_e, extrapolate=True)(ring) # <-- tck_pa_bs[1] coeffs (0 ~ 360)
        _params['_pa_init'] = _pa_bs # update _params
    else:
        _pa_bs_e = _pa_e

    if n_coeffs_incl_bs != 0: # not constant
        _incl_bs = splev(ring, tck_incl_bs)

        _incl_bs_e = BSpline(*tck_incl_bs_e, extrapolate=True)(ring) # <-- tck_incl_bs[1] coeffs (0 ~ 90)
        _params['_incl_init'] = _incl_bs # update _params
    else:
        _incl_bs_e = _incl_e

    if n_coeffs_vrot_bs != 0: # not constant
        _vrot_bs = splev(ring, tck_vrot_bs)
        _vrot_bs_e = BSpline(*tck_vrot_bs_e, extrapolate=True)(ring)
        _params['_vrot_init'] = _vrot_bs # update _params
    else:
        _vrot_bs_e = _vrot_e

    if n_coeffs_vrad_bs != 0: # not constant
        _vrad_bs = BSpline(*tck_vrad_bs, extrapolate=True)(ring)
        _vrad_bs_e = BSpline(*tck_vrad_bs_e, extrapolate=True)(ring)
        _params['_vrad_init'] = _vrad_bs # update _params
    else:
        _vrad_bs_e = _vrad_e

    return _sigma, _sigma_e, _xpos, _xpos_e, _ypos, _ypos_e, _vsys, _vsys_e, _pa_bs, _pa_bs_e, _incl_bs, _incl_bs_e, _vrot_bs, _vrot_bs_e, _vrad_bs, _vrad_bs_e




def extract_tr2dfit_params_part_for_make_model_vf(_tr2dfit_results, _params, fit_opt_2d):





    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    deg_to_rad = np.pi / 180.

    n_coeffs_pa_bs, tck_pa_bs = bspline_ncoeffs_tck(_params, 'pa', _params['nrings_intp'])
    n_coeffs_pa_bs, tck_pa_bs_e = bspline_ncoeffs_tck(_params, 'pa', _params['nrings_intp'])
    n_coeffs_incl_bs, tck_incl_bs = bspline_ncoeffs_tck(_params, 'incl', _params['nrings_intp'])
    n_coeffs_incl_bs, tck_incl_bs_e = bspline_ncoeffs_tck(_params, 'incl', _params['nrings_intp'])
    n_coeffs_vrot_bs, tck_vrot_bs = bspline_ncoeffs_tck(_params, 'vrot', _params['nrings_intp'])
    n_coeffs_vrot_bs, tck_vrot_bs_e = bspline_ncoeffs_tck(_params, 'vrot', _params['nrings_intp'])
    n_coeffs_vrad_bs, tck_vrad_bs = bspline_ncoeffs_tck(_params, 'vrad', _params['nrings_intp'])
    n_coeffs_vrad_bs, tck_vrad_bs_e = bspline_ncoeffs_tck(_params, 'vrad', _params['nrings_intp'])


    n_ring_params_free = 0 # starting from sigma
    if fit_opt_2d[0] == 1: # sigma fixed:0 free:1
        _sigma = _tr2dfit_results[n_ring_params_free]
        _params['_sigma_init'] = _sigma # update params
        n_ring_params_free += 1
    else:
        _sigma = _params['_sigma_init'] # fixed to the previous fitting result
    if fit_opt_2d[1] == 1: # xpos fixed:0 free:1
        _xpos = _tr2dfit_results[n_ring_params_free]
        _params['_xpos_init'] = _xpos # update params
        n_ring_params_free += 1
    else:
        _xpos = _params['_xpos_init'] # fixed to the previous fitting result
    if fit_opt_2d[2] == 1: # ypos fixed:0 free:1
        _ypos = _tr2dfit_results[n_ring_params_free]
        _params['_ypos_init'] = _ypos # update params
        n_ring_params_free += 1
    else:
        _ypos = _params['_ypos_init'] # fixed to the previous fitting result
    if fit_opt_2d[3] == 1: # vsys fixed:0 free:1
        _vsys = _tr2dfit_results[n_ring_params_free]
        _params['_vsys_init'] = _vsys # update params
        n_ring_params_free += 1
    else:
        _vsys = _params['_vsys_init'] # fixed to the previous fitting result
    if fit_opt_2d[4] == 1 and n_coeffs_pa_bs == 0: # pa fixed:0 free:1
        _pa = _tr2dfit_results[n_ring_params_free]
        _params['_pa_init'] = _pa # update params
        n_ring_params_free += 1
    else:
        _pa = _params['_pa_init'] # fixed to the previous fitting result
    if fit_opt_2d[5] == 1 and n_coeffs_incl_bs == 0 : # incl fixed:0 free:1
        _incl = _tr2dfit_results[n_ring_params_free]
        _params['_incl_init'] = _incl # update params
        n_ring_params_free += 1
    else:
        _incl = _params['_incl_init'] # fixed to the previous fitting result
    if fit_opt_2d[6] == 1 and n_coeffs_vrot_bs == 0: # vrot fixed:0 free:1
        _vrot = _tr2dfit_results[n_ring_params_free]
        _params['_vrot_init'] = _vrot # update params
        n_ring_params_free += 1
    else:
        _vrot = _params['_vrot_init'] # fixed to the previous fitting result
    if fit_opt_2d[7] == 1 and n_coeffs_vrad_bs == 0: # vrad fixed:0 free:1
        _vrad = _tr2dfit_results[n_ring_params_free]
        _params['_vrad_init'] = _vrad # update params
        n_ring_params_free += 1
    else:
        _vrad = _params['_vrad_init'] # fixed to the previous fitting result



    ndim_total = n_ring_params_free + n_coeffs_pa_bs + n_coeffs_incl_bs + n_coeffs_vrot_bs + n_coeffs_vrad_bs

    n_ring_params_free = 0 # starting from sigma-e
    if fit_opt_2d[0] == 1: # sigma fixed:0 free:1
        _sigma_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _sigma_e = 999
    if fit_opt_2d[1] == 1: # xpos fixed:0 free:1
        _xpos_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _xpos_e = 999
    if fit_opt_2d[2] == 1: # ypos fixed:0 free:1
        _ypos_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _ypos_e = 999
    if fit_opt_2d[3] == 1: # vsys fixed:0 free:1
        _vsys_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _vsys_e = 999
    if fit_opt_2d[4] == 1 and n_coeffs_pa_bs == 0: # pa fixed:0 free:1
        _pa_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _pa_e = 999
    if fit_opt_2d[5] == 1 and n_coeffs_incl_bs == 0 : # incl fixed:0 free:1
        _incl_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _incl_e = 999
    if fit_opt_2d[6] == 1 and n_coeffs_vrot_bs == 0: # vrot fixed:0 free:1
        _vrot_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _vrot_e = 999
    if fit_opt_2d[7] == 1 and n_coeffs_vrad_bs == 0: # vrot fixed:0 free:1: # vrad fixed:0 free:1ssssss
        _vrad_e = _tr2dfit_results[n_ring_params_free+ndim_total]
        n_ring_params_free += 1
    else:
        _vrad_e = 999






    if n_coeffs_pa_bs != 0:
        _is = n_ring_params_free
        _ie = _is + n_coeffs_pa_bs 
        tck_pa_bs[1][:n_coeffs_pa_bs] = _tr2dfit_results[_is:_ie]
        tck_pa_bs_e[1][:n_coeffs_pa_bs] = _tr2dfit_results[_is+ndim_total:_ie+ndim_total]

    if n_coeffs_incl_bs != 0:
        _is = n_ring_params_free + n_coeffs_pa_bs
        _ie = _is + n_coeffs_incl_bs 
        tck_incl_bs[1][:n_coeffs_incl_bs] = _tr2dfit_results[_is:_ie]
        tck_incl_bs_e[1][:n_coeffs_incl_bs] = _tr2dfit_results[_is+ndim_total:_ie+ndim_total]

    if n_coeffs_vrot_bs != 0: # not constant
        _is = n_ring_params_free + n_coeffs_pa_bs + n_coeffs_incl_bs 
        _ie = _is + n_coeffs_vrot_bs 
        tck_vrot_bs[1][:n_coeffs_vrot_bs] = _tr2dfit_results[_is:_ie]
        tck_vrot_bs_e[1][:n_coeffs_vrot_bs] = _tr2dfit_results[_is+ndim_total:_ie+ndim_total]


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












def ellipse_perimeter(a, b):
    """
    타원의 둘레를 근사적으로 계산하는 함수입니다.
    :param a: 타원의 장축의 반지름
    :param b: 타원의 단축의 반지름
    :return: 타원의 둘레 근사치
    """
    h = ((a - b)**2) / ((a + b)**2)
    perimeter = np.pi * (a + b) * (1 + (3*h) / (10 + np.sqrt(4 - 3*h)))
    return perimeter









def derive_vlos_model_2d_sc(_dyn_params, _tr_model_vf, _norm_vrot_sini_costh_2d, ij_aring, fit_opt, _params, _r_galaxy_plane_2d, r_galaxy_plane, n_coeffs_pa_bs, \
                            tck_pa_bs, n_coeffs_incl_bs, tck_incl_bs, \
                            n_coeffs_vrot_bs, tck_vrot_bs, \
                            _wt_2d_geo, _input_vf):


    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    deg_to_rad = np.pi / 180.

    n_ring_params_free = 0 # starting from sigma
    if fit_opt[0] == 1: # sigma fixed:0 free:1
        _sigma = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _sigma = _params['_sigma_init'] # fixed to the previous fitting result
    if fit_opt[1] == 1: # xpos fixed:0 free:1
        _xpos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _xpos = _params['_xpos_init'] # fixed to the previous fitting result
    if fit_opt[2] == 1: # ypos fixed:0 free:1
        _ypos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _ypos = _params['_ypos_init'] # fixed to the previous fitting result
    if fit_opt[3] == 1: # vsys fixed:0 free:1
        _vsys = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vsys = _params['_vsys_init'] # fixed to the previous fitting result
    if fit_opt[4] == 1 and n_coeffs_pa_bs == 0: # pa fixed:0 free:1
        _pa = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _pa = _params['_pa_init'] # fixed to the previous fitting result
    if fit_opt[5] == 1 and n_coeffs_incl_bs == 0 : # incl fixed:0 free:1
        _incl = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _incl = _params['_incl_init'] # fixed to the previous fitting result
    if fit_opt[6] == 1 and n_coeffs_vrot_bs == 0: # vrot fixed:0 free:1
        _vrot = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrot = _params['_vrot_init'] # fixed to the previous fitting result
    if fit_opt[7] == 1: # vrad fixed:0 free:1
        _vrad = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrad = _params['_vrad_init'] # fixed to the previous fitting result


    cdelt1 = 1
    cdelt2 = 1
    del_x = (ij_aring[:, 0] - _xpos)*cdelt1 # calculate x
    del_y = (ij_aring[:, 1] - _ypos)*cdelt2 # calculate y

    if n_coeffs_pa_bs != 0:
        _is = n_ring_params_free
        _ie = _is + n_coeffs_pa_bs 
        tck_pa_bs[1][:n_coeffs_pa_bs] = _dyn_params[_is:_ie]
    else: # constant PA
        _pa_bs = _pa

    if n_coeffs_incl_bs != 0:
        _is = n_ring_params_free + n_coeffs_pa_bs
        _ie = _is + n_coeffs_incl_bs 
        tck_incl_bs[1][:n_coeffs_incl_bs] = _dyn_params[_is:_ie]
    else: # constant INCL
        _incl_bs = _incl

    if n_coeffs_vrot_bs != 0: # not constant
        _is = n_ring_params_free + n_coeffs_pa_bs + n_coeffs_incl_bs 
        _ie = _is + n_coeffs_vrot_bs
        tck_vrot_bs[1][:n_coeffs_vrot_bs] = _dyn_params[_is:_ie]

    else: # constant VROT
        _vrot_bs = _vrot



    r_galaxy_plane, _nxy = solve_r_galaxy_plane_newton_sc(fit_opt, _params, ij_aring, _xpos, _ypos, _pa, _incl, 0, \
                                                          n_coeffs_pa_bs, tck_pa_bs, n_coeffs_incl_bs, tck_incl_bs)

    if n_coeffs_pa_bs != 0: # not constant
        _pa_bs = BSpline(*tck_pa_bs, extrapolate=True)(r_galaxy_plane) # <-- tck_pa_bs[1] coeffs (0 ~ 360)


    if n_coeffs_incl_bs != 0: # not constant
        _incl_bs = splev(r_galaxy_plane, tck_incl_bs)

    else:
        _incl_bs_max = _incl

    if n_coeffs_vrot_bs != 0: # not constant
        _vrot_bs = splev(r_galaxy_plane, tck_vrot_bs)

    cosp1 = np.cos(deg_to_rad*_pa_bs)
    sinp1 = np.sin(deg_to_rad*_pa_bs)

    cosi1 = np.cos(deg_to_rad*_incl_bs) #cosine
    sini1 = np.sin(deg_to_rad*_incl_bs) # sine

    x_galaxy_plane = (-del_x * sinp1 + del_y * cosp1) # x in plane of galaxy
    y_galaxy_plane = (-del_x * cosp1 - del_y * sinp1) / cosi1 # y in plane of galaxy

    cost1 = x_galaxy_plane / r_galaxy_plane # cosine of angle in plane of galaxy 
    sint1 = y_galaxy_plane / r_galaxy_plane # sine of angle in plane of galaxy





    xr = (-del_x * sinp1 + del_y * cosp1)
    yr = (-del_x * cosp1 - del_y * sinp1) / cosi1

    r = np.sqrt(xr**2 + yr**2)
    theta = np.where(r < 0.1, 0.0, np.arctan2(yr, xr))  # in radians

    costh = np.abs(np.cos(theta))  # in radians

    cosine_weight = costh ** _params['cosine_weight_power']


    _tr_model_vf[ij_aring[:, 1].astype(int), ij_aring[:, 0].astype(int)] = \
        _vsys + (_vrot_bs * cost1 + _vrad * sint1) * sini1


    return _tr_model_vf









def derive_vlos_model_2d_sc_0325(_dyn_params, _tr_model_vf, _norm_vrot_sini_costh_2d, ij_aring, fit_opt, _params, _r_galaxy_plane_2d, r_galaxy_plane, n_coeffs_pa_bs, tck_pa_bs, n_coeffs_incl_bs, tck_incl_bs, n_coeffs_vrot_bs, tck_vrot_bs, _wt_2d_geo, _input_vf):





    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    deg_to_rad = np.pi / 180.




    n_ring_params_free = 0 # starting from sigma
    if fit_opt[0] == 1: # sigma fixed:0 free:1
        _sigma = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _sigma = _params['_sigma_init'] # fixed to the previous fitting result
    if fit_opt[1] == 1: # xpos fixed:0 free:1
        _xpos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _xpos = _params['_xpos_init'] # fixed to the previous fitting result
    if fit_opt[2] == 1: # ypos fixed:0 free:1
        _ypos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _ypos = _params['_ypos_init'] # fixed to the previous fitting result
    if fit_opt[3] == 1: # vsys fixed:0 free:1
        _vsys = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vsys = _params['_vsys_init'] # fixed to the previous fitting result
    if fit_opt[4] == 1 and n_coeffs_pa_bs == 0: # pa fixed:0 free:1
        _pa = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _pa = _params['_pa_init'] # fixed to the previous fitting result
    if fit_opt[5] == 1 and n_coeffs_incl_bs == 0 : # incl fixed:0 free:1
        _incl = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _incl = _params['_incl_init'] # fixed to the previous fitting result
    if fit_opt[6] == 1 and n_coeffs_vrot_bs == 0: # vrot fixed:0 free:1
        _vrot = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrot = _params['_vrot_init'] # fixed to the previous fitting result
    if fit_opt[7] == 1: # vrad fixed:0 free:1
        _vrad = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrad = _params['_vrad_init'] # fixed to the previous fitting result


    cdelt1 = 1
    cdelt2 = 1
    del_x = (ij_aring[:, 0] - _xpos)*cdelt1 # calculate x
    del_y = (ij_aring[:, 1] - _ypos)*cdelt2 # calculate y

    if n_coeffs_pa_bs != 0:
        _is = n_ring_params_free
        _ie = _is + n_coeffs_pa_bs 
        tck_pa_bs[1][:n_coeffs_pa_bs] = _dyn_params[_is:_ie]
    else: # constant PA
        _pa_bs = _pa

    if n_coeffs_incl_bs != 0:
        _is = n_ring_params_free + n_coeffs_pa_bs
        _ie = _is + n_coeffs_incl_bs 
        tck_incl_bs[1][:n_coeffs_incl_bs] = _dyn_params[_is:_ie]
    else: # constant INCL
        _incl_bs = _incl

    if n_coeffs_vrot_bs != 0: # not constant
        _is = n_ring_params_free + n_coeffs_pa_bs + n_coeffs_incl_bs 
        _ie = _is + n_coeffs_vrot_bs 
        tck_vrot_bs[1][:n_coeffs_vrot_bs] = _dyn_params[_is:_ie]

    else: # constant VROT
        _vrot_bs = _vrot
        _vrot_bs_max = _vrot # constant vrot


    r_galaxy_plane, _nxy = solve_r_galaxy_plane_newton_sc(fit_opt, _params, ij_aring, _xpos, _ypos, _pa, _incl, 0, \
                                                          n_coeffs_pa_bs, tck_pa_bs, n_coeffs_incl_bs, tck_incl_bs)







    if n_coeffs_pa_bs != 0: # not constant
        _pa_bs = BSpline(*tck_pa_bs, extrapolate=True)(r_galaxy_plane) # <-- tck_pa_bs[1] coeffs (0 ~ 360)


    if n_coeffs_incl_bs != 0: # not constant
        _incl_bs = splev(r_galaxy_plane, tck_incl_bs)
        _incl_bs_max = BSpline(*tck_incl_bs, extrapolate=True)(_params['r_galaxy_plane_e']*0.5) # <-- tck_incl_bs[1] coeffs (0 ~ 90)
    else:
        _incl_bs_max = _incl

    if n_coeffs_vrot_bs != 0: # not constant
        _vrot_bs = splev(r_galaxy_plane, tck_vrot_bs)
        _vrot_bs_max = np.nanmax(_vrot_bs)

    


    



    cosp1 = np.cos(deg_to_rad*_pa_bs)

    sinp1 = np.sin(deg_to_rad*_pa_bs)

    cosi1 = np.cos(deg_to_rad*_incl_bs) #cosine
    cosi1_max = np.cos(deg_to_rad*_incl_bs_max) #cosine

    sini1 = np.sin(deg_to_rad*_incl_bs) # sine
    sini1_max = np.sin(deg_to_rad*_incl_bs_max) # sine

    _A = (-del_x * sinp1 + del_y * cosp1)
    _B = (-del_x * cosp1 - del_y * sinp1)

    x_galaxy_plane = (-del_x * sinp1 + del_y * cosp1) # x in plane of galaxy
    x_galaxy_plane_max = np.nanmax(x_galaxy_plane)
    r_galaxy_plane_max = np.nanmax(r_galaxy_plane)
    y_galaxy_plane = (-del_x * cosp1 - del_y * sinp1) / cosi1 # y in plane of galaxy


    cost1 = x_galaxy_plane / r_galaxy_plane # cosine of angle in plane of galaxy 
    cost1_max = x_galaxy_plane / (_params['r_galaxy_plane_e'] * 0.5)

    sint1 = y_galaxy_plane / r_galaxy_plane # sine of angle in plane of galaxy


    _alpha = (_vrot_bs / _vrot_bs_max ) * (sini1 / r_galaxy_plane)
    _norm_vrot_sini_costh_2d[ij_aring[:, 1].astype(int), ij_aring[:, 0].astype(int)] = _alpha







    del_x_w = (ij_aring[:, 0] - _params['_xpos_el'])*cdelt1 # calculate x
    del_y_w = (ij_aring[:, 1] - _params['_ypos_el'])*cdelt2 # calculate y

    cosp1_w = np.cos(deg_to_rad*_params['_pa_el'])

    sinp1_w = np.sin(deg_to_rad*_params['_pa_el'])

    cosi1_w = np.cos(deg_to_rad*_params['_incl_el']) #cosine

    sini1_w = np.sin(deg_to_rad*_params['_incl_el']) # sine


    xr = (-del_x * sinp1 + del_y * cosp1)
    yr = (-del_x * cosp1 - del_y * sinp1) / cosi1

    r = np.sqrt(xr**2 + yr**2)
    theta = np.where(r < 0.1, 0.0, np.arctan2(yr, xr))  # in radians

    costh = np.abs(np.cos(theta))  # in radians

    cosine_weight = costh ** _params['cosine_weight_power']



    _bi = r_galaxy_plane * np.cos(_incl_bs*deg_to_rad) # along minor axis

    _li = ellipse_perimeter(r_galaxy_plane, _bi)

    _r_galaxy_min = np.nanmin(r_galaxy_plane)
    _bi_min = _r_galaxy_min * np.cos(_incl_bs*deg_to_rad) # along minor axis
    _li_min = ellipse_perimeter(_r_galaxy_min, _bi_min)



    _r_galaxy_plane_2d[ij_aring[:, 1].astype(int), ij_aring[:, 0].astype(int)] = _vrot_bs

































    
    _tr_model_vf[ij_aring[:, 1].astype(int), ij_aring[:, 0].astype(int)] = \
        _vsys + (_vrot_bs * cost1 + _vrad * sint1) * sini1   # model LOS velocity
    
    










    
    


        
        





    _input_vf_ij = _input_vf[ij_aring[:, 1].astype(int), ij_aring[:, 0].astype(int)] 




    _a = np.nanmax(r_galaxy_plane)
    _b = _a * np.cos(_incl_bs*deg_to_rad) # along minor axis
    _area = 3.141592 * _a * _b




    return _tr_model_vf, _norm_vrot_sini_costh_2d, _vsys, _r_galaxy_plane_2d, _wt_2d_geo, _vrot_bs_max, sini1_max








    _bsfit_vf[_ij_aring_nogrid_available[:, 1].astype(int), _ij_aring_nogrid_available[:, 0].astype(int)] = \
       _vsys + (_vrot_bs * cost1 + _vrad_bs * sint1) * sini1  # model LOS velocity

    




    _input_nogrid_m_trfit_vf[:] = _input_vf_nogrid[:] - _bsfit_vf[:]

    _input_m_bsfit_vf_grid[:] = _input_vf_tofit_2d[:] - _bsfit_vf[:]

    res_median = np.nanmedian(_input_m_bsfit_vf_grid)
    res_std = np.nanstd(_input_m_bsfit_vf_grid)

    clip_min = res_median - 3 * res_std
    clip_max = res_median + 3 * res_std
    _input_m_bsfit_vf_grid = np.where( ((_input_m_bsfit_vf_grid > clip_min) & (_input_m_bsfit_vf_grid < clip_max)), _input_m_bsfit_vf_grid, np.nan)

    write_fits_images(_params, _bsfit_vf, _2dbat_run_i, 'bsfit_vf.fits')
    write_fits_images(_params, _input_m_bsfit_vf_grid, _2dbat_run_i, 'input_m_bsfit_vf_grid_x%dy%d.fits' % (_params['x_grid_tr'], _params['y_grid_tr']))
    write_fits_images(_params, _input_nogrid_m_trfit_vf, _2dbat_run_i, 'input_m_trfit_vf_grid_x1y1.fits')



    return _bsfit_vf




def extract_vrot_bs_tr_rings_given_dyn_params(_input_vf_nogrid, _input_vf_tofit_2d, _tr2dfit_results, _params, fit_opt_2d, _bsfit_vf, _2dbat_run_i):





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


    nrings_reliable = 3*_params['nrings_reliable']
    ring_w = _params['ring_w']
    _tr_rings = np.arange(0, nrings_reliable*ring_w, ring_w, dtype=np.float64)
    _vrot_bs = np.zeros(nrings_reliable, dtype=np.float64)
    _vrot_bs_e = np.zeros(nrings_reliable, dtype=np.float64)

    _tr_rings = np.where(_tr_rings < _params['r_galaxy_plane_e'], _tr_rings, _params['r_galaxy_plane_e'])


    if n_coeffs_vrot_bs != 0 and tck_vrot_bs[2] != 0: # not constant
        _vrot_bs[:] = BSpline(*tck_vrot_bs, extrapolate=True)(_tr_rings)
        _vrot_bs_e[:] = BSpline(*tck_vrot_bs_e, extrapolate=True)(_tr_rings)

    elif n_coeffs_vrot_bs != 0 and tck_vrot_bs[2] == 0: # 0th-order
        _vrot_bs[:] = evaluate_constant_spline_vectorized(_tr_rings, tck_vrot_bs)
        _vrot_bs_e[:] = evaluate_constant_spline_vectorized(_tr_rings, tck_vrot_bs_e)

    else:
        _vrot_bs[:] = _vrot
        _vrot_bs_e[:] = _vrot_e

    return _vrot_bs, _vrot_bs_e




def make_vlos_model_vf_given_dyn_params(_input_vf_nogrid, _input_vf_tofit_2d, _tr2dfit_results, _params, fit_opt_2d, _bsfit_vf, _2dbat_run_i):





    _naxis1 = _params['naxis1']
    _naxis2 = _params['naxis2']
    ring_w = _params['ring_w']
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


    npoints_in_a_ring_total_including_blanks_t, npoints_in_ring_t, _ij_aring_nogrid_available, _wt_2d = define_tilted_ring(_input_vf_nogrid, _xpos, _ypos, 0, 1, 0, 5*_params['r_galaxy_plane_e'], 999, _params)



    r_galaxy_plane, _nxy = solve_r_galaxy_plane_newton_sc(fit_opt_2d, _params, _ij_aring_nogrid_available, _xpos, _ypos, _pa, _incl, 0, \
                                                          n_coeffs_pa_bs, tck_pa_bs, n_coeffs_incl_bs, tck_incl_bs)

    cdelt1 = 1
    cdelt2 = 1
    del_x = (_ij_aring_nogrid_available[:, 0] - _xpos)*cdelt1 # calculate x
    del_y = (_ij_aring_nogrid_available[:, 1] - _ypos)*cdelt2 # calculate y



    if n_coeffs_pa_bs != 0: # not constant
        _pa_bs = BSpline(*tck_pa_bs, extrapolate=True)(r_galaxy_plane) # <-- tck_pa_bs[1] coeffs (0 ~ 360)
    else:
        _pa_bs = _pa

    if n_coeffs_incl_bs != 0: # not constant
        _incl_bs = BSpline(*tck_incl_bs, extrapolate=True)(r_galaxy_plane) # <-- tck_incl_bs[1] coeffs (0 ~ 90)
    else:
        _incl_bs = _incl

    if n_coeffs_vrot_bs != 0: # not constant
        _vrot_bs = BSpline(*tck_vrot_bs, extrapolate=True)(r_galaxy_plane)

    else:
        _vrot_bs = _vrot

        


    if n_coeffs_vrad_bs != 0: # not constant
        _vrad_bs = BSpline(*tck_vrad_bs, extrapolate=True)(r_galaxy_plane)
    else:
        _vrad_bs = _vrad



    cosp1 = np.cos(deg_to_rad*_pa_bs)
    sinp1 = np.sin(deg_to_rad*_pa_bs)
    cosi1 = np.cos(deg_to_rad*_incl_bs) #cosine
    sini1 = np.sin(deg_to_rad*_incl_bs) # sine

    x_galaxy_plane = (-del_x * sinp1 + del_y * cosp1) # x in plane of galaxy
    y_galaxy_plane = (-del_x * cosp1 - del_y * sinp1) / cosi1 # y in plane of galaxy

    cost1 = x_galaxy_plane / r_galaxy_plane # cosine of angle in plane of galaxy 
    sint1 = y_galaxy_plane / r_galaxy_plane # sine of angle in plane of galaxy











    _bsfit_vf[_ij_aring_nogrid_available[:, 1].astype(int), _ij_aring_nogrid_available[:, 0].astype(int)] = \
        _vsys + (_vrot_bs * cost1 + _vrad_bs * sint1) * sini1 # model LOS velocity


    _input_nogrid_m_trfit_vf[:] = _input_vf_nogrid[:] - _bsfit_vf[:]
    _input_m_bsfit_vf_grid[:] = _input_vf_tofit_2d[:] - _bsfit_vf[:]

    write_fits_images(_params, _bsfit_vf, _2dbat_run_i, 'bsfit_vf.fits')
    write_fits_images(_params, _input_m_bsfit_vf_grid, _2dbat_run_i, 'input_m_bsfit_vf_grid_x%dy%d.fits' % (_params['x_grid_tr'], _params['y_grid_tr']))
    write_fits_images(_params, _input_nogrid_m_trfit_vf, _2dbat_run_i, 'input_m_bsfit_vf_grid_x1y1.fits')

    return _bsfit_vf


def write_fits_images(_params, _tr_model_vf, _2dbat_run_i, fitsfile_name):
    _dir_2dbat_PI_output = _params['wdir'] + '/' + _params['_2dbatdir'] + ".%d" % _2dbat_run_i
    if not os.path.exists("%s" % _dir_2dbat_PI_output):
        make_dirs("%s" % _dir_2dbat_PI_output)



    with fits.open(_params['wdir'] + '/' + _params['input_vf'], 'update') as ref_fits:
        input_header = ref_fits[0].header

    hdu = fits.PrimaryHDU(_tr_model_vf, header=input_header)

    output_filename = os.path.join(_dir_2dbat_PI_output, fitsfile_name)  # Use a FITS file extension
    hdu.writeto(output_filename, overwrite=True)





def radius_galaxy_plane(radius, x, y, pa, incl):
    deg_to_rad = np.pi / 180.


    xr = (-x * np.sin(pa * deg_to_rad) + y * np.cos(pa * deg_to_rad))
    yr = (-x * np.cos(pa * deg_to_rad) - y * np.sin(pa * deg_to_rad)) / np.cos(incl * deg_to_rad)

    return radius - np.sqrt(xr * xr + yr * yr)


def make_vlos_model_vf_given_dyn_params_trfit_final_vrot(_input_vf_nogrid, _input_vf_tofit_2d, _tr2dfit_results, \
                                                         _xpos_f_b, _ypos_f_b, _vsys_f_b, _pa_f_b, _incl_f_b, _vrot_f_b, _vrad_f_b, \
                                                         _params, fit_opt_2d, _tr_model_vf, _2dbat_run_i):




    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    deg_to_rad = np.pi / 180.

    _sigma, _sigma_e, _xpos, _xpos_e, _ypos, _ypos_e, _vsys, _vsys_e, \
        _pa, _pa_e, _incl, _incl_e, _vrot, _vrot_e, _vrad, _vrad_e, \
        n_coeffs_pa_bs, tck_pa_bs, tck_pa_bs_e, \
        n_coeffs_incl_bs, tck_incl_bs, tck_incl_bs_e, \
        n_coeffs_vrot_bs, tck_vrot_bs, tck_vrot_bs_e, \
        n_coeffs_vrad_bs, tck_vrad_bs, tck_vrad_bs_e = extract_tr2dfit_params_part_for_make_model_vf(_tr2dfit_results, _params, fit_opt_2d)

    npoints_in_a_ring_total_including_blanks_t, npoints_in_ring_t, _ij_aring_nogrid_available, _wt_2d, = define_tilted_ring(_input_vf_nogrid, _xpos, _ypos, 0, 1, 0, 5*_params['r_galaxy_plane_e'], 0, _params)


    cdelt1 = 1
    cdelt2 = 1
    del_x = (_ij_aring_nogrid_available[:, 0] - _xpos)*cdelt1 # calculate x
    del_y = (_ij_aring_nogrid_available[:, 1] - _ypos)*cdelt2 # calculate y

    _naxis1 = _params['naxis1']
    _naxis2 = _params['naxis2']
    nrings_reliable = _params['nrings_reliable']
    ring_s = 0
    ring_w = _params['ring_w']
    flt_epsilon = 0.000000000000001
    _trfit_final_model_vf = np.full((_naxis2, _naxis1), fill_value=np.nan, dtype=np.float64)
    _trfit_final_model_vf_full = np.full((_naxis2, _naxis1), fill_value=np.nan, dtype=np.float64)
    _input_m_trfit_vf = np.full((_naxis2, _naxis1), fill_value=np.nan, dtype=np.float64)

    _pa_f = np.zeros(nrings_reliable*10, dtype=np.float64)
    _incl_f = np.zeros(nrings_reliable*10, dtype=np.float64)
    _vsys_f = np.zeros(nrings_reliable*10, dtype=np.float64)
    _vrot_f = np.zeros(nrings_reliable*10, dtype=np.float64)
    _vrad_f = np.zeros(nrings_reliable*10, dtype=np.float64)

    _pa_f[:nrings_reliable] = _pa_f_b[:nrings_reliable]
    _incl_f[:nrings_reliable] = _incl_f_b[:nrings_reliable]
    _vrad_f[:nrings_reliable] = _vrad_f_b[:nrings_reliable]
    _vsys_f[:nrings_reliable] = _vsys_f_b[:nrings_reliable]
    _vrot_f[:nrings_reliable] = _vrot_f_b[:nrings_reliable]


    for i in range(_naxis1):
        for j in range(_naxis2):
            x_pixel_from_xpos = i - _xpos
            y_pixel_from_ypos = j - _ypos
            r_pixel_from_centre = np.sqrt(x_pixel_from_xpos**2 + y_pixel_from_ypos**2)
            
            if r_pixel_from_centre > 10 * _params['r_galaxy_plane_e']:  # if radius is outside the outermost ring
                _trfit_final_model_vf[j, i] = np.nan # blank
            else:
                d1 = radius_galaxy_plane(0, x_pixel_from_xpos, y_pixel_from_ypos, _pa_f_b[0], _incl_f_b[0])
                d2 = radius_galaxy_plane(10 * _params['r_galaxy_plane_e'], x_pixel_from_xpos, y_pixel_from_ypos, _pa_f_b[nrings_reliable-1], _incl_f_b[nrings_reliable-1])
                
                if d1 * d2 > 0.0:
                   _trfit_final_model_vf[j, i] = np.nan # blank
                else:
                    for n in range(1, 10 * nrings_reliable):
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

                    cosp1 = np.cos(deg_to_rad*_pa_f[n]) # cosine
                    sinp1 = np.sin(deg_to_rad*_pa_f[n]) # sine
                    cosi1 = np.cos(deg_to_rad*_incl_f[n]) #cosine
                    sini1 = np.sin(deg_to_rad*_incl_f[n]) # sine

                    x_galaxy_plane = (-x_pixel_from_xpos * sinp1 + y_pixel_from_ypos * cosp1) # x in plane of galaxy
                    y_galaxy_plane = (-x_pixel_from_xpos * cosp1 - y_pixel_from_ypos * sinp1) / cosi1 # y in plane of galaxy
                    r_galaxy_plane = (x_galaxy_plane**2 + y_galaxy_plane**2)**0.5

                    if r_galaxy_plane < ring_w and r_galaxy_plane >= 1: # linear interpolation of vrot
                        vrot01 = (_vrot_f[1] / ring_w) * r_galaxy_plane
                    if r_galaxy_plane < 1:
                        vrot01 = 0

                    _trfit_final_model_vf_full[j, i] = vsys01 + (vrot01 * cost + vrad01 * sint) * np.sin(incl01*deg_to_rad)

                    if not np.isinf(_input_vf_nogrid[j, i]) and not np.isnan(_input_vf_nogrid[j, i]):  # no blank
                        _trfit_final_model_vf[j, i] = vsys01 + (vrot01 * cost + vrad01 * sint) * np.sin(incl01*deg_to_rad) 
                    else:
                        _trfit_final_model_vf[j, i] = np.nan
            
            _input_m_trfit_vf[j, i] = _input_vf_nogrid[j, i] - _trfit_final_model_vf[j, i]

    
    write_fits_images(_params, _input_vf_nogrid, _2dbat_run_i, 'input_vf_grid_x1y1.fits')
    write_fits_images(_params, _input_vf_tofit_2d, _2dbat_run_i, 'input_vf_grid_x%dy%d.fits' % (_params['x_grid_2d'], _params['y_grid_2d']))
    write_fits_images(_params, _trfit_final_model_vf, _2dbat_run_i, 'trfit_vf.fits')
    write_fits_images(_params, _trfit_final_model_vf_full, _2dbat_run_i, 'trfit_vf_full.fits')
    write_fits_images(_params, _input_m_trfit_vf, _2dbat_run_i, 'input_m_trfit_vf.fits')



    return _trfit_final_model_vf



















def derive_vlos_model_2d_sc_checkplot(_dyn_params, _tr_model_vf, _wi_2d, ij_aring, fit_opt, _params, nrings_reliable, r_galaxy_plane):





    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    deg_to_rad = np.pi / 180.


    n_coeffs_vrot_bs, tck_vrot_bs = bspline_ncoeffs_tck(_params, 'vrot', _params['nrings_intp'])
    n_coeffs_pa_bs, tck_pa_bs = bspline_ncoeffs_tck(_params, 'pa', _params['nrings_intp'])
    n_coeffs_incl_bs, tck_incl_bs = bspline_ncoeffs_tck(_params, 'incl', _params['nrings_intp'])


    n_ring_params_free = 0 # starting from sigma
    if fit_opt[0] == 1: # sigma fixed:0 free:1
        _sigma = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _sigma = _params['_sigma_init'] # fixed to the previous fitting result
    if fit_opt[1] == 1: # xpos fixed:0 free:1
        _xpos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _xpos = _params['_xpos_init'] # fixed to the previous fitting result
    if fit_opt[2] == 1: # ypos fixed:0 free:1
        _ypos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _ypos = _params['_ypos_init'] # fixed to the previous fitting result
    if fit_opt[3] == 1: # vsys fixed:0 free:1
        _vsys = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vsys = _params['_vsys_init'] # fixed to the previous fitting result
    if fit_opt[4] == 1 and n_coeffs_pa_bs == 0: # pa fixed:0 free:1
        _pa = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _pa = _params['_pa_init'] # fixed to the previous fitting result
    if fit_opt[5] == 1 and n_coeffs_incl_bs == 0 : # incl fixed:0 free:1
        _incl = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _incl = _params['_incl_init'] # fixed to the previous fitting result
    if fit_opt[6] == 1 and n_coeffs_vrot_bs == 0: # vrot fixed:0 free:1
        _vrot = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrot = _params['_vrot_init'] # fixed to the previous fitting result
    if fit_opt[7] == 1: # vrad fixed:0 free:1
        _vrad = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrad = _params['_vrad_init'] # fixed to the previous fitting result



    cdelt1 = 1
    cdelt2 = 1
    del_x = (ij_aring[:, 0] - _xpos)*cdelt1 # calculate x
    del_y = (ij_aring[:, 1] - _ypos)*cdelt2 # calculate y

    r_galaxy_plane, _nxy = solve_r_galaxy_plane_newton_sc(fit_opt, _params, ij_aring, _xpos, _ypos, _pa, _incl, 0)
    r_galaxy_plane = np.where(r_galaxy_plane <= 0, 0.1, r_galaxy_plane)


    if n_coeffs_pa_bs != 0: # not constant
        for _nbs in range(0, n_coeffs_pa_bs):
            tck_pa_bs[1][_nbs] = _dyn_params[n_ring_params_free + _nbs]

        _pa_bs = BSpline(*tck_pa_bs, extrapolate=True)(r_galaxy_plane) # <-- tck_pa_bs[1] coeffs (0 ~ 360)

    else:
        _pa_bs = _pa



    if n_coeffs_incl_bs != 0: # not constant
        for _nbs in range(0, n_coeffs_incl_bs):
            tck_incl_bs[1][_nbs] = _dyn_params[n_ring_params_free + n_coeffs_pa_bs + _nbs]

        _incl_bs = splev(r_galaxy_plane, tck_incl_bs)
    else:
        _incl_bs = _incl

    cosp1 = np.cos(deg_to_rad*_pa_bs)
    sinp1 = np.sin(deg_to_rad*_pa_bs)
    cosi1 = np.cos(deg_to_rad*_incl_bs) #cosine
    sini1 = np.sin(deg_to_rad*_incl_bs) # sine

    x_galaxy_plane = (-del_x * sinp1 + del_y * cosp1) # x in plane of galaxy
    y_galaxy_plane = (-del_x * cosp1 - del_y * sinp1) / cosi1 # y in plane of galaxy

    cost1 = x_galaxy_plane / r_galaxy_plane # cosine of angle in plane of galaxy 
    sint1 = y_galaxy_plane / r_galaxy_plane # sine of angle in plane of galaxy







    if n_coeffs_vrot_bs != 0: # not constant
        for _nbs in range(0, n_coeffs_vrot_bs):
            tck_vrot_bs[1][_nbs] = _dyn_params[n_ring_params_free + n_coeffs_pa_bs + n_coeffs_incl_bs + _nbs]
        _vrot_bs = splev(r_galaxy_plane, tck_vrot_bs)
    else:
        _vrot_bs = _vrot


    _tr_model_vf[:, :] = np.nan # initialize with a blank value
    _tr_model_vf[ij_aring[:, 1].astype(int), ij_aring[:, 0].astype(int)] = \
        _vsys + (_vrot_bs * cost1 + _vrad * sint1) * sini1 # model LOS velocity

    return r_galaxy_plane, _vrot_bs








def derive_vlos_model_2d_confirmed(_dyn_params, _tr_model_vf, ij_aring, fit_opt, _params, nrings_reliable):





    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    deg_to_rad = np.pi / 180.

    n_ring_params_free = 0 # starting from sigma
    if fit_opt[0] == 1: # sigma fixed:0 free:1
        _sigma = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _sigma = 10 # fixed to the previous fitting result
    if fit_opt[1] == 1: # xpos fixed:0 free:1
        _xpos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _xpos = 20 # fixed to the previous fitting result
    if fit_opt[2] == 1: # ypos fixed:0 free:1
        _ypos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _ypos = 20 # fixed to the previous fitting result
    if fit_opt[3] == 1: # vsys fixed:0 free:1
        _vsys = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vsys = 1040 # fixed to the previous fitting result
    if fit_opt[4] == 1: # pa fixed:0 free:1
        _pa = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _pa = 45 # fixed to the previous fitting result
    if fit_opt[5] == 1: # incl fixed:0 free:1
        _incl = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _incl = 60 # fixed to the previous fitting result
    if fit_opt[6] == 1: # vrot fixed:0 free:1
        _vrot = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrot = 50 # fixed to the previous fitting result
    if fit_opt[7] == 1: # vrad fixed:0 free:1
        _vrad = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrad = 0 # fixed to the previous fitting result


    n_coeffs_vrot_bs, tck_vrot_bs = bspline_ncoeffs_tck(_params, 'vrot', _params['nrings_intp'])
    n_coeffs_pa_bs, tck_pa_bs = bspline_ncoeffs_tck(_params, 'pa', _params['nrings_intp'])


    cdelt1 = 1
    cdelt2 = 1
    del_x = (ij_aring[:, 0] - _xpos)*cdelt1 # calculate x
    del_y = (ij_aring[:, 1] - _ypos)*cdelt2 # calculate y



    fx_r_galaxy_plane_id = ray.put(fx_r_galaxy_plane)
    _dyn_params_id = ray.put(_dyn_params)
    fit_opt_id = ray.put(fit_opt)
    _params_id = ray.put(_params)
    ij_aring_id = ray.put(ij_aring)
    nrings_reliable_id = ray.put(nrings_reliable)
    _xpos_id = ray.put(_xpos)
    _ypos_id = ray.put(_ypos)











    for _nbs in range(0, n_coeffs_pa_bs):
        tck_pa_bs[1][_nbs] = _dyn_params[n_ring_params_free + n_coeffs_vrot_bs + _nbs]

    _pa_bs = BSpline(*tck_pa_bs, extrapolate=True)(r_galaxy_plane) # normalised PA_bs [0 ~ 1] <-- tck_pa_bs[1] coeffs (0 ~ 1)

    cosp1 = np.cos(deg_to_rad*_pa_bs)
    sinp1 = np.sin(deg_to_rad*_pa_bs)
    cosi1 = np.cos(deg_to_rad*_incl) #cosine
    sini1 = np.sin(deg_to_rad*_incl) # sine

    x_galaxy_plane = (-del_x * sinp1 + del_y * cosp1) # x in plane of galaxy
    y_galaxy_plane = (-del_x * cosp1 - del_y * sinp1) / cosi1 # y in plane of galaxy

    cost1 = x_galaxy_plane / r_galaxy_plane # cosine of angle in plane of galaxy 
    sint1 = y_galaxy_plane / r_galaxy_plane # sine of angle in plane of galaxy


    for _nbs in range(0, n_coeffs_vrot_bs):
        tck_vrot_bs[1][_nbs] = _dyn_params[n_ring_params_free+_nbs]
    _vrot_bs = splev(r_galaxy_plane, tck_vrot_bs)

    ij_aring[:,2] = _vsys + (_vrot_bs * cost1 + _vrad * sint1) * sini1 # model LOS velocity

    _tr_model_vf[:, :] = np.nan # initialize with a blank value
    _tr_model_vf[ij_aring[:, 1].astype(int), ij_aring[:, 0].astype(int)] = ij_aring[:, 2]

    return _tr_model_vf


def derive_vlos_model_2d_single_core_old(_dyn_params, _tr_model_vf, ij_aring, fit_opt, _params, nrings_reliable):




    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    deg_to_rad = np.pi / 180.

    n_ring_params_free = 0 # starting from sigma
    if fit_opt[0] == 1: # sigma fixed:0 free:1
        _sigma = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _sigma = 10 # fixed to the previous fitting result
    if fit_opt[1] == 1: # xpos fixed:0 free:1
        _xpos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _xpos = 20 # fixed to the previous fitting result
    if fit_opt[2] == 1: # ypos fixed:0 free:1
        _ypos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _ypos = 20 # fixed to the previous fitting result
    if fit_opt[3] == 1: # vsys fixed:0 free:1
        _vsys = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vsys = 1040 # fixed to the previous fitting result
    if fit_opt[4] == 1: # pa fixed:0 free:1
        _pa = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _pa = 45 # fixed to the previous fitting result
    
    if fit_opt[5] == 1: # incl fixed:0 free:1
        _incl = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _incl = 60 # fixed to the previous fitting result
    if fit_opt[6] == 1: # vrot fixed:0 free:1
        _vrot = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrot = 50 # fixed to the previous fitting result
    if fit_opt[7] == 1: # vrad fixed:0 free:1
        _vrad = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrad = 0 # fixed to the previous fitting result


    n_coeffs_vrot_bs, tck_vrot_bs = bspline_ncoeffs_tck(_params, 'vrot', _params['nrings_intp'])
    n_coeffs_pa_bs, tck_pa_bs = bspline_ncoeffs_tck(_params, 'pa', _params['nrings_intp'])


    cdelt1 = 1
    cdelt2 = 1
    del_x = (ij_aring[:, 0] - _xpos)*cdelt1 # calculate x
    del_y = (ij_aring[:, 1] - _ypos)*cdelt2 # calculate y

    _r_galaxy_plane_init = (del_x**2 + del_y**2)**0.5 # actually minimum r_galaxy_plane given PA / INCL
    r_galaxy_plane = optimize.newton(fx_r_galaxy_plane, _r_galaxy_plane_init, args=(_dyn_params, fit_opt, _params, ij_aring[:, 0], ij_aring[:, 1], nrings_reliable), tol=1E-1, maxiter=10000, disp=True) 

    for _nbs in range(0, n_coeffs_pa_bs):
        tck_pa_bs[1][_nbs] = _dyn_params[n_ring_params_free + n_coeffs_vrot_bs + _nbs]
    _pa_bs = BSpline(*tck_pa_bs, extrapolate=True)(r_galaxy_plane)

    _pa_bs = BSpline(*tck_pa_bs, extrapolate=True)(r_galaxy_plane) # normalised PA_bs [0 ~ 1] <-- tck_pa_bs[1] coeffs (0 ~ 1)

    cosp1 = np.cos(deg_to_rad*_pa_bs)
    sinp1 = np.sin(deg_to_rad*_pa_bs)
    cosi1 = np.cos(deg_to_rad*_incl) #cosine
    sini1 = np.sin(deg_to_rad*_incl) # sine

    x_galaxy_plane = (-del_x * sinp1 + del_y * cosp1) # x in plane of galaxy
    y_galaxy_plane = (-del_x * cosp1 - del_y * sinp1) / cosi1 # y in plane of galaxy

    cost1 = x_galaxy_plane / r_galaxy_plane # cosine of angle in plane of galaxy 
    sint1 = y_galaxy_plane / r_galaxy_plane # sine of angle in plane of galaxy


    for _nbs in range(0, n_coeffs_vrot_bs):
        tck_vrot_bs[1][_nbs] = _dyn_params[n_ring_params_free+_nbs]
    _vrot_bs = BSpline(*tck_vrot_bs, extrapolate=True)(r_galaxy_plane)

    ij_aring[:,2] = _vsys + (_vrot_bs * cost1 + _vrad * sint1) * sini1 # model LOS velocity

    _tr_model_vf[:, :] = np.nan # initialize with a blank value
    _tr_model_vf[ij_aring[:, 1].astype(int), ij_aring[:, 0].astype(int)] = ij_aring[:, 2]

    return _tr_model_vf


def derive_vlos_model_2d_test(_dyn_params, _tr_model_vf, ij_aring, fit_opt, _params, nrings_reliable, r_galaxy_plane):




    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    deg_to_rad = np.pi / 180.

    n_ring_params_free = 0 # starting from sigma
    if fit_opt[0] == 1: # sigma fixed:0 free:1
        _sigma = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _sigma = 10 # fixed to the previous fitting result
    if fit_opt[1] == 1: # xpos fixed:0 free:1
        _xpos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _xpos = 20 # fixed to the previous fitting result
    if fit_opt[2] == 1: # ypos fixed:0 free:1
        _ypos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _ypos = 20 # fixed to the previous fitting result
    if fit_opt[3] == 1: # vsys fixed:0 free:1
        _vsys = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vsys = 1040 # fixed to the previous fitting result
    if fit_opt[4] == 1: # pa fixed:0 free:1
        _pa = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _pa = 45 # fixed to the previous fitting result
    if fit_opt[5] == 1: # incl fixed:0 free:1
        _incl = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _incl = 60 # fixed to the previous fitting result
    if fit_opt[6] == 1: # vrot fixed:0 free:1
        _vrot = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrot = 50 # fixed to the previous fitting result
    if fit_opt[7] == 1: # vrad fixed:0 free:1
        _vrad = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrad = 0 # fixed to the previous fitting result


    n_coeffs_pa_bs, tck_pa_bs = bspline_ncoeffs_tck(_params, 'pa', _params['nrings_intp'])
    
    n_coeffs_vrot_bs, tck_vrot_bs = bspline_ncoeffs_tck(_params, 'vrot', _params['nrings_intp'])


    cdelt1 = 1
    cdelt2 = 1
    del_x = (ij_aring[:, 0] - _xpos)*cdelt1 # calculate x
    del_y = (ij_aring[:, 1] - _ypos)*cdelt2 # calculate y
    _r_galaxy_plane_init = (del_x**2 + del_y**2)**0.5 # actually minimum r_galaxy_plane given PA / INCL

    r_galaxy_plane = (del_x**2 + del_y**2)**0.5 # actually minimum r_galaxy_plane given PA / INCL

    for _nbs in range(0, n_coeffs_pa_bs):
        tck_pa_bs[1][_nbs] = _dyn_params[n_ring_params_free + n_coeffs_vrot_bs + _nbs]

    _pa_bs = BSpline(*tck_pa_bs, extrapolate=True)(r_galaxy_plane) # normalised PA_bs [0 ~ 1] <-- tck_pa_bs[1] coeffs (0 ~ 1)

    cosp1 = np.cos(deg_to_rad*_pa_bs)
    sinp1 = np.sin(deg_to_rad*_pa_bs)
    cosi1 = np.cos(deg_to_rad*_incl) #cosine
    sini1 = np.sin(deg_to_rad*_incl) # sine

    x_galaxy_plane = (-del_x * sinp1 + del_y * cosp1) # x in plane of galaxy
    y_galaxy_plane = (-del_x * cosp1 - del_y * sinp1) / cosi1 # y in plane of galaxy

    cost1 = x_galaxy_plane / r_galaxy_plane # cosine of angle in plane of galaxy 
    sint1 = y_galaxy_plane / r_galaxy_plane # sine of angle in plane of galaxy


    for _nbs in range(0, n_coeffs_vrot_bs):
        tck_vrot_bs[1][_nbs] = _dyn_params[n_ring_params_free+_nbs]
    _vrot_bs = BSpline(*tck_vrot_bs, extrapolate=True)(r_galaxy_plane)

    ij_aring[:,2] = _vsys + (_vrot_bs * cost1 + _vrad * sint1) * sini1 # model LOS velocity

    _tr_model_vf[:, :] = np.nan # initialize with a blank value
    _tr_model_vf[ij_aring[:, 1].astype(int), ij_aring[:, 0].astype(int)] = ij_aring[:, 2]

    return r_galaxy_plane, _pa_bs



def derive_vlos_model_2d_vrot_bs_org(_dyn_params, _tr_model_vf, ij_aring, fit_opt, _params):




    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    deg_to_rad = np.pi / 180.

    n_ring_params_free = 0 # starting from sigma
    if fit_opt[0] == 1: # sigma fixed:0 free:1
        _sigma = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _sigma = 10 # fixed to the previous fitting result
    if fit_opt[1] == 1: # xpos fixed:0 free:1
        _xpos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _xpos = 20 # fixed to the previous fitting result
    if fit_opt[2] == 1: # ypos fixed:0 free:1
        _ypos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _ypos = 20 # fixed to the previous fitting result
    if fit_opt[3] == 1: # vsys fixed:0 free:1
        _vsys = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vsys = 173 # fixed to the previous fitting result
    if fit_opt[4] == 1: # pa fixed:0 free:1
        _pa = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _pa = 45 # fixed to the previous fitting result
    if fit_opt[5] == 1: # incl fixed:0 free:1
        _incl = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _incl = 60 # fixed to the previous fitting result
    if fit_opt[6] == 1: # vrot fixed:0 free:1
        _vrot = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrot = 50 # fixed to the previous fitting result
    if fit_opt[7] == 1: # vrad fixed:0 free:1
        _vrad = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrad = 0 # fixed to the previous fitting result


    n_coeffs_vrot_bs, tck_vrot_bs = bspline_ncoeffs_tck(_params, 'vrot')


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

    for _nbs in range(0, n_coeffs_vrot_bs):
        tck_vrot_bs[1][_nbs] = _dyn_params[n_ring_params_free+_nbs]
    _vrot_bs = BSpline(*tck_vrot_bs, extrapolate=True)(r_galaxy_plane)

    ij_aring[:,2] = _vsys + (_vrot_bs * cost1 + _vrad * sint1) * sini1 # model LOS velocity

    _tr_model_vf[:, :] = np.nan 
    _tr_model_vf[ij_aring[:, 1].astype(int), ij_aring[:, 0].astype(int)] = ij_aring[:, 2]

    return _tr_model_vf



def derive_vlos_model_org(_dyn_params, i, j, fit_opt):


    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    deg_to_rad = np.pi / 180.

    n_ring_params_free = 0 # starting from sigma
    if fit_opt[0] == 1: # xpos fixed:0 free:1
        _sigma = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _sigma = 10 # fixed to the previous fitting result
    if fit_opt[1] == 1: # xpos fixed:0 free:1
        _xpos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _xpos = 20 # fixed to the previous fitting result
    if fit_opt[2] == 1: # ypos fixed:0 free:1
        _ypos = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _ypos = 20 # fixed to the previous fitting result
    if fit_opt[3] == 1: # vsys fixed:0 free:1
        _vsys = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vsys = 173 # fixed to the previous fitting result
    if fit_opt[4] == 1: # pa fixed:0 free:1
        _pa = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _pa = 45 # fixed to the previous fitting result
    if fit_opt[5] == 1: # incl fixed:0 free:1
        _incl = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _incl = 60 # fixed to the previous fitting result
    if fit_opt[6] == 1: # vrot fixed:0 free:1
        _vrot = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrot = 50 # fixed to the previous fitting result
    if fit_opt[7] == 1: # vrad fixed:0 free:1
        _vrad = _dyn_params[n_ring_params_free]
        n_ring_params_free += 1
    else:
        _vrad = 0 # fixed to the previous fitting result

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




def run_nested_sampler_trfit(_input_vf, _tr_model_vf, _wt_2d, _ij_aring, _params, fit_opt, ndim, tr_params_priors_init):

    bspline_opt = np.zeros(8, dtype=np.int32)

    _naxis1 = _params['naxis1']
    _naxis2 = _params['naxis2']
    _queue_size = int(_params['num_cpus_tr_dyn'])
    _tr_model_vf = np.full((_naxis2, _naxis1), fill_value=np.nan, dtype=np.float64)
    if _params['_dynesty_class_tr'] == 'static': # recommended calss option



        with mp.Pool(_queue_size) as pool:
            sampler = NestedSampler(loglike_trfit, optimal_prior_trfit, ndim,
                nlive=_params['nlive'],
                update_interval=_params['update_interval'],
                sample=_params['sample'],
                pool=pool,
                queue_size=_queue_size,
                bound=_params['bound'],
                facc=_params['facc'],
                fmove=_params['fmove'],
                max_move=_params['max_move'],
                logl_args=[_input_vf, _tr_model_vf, _wt_2d, _ij_aring, fit_opt, _params, bspline_opt], ptform_args=[fit_opt, bspline_opt, tr_params_priors_init])

            sampler.run_nested(dlogz=_params['dlogz_tr'], maxiter=_params['maxiter_tr'], maxcall=_params['maxcall'], print_progress=_params['print_progress_tr'])

    elif _params['_dynesty_class_tr'] == 'dynamic':



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


    _trfit_results_temp, _logz = get_dynesty_sampler_results(sampler)
    _trfit_results = np.zeros(2*ndim, dtype=np.float32)
    _trfit_results[:2*ndim] = _trfit_results_temp

    return _trfit_results, ndim

    
    



def run_nested_sampler_trfit_2d(_input_vf, combined_vfe_res_w, _tr_model_vf, _input_int_w, _input_vdisp, _wt_2d_geo, _ij_aring, \
                                _params, tr_params_bounds, _r_galaxy_plane_2d, r_galaxy_plane, \
                                tck_vrot_bs_init_from_trfit, tck_pa_bs_init_from_trfit, tck_incl_bs_init_from_trfit, \
                                _2dbat_run_i, region):

    n_ij_aring = _ij_aring.shape[0]

    fit_opt_2d = np.zeros(8, dtype=np.int32)

    _ring_ns_t = np.zeros(_params['nrings']+2, dtype=np.float64)
    _vrot_ns_t = np.zeros(_params['nrings']+2, dtype=np.float64)

    if region == 'entire':
        n_coeffs_vrot_bs, tck_vrot_bs = bspline_ncoeffs_tck(_params, 'vrot', _params['nrings_intp'])
        n_coeffs_pa_bs, tck_pa_bs = bspline_ncoeffs_tck(_params, 'pa', _params['nrings_intp'])
        n_coeffs_incl_bs, tck_incl_bs = bspline_ncoeffs_tck(_params, 'incl', _params['nrings_intp'])

    elif region == 'sector':
        n_coeffs_vrot_bs, tck_vrot_bs = bspline_ncoeffs_tck(_params, 'vrot', _params['nrings_intp'])

        n_coeffs_pa_bs, tck_pa_bs = bspline_ncoeffs_tck(_params, 'pa', _params['nrings_intp'])
        n_coeffs_incl_bs, tck_incl_bs = bspline_ncoeffs_tck(_params, 'incl', _params['nrings_intp'])
        n_coeffs_vrot_bs = 0
        n_coeffs_pa_bs = 0
        n_coeffs_incl_bs = 0


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


    ndim_t += n_coeffs_pa_bs
    ndim_t += n_coeffs_incl_bs
    ndim_t += n_coeffs_vrot_bs

    ndim = ndim_t
    nparams = ndim_t

    bspline_opt = np.zeros(8, dtype=np.int32)


    if _params['_dynesty_class_2d'] == 'static': # recommended calss option



        _queue_size = int(_params['num_cpus_2d_dyn'])
        rstate = np.random.default_rng(2)



        naxis1 = _params['naxis1']
        naxis2 = _params['naxis2']
        _norm = np.full((naxis2, naxis1), fill_value=np.nan, dtype=np.float64)
        _trmodel_vsys = np.full((naxis2, naxis1), fill_value=np.nan, dtype=np.float64)

        with mp.Pool(_queue_size) as pool:
            sampler = NestedSampler(loglike_trfit_2d, optimal_prior_trfit_2d, ndim,
                nlive=_params['nlive'],
                update_interval=_params['update_interval'],
                sample=_params['sample'],
                pool=pool,
                queue_size=_queue_size,
                rstate=rstate,
                first_update={
                    'min_eff': 10,
                    'min_ncall': 200},
                bound=_params['bound'],
                facc=_params['facc'],
                fmove=_params['fmove'],
                max_move=_params['max_move'],
                logl_args=[_input_vf, _tr_model_vf, _ij_aring, fit_opt_2d, bspline_opt, _params, _r_galaxy_plane_2d, r_galaxy_plane, _input_vdisp, \
                           n_coeffs_pa_bs, tck_pa_bs, n_coeffs_incl_bs, tck_incl_bs, \
                            n_coeffs_vrot_bs, tck_vrot_bs, \
                                _trmodel_vsys, combined_vfe_res_w], \
                ptform_args=[fit_opt_2d, bspline_opt, _params, tr_params_bounds, _r_galaxy_plane_2d, \
                             tck_vrot_bs_init_from_trfit, tck_pa_bs_init_from_trfit, tck_incl_bs_init_from_trfit, _input_vf, _ring_ns_t, _vrot_ns_t, \
                             n_coeffs_pa_bs, tck_pa_bs, n_coeffs_incl_bs, tck_incl_bs, n_coeffs_vrot_bs, tck_vrot_bs])


            sampler.run_nested(dlogz=_params['dlogz_2d'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=_params['print_progress_2d'])


            _dir_2dbat_PI_output = _params['wdir'] + '/' + _params['_2dbatdir'] + ".%d" % _2dbat_run_i
            if not os.path.exists("%s" % _dir_2dbat_PI_output):
                make_dirs("%s" % _dir_2dbat_PI_output)

            dyn_sampler_bin = 'dyn_sample.pkl'  # file name
            with open(_dir_2dbat_PI_output + '/' + dyn_sampler_bin, 'wb') as file:
                pickle.dump(sampler.results, file)


    elif _params['_dynesty_class_2d'] == 'dynamic':




        _queue_size = int(_params['num_cpus_2d_dyn'])
        rstate = np.random.default_rng(2)
        with mp.Pool(_queue_size) as pool:
            sampler = DynamicNestedSampler(loglike_trfit_2d, optimal_prior_trfit_2d, ndim,
                nlive=_params['nlive'],
                update_interval=_params['update_interval'],
                sample=_params['sample'],
                pool=pool,
                queue_size=_queue_size,
                rstate=rstate,
                first_update={
                    'min_eff': 10,
                    'min_ncall': 200},
                bound=_params['bound'],
                facc=_params['facc'],
                fmove=_params['fmove'],
                max_move=_params['max_move'],
                logl_args=[_input_vf, _tr_model_vf, _ij_aring, fit_opt_2d, bspline_opt, _params, _r_galaxy_plane_2d, r_galaxy_plane, _wt_2d, n_coeffs_pa_bs, tck_pa_bs, n_coeffs_incl_bs, tck_incl_bs, n_coeffs_vrot_bs, tck_vrot_bs], \
                ptform_args=[fit_opt_2d, bspline_opt, _params, tr_params_bounds, _r_galaxy_plane_2d, tck_vrot_bs_init_from_trfit, tck_pa_bs_init_from_trfit, tck_incl_bs_init_from_trfit, _input_vf, _ring_ns_t, _vrot_ns_t, \
                             n_coeffs_pa_bs, tck_pa_bs, n_coeffs_incl_bs, tck_incl_bs, n_coeffs_vrot_bs, tck_vrot_bs])


            sampler.run_nested(dlogz_init=_params['dlogz_2d'], nlive_init=100, nlive_batch=50, n_effective = 20000, maxiter_init=_params['maxiter'], maxcall_init=_params['maxcall'], print_progress=_params['print_progress_2d'])





    res0 = sampler.results
    _trfit_results_temp, _logz = get_dynesty_sampler_results(sampler)

    samples, weights = sampler.results.samples, np.exp(sampler.results.logwt - sampler.results.logz[-1])

    res_total = []
    rstate = np.random.default_rng(2)
    for i in range(100):
        res_t = dyfunc.resample_run(res0, rstate=rstate)
        res_total.append(res_t)

    x_arr = np.array([dyfunc.mean_and_cov(res_t.samples,
                      weights=np.exp(res_t.logwt - res_t.logz[-1]))[1]
                      for res_t in res_total])

    x_arr = [np.sqrt(np.diag(x)) for x in x_arr]

    std_resample_run = np.round(np.mean(x_arr, axis=0), 3)
    x_std = np.round(np.std(x_arr, axis=0), 3)


    _trfit_results = np.zeros(2*ndim, dtype=np.float32)
    _trfit_results[:2*ndim] = _trfit_results_temp

    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        sampler.results.summary()
        dyn_summary = buf.getvalue()

    _dir_2dbat_PI_output = os.path.join(_params['wdir'], _params['_2dbatdir'] + ".%d" % _2dbat_run_i)

    if not os.path.exists(_dir_2dbat_PI_output):
        os.makedirs(_dir_2dbat_PI_output)

    summary_file_path = os.path.join(_dir_2dbat_PI_output, '2dbat_PI_dynesty.summary.txt')
    with open(summary_file_path, 'w') as file:
        file.write(dyn_summary)

    _fig1, _axes = dyplot.cornerplot(res0, show_titles=True)
    _fig1.savefig('%s/dyn_posteriors_cornerplot.png' % _dir_2dbat_PI_output)


    _fig3, _axes = dyplot.traceplot(res0, show_titles=True)
    _fig3.savefig('%s/dyn_traceplot.png' % _dir_2dbat_PI_output)



    return _trfit_results, ndim, fit_opt_2d, std_resample_run

    
    




























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
    gfit_priors_init = np.zeros(2*5, dtype=np.float32)
    _x_boundaries = np.full(2*ngauss, fill_value=-1E11, dtype=np.float32)
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
    
                if(ndim * (ndim + 1) // 2 > _params['nlive']):
                    _params['nlive'] = 1 + ndim * (ndim + 1) // 2 # optimal minimum nlive
    


                if _params['_dynesty_class_'] == 'static':
                    sampler = NestedSampler(loglike_d, optimal_prior, ndim,
                        nlive=_params['nlive'],
                        update_interval=_params['update_interval'],
                        sample=_params['sample'],
                        bound=_params['bound'],
                        facc=_params['facc'],
                        fmove=_params['fmove'],
                        max_move=_params['max_move'],
                        logl_args=[(_inputDataCube[:, j, i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])

                    sampler.run_nested(dlogz=_params['dlogz_2d'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=True)

                elif _params['_dynesty_class_'] == 'dynamic':
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

                _gfit_results_temp, _logz = get_dynesty_sampler_results(sampler)
    
    
                _x_boundaries[0:ngauss] = _gfit_results_temp[2:nparams:3] - 5*_gfit_results_temp[3:nparams:3] # x - 3*std
    
                _x_boundaries[ngauss:2*ngauss] = _gfit_results_temp[2:nparams:3] + 5*_gfit_results_temp[3:nparams:3] # x + 3*std
    
                _x_boundaries_ft = np.delete(_x_boundaries, np.where(_x_boundaries < -1E3))
                _x_lower = np.sort(_x_boundaries_ft)[0]
                _x_upper = np.sort(_x_boundaries_ft)[-1]
                _x_lower = _x_lower if _x_lower > 0 else 0
                _x_upper = _x_upper if _x_upper < 1 else 1
    
                _f_ngfit = f_gaussian_model(_x, _gfit_results_temp, ngauss)
                _res_spect = ((_inputDataCube[:, j, i]-_f_min)/(_f_max-_f_min) - _f_ngfit)
                _index_t = np.argwhere((_x < _x_lower) | (_x > _x_upper))
                _res_spect_ft = np.delete(_res_spect, _index_t)
    
                _rms[k] = np.std(_res_spect_ft)*(_f_max - _f_min)
                _bg[k] = _gfit_results_temp[1]*(_f_max - _f_min) + _f_min # bg
                print(i, j, _rms[k], _bg[k])
                k += 1

    zero_to_nan_rms = np.where(_rms == 0.0, np.nan, _rms)
    zero_to_nan_bg = np.where(_bg == 0.0, np.nan, _bg)

    _rms_med = np.nanmedian(zero_to_nan_rms)
    _bg_med = np.nanmedian(zero_to_nan_bg)
    _params['_rms_med'] = _rms_med
    _params['_bg_med'] = _bg_med
    print("rms_med:_", _rms_med)
    print("bg_med:_", _bg_med)


def little_derive_rms_npoints(_inputDataCube, i, j, _x, _f_min, _f_max, ngauss, _gfit_results_temp):

    ndim = 3*ngauss + 2
    nparams = ndim

    _x_boundaries = np.full(2*ngauss, fill_value=-1E11, dtype=np.float32)
    _x_boundaries[0:ngauss] = _gfit_results_temp[2:nparams:3] - 5*_gfit_results_temp[3:nparams:3] # x - 3*std

    _x_boundaries[ngauss:2*ngauss] = _gfit_results_temp[2:nparams:3] + 5*_gfit_results_temp[3:nparams:3] # x + 3*std

    _x_boundaries_ft = np.delete(_x_boundaries, np.where(_x_boundaries < -1E3))
    _x_lower = np.sort(_x_boundaries_ft)[0]
    _x_upper = np.sort(_x_boundaries_ft)[-1]
    _x_lower = _x_lower if _x_lower > 0 else 0
    _x_upper = _x_upper if _x_upper < 1 else 1

    _f_ngfit = f_gaussian_model(_x, _gfit_results_temp, ngauss)
    _res_spect = ((_inputDataCube[:, j, i]-_f_min)/(_f_max-_f_min) - _f_ngfit)
    _index_t = np.argwhere((_x < _x_lower) | (_x > _x_upper))
    _res_spect_ft = np.delete(_res_spect, _index_t)

    _rms_ngfit = np.std(_res_spect_ft) # normalised


    del(_x_boundaries, _x_boundaries_ft, _index_t, _res_spect_ft)
    gc.collect()

    return _rms_ngfit # resturn normalised _rms



def run_dynesty_sampler_optimal_priors(_inputDataCube, _x, _peak_sn_map, _sn_int_map, _params, _is, _ie, i, _js, _je, _cube_mask_2d):

    _max_ngauss = _params['max_ngauss']
    _vel_min = _params['vel_min']
    _vel_max = _params['vel_max']
    _cdelt3 = _params['cdelt3']

    gfit_results = np.zeros(((_je-_js), _max_ngauss, 2*(2+3*_max_ngauss) + 7 + _max_ngauss), dtype=np.float32)
    _x_boundaries = np.full(2*_max_ngauss, fill_value=-1E11, dtype=np.float32)

    for j in range(0, _je -_js):

        _f_max = np.max(_inputDataCube[:,j+_js,i]) # peak flux : being used for normalization
        _f_min = np.min(_inputDataCube[:,j+_js,i]) # lowest flux : being used for normalization

        gfit_priors_init = np.zeros(2*5, dtype=np.float32)
        gfit_priors_init = [0.0, 0.0, 0.001, 0.001, 0.001, 0.5, 0.6, 0.999, 0.999, 1.01]

        if _cube_mask_2d[j+_js, i] <= 0 : # if masked, then skip : NOTE THE MASK VALUE SHOULD BE zero or negative.
            print("mask filtered: %d %d | peak S/N: %.1f :: S/N limit: %.1f | integrated S/N: %.1f :: S/N limit: %.1f :: f_min: %e :: f_max: %e" \
                % (i, j+_js, _peak_sn_map[j+_js, i], _params['peak_sn_limit'], _sn_int_map[j+_js, i], _params['int_sn_limit'], _f_min, _f_max))

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



            if _params['_dynesty_class_'] == 'static':
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

            elif _params['_dynesty_class_'] == 'dynamic':
                dsampler = DynamicNestedSampler(loglike_d, optimal_prior, ndim,
                    nlive=_params['nlive'],
                    sample=_params['sample'],
                    bound=_params['bound'],
                    facc=_params['facc'],
                    fmove=_params['fmove'],
                    max_move=_params['max_move'],
                    logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])
                dsampler.run_nested(dlogz_init=_params['dlogz_2d'], maxiter_init=_params['maxiter'], maxcall_init=_params['maxcall'], print_progress=_params['print_progress'])

            _gfit_results_temp, _logz = get_dynesty_sampler_results(sampler)

            gfit_results[j][k][:2*nparams] = _gfit_results_temp

            _rms_ngfit = little_derive_rms_npoints(_inputDataCube, i, j+_js, _x, _f_min, _f_max, ngauss, _gfit_results_temp)

            if ngauss == 1: # check the peak s/n

                nparams_n = 3*(ngauss+1) + 2
                gfit_priors_init = np.zeros(2*nparams_n, dtype=np.float32)
                gfit_priors_init[:nparams] = _gfit_results_temp[:nparams] - _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]
                gfit_priors_init[nparams+3:2*nparams+3] = _gfit_results_temp[:nparams] + _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]


                _bg_sgfit = _gfit_results_temp[1]
                _p_sgfit = _gfit_results_temp[4] # bg already subtracted
                _peak_sn_sgfit = _p_sgfit/_rms_ngfit

                if _peak_sn_sgfit < _params['peak_sn_limit']: 
                    print("skip the rest of Gaussian fits: %d %d | rms:%.1f | bg:%.1f | peak:%.1f | peak_sgfit s/n: %.1f < %.1f" % (i, j+_js, _rms_ngfit, _bg_sgfit, _p_sgfit, _peak_sn_sgfit, _params['peak_sn_limit']))

                    for l in range(0, _max_ngauss):
                        if l == 0:
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

                    gfit_results[j][k][0] = gfit_results[j][k][0]*(_f_max - _f_min) # sigma-flux
                    gfit_results[j][k][1] = gfit_results[j][k][1]*(_f_max - _f_min) + _f_min # bg-flux
                    gfit_results[j][k][6 + 3*k] = gfit_results[j][k][6 + 3*k]*(_f_max - _f_min) # bg-flux-e
                    _bg_flux = gfit_results[j][k][1]
        
                    for m in range(0, k+1):
                        if _cdelt3 > 0: # if velocity axis is with increasing order
                            gfit_results[j][k][2 + 3*m] = gfit_results[j][k][2 + 3*m]*(_vel_max - _vel_min) + _vel_min # velocity
                        elif _cdelt3 < 0: # if velocity axis is with decreasing order
                            gfit_results[j][k][2 + 3*m] = gfit_results[j][k][2 + 3*m]*(_vel_min - _vel_max) + _vel_max # velocity

                        gfit_results[j][k][3 + 3*m] = gfit_results[j][k][3 + 3*m]*(_vel_max - _vel_min) # velocity-dispersion

                        gfit_results[j][k][4 + 3*m] = gfit_results[j][k][4 + 3*m]*(_f_max - _f_min) # peak flux
        
                        gfit_results[j][k][7 + 3*(m+k)] = gfit_results[j][k][7 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-e
                        gfit_results[j][k][8 + 3*(m+k)] = gfit_results[j][k][8 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-dispersion-e
                        gfit_results[j][k][9 + 3*(m+k)] = gfit_results[j][k][9 + 3*(m+k)]*(_f_max - _f_min) # flux-e

                    gfit_results[j][k][2*(3*_max_ngauss+2)+k] = gfit_results[j][k][2*(3*_max_ngauss+2)+k]*(_f_max - _f_min) # rms-(k+1)gfit
                    continue


            if ngauss < _max_ngauss:
                nparams_n = 3*(ngauss+1) + 2
                gfit_priors_init = np.zeros(2*nparams_n, dtype=np.float32)
                gfit_priors_init[:nparams] = _gfit_results_temp[:nparams] - _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]
                gfit_priors_init[nparams+3:2*nparams+3] = _gfit_results_temp[:nparams] + _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]
    
                _x_min_t = _gfit_results_temp[2:nparams:3].min()
                _x_max_t = _gfit_results_temp[2:nparams:3].max()
                _std_min_t = _gfit_results_temp[3:nparams:3].min()
                _std_max_t = _gfit_results_temp[3:nparams:3].max()
                _p_min_t = _gfit_results_temp[4:nparams:3].min()
                _p_max_t = _gfit_results_temp[4:nparams:3].max()

                gfit_priors_init[0] = _params['sigma_prior_lowerbound_factor']*_gfit_results_temp[0]
                gfit_priors_init[nparams_n] = _params['sigma_prior_upperbound_factor']*_gfit_results_temp[0]

                gfit_priors_init[1] = _params['bg_prior_lowerbound_factor']*_gfit_results_temp[1]
                gfit_priors_init[nparams_n + 1] = _params['bg_prior_upperbound_factor']*_gfit_results_temp[1]


                if ngauss == 1:
                    gfit_priors_init[nparams] = _params['x_lowerbound_gfit']
                    gfit_priors_init[2*nparams+3] = _params['x_upperbound_gfit']
                else:
                    gfit_priors_init[nparams] = _x_min_t - _params['x_prior_lowerbound_factor']*_std_max_t
                    gfit_priors_init[2*nparams+3] = _x_max_t + _params['x_prior_upperbound_factor']*_std_max_t

                gfit_priors_init[nparams+1] = _params['std_prior_lowerbound_factor']*_std_min_t
                gfit_priors_init[2*nparams+4] = _params['std_prior_upperbound_factor']*_std_max_t
    
                gfit_priors_init[nparams+2] = _params['p_prior_lowerbound_factor']*_p_max_t # 5% of the maxium flux
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







            gfit_results[j][k][0] = gfit_results[j][k][0]*(_f_max - _f_min) # sigma-flux
            gfit_results[j][k][1] = gfit_results[j][k][1]*(_f_max - _f_min) + _f_min # bg-flux
            gfit_results[j][k][6 + 3*k] = gfit_results[j][k][6 + 3*k]*(_f_max - _f_min) # bg-flux-e
            _bg_flux = gfit_results[j][k][1]

            for m in range(0, k+1):

                if _cdelt3 > 0: # if velocity axis is with increasing order
                    gfit_results[j][k][2 + 3*m] = gfit_results[j][k][2 + 3*m]*(_vel_max - _vel_min) + _vel_min # velocity
                elif _cdelt3 < 0: # if velocity axis is with decreasing order
                    gfit_results[j][k][2 + 3*m] = gfit_results[j][k][2 + 3*m]*(_vel_min - _vel_max) + _vel_max # velocity

                gfit_results[j][k][3 + 3*m] = gfit_results[j][k][3 + 3*m]*(_vel_max - _vel_min) # velocity-dispersion

                gfit_results[j][k][4 + 3*m] = gfit_results[j][k][4 + 3*m]*(_f_max - _f_min) # peak flux

                gfit_results[j][k][7 + 3*(m+k)] = gfit_results[j][k][7 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-e
                gfit_results[j][k][8 + 3*(m+k)] = gfit_results[j][k][8 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-dispersion-e

                gfit_results[j][k][9 + 3*(m+k)] = gfit_results[j][k][9 + 3*(m+k)]*(_f_max - _f_min) # flux-e

            gfit_results[j][k][2*(3*_max_ngauss+2)+k] = gfit_results[j][k][2*(3*_max_ngauss+2)+k]*(_f_max - _f_min) # rms-(k+1)gfit


            print(gfit_results)



    return gfit_results

    
    

def run_dynesty_sampler_optimal_priors_org(_inputDataCube, _x, _peak_sn_map, _sn_int_map, _params, _is, _ie, i, _js, _je):

    _max_ngauss = _params['max_ngauss']
    _vel_min = _params['vel_min']
    _vel_max = _params['vel_max']
    gfit_results = np.zeros(((_je-_js), _max_ngauss, 2*(2+3*_max_ngauss) + 7 + _max_ngauss), dtype=np.float32)
    _x_boundaries = np.full(2*_max_ngauss, fill_value=-1E11, dtype=np.float32)

    for j in range(0, _je -_js):

        _f_max = np.max(_inputDataCube[:,j+_js,i]) # peak flux : being used for normalization
        _f_min = np.min(_inputDataCube[:,j+_js,i]) # lowest flux : being used for normalization

        gfit_priors_init = np.zeros(2*5, dtype=np.float32)
        gfit_priors_init = [0.0, 0.0, 0.01, 0.01, 0.01, 0.5, 0.6, 0.99, 0.6, 1.01]

        if _sn_int_map[j+_js, i] < _params['int_sn_limit'] or _peak_sn_map[j+_js, i] < _params['peak_sn_limit'] \
            or np.isnan(_f_max) or np.isnan(_f_min) \
            or np.isinf(_f_min) or np.isinf(_f_min):

            print("low S/N: %d %d | peak S/N: %.1f :: S/N limit: %.1f | integrated S/N: %.1f :: S/N limit: %.1f :: f_min: %e :: f_max: %e" \
                % (i, j+_js, _peak_sn_map[j+_js, i], _params['peak_sn_limit'], _sn_int_map[j+_js, i], _params['int_sn_limit'], _f_min, _f_max))
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



            if _params['_dynesty_class_'] == 'static':
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

            elif _params['_dynesty_class_'] == 'dynamic':
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


            _gfit_results_temp, _logz = get_dynesty_sampler_results(sampler)

            gfit_results[j][k][:2*nparams] = _gfit_results_temp

            _rms_ngfit = little_derive_rms_npoints(_inputDataCube, i, j+_js, _x, _f_min, _f_max, ngauss, _gfit_results_temp)

            if ngauss == 1: # check the peak s/n

                nparams_n = 3*(ngauss+1) + 2
                gfit_priors_init = np.zeros(2*nparams_n, dtype=np.float32)
                gfit_priors_init[:nparams] = _gfit_results_temp[:nparams] - _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]
                gfit_priors_init[nparams+3:2*nparams+3] = _gfit_results_temp[:nparams] + _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]


                _bg_sgfit = _gfit_results_temp[1]
                _p_sgfit = _gfit_results_temp[4] # bg already subtracted
                _peak_sn_sgfit = _p_sgfit/_rms_ngfit

                if _peak_sn_sgfit < _params['peak_sn_limit']: 
                    print("skip the rest of Gaussian fits: %d %d | rms:%.1f | bg:%.1f | peak:%.1f | peak_sgfit s/n: %.1f < %.1f" % (i, j+_js, _rms_ngfit, _bg_sgfit, _p_sgfit, _peak_sn_sgfit, _params['peak_sn_limit']))

                    for l in range(0, _max_ngauss):
                        if l == 0:
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

                    gfit_results[j][k][0] = gfit_results[j][k][0]*(_f_max - _f_min) # sigma-flux
                    gfit_results[j][k][1] = gfit_results[j][k][1]*(_f_max - _f_min) + _f_min # bg-flux
                    gfit_results[j][k][6 + 3*k] = gfit_results[j][k][6 + 3*k]*(_f_max - _f_min) # bg-flux-e
                    _bg_flux = gfit_results[j][k][1]
        
                    for m in range(0, k+1):
                        gfit_results[j][k][2 + 3*m] = gfit_results[j][k][2 + 3*m]*(_vel_max - _vel_min) + _vel_min # velocity
                        gfit_results[j][k][3 + 3*m] = gfit_results[j][k][3 + 3*m]*(_vel_max - _vel_min) # velocity-dispersion
                        gfit_results[j][k][4 + 3*m] = gfit_results[j][k][4 + 3*m]*(_f_max - _f_min) # flux
        
                        gfit_results[j][k][7 + 3*(m+k)] = gfit_results[j][k][7 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-e
                        gfit_results[j][k][8 + 3*(m+k)] = gfit_results[j][k][8 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-dispersion-e
                        gfit_results[j][k][9 + 3*(m+k)] = gfit_results[j][k][9 + 3*(m+k)]*(_f_max - _f_min) # flux-e

                    gfit_results[j][k][2*(3*_max_ngauss+2)+k] = gfit_results[j][k][2*(3*_max_ngauss+2)+k]*(_f_max - _f_min) # rms-(k+1)gfit
                    continue


            if ngauss < _max_ngauss:
                nparams_n = 3*(ngauss+1) + 2
                gfit_priors_init = np.zeros(2*nparams_n, dtype=np.float32)
                gfit_priors_init[:nparams] = _gfit_results_temp[:nparams] - _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]
                gfit_priors_init[nparams+3:2*nparams+3] = _gfit_results_temp[:nparams] + _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]
    
                _x_min_t = _gfit_results_temp[2:nparams:3].min()
                _x_max_t = _gfit_results_temp[2:nparams:3].max()
                _std_min_t = _gfit_results_temp[3:nparams:3].min()
                _std_max_t = _gfit_results_temp[3:nparams:3].max()
                _p_min_t = _gfit_results_temp[4:nparams:3].min()
                _p_max_t = _gfit_results_temp[4:nparams:3].max()

                gfit_priors_init[0] = _params['sigma_prior_lowerbound_factor']*_gfit_results_temp[0]
                gfit_priors_init[nparams_n] = _params['sigma_prior_upperbound_factor']*_gfit_results_temp[0]

                gfit_priors_init[1] = _params['bg_prior_lowerbound_factor']*_gfit_results_temp[1]
                gfit_priors_init[nparams_n + 1] = _params['bg_prior_upperbound_factor']*_gfit_results_temp[1]


                if ngauss == 1:
                    gfit_priors_init[nparams] = _params['x_lowerbound_gfit']
                    gfit_priors_init[2*nparams+3] = _params['x_upperbound_gfit']
                else:
                    gfit_priors_init[nparams] = _x_min_t - _params['x_prior_lowerbound_factor']*_std_max_t
                    gfit_priors_init[2*nparams+3] = _x_max_t + _params['x_prior_upperbound_factor']*_std_max_t

                gfit_priors_init[nparams+1] = _params['std_prior_lowerbound_factor']*_std_min_t
                gfit_priors_init[2*nparams+4] = _params['std_prior_upperbound_factor']*_std_max_t
    
                gfit_priors_init[nparams+2] = _params['p_prior_lowerbound_factor']*_p_max_t # 5% of the maxium flux
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






            gfit_results[j][k][0] = gfit_results[j][k][0]*(_f_max - _f_min) # sigma-flux
            gfit_results[j][k][1] = gfit_results[j][k][1]*(_f_max - _f_min) + _f_min # bg-flux
            gfit_results[j][k][6 + 3*k] = gfit_results[j][k][6 + 3*k]*(_f_max - _f_min) # bg-flux-e
            _bg_flux = gfit_results[j][k][1]

            for m in range(0, k+1):
                gfit_results[j][k][2 + 3*m] = gfit_results[j][k][2 + 3*m]*(_vel_max - _vel_min) + _vel_min # velocity
                gfit_results[j][k][3 + 3*m] = gfit_results[j][k][3 + 3*m]*(_vel_max - _vel_min) # velocity-dispersion
                gfit_results[j][k][4 + 3*m] = gfit_results[j][k][4 + 3*m]*(_f_max - _f_min)  # peak flux

                gfit_results[j][k][7 + 3*(m+k)] = gfit_results[j][k][7 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-e
                gfit_results[j][k][8 + 3*(m+k)] = gfit_results[j][k][8 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-dispersion-e
                gfit_results[j][k][9 + 3*(m+k)] = gfit_results[j][k][9 + 3*(m+k)]*(_f_max - _f_min) # peak flux-e

            gfit_results[j][k][2*(3*_max_ngauss+2)+k] = gfit_results[j][k][2*(3*_max_ngauss+2)+k]*(_f_max - _f_min) # rms-(k+1)gfit


    return gfit_results

    
    

def run_dynesty_sampler_uniform_priors(_x, _inputDataCube, _is, _ie, i, _js, _je, _max_ngauss, _vel_min, _vel_max):

    gfit_results = np.zeros(((_je-_js), _max_ngauss, 2*(2+3*_max_ngauss)+7), dtype=np.float32)
    for j in range(0, _je -_js):

        _f_max = np.max(_inputDataCube[:,j+_js,i]) # peak flux : being used for normalization
        _f_min = np.min(_inputDataCube[:,j+_js,i]) # lowest flux : being used for normalization

        gfit_priors_init = np.zeros(2*5, dtype=np.float32)
        gfit_priors_init = [0.0, 0.0, 0.01, 0.01, 0.01, 0.5, 0.5, 0.9, 0.6, 1.01]
        for k in range(0, _max_ngauss):
            ngauss = k+1  # set the number of gaussian
            ndim = 3*ngauss + 2
            nparams = ndim

    
            print("processing: %d %d gauss-%d" % (i, j+_js, ngauss))



            if _params['_dynesty_class_'] == 'static':
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

            elif _params['_dynesty_class_'] == 'dynamic':
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

            _gfit_results_temp, _logz = get_dynesty_sampler_results(sampler)

            gfit_results[j][k][:2*nparams] = _gfit_results_temp

            print(gfit_priors_init)

            gfit_results[j][k][2*(3*_max_ngauss+2)+0] = _logz
            gfit_results[j][k][2*(3*_max_ngauss+2)+1] = _is
            gfit_results[j][k][2*(3*_max_ngauss+2)+2] = _ie
            gfit_results[j][k][2*(3*_max_ngauss+2)+3] = _js
            gfit_results[j][k][2*(3*_max_ngauss+2)+4] = _je
            gfit_results[j][k][2*(3*_max_ngauss+2)+5] = i
            gfit_results[j][k][2*(3*_max_ngauss+2)+6] = _js + j





            gfit_results[j][k][1] = gfit_results[j][k][1]*(_f_max - _f_min) + _f_min # bg-flux
            gfit_results[j][k][6 + 3*k] = gfit_results[j][k][6 + 3*k]*(_f_max - _f_min) # bg-flux-e

            for m in range(0, k+1):
                gfit_results[j][k][2 + 3*m] = gfit_results[j][k][2 + 3*m]*(_vel_max - _vel_min) + _vel_min # velocity
                gfit_results[j][k][3 + 3*m] = gfit_results[j][k][3 + 3*m]*(_vel_max - _vel_min) # velocity-dispersion
                gfit_results[j][k][4 + 3*m] = gfit_results[j][k][4 + 3*m]*(_f_max - _f_min) + _f_min # flux

                gfit_results[j][k][7 + 3*(m+k)] = gfit_results[j][k][7 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-e
                gfit_results[j][k][8 + 3*(m+k)] = gfit_results[j][k][8 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-dispersion-e
                gfit_results[j][k][9 + 3*(m+k)] = gfit_results[j][k][9 + 3*(m+k)]*(_f_max - _f_min) # flux-e

    
    del(ndim, nparams, ngauss, sampler)
    gc.collect()

    return gfit_results

    
    



def get_dynesty_sampler_results(_sampler):
    samples = _sampler.results.samples  # samples
    weights = exp(_sampler.results.logwt - _sampler.results.logz[-1])  # normalized weights


    quantiles = [dyfunc.quantile(samps, [0.1, 0.9], weights=weights)
                for samps in samples.T]
    
    mean, cov = dyfunc.mean_and_cov(samples, weights)
    std = np.sqrt(np.diag(cov)) 
    bestfit_results = _sampler.results.samples[-1, :]
    log_Z = _sampler.results.logz[-1]




    del(samples, weights, quantiles)
    gc.collect()

    return concatenate((mean, std)), log_Z



def multi_gaussian_model_d(_x, _params, ngauss): # params: cube
    try:
        g = ((_params[3*i+4] * exp( -0.5*((_x - _params[3*i+2]) / _params[3*i+3])**2)) \
            for i in range(0, ngauss) \
            if _params[3*i+3] != 0 and not np.isnan(_params[3*i+3]) and not np.isinf(_params[3*i+3]))
    except:
        g = 1E9 * exp( -0.5*((_x - 0) / 1)**2)
        print(g)

    return sum(g, axis=1) + _params[1]

def f_gaussian_model(_x, gfit_results, ngauss):
    try:
        g = ((gfit_results[3*i+4] * exp( -0.5*((_x - gfit_results[3*i+2]) / gfit_results[3*i+3])**2)) \
            for i in range(0, ngauss) \
            if gfit_results[3*i+3] != 0 and not np.isnan(gfit_results[3*i+3]) and not np.isinf(gfit_results[3*i+3]))
    except:
        g = 1E9 * exp( -0.5*((_x - 0) / 1)**2)
        print(g)

    return sum(g, axis=1) + gfit_results[1]


def multi_gaussian_model_d_new(_x, _params, ngauss): # _x: global array, params: cube

    _gparam = _params[2:].reshape(ngauss, 3).T
    return (_gparam[2].reshape(ngauss, 1)*exp(-0.5*((_x-_gparam[0].reshape(ngauss, 1)) / _gparam[1].reshape(ngauss, 1))**2)).sum(axis=0) + _params[1]


def multi_gaussian_model_d_classic(_x, _params, ngauss): # params: cube
    _bg0 = _params[1]
    _y = np.zeros_like(_x, dtype=np.float32)
    for i in range(0, ngauss):
        _x0 = _params[3*i+2]
        _std0 = _params[3*i+3]
        _p0 = _params[3*i+4]

        _y += _p0 * exp( -0.5*((_x - _x0) / _std0)**2)
    _y += _bg0
    return _y



def optimal_prior_trfit(*args):







    _sigma0 = args[3][0]
    _sigma1 = args[3][8]
    _xpos0 = args[3][1]
    _xpos1 = args[3][9]
    _ypos0 = args[3][2]
    _ypos1 = args[3][10]
    _vsys0 = args[3][3]
    _vsys1 = args[3][11]
    _pa0 = args[3][4]
    _pa1 = args[3][12]
    _incl0 = args[3][5]
    _incl1 = args[3][13]

    _vrot0 = args[3][6]
    _vrot1 = args[3][14]
    _vrad0 = args[3][7]
    _vrad1 = args[3][15]


    n_ring_params_free = 0
    if args[1][0] == 1: # sigma fixed:0 free:1 
        args[0][n_ring_params_free] = _sigma0 + args[0][n_ring_params_free]*(_sigma1 - _sigma0)   # sigma: uniform prior
        n_ring_params_free += 1
    if args[1][1] == 1: # xpos fixed:0 free:1 
        args[0][n_ring_params_free] = _xpos0 + args[0][n_ring_params_free]*(_xpos1 - _xpos0)      # xpos: uniform prior
        n_ring_params_free += 1
    if args[1][2] == 1: # ypos fixed:0 free:1
        args[0][n_ring_params_free] = _ypos0 + args[0][n_ring_params_free]*(_ypos1 - _ypos0)      # ypos: uniform prior
        n_ring_params_free += 1
    if args[1][3] == 1: # vsys fixed:0 free:1
        args[0][n_ring_params_free] = _vsys0 + args[0][n_ring_params_free]*(_vsys1 - _vsys0)      # vsys: uniform prior
        n_ring_params_free += 1
    if args[1][4] == 1: # pa fixed:0 free:1
        args[0][n_ring_params_free] = _pa0 + args[0][n_ring_params_free]*(_pa1 - _pa0)      # pa: uniform prior
        n_ring_params_free += 1
    if args[1][5] == 1: # incl fixed:0 free:1
        args[0][n_ring_params_free] = _incl0 + args[0][n_ring_params_free]*(_incl1 - _incl0)      # incl: uniform prior
        n_ring_params_free += 1
    if args[1][6] == 1: # vrot fixed:0 free:1
        args[0][n_ring_params_free] = _vrot0 + args[0][n_ring_params_free]*(_vrot1 - _vrot0)      # vrot: uniform prior
        n_ring_params_free += 1

    if args[1][7] == 1: # vrad fixed:0 free:1
        args[0][n_ring_params_free] = _vrad0 + args[0][n_ring_params_free]*(_vrad1 - _vrad0)      # vrad: uniform prior
        n_ring_params_free += 1

    return args[0]



def optimal_prior_trfit_2d(*args):











    _sigma0 = args[4][0, 0]
    _sigma1 = args[4][0, 1]
    _xpos0 = args[4][1, 0]
    _xpos1 = args[4][1, 1]
    _ypos0 = args[4][2, 0]
    _ypos1 = args[4][2, 1]
    _vsys0 = args[4][3, 0]
    _vsys1 = args[4][3, 1]
    _pa0 = args[4][4, 0]
    _pa1 = args[4][4, 1]
    _incl0 = args[4][5, 0]
    _incl1 = args[4][5, 1]
    _vrot0 = args[4][6, 0]
    _vrot1 = args[4][6, 1]
    _vrad0 = args[4][7, 0]
    _vrad1 = args[4][7, 1]






    n_coeffs_pa_bs = args[12]
    n_coeffs_incl_bs = args[14]
    n_coeffs_vrot_bs = args[16]


    n_ring_params_free = 0
    if args[1][0] == 1: # sigma fixed:0 free:1 
        args[0][n_ring_params_free] = _sigma0 + args[0][n_ring_params_free]*(_sigma1 - _sigma0)   # sigma: uniform prior
        n_ring_params_free += 1

    if args[1][1] == 1: # xpos fixed:0 free:1 
        args[0][n_ring_params_free] = _xpos0 + args[0][n_ring_params_free]*(_xpos1 - _xpos0)      # xpos: uniform prior
        n_ring_params_free += 1

    if args[1][2] == 1: # ypos fixed:0 free:1
        args[0][n_ring_params_free] = _ypos0 + args[0][n_ring_params_free]*(_ypos1 - _ypos0)      # ypos: uniform prior
        n_ring_params_free += 1

    if args[1][3] == 1: # vsys fixed:0 free:1
        args[0][n_ring_params_free] = _vsys0 + args[0][n_ring_params_free]*(_vsys1 - _vsys0)      # vsys: uniform prior
        n_ring_params_free += 1

    if args[1][4] == 1 and n_coeffs_pa_bs == 0: # constant PA : # pa fixed:0 free:1
        args[0][n_ring_params_free] = _pa0 + args[0][n_ring_params_free]*(_pa1 - _pa0)      # pa: uniform prior
        n_ring_params_free += 1


    if args[1][5] == 1 and n_coeffs_incl_bs == 0: # constant INCL : # incl fixed:0 free:1
        args[0][n_ring_params_free] = _incl0 + args[0][n_ring_params_free]*(_incl1 - _incl0)      # incl: uniform prior
        n_ring_params_free += 1


    if args[1][6] == 1 and n_coeffs_vrot_bs == 0: # constant VROT : # vrot fixed:0 free:1
        args[0][n_ring_params_free] = _vrot0 + args[0][n_ring_params_free]*(_vrot1 - _vrot0)      # vrot: uniform prior
        n_ring_params_free += 1

    if args[1][7] == 1: # vrad fixed:0 free:1
       args[0][n_ring_params_free] = _vrad0 + args[0][n_ring_params_free]*(_vrad1 - _vrad0)      # vrad: uniform prior
       n_ring_params_free += 1


    if n_coeffs_pa_bs != 0:
        _is = n_ring_params_free
        _ie = _is + n_coeffs_pa_bs 

        args[0][_is:_ie] = args[4][7+1:7+1+n_coeffs_pa_bs, 0] \
                         + args[0][_is:_ie] \
                         * (args[4][7+1:7+1+n_coeffs_pa_bs, 1] - args[4][7+1:7+1+n_coeffs_pa_bs, 0]) # pa_c_bs: uniform prior





    if n_coeffs_incl_bs != 0:
        _is = n_ring_params_free + n_coeffs_pa_bs
        _ie = _is + n_coeffs_incl_bs 

        args[0][_is:_ie] = args[4][7+1+n_coeffs_pa_bs:7+1+n_coeffs_pa_bs+n_coeffs_incl_bs, 0] \
                         + args[0][_is:_ie] \
                         * (args[4][7+1+n_coeffs_pa_bs:7+1+n_coeffs_pa_bs+n_coeffs_incl_bs, 1] - args[4][7+1+n_coeffs_pa_bs:7+1+n_coeffs_pa_bs+n_coeffs_incl_bs, 0]) # incl_c_bs: uniform prior





    if n_coeffs_vrot_bs != 0:
        _is = n_ring_params_free + n_coeffs_pa_bs + n_coeffs_incl_bs
        _ie = _is + n_coeffs_vrot_bs


        args[0][_is:_ie] = args[4][7+1+n_coeffs_pa_bs+n_coeffs_incl_bs:7+1+n_coeffs_pa_bs+n_coeffs_incl_bs+n_coeffs_vrot_bs, 0] \
                         + args[0][_is:_ie] \
                         * (args[4][7+1+n_coeffs_pa_bs+n_coeffs_incl_bs:7+1+n_coeffs_pa_bs+n_coeffs_incl_bs+n_coeffs_vrot_bs, 1] \
                            - args[4][7+1+n_coeffs_pa_bs+n_coeffs_incl_bs:7+1+n_coeffs_pa_bs+n_coeffs_incl_bs+n_coeffs_vrot_bs, 0]) # incl_c_bs: uniform prior

        args[0][_is] = 0.0









    return args[0]











def optimal_prior_trfit_2d_org(*args):











    _sigma0 = args[4][0, 0]
    _sigma1 = args[4][0, 1]
    _xpos0 = args[4][1, 0]
    _xpos1 = args[4][1, 1]
    _ypos0 = args[4][2, 0]
    _ypos1 = args[4][2, 1]
    _vsys0 = args[4][3, 0]
    _vsys1 = args[4][3, 1]
    _pa0 = args[4][4, 0]
    _pa1 = args[4][4, 1]
    _incl0 = args[4][5, 0]
    _incl1 = args[4][5, 1]
    _vrot0 = args[4][6, 0]
    _vrot1 = args[4][6, 1]
    _vrad0 = args[4][7, 0]
    _vrad1 = args[4][7, 1]






    n_coeffs_pa_bs = args[12]
    n_coeffs_incl_bs = args[14]
    n_coeffs_vrot_bs = args[16]


    n_ring_params_free = 0
    if args[1][0] == 1: # sigma fixed:0 free:1 
        args[0][n_ring_params_free] = _sigma0 + args[0][n_ring_params_free]*(_sigma1 - _sigma0)   # sigma: uniform prior
        n_ring_params_free += 1

    if args[1][1] == 1: # xpos fixed:0 free:1 
        args[0][n_ring_params_free] = _xpos0 + args[0][n_ring_params_free]*(_xpos1 - _xpos0)      # xpos: uniform prior
        n_ring_params_free += 1

    if args[1][2] == 1: # ypos fixed:0 free:1
        args[0][n_ring_params_free] = _ypos0 + args[0][n_ring_params_free]*(_ypos1 - _ypos0)      # ypos: uniform prior
        n_ring_params_free += 1

    if args[1][3] == 1: # vsys fixed:0 free:1
        args[0][n_ring_params_free] = _vsys0 + args[0][n_ring_params_free]*(_vsys1 - _vsys0)      # vsys: uniform prior
        n_ring_params_free += 1

    if args[1][4] == 1 and n_coeffs_pa_bs == 0: # constant PA : # pa fixed:0 free:1
        args[0][n_ring_params_free] = _pa0 + args[0][n_ring_params_free]*(_pa1 - _pa0)      # pa: uniform prior
        args[0][n_ring_params_free] = 330.
        n_ring_params_free += 1



    if args[1][5] == 1 and n_coeffs_incl_bs == 0: # constant INCL : # incl fixed:0 free:1
        args[0][n_ring_params_free] = _incl0 + args[0][n_ring_params_free]*(_incl1 - _incl0)      # incl: uniform prior
        n_ring_params_free += 1


    if args[1][6] == 1 and n_coeffs_vrot_bs == 0: # constant VROT : # vrot fixed:0 free:1
        args[0][n_ring_params_free] = _vrot0 + args[0][n_ring_params_free]*(_vrot1 - _vrot0)      # vrot: uniform prior
        n_ring_params_free += 1

    if args[1][7] == 1: # vrad fixed:0 free:1
       args[0][n_ring_params_free] = _vrad0 + args[0][n_ring_params_free]*(_vrad1 - _vrad0)      # vrad: uniform prior
       n_ring_params_free += 1


    if n_coeffs_pa_bs != 0:
        _is = n_ring_params_free
        _ie = _is + n_coeffs_pa_bs 

        args[0][_is:_ie] = args[4][7+1:7+1+n_coeffs_pa_bs, 0] \
                         + args[0][_is:_ie] \
                         * (args[4][7+1:7+1+n_coeffs_pa_bs, 1] - args[4][7+1:7+1+n_coeffs_pa_bs, 0]) # pa_c_bs: uniform prior






    if n_coeffs_incl_bs != 0:
        _is = n_ring_params_free + n_coeffs_pa_bs
        _ie = _is + n_coeffs_incl_bs 

        args[0][_is:_ie] = args[4][7+1+n_coeffs_pa_bs:7+1+n_coeffs_pa_bs+n_coeffs_incl_bs, 0] \
                         + args[0][_is:_ie] \
                         * (args[4][7+1+n_coeffs_pa_bs:7+1+n_coeffs_pa_bs+n_coeffs_incl_bs, 1] - args[4][7+1+n_coeffs_pa_bs:7+1+n_coeffs_pa_bs+n_coeffs_incl_bs, 0]) # incl_c_bs: uniform prior
        








    if n_coeffs_vrot_bs != 0:
        _is = n_ring_params_free + n_coeffs_pa_bs + n_coeffs_incl_bs
        _ie = _is + n_coeffs_vrot_bs 
        args[0][_is:_ie] = args[4][7+1+n_coeffs_pa_bs+n_coeffs_incl_bs:7+1+n_coeffs_pa_bs+n_coeffs_incl_bs+n_coeffs_vrot_bs, 0] \
                         + args[0][_is:_ie] \
                         * (args[4][7+1+n_coeffs_pa_bs+n_coeffs_incl_bs:7+1+n_coeffs_pa_bs+n_coeffs_incl_bs+n_coeffs_vrot_bs, 1] - args[4][7+1+n_coeffs_pa_bs+n_coeffs_incl_bs:7+1+n_coeffs_pa_bs+n_coeffs_incl_bs+n_coeffs_vrot_bs, 0]) # incl_c_bs: uniform prior
        

        args[0][_is] = 0.0









    return args[0]


def trfit_aring(_vrot_bs, _A, _B):
    return np.sum((_A - _vrot_bs * _B)**2)

def optimal_prior(*args):


    _sigma0 = args[2][0]
    _sigma1 = args[2][2+3*args[1]] # args[1]=ngauss
    _bg0 = args[2][1]
    _bg1 = args[2][3+3*args[1]] # args[1]=ngauss

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


    args[0][0] = _sigma0 + args[0][0]*(_sigma1 - _sigma0)   # sigma: uniform prior between 0:1
    args[0][1] = _bg0 + args[0][1]*(_bg1 - _bg0)            # bg: uniform prior betwargs[1]een 0:1

    params_t[0] = (_xn_0 + params_t[0].T*(_xn_1 - _xn_0)).T
    params_t[1] = (_stdn_0 + params_t[1].T*(_stdn_1 - _stdn_0)).T
    params_t[2] = (_pn_0 + params_t[2].T*(_pn_1 - _pn_0)).T

    params_t_conc = np.hstack((params_t[0].reshape(args[1], 1), params_t[1].reshape(args[1], 1), params_t[2].reshape(args[1], 1))).reshape(1, 3*args[1])
    args[0][2:] = params_t_conc

    return args[0]

def uniform_prior(*args):


    _sigma0 = args[2][0]
    _sigma1 = args[2][5]
    _bg0 = args[2][1]
    _bg1 = args[2][6]
    _x0 = args[2][2]
    _x1 = args[2][7]
    _std0 = args[2][3]
    _std1 = args[2][8]
    _p0 = args[2][4]
    _p1 = args[2][9]


    params_t = args[0][2:].reshape(args[1], 3).T

    args[0][0] = _sigma0 + args[0][0]*(_sigma1 - _sigma0)            # bg: uniform prior between 0:1
    args[0][1] = _bg0 + args[0][1]*(_bg1 - _bg0)            # bg: uniform prior betwargs[1]een 0:1

    params_t[0] = (_x0 + params_t[0].T*(_x1 - _x0)).T
    params_t[1] = (_std0 + params_t[1].T*(_std1 - _std0)).T
    params_t[2] = (_p0 + params_t[2].T*(_p1 - _p0)).T


    params_t_conc = np.hstack((params_t[0].reshape(args[1], 1), params_t[1].reshape(args[1], 1), params_t[2].reshape(args[1], 1))).reshape(1, 3*args[1])
    args[0][2:] = params_t_conc

    return args[0]



def uniform_prior_d_pre(*args):


    _sigma0 = args[2][0]
    _sigma1 = args[2][5]
    _bg0 = args[2][1]
    _bg1 = args[2][6]
    _x0 = args[2][2]
    _x1 = args[2][7]
    _std0 = args[2][3]
    _std1 = args[2][8]
    _p0 = args[2][4]
    _p1 = args[2][9]


    params_t = args[0][2:].reshape(args[1], 3).T

    args[0][0] = _sigma0 + args[0][0]*(_sigma1 - _sigma0)            # bg: uniform prior between 0:1
    args[0][1] = _bg0 + args[0][1]*(_bg1 - _bg0)            # bg: uniform prior betwargs[1]een 0:1

    params_t[0] = _x0 + params_t[0]*(_x1 - _x0)
    params_t[1] = _std0 + params_t[1]*(_std1 - _std0)
    params_t[2] = _p0 + params_t[2]*(_p1 - _p0)


    params_t_conc = np.hstack((params_t[0].reshape(args[1], 1), params_t[1].reshape(args[1], 1), params_t[2].reshape(args[1], 1))).reshape(1, 3*args[1])
    args[0][2:] = params_t_conc

    return args[0]



def loglike_d(*args):

    npoints = args[2].size
    sigma = args[0][0] # loglikelihoood sigma

    gfit = multi_gaussian_model_d(args[2], args[0], args[3])

    log_n_sigma = -0.5*npoints*log(2.0*pi) - 1.0*npoints*log(sigma)
    chi2 = sum((-1.0 / (2*sigma**2)) * ((gfit - args[1])**2))

    return log_n_sigma + chi2



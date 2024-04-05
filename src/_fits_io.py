#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#|-----------------------------------------|
#| _2dbat.py
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
from _nested_sampler import define_tilted_ring, find_maximum_radius, define_tilted_ring_geo, define_tilted_ring_w, min_val, max_val, write_fits_images

import sys
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS


from scipy.optimize import curve_fit
from scipy.stats import norm



from scipy.interpolate import BSpline, splrep, splev
from scipy import optimize
from scipy.interpolate import interp1d

from itertools import zip_longest
from numba import njit


import ray

from skimage.measure import EllipseModel
from scipy.spatial import distance

from _dirs_files import make_dirs
import os



from scipy import ndimage






def find_the_largest_blob(_input_vf, threshold):

    binary_data = (_input_vf > threshold).astype(int)

    labeled_data, num_features = ndimage.label(binary_data)

    region_sizes = [np.sum(labeled_data == label_id) for label_id in range(1, num_features + 1)]

    largest_blob_label = np.argmax(region_sizes) + 1

    largest_blob_region = labeled_data == largest_blob_label

    result_data = np.where(largest_blob_region, _input_vf, np.nan)

    return result_data





def find_the_largest_blob_write_fits_version(_params):

    fits_file = fits.open(_params['wdir'] + '/' + _params['input_vf'], 'update')
    data = fits_file[0].data

    threshold = 50  # 원하는 임계값 설정
    binary_data = (data > threshold).astype(int)

    labeled_data, num_features = ndimage.label(binary_data)

    region_sizes = [np.sum(labeled_data == label_id) for label_id in range(1, num_features + 1)]

    largest_blob_label = np.argmax(region_sizes) + 1

    largest_blob_region = labeled_data == largest_blob_label

    result_data = np.where(largest_blob_region, data, np.nan)

    new_fits = fits.PrimaryHDU(result_data, header=fits_file[0].header)
    new_fits.writeto(_params['wdir'] + '/' + _params['input_vf_blobbed'], overwrite=True)

    return result_data



def ellipsefit_2dpoints(_input_vf_tofit, _input_int_vdisp_w_50percentile, _wi_2d, _params, xpos, ypos, pa, incl, ri, ro, side):







    npoints_in_a_ring_total_including_blanks, npoints_in_ring, _ij_aring_tofit, _wt_2d = define_tilted_ring(_input_vf_tofit, xpos, ypos, pa, incl, ri, ro, side, _params)

    a_points_t = np.array((_ij_aring_tofit[:, 0], _ij_aring_tofit[:, 1]))
    a_points = np.transpose(a_points_t)

    x = a_points[:, 0]
    y = a_points[:, 1]

    ell = EllipseModel()
    ell.estimate(a_points)
    _xc, _yc, _a, _b, _theta = ell.params
    _i = np.arccos((_b/_a)) * 180. / np.pi # in degree
    _theta = (_theta * 180. / np.pi) # w.r.t. W-E node in degree

    _centre = np.array((_xc, _yc))
    _centre = [(_xc, _yc)]
    dists = distance.cdist(_centre, a_points, 'euclidean')
    r_max = np.max(dists)

    npoints_in_a_ring_total_including_blanks, npoints_in_ring, _ij_aring_central, _wt_2d = define_tilted_ring(_input_vf_tofit, _xc, _yc, _theta, _i, 0, 0.2*r_max, side, _params)
    _it = _ij_aring_central[:, 0]
    _jt = _ij_aring_central[:, 1]
    vlos = _input_vf_tofit[_jt.astype(int), _it.astype(int)]
    vsys_init = np.nanmedian(vlos)


    def is_in_ellipse_and_left_half(x, y, xc, yc, a, b, theta):
        cos_angle = np.cos(np.radians(360-theta))
        sin_angle = np.sin(np.radians(360-theta))

        x_rot = cos_angle * (x - xc) - sin_angle * (y - yc)
        y_rot = sin_angle * (x - xc) + cos_angle * (y - yc)

        inside_ellipse = (x_rot**2 / a**2 + y_rot**2 / b**2) <= 1
        in_left_half = x_rot < 0 # left side

        return inside_ellipse and in_left_half




    def is_in_ellipse_and_right_half(x, y, xc, yc, a, b, theta):
        cos_angle = np.cos(np.radians(360-theta))
        sin_angle = np.sin(np.radians(360-theta))

        x_rot = cos_angle * (x - xc) - sin_angle * (y - yc)
        y_rot = sin_angle * (x - xc) + cos_angle * (y - yc)

        inside_ellipse = (x_rot**2 / a**2 + y_rot**2 / b**2) <= 1
        in_right_half = x_rot >= 0 # right side

        return inside_ellipse and in_right_half



    inside_left_half_ellipse_points = []
    cols, rows = _input_vf_tofit.shape
    for i in range(rows):
        for j in range(cols):
            if is_in_ellipse_and_left_half(i, j, _xc, _yc, _a, _b, _theta):
                inside_left_half_ellipse_points.append((i, j))

    x_left, y_left = zip(*inside_left_half_ellipse_points) # x, y 좌표 분리
    x_l = np.array(x_left)
    y_l = np.array(y_left)
    median_vlos_side_l = np.nanmedian(_input_vf_tofit[y_l.astype(int), x_l.astype(int)])


    inside_right_half_ellipse_points = []
    cols, rows = _input_vf_tofit.shape
    for i in range(rows):
        for j in range(cols):
            if is_in_ellipse_and_right_half(i, j, _xc, _yc, _a, _b, _theta):
                inside_right_half_ellipse_points.append((i, j))

    x_right, y_right = zip(*inside_right_half_ellipse_points) # x, y 좌표 분리
    x_r = np.array(x_right)
    y_r = np.array(y_right)
    median_vlos_side_r = np.nanmedian(_input_vf_tofit[y_r.astype(int), x_r.astype(int)])


    _theta += 90 # w.r.t. N-S node
    if(median_vlos_side_l < median_vlos_side_r):
        _theta += 180

    if _theta > 360: _theta -= 360

    print('-'*50)
    print("+ Ellipse Fit results")
    print('-'*50)

    print(" XY = ",  (round(_xc, 2), round(_yc, 2)))
    print(" PA = ", round(_theta, 2))
    print(" R_major, R_minor = ", (round(_a, 2), round(_b, 2)))
    print(" VSYS = ", round(vsys_init, 2))
    print(" INCL = ", round(np.arccos(_b/_a) * 180. / np.pi, 2))
    print('-'*50)
    print("")
    print("+ median-vlos_left_side: ", round(median_vlos_side_l, 2))
    print("+ median-vlos_right side: ", round(median_vlos_side_r, 2))
    print('-'*50)
    print('-'*50)

    
    return _xc, _yc, vsys_init, _theta, _i, r_max


def ellipsefit_2dpoints1(_input_vf_tofit, _wi_2d, _params, xpos, ypos, pa, incl, ri, ro, side):

    npoints_in_a_ring_total_including_blanks, npoints_in_ring, _ij_aring_tofit, _wt_2d = define_tilted_ring(_input_vf_tofit, xpos, ypos, pa, incl, ri, ro, side, _params)

    a_points_t = np.array((_ij_aring_tofit[:, 0], _ij_aring_tofit[:, 1]))
    a_points = np.transpose(a_points_t)

    x = a_points[:, 0]
    y = a_points[:, 1]

    ell = EllipseModel()
    ell.estimate(a_points)
    _xc, _yc, _a, _b, _theta = ell.params
    _i = np.arccos((_b/_a)) * 180. / np.pi # in degree
    _theta = (_theta * 180. / np.pi) # w.r.t. W-E node in degree

    _centre = np.array((_xc, _yc))
    _centre = [(_xc, _yc)]
    dists = distance.cdist(_centre, a_points, 'euclidean')
    r_max = np.max(dists)

    npoints_in_a_ring_total_including_blanks, npoints_in_ring, _ij_aring_central, _wt_2d = define_tilted_ring(_input_vf_tofit, _xc, _yc, _theta, _i, 0, 0.2*r_max, side, _params)
    _it = _ij_aring_central[:, 0]
    _jt = _ij_aring_central[:, 1]
    vlos = _input_vf_tofit[_jt.astype(int), _it.astype(int)]
    vsys_init = np.nanmedian(vlos)


    def is_in_ellipse_and_left_half(x, y, xc, yc, a, b, theta):
        cos_angle = np.cos(np.radians(360-theta))
        sin_angle = np.sin(np.radians(360-theta))

        x_rot = cos_angle * (x - xc) - sin_angle * (y - yc)
        y_rot = sin_angle * (x - xc) + cos_angle * (y - yc)

        inside_ellipse = (x_rot**2 / a**2 + y_rot**2 / b**2) <= 1
        in_left_half = x_rot < 0 # left side

        return inside_ellipse and in_left_half




    def is_in_ellipse_and_right_half(x, y, xc, yc, a, b, theta):
        cos_angle = np.cos(np.radians(360-theta))
        sin_angle = np.sin(np.radians(360-theta))

        x_rot = cos_angle * (x - xc) - sin_angle * (y - yc)
        y_rot = sin_angle * (x - xc) + cos_angle * (y - yc)

        inside_ellipse = (x_rot**2 / a**2 + y_rot**2 / b**2) <= 1
        in_right_half = x_rot >= 0 # right side

        return inside_ellipse and in_right_half



    inside_left_half_ellipse_points = []
    cols, rows = _input_vf_tofit.shape
    for i in range(rows):
        for j in range(cols):
            if is_in_ellipse_and_left_half(i, j, _xc, _yc, _a, _b, _theta):
                inside_left_half_ellipse_points.append((i, j))

    x_left, y_left = zip(*inside_left_half_ellipse_points) # x, y 좌표 분리
    x_l = np.array(x_left)
    y_l = np.array(y_left)
    median_vlos_side_l = np.nanmedian(_input_vf_tofit[y_l.astype(int), x_l.astype(int)])


    inside_right_half_ellipse_points = []
    cols, rows = _input_vf_tofit.shape
    for i in range(rows):
        for j in range(cols):
            if is_in_ellipse_and_right_half(i, j, _xc, _yc, _a, _b, _theta):
                inside_right_half_ellipse_points.append((i, j))

    x_right, y_right = zip(*inside_right_half_ellipse_points) # x, y 좌표 분리
    x_r = np.array(x_right)
    y_r = np.array(y_right)
    median_vlos_side_r = np.nanmedian(_input_vf_tofit[y_r.astype(int), x_r.astype(int)])


    _theta += 90 # w.r.t. N-S node
    if(median_vlos_side_l < median_vlos_side_r):
        _theta += 180

    if _theta > 360: _theta -= 360

    print('-'*50)
    print("+ Ellipse Fit results")
    print('-'*50)

    print(" XY = ",  (round(_xc, 2), round(_yc, 2)))
    print(" PA = ", round(_theta, 2))
    print(" R_major, R_minor = ", (round(_a, 2), round(_b, 2)))
    print(" VSYS = ", round(vsys_init, 2))
    print(" INCL = ", round(np.arccos(_b/_a) * 180. / np.pi, 2))
    print('-'*50)
    print("")
    print("+ median-vlos_left_side: ", round(median_vlos_side_l, 2))
    print("+ median-vlos_right side: ", round(median_vlos_side_r, 2))
    print('-'*50)
    print('-'*50)


    return _xc, _yc, vsys_init, _theta, _i, r_max









def ellipsefit_2dpoints_test(_input_vf_tofit, _wi_2d, _params, xpos, ypos, pa, incl, ri, ro, side):

    npoints_in_a_ring_total_including_blanks, npoints_in_ring, _ij_aring_tofit, _wt_2d = define_tilted_ring(_input_vf_tofit, xpos, ypos, pa, incl, ri, ro, side, _params)

    a_points_t = np.array((_ij_aring_tofit[:, 0], _ij_aring_tofit[:, 1]))
    a_points = np.transpose(a_points_t)

    x = a_points[:, 0]
    y = a_points[:, 1]

    ell = EllipseModel()
    ell.estimate(a_points)
    _xc, _yc, _a, _b, _theta = ell.params
    _i = np.arccos((_b/_a)) * 180. / np.pi # in degree
    _theta = (_theta * 180. / np.pi) # w.r.t. W-E node in degree

    _centre = np.array((_xc, _yc))
    _centre = [(_xc, _yc)]
    dists = distance.cdist(_centre, a_points, 'euclidean')
    r_max = np.max(dists)

    npoints_in_a_ring_total_including_blanks, npoints_in_ring, _ij_aring_central, _wt_2d = define_tilted_ring(_input_vf_tofit, _xc, _yc, _theta, _i, 0, 0.2*r_max, side, _params)
    _it = _ij_aring_central[:, 0]
    _jt = _ij_aring_central[:, 1]
    vlos = _input_vf_tofit[_jt.astype(int), _it.astype(int)]
    vsys_init = np.nanmedian(vlos)


    def is_in_ellipse_and_right_half(x, y, xc, yc, a, b, theta):
        cos_angle = np.cos(np.radians(theta))
        sin_angle = np.sin(np.radians(theta))

        x_rot = cos_angle * (x - xc) - sin_angle * (y - yc)
        y_rot = sin_angle * (x - xc) + cos_angle * (y - yc)

        inside_ellipse = (x_rot**2 / a**2 + y_rot**2 / b**2) <= 1
        in_right_half = x_rot >= 0  and y_rot >= 0 # right side

        return inside_ellipse and in_right_half

    def is_in_ellipse_and_left_half(x, y, xc, yc, a, b, theta):
        cos_angle = np.cos(np.radians(theta))
        sin_angle = np.sin(np.radians(theta))

        x_rot = cos_angle * (x - xc) - sin_angle * (y - yc)
        y_rot = sin_angle * (x - xc) + cos_angle * (y - yc)

        inside_ellipse = (x_rot**2 / a**2 + y_rot**2 / b**2) <= 1
        in_left_half = x_rot < 0  # left side

        return inside_ellipse and in_left_half
    

    def rotate_point_back(x_rot, y_rot, xc, yc, theta):
        x_trans = x_rot - xc
        y_trans = y_rot - yc

        cos_angle = np.cos(np.radians(-theta + 0)) # w.r.t., N-S node
        sin_angle = np.sin(np.radians(-theta + 0))

        x = cos_angle * x_trans + sin_angle * y_trans + xc
        y = -sin_angle * x_trans + cos_angle * y_trans + yc


        return x, y




    inside_left_half_ellipse_points = []
    rows, cols = _input_vf_tofit.shape
    for i in range(rows):
        for j in range(cols):
            if is_in_ellipse_and_left_half(i, j, _xc, _yc, _a, _b, _theta):
                inside_left_half_ellipse_points.append((i, j))

    inside_left_half_ellipse_points_rotated_back = []
    for point in inside_left_half_ellipse_points:
        i_rot, j_rot = point
        i_original, j_original = rotate_point_back(i_rot, j_rot, _xc, _yc, _theta)
        inside_left_half_ellipse_points_rotated_back.append((i_original, j_original))

    x_left, y_left = zip(*inside_left_half_ellipse_points_rotated_back) # x, y 좌표 분리
    x_l = np.array(x_left)
    y_l = np.array(y_left)
    median_vlos_side_l = np.nanmedian(_input_vf_tofit[y_l.astype(int), x_l.astype(int)])

    inside_right_half_ellipse_points = []
    rows, cols = _input_vf_tofit.shape
    for i in range(rows):
        for j in range(cols):
            if is_in_ellipse_and_right_half(i, j, _xc, _yc, _a, _b, _theta):
                inside_right_half_ellipse_points.append((i, j))

    inside_right_half_ellipse_points_rotated_back = []
    for point in inside_right_half_ellipse_points:
        i_rot, j_rot = point
        i_original, j_original = rotate_point_back(i_rot, j_rot, _xc, _yc, _theta)
        inside_right_half_ellipse_points_rotated_back.append((i_original, j_original))

    x_right, y_right = zip(*inside_right_half_ellipse_points_rotated_back) # x, y 좌표 분리
    x_r = np.array(x_right)
    y_r = np.array(y_right)
    median_vlos_side_r = np.nanmedian(_input_vf_tofit[y_r.astype(int), x_r.astype(int)])

    print("center = ",  (_xc, _yc))
    print("angle of rotation = ",  _theta)
    print("axes = ", (_a, _b))
    print("vsys = ", vsys_init)
    print("i = ", np.arccos((_b/_a)) * 180. / np.pi  )


    print(median_vlos_side_l)
    print(median_vlos_side_r)



    _theta += 90 # w.r.t. N-S node
    if(median_vlos_side_l < median_vlos_side_r):
        _theta += 180

    print(_theta)





    


    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=30, edgecolors='b', facecolors='none') # s는 점의 크기
    plt.scatter(x_r, y_r, s=20, edgecolors='g', facecolors='red') # s는 점의 크기
    plt.scatter(x_l, y_l, s=20, edgecolors='r', facecolors='blue') # s는 점의 크기
    plt.xlim(0, 100) # x축 범위 설정
    plt.ylim(0, 100) # y축 범위 설정
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Half Ellipse Points')
    plt.grid(True)
    plt.show()
    sys.exit()


    return _xc, _yc, vsys_init, _theta, _i, r_max, half_ellipse_points





def ellipsefit_2dpoints_org(_input_vf_tofit, _wi_2d, _params, xpos, ypos, pa, incl, ri, ro, side):

    npoints_in_a_ring_total_including_blanks, npoints_in_ring, _ij_aring_tofit, _wt_2d = define_tilted_ring(_input_vf_tofit, xpos, ypos, pa, incl, ri, ro, side, _params)

    a_points_t = np.array((_ij_aring_tofit[:, 0], _ij_aring_tofit[:, 1]))
    a_points = np.transpose(a_points_t)

    x = a_points[:, 0]
    y = a_points[:, 1]

    ell = EllipseModel()
    ell.estimate(a_points)
    _xc, _yc, _a, _b, _theta = ell.params
    _i = np.arccos((_b/_a)) * 180. / np.pi # in degree
    _theta = (_theta * 180. / np.pi) + 90 # in degree

    _centre = [(_xc, _yc)]
    dists = distance.cdist(_centre, a_points, 'euclidean')
    r_max = np.max(dists)

    npoints_in_a_ring_total_including_blanks, npoints_in_ring, _ij_aring_central, _wt_2d = define_tilted_ring(_input_vf_tofit, _xc, _yc, _theta, _i, 0, 0.2*r_max, side, _params)
    _it = _ij_aring_central[:, 0]
    _jt = _ij_aring_central[:, 1]
    vlos = _input_vf_tofit[_jt.astype(int), _it.astype(int)]
    vsys_init = np.nanmedian(vlos)

    return _xc, _yc, vsys_init, _theta, _i, r_max




def set_vrot_bs_coefficients_bounds(_vrot_init_t, del_vrot, _params, _ring_t):

    nrings_reliable = _params['nrings_reliable']

    xs_bs = _params['r_galaxy_plane_s']
    xe_bs = _ring_t[nrings_reliable-1]
    scipy_interp1d = interp1d(_ring_t[:nrings_reliable], _vrot_init_t[:nrings_reliable], kind='linear')  # 선형 보간

    r_fine = np.linspace(xs_bs, xe_bs, _params['nrings_intp'], endpoint=True)

    vrot_fine = scipy_interp1d(r_fine)

    _vrot_bound_l = np.zeros(vrot_fine.shape[0], dtype=np.float64)
    _vrot_bound_u = np.zeros(vrot_fine.shape[0], dtype=np.float64)

    for _i in range(len(vrot_fine)):
        _vrot_bound_l[_i] = vrot_fine[_i] - del_vrot
        _vrot_bound_u[_i] = vrot_fine[_i] + del_vrot

    n_knots_inner = _params['n_vrot_bs_knots_inner'] # 0, 1, 2, ...
    k_bs = _params['k_vrot_bs'] # 0, 1, 2, ...
    if k_bs == 0: k_bs = 1

    vrot_bs_knots_inner = np.linspace(xs_bs, xe_bs, (2+n_knots_inner))[1:-1]
    tck_vrot_bs_bound_l = splrep(r_fine, _vrot_bound_l, t=vrot_bs_knots_inner, k=k_bs)
    tck_vrot_bs_bound_u = splrep(r_fine, _vrot_bound_u, t=vrot_bs_knots_inner, k=k_bs)

    return tck_vrot_bs_bound_l, tck_vrot_bs_bound_u





def set_incl_bs_coefficients_bounds(_incl_init_t, del_incl, _params, _ring_t):

    nrings_reliable = _params['nrings_reliable']

    xs_bs = _params['r_galaxy_plane_s']
    xe_bs = _ring_t[nrings_reliable-1]
    scipy_interp1d = interp1d(_ring_t[:nrings_reliable], _incl_init_t[:nrings_reliable], kind='linear')  # 선형 보간

    r_fine = np.linspace(xs_bs, xe_bs, _params['nrings_intp'], endpoint=True)

    incl_fine = scipy_interp1d(r_fine)

    _incl_bound_l = np.zeros(incl_fine.shape[0], dtype=np.float64)
    _incl_bound_u = np.zeros(incl_fine.shape[0], dtype=np.float64)

    for _i in range(len(incl_fine)):
        _incl_bound_l[_i] = incl_fine[_i] - del_incl
        _incl_bound_u[_i] = incl_fine[_i] + del_incl

    n_knots_inner = _params['n_incl_bs_knots_inner'] # 0, 1, 2, ...
    k_bs = _params['k_incl_bs'] # 0, 1, 2, ...
    if k_bs == 0: k_bs = 1

    incl_bs_knots_inner = np.linspace(xs_bs, xe_bs, (2+n_knots_inner))[1:-1]
    tck_incl_bs_bound_l = splrep(r_fine, _incl_bound_l, t=incl_bs_knots_inner, k=k_bs)
    tck_incl_bs_bound_u = splrep(r_fine, _incl_bound_u, t=incl_bs_knots_inner, k=k_bs)

    return tck_incl_bs_bound_l, tck_incl_bs_bound_u


def set_pa_bs_coefficients_bounds(_pa_init_t, del_pa, _params, _ring_t):

    nrings_reliable = _params['nrings_reliable']

    xs_bs = _params['r_galaxy_plane_s']
    xe_bs = _ring_t[nrings_reliable-1]
    scipy_interp1d = interp1d(_ring_t[:nrings_reliable], _pa_init_t[:nrings_reliable], fill_value=(_pa_init_t[0], _pa_init_t[nrings_reliable-1]), kind='linear')  # linear intp.

    r_fine = np.linspace(xs_bs, xe_bs, _params['nrings_intp'], endpoint=True)

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


    n_knots_inner = _params['n_vrot_bs_knots_inner'] # 0, 1, 2, ...
    k_bs = _params['k_vrot_bs'] # 0, 1, 2, ...
    vrot_bs_knots_inner = np.linspace(xs_bs, xe_bs, (2+n_knots_inner))[1:-1]
    tck_vrot_bs_bound_l = splrep(_ring_t, _vrot_bound_l, t=vrot_bs_knots_inner, k=k_bs)
    tck_vrot_bs_bound_u = splrep(_ring_t, _vrot_bound_u, t=vrot_bs_knots_inner, k=k_bs)

    return tck_vrot_bs_bound_l, tck_vrot_bs_bound_u






@ray.remote(num_cpus=1)
def define_tilted_ring_i(_input_vf, _xpos, ypos, pa, incl, ri, ro, side, _params):

    naxis1 = _params['naxis1']
    naxis2 = _params['naxis2']

    x0 = 0
    y0 = 0
    x1 = naxis1
    y1 = naxis2

    cdelt1 = 1 # we will fix with 1 as pixel unit is used here
    cdelt2 = 1 # we will fix with 1 as pixel unit is used here
    ri_deg = cdelt1*ri
    ro_deg = cdelt1*ro

    deg_to_rad = np.pi / 180.

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

    x0 = 0
    y0 = 0
    x1 = naxis1
    y1 = naxis2
		
    wpow = 1
    npoints_in_ring_t = 0
    for i0 in range(i0_lo, i0_up):
        for j0 in range(j0_lo, j0_up):

            rx = cdelt1 * i0  # X position in plane of galaxy
            ry = cdelt2 * j0  # Y position in plane of galaxy
            if _input_vf[j0, i0] > -1E8 and _input_vf[j0, i0] < 1E8:

                xr = ( - ( rx - cdelt1 * xpos ) * sinp + ( ry - cdelt2 * ypos ) * cosp )
                yr = ( - ( rx - cdelt1 * xpos ) * cosp - ( ry - cdelt2 * ypos ) * sinp ) / cosi
                r = ( xr**2 + yr**2 )**0.5  # distance from centre
                if r < 0.1:
                    theta = 0.0
                else:
                    theta = math.atan2( yr, xr ) / deg_to_rad # in degree

                costh = np.fabs( np.cos ( deg_to_rad * theta ) ) # in radian

                if r > ri_deg and r < ro_deg and costh > sine_free_angle: # both sides
                    npoints_in_ring_t += 1

    ij_tilted_ring = np.array([], dtype=np.float64)
    ij_tilted_ring.shape = (0, 3)

    x0 = 0
    y0 = 0
    x1 = naxis1
    y1 = naxis2
    npoints_in_ring_t = 0
    for i0 in range(i0_lo, i0_up):
        for j0 in range(j0_lo, j0_up):
            rx = cdelt1 * i0  # X position in plane of galaxy
            ry = cdelt2 * j0  # Y position in plane of galaxy
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
                    ij_point = np.array([[i0, j0, -1E10]])
                    ij_tilted_ring = np.concatenate( (ij_tilted_ring, ij_point) )
                    npoints_in_ring_t += 1
                    



    return ij_tilted_ring


def trfit_ring_by_ring(_input_vf, _input_int, _tr_model_vf, _wi_2d, _params, fit_opt, ndim, tr_params_priors_init, _intp_index):

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

    _ring_t = np.zeros(nrings+2, dtype=np.float64)
    _sigma_t = np.zeros(nrings+2, dtype=np.float64)
    _xpos_t = np.zeros(nrings+2, dtype=np.float64)
    _ypos_t = np.zeros(nrings+2, dtype=np.float64)
    _vsys_t = np.zeros(nrings+2, dtype=np.float64)
    _pa_t = np.zeros(nrings+2, dtype=np.float64)
    _incl_t = np.zeros(nrings+2, dtype=np.float64)
    _vrot_t = np.zeros(nrings+2, dtype=np.float64)
    _vrad_t = np.zeros(nrings+2, dtype=np.float64)

    _sigma_et = np.zeros(nrings+2, dtype=np.float64)
    _xpos_et = np.zeros(nrings+2, dtype=np.float64)
    _ypos_et = np.zeros(nrings+2, dtype=np.float64)
    _vsys_et = np.zeros(nrings+2, dtype=np.float64)
    _pa_et = np.zeros(nrings+2, dtype=np.float64)
    _incl_et = np.zeros(nrings+2, dtype=np.float64)
    _vrot_et = np.zeros(nrings+2, dtype=np.float64)
    _vrad_et = np.zeros(nrings+2, dtype=np.float64)

    _nrings_reliable_t = np.zeros(nrings+2, dtype=np.float64)
    _npoints_in_a_ring_total = np.zeros(nrings+2, dtype=np.float64)
    _npoints_in_a_ring = np.zeros(nrings+2, dtype=np.float64)

    _input_vf_id = ray.put(_input_vf)
    _input_int_id = ray.put(_input_int)
    _tr_model_vf_id = ray.put(_tr_model_vf)
    _wi_2d_id = ray.put(_wi_2d)
    _params_id = ray.put(_params)
    _i = 0
    _i_id = ray.put(_i)

    fit_opt_id = ray.put(fit_opt)
    ndim_id = ray.put(ndim)
    tr_params_priors_init_id = ray.put(tr_params_priors_init)

    results_ids = [trfit_ring_i.remote(_input_vf_id, _input_int_id, _tr_model_vf_id, _wi_2d_id, _params_id, fit_opt_id, ndim_id, tr_params_priors_init_id, _i_id) for _i_id in range(0, nrings)]

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


            _ring_t[_i_ti+1] = ring
            _sigma_t[_i_ti+1] = _sigma_ti
            _sigma_et[_i_ti+1] = _sigma_eti
            _xpos_t[_i_ti+1] = _xpos_ti
            _xpos_et[_i_ti+1] = _xpos_eti
            _ypos_t[_i_ti+1] = _ypos_ti
            _ypos_et[_i_ti+1] = _ypos_eti
            _vsys_t[_i_ti+1] = _vsys_ti
            _vsys_et[_i_ti+1] = _vsys_eti
            _pa_t[_i_ti+1] = _pa_ti
            _pa_et[_i_ti+1] = _pa_eti
            _incl_t[_i_ti+1] = _incl_ti
            _incl_et[_i_ti+1] = _incl_eti
            _vrot_t[_i_ti+1] = _vrot_ti
            _vrot_et[_i_ti+1] = _vrot_eti
            _vrad_t[_i_ti+1] = _vrad_ti
            _vrad_et[_i_ti+1] = _vrad_eti
            _nrings_reliable_t[_i_ti] = _i_t_reliable

            _npoints_in_a_ring_total[_i_ti+1] = 999
            _npoints_in_a_ring[_i_ti+1] = 999


    nrings_reliable = int(np.max(_nrings_reliable_t)) + 1

    _r_max_tr = _params['ring_w']*nrings_reliable
    _params['r_galaxy_plane_e'] = _r_max_tr


    if _intp_index == 'True':

        _ring_t[0] = 0
        _sigma_t[0] = _sigma_t[1]
        _xpos_t[0] = _xpos_t[1]
        _ypos_t[0] = _ypos_t[1]
        _vsys_t[0] = _vsys_t[1]
        _pa_t[0] = _pa_t[1]
        _incl_t[0] = _incl_t[1]
        _vrot_t[0] = 0
        _vrad_t[0] = _vrad_t[1]

        _sigma_et[0] = _sigma_et[1]
        _xpos_et[0] = _xpos_et[1]
        _ypos_et[0] = _ypos_et[1]
        _vsys_et[0] = _vsys_et[1]
        _pa_et[0] = _pa_et[1]
        _incl_et[0] = _incl_et[1]
        _vrot_et[0] = _vrot_et[1]
        _vrad_et[0] = _vrad_et[1]


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

        nrings_reliable += 1 # add extra outer ring

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
        nrings_reliable, (_ring_t[:nrings_reliable], \
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
        nrings_reliable)




def trfit_ring_by_ring1(_input_vf, _tr_model_vf, _wi_2d, _params, fit_opt, ndim, tr_params_priors_init, _intp_index):

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

    _ring_t = np.zeros(nrings+2, dtype=np.float64)
    _sigma_t = np.zeros(nrings+2, dtype=np.float64)
    _xpos_t = np.zeros(nrings+2, dtype=np.float64)
    _ypos_t = np.zeros(nrings+2, dtype=np.float64)
    _vsys_t = np.zeros(nrings+2, dtype=np.float64)
    _pa_t = np.zeros(nrings+2, dtype=np.float64)
    _incl_t = np.zeros(nrings+2, dtype=np.float64)
    _vrot_t = np.zeros(nrings+2, dtype=np.float64)
    _vrad_t = np.zeros(nrings+2, dtype=np.float64)

    _sigma_et = np.zeros(nrings+2, dtype=np.float64)
    _xpos_et = np.zeros(nrings+2, dtype=np.float64)
    _ypos_et = np.zeros(nrings+2, dtype=np.float64)
    _vsys_et = np.zeros(nrings+2, dtype=np.float64)
    _pa_et = np.zeros(nrings+2, dtype=np.float64)
    _incl_et = np.zeros(nrings+2, dtype=np.float64)
    _vrot_et = np.zeros(nrings+2, dtype=np.float64)
    _vrad_et = np.zeros(nrings+2, dtype=np.float64)

    _nrings_reliable_t = np.zeros(nrings+2, dtype=np.float64)


    _input_vf_id = ray.put(_input_vf)
    _tr_model_vf_id = ray.put(_tr_model_vf)
    _wi_2d_id = ray.put(_wi_2d)
    _params_id = ray.put(_params)
    _i = 0
    _i_id = ray.put(_i)

    fit_opt_id = ray.put(fit_opt)
    ndim_id = ray.put(ndim)
    tr_params_priors_init_id = ray.put(tr_params_priors_init)

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


            _ring_t[_i_ti+1] = ring
            _sigma_t[_i_ti+1] = _sigma_ti
            _sigma_et[_i_ti+1] = _sigma_eti
            _xpos_t[_i_ti+1] = _xpos_ti
            _xpos_et[_i_ti+1] = _xpos_eti
            _ypos_t[_i_ti+1] = _ypos_ti
            _ypos_et[_i_ti+1] = _ypos_eti
            _vsys_t[_i_ti+1] = _vsys_ti
            _vsys_et[_i_ti+1] = _vsys_eti
            _pa_t[_i_ti+1] = _pa_ti
            _pa_et[_i_ti+1] = _pa_eti
            _incl_t[_i_ti+1] = _incl_ti
            _incl_et[_i_ti+1] = _incl_eti
            _vrot_t[_i_ti+1] = _vrot_ti
            _vrot_et[_i_ti+1] = _vrot_eti
            _vrad_t[_i_ti+1] = _vrad_ti
            _vrad_et[_i_ti+1] = _vrad_eti
            _nrings_reliable_t[_i_ti] = _i_t_reliable


    nrings_reliable = int(np.max(_nrings_reliable_t)) + 1

    _r_max_tr = _params['ring_w']*nrings_reliable
    _params['r_galaxy_plane_e'] = _r_max_tr


    if _intp_index == 'True':

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

        nrings_reliable += 1 # add extra outer ring

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




def trfit_ring_by_ring_final_org(_input_vf, _tr_model_vf, _wi_2d, _params, fit_opt, fit_opt_2d, ndim, tr_params_priors_init, _tr2dfit_results, side, _intp_index):


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

    _ring_t = np.zeros(nrings+2, dtype=np.float64)

    _sigma_t = np.zeros(nrings+2, dtype=np.float64)
    _xpos_t = np.zeros(nrings+2, dtype=np.float64)
    _ypos_t = np.zeros(nrings+2, dtype=np.float64)
    _vsys_t = np.zeros(nrings+2, dtype=np.float64)
    _pa_t = np.zeros(nrings+2, dtype=np.float64)
    _incl_t = np.zeros(nrings+2, dtype=np.float64)
    _vrot_t = np.zeros(nrings+2, dtype=np.float64)
    _vrad_t = np.zeros(nrings+2, dtype=np.float64)

    _sigma_et = np.zeros(nrings+2, dtype=np.float64)
    _xpos_et = np.zeros(nrings+2, dtype=np.float64)
    _ypos_et = np.zeros(nrings+2, dtype=np.float64)
    _vsys_et = np.zeros(nrings+2, dtype=np.float64)
    _pa_et = np.zeros(nrings+2, dtype=np.float64)
    _incl_et = np.zeros(nrings+2, dtype=np.float64)
    _vrot_et = np.zeros(nrings+2, dtype=np.float64)
    _vrad_et = np.zeros(nrings+2, dtype=np.float64)

    _nrings_reliable_t = np.zeros(nrings+2, dtype=np.float64)
    _npoints_in_a_ring_total = np.zeros(nrings+2, dtype=np.float64)
    _npoints_in_a_ring = np.zeros(nrings+2, dtype=np.float64)

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

            _ring_t[_i_ti+1] = ring
            _sigma_t[_i_ti+1] = _sigma_ti
            _sigma_et[_i_ti+1] = _sigma_eti
            _xpos_t[_i_ti+1] = _xpos_ti
            _xpos_et[_i_ti+1] = _xpos_eti
            _ypos_t[_i_ti+1] = _ypos_ti
            _ypos_et[_i_ti+1] = _ypos_eti
            _vsys_t[_i_ti+1] = _vsys_ti
            _vsys_et[_i_ti+1] = _vsys_eti
            _pa_t[_i_ti+1] = _pa_ti
            _pa_et[_i_ti+1] = _pa_eti
            _incl_t[_i_ti+1] = _incl_ti
            _incl_et[_i_ti+1] = _incl_eti
            _vrot_t[_i_ti+1] = _vrot_ti
            _vrot_et[_i_ti+1] = _vrot_eti
            _vrad_t[_i_ti+1] = _vrad_ti
            _vrad_et[_i_ti+1] = _vrad_eti
            _nrings_reliable_t[_i_ti] = _i_t_reliable

            _npoints_in_a_ring_total[_i_ti+1] = npoints_in_a_ring_total_including_blanks
            _npoints_in_a_ring[_i_ti+1] = npoints_in_a_ring

    nrings_reliable = int(np.max(_nrings_reliable_t)) + 1


    if _intp_index == 'True':

        _ring_t[0] = 0
        _sigma_t[0] = _sigma_t[1]
        _xpos_t[0] = _xpos_t[1]
        _ypos_t[0] = _ypos_t[1]
        _vsys_t[0] = _vsys_t[1]
        _pa_t[0] = _pa_t[1]
        _incl_t[0] = _incl_t[1]
        _vrot_t[0] = 0
        _vrad_t[0] = _vrad_t[1]


        _ring_t[nrings_reliable] = _ring_t[nrings_reliable-1] + ring_w # --> this will be replaced to _r_galaxy_plane_e after in _2dbat.py 
        _sigma_t[nrings_reliable] = _sigma_t[nrings_reliable-1]
        _xpos_t[nrings_reliable] = _xpos_t[nrings_reliable-1]
        _ypos_t[nrings_reliable] = _ypos_t[nrings_reliable-1]
        _vsys_t[nrings_reliable] = _vsys_t[nrings_reliable-1]
        _pa_t[nrings_reliable] = _pa_t[nrings_reliable-1]
        _incl_t[nrings_reliable] = _incl_t[nrings_reliable-1]
        _vrot_t[nrings_reliable] = _vrot_t[nrings_reliable-1]
        _vrad_t[nrings_reliable] = _vrad_t[nrings_reliable-1]

        nrings_reliable += 1 # add extra outer ring

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


def trfit_ring_by_ring_final(_input_vf, _tr_model_vf, _wi_2d, _params, fit_opt, fit_opt_2d, ndim, tr_params_priors_init, _tr2dfit_results, side, _intp_index):


    n_cpus = int(_params['num_cpus_tr_ray'])

    nrings = 3*_params['nrings_reliable']
    ring_w = _params['ring_w']

    _sigma = _params['_sigma_init']
    _xpos = _params['_xpos_init']
    _ypos = _params['_ypos_init']
    _vsys = _params['_vsys_init']
    _pa = _params['_pa_init']
    _incl = _params['_incl_init']
    _vrot = _params['_vrot_init']
    _vrad = _params['_vrad_init']

    _ring_t = np.zeros(nrings+2, dtype=np.float64)

    _sigma_t = np.zeros(nrings+2, dtype=np.float64)
    _xpos_t = np.zeros(nrings+2, dtype=np.float64)
    _ypos_t = np.zeros(nrings+2, dtype=np.float64)
    _vsys_t = np.zeros(nrings+2, dtype=np.float64)
    _pa_t = np.zeros(nrings+2, dtype=np.float64)
    _incl_t = np.zeros(nrings+2, dtype=np.float64)
    _vrot_t = np.zeros(nrings+2, dtype=np.float64)
    _vrad_t = np.zeros(nrings+2, dtype=np.float64)

    _sigma_et = np.zeros(nrings+2, dtype=np.float64)
    _xpos_et = np.zeros(nrings+2, dtype=np.float64)
    _ypos_et = np.zeros(nrings+2, dtype=np.float64)
    _vsys_et = np.zeros(nrings+2, dtype=np.float64)
    _pa_et = np.zeros(nrings+2, dtype=np.float64)
    _incl_et = np.zeros(nrings+2, dtype=np.float64)
    _vrot_et = np.zeros(nrings+2, dtype=np.float64)
    _vrad_et = np.zeros(nrings+2, dtype=np.float64)

    _nrings_reliable_t = np.zeros(nrings+2, dtype=np.float64)
    _npoints_in_a_ring_total = np.zeros(nrings+2, dtype=np.float64)
    _npoints_in_a_ring = np.zeros(nrings+2, dtype=np.float64)

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

            _ring_t[_i_ti+1] = ring
            _sigma_t[_i_ti+1] = _sigma_ti
            _sigma_et[_i_ti+1] = _sigma_eti
            _xpos_t[_i_ti+1] = _xpos_ti
            _xpos_et[_i_ti+1] = _xpos_eti
            _ypos_t[_i_ti+1] = _ypos_ti
            _ypos_et[_i_ti+1] = _ypos_eti
            _vsys_t[_i_ti+1] = _vsys_ti
            _vsys_et[_i_ti+1] = _vsys_eti
            _pa_t[_i_ti+1] = _pa_ti
            _pa_et[_i_ti+1] = _pa_eti
            _incl_t[_i_ti+1] = _incl_ti
            _incl_et[_i_ti+1] = _incl_eti
            _vrot_t[_i_ti+1] = _vrot_ti
            _vrot_et[_i_ti+1] = _vrot_eti
            _vrad_t[_i_ti+1] = _vrad_ti
            _vrad_et[_i_ti+1] = _vrad_eti
            _nrings_reliable_t[_i_ti] = _i_t_reliable

            _npoints_in_a_ring_total[_i_ti+1] = npoints_in_a_ring_total_including_blanks
            _npoints_in_a_ring[_i_ti+1] = npoints_in_a_ring

    nrings_reliable = int(np.max(_nrings_reliable_t)) + 1


    if _intp_index == 'True':

        _ring_t[0] = 0
        _sigma_t[0] = _sigma_t[1]
        _xpos_t[0] = _xpos_t[1]
        _ypos_t[0] = _ypos_t[1]
        _vsys_t[0] = _vsys_t[1]
        _pa_t[0] = _pa_t[1]
        _incl_t[0] = _incl_t[1]
        _vrot_t[0] = 0
        _vrad_t[0] = _vrad_t[1]


        _ring_t[nrings_reliable] = _ring_t[nrings_reliable-1] + ring_w # --> this will be replaced to _r_galaxy_plane_e after in _2dbat.py 
        _sigma_t[nrings_reliable] = _sigma_t[nrings_reliable-1]
        _xpos_t[nrings_reliable] = _xpos_t[nrings_reliable-1]
        _ypos_t[nrings_reliable] = _ypos_t[nrings_reliable-1]
        _vsys_t[nrings_reliable] = _vsys_t[nrings_reliable-1]
        _pa_t[nrings_reliable] = _pa_t[nrings_reliable-1]
        _incl_t[nrings_reliable] = _incl_t[nrings_reliable-1]
        _vrot_t[nrings_reliable] = _vrot_t[nrings_reliable-1]
        _vrad_t[nrings_reliable] = _vrad_t[nrings_reliable-1]

        nrings_reliable += 1 # add extra outer ring



    return (_ring_t[:nrings_reliable], \
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
        nrings_reliable)





@ray.remote(num_cpus=1)
def trfit_ring_i_final(_input_vf, _tr_model_vf, _wi_2d, _params, fit_opt, fit_opt_2d, ndim, tr_params_priors_init, _tr2dfit_results, side, _i):

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

    ring = (ri+ro)/2.0

    _sigma, _sigma_e, _xpos, _xpos_e, _ypos, _ypos_e, _vsys, _vsys_e, _pa, _pa_e, _incl, _incl_e, _vrot, _vrot_e, _vrad, _vrad_e \
        = extract_tr2dfit_params(_tr2dfit_results, _params, fit_opt_2d, ring, 'entire')

    npoints_in_a_ring_total_including_blanks, npoints_in_ring, _ij_aring, _wt_2d = define_tilted_ring(_input_vf, _xpos, _ypos, _pa, _incl, ri, ro, side, _params)
    n_ij_aring = _ij_aring.shape[0]

    _a = ring # along major axis
    _b = _a * np.cos(_incl*deg_to_rad) # along minor axis
    _l = np.pi * (3.0*(_a + _b) - ((3*_a + _b) * (_a + 3*_b))**0.5) # ellipse perimeter approximation
    _ring_area = _l * ring_w
    _f_npoints = n_ij_aring / _ring_area

    if n_ij_aring > 10 or (n_ij_aring >= 10 and _i < nrings_outto_rmax): # dynesty run

        print("ring----- ", _i, ":", n_ij_aring)
        _trfit_results, _n_dim = run_nested_sampler_trfit(_input_vf, _tr_model_vf, _wt_2d, _ij_aring, _params, fit_opt, ndim, tr_params_priors_init)

        n_ring_params_free = 0
        if _params['sigma_fitting'] == 'free':
            _sigma_t = _trfit_results[n_ring_params_free]
            _sigma_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _sigma_t = _sigma
            _sigma_et = _sigma_e

        if _params['xpos_fitting'] == 'free':
            _xpos_t = _trfit_results[n_ring_params_free]
            _xpos_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _xpos_t = _xpos
            _xpos_et = _xpos_e

        if _params['ypos_fitting'] == 'free':
            _ypos_t = _trfit_results[n_ring_params_free]
            _ypos_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _ypos_t = _ypos
            _ypos_et = _ypos_e

        if _params['vsys_fitting'] == 'free':
            _vsys_t = _trfit_results[n_ring_params_free]
            _vsys_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _vsys_t = _vsys
            _vsys_et = _vsys_e

        if _params['pa_fitting'] == 'free':
            _pa_t = _trfit_results[n_ring_params_free]
            _pa_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _pa_t = _pa
            _pa_et = _pa_e

        if _params['incl_fitting'] == 'free':
            _incl_t = _trfit_results[n_ring_params_free]
            _incl_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _incl_t = _incl
            _incl_et = _incl_e

        if _params['vrot_fitting'] == 'free':
            _vrot_t = _trfit_results[n_ring_params_free]
            _vrot_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _vrot_t = _vrot
            _vrot_et = _vrot_e

        if _params['vrad_fitting'] == 'free':
            _vrad_t = _trfit_results[n_ring_params_free]
            _vrad_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _vrad_t = _vrad
            _vrad_et = _vrad_e

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




@ray.remote(num_cpus=1)
def trfit_ring_i(_input_vf, _input_int, _tr_model_vf, _wi_2d, _params, fit_opt, ndim, tr_params_priors_init, _i):

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

    side = 999
    ring_s = ring_w
    ri = ring_s + _i*ring_w - 0.5*ring_w
    ro = ring_s + _i*ring_w + 0.5*ring_w
    ring = (ri+ro)/2.0

    print("")
    print("define rings PA:", _pa)
    print("define rings INCL:", _incl)
    print("")
    npoints_in_a_ring_total_including_blanks, npoints_in_ring, _ij_aring, _wt_2d = define_tilted_ring(_input_vf, _xpos, _ypos, _pa, _incl, ri, ro, side, _params)

    npoints_in_a_ring_total_including_blanks_wt, npoints_in_ring_wt, _ij_aring_wt, _wt_2d_int = define_tilted_ring_geo(_input_int, _xpos, _ypos, _params['_theta_el'], _params['_i_el'], ri, ro, 999, _params)

    n_ij_aring = _ij_aring.shape[0]

    _a = ring # along major axis
    _b = _a * np.cos(_incl*deg_to_rad) # along minor axis
    _l = np.pi * (3.0*(_a + _b) - ((3*_a + _b) * (_a + 3*_b))**0.5) # ellipse perimeter approximation
    _ring_area = _l * ring_w
    _f_npoints = n_ij_aring / _ring_area

    if n_ij_aring > 10 or (n_ij_aring >= 10 and _i < nrings_outto_rmax): # dynesty run
        print("ring----- ", _i, ":", n_ij_aring)
        _trfit_results, _n_dim = run_nested_sampler_trfit(_input_vf, _tr_model_vf, _wt_2d_int, _ij_aring, _params, fit_opt, ndim, tr_params_priors_init)

        n_ring_params_free = 0
        if _params['sigma_fitting'] == 'free':
            _sigma_t = _trfit_results[n_ring_params_free]
            _sigma_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _sigma_t = _sigma
            _sigma_et = 999

        if _params['xpos_fitting'] == 'free':
            _xpos_t = _trfit_results[n_ring_params_free]
            _xpos_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _xpos_t = _xpos
            _xpos_et = 999

        if _params['ypos_fitting'] == 'free':
            _ypos_t = _trfit_results[n_ring_params_free]
            _ypos_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _ypos_t = _ypos
            _ypos_et = 999

        if _params['vsys_fitting'] == 'free':
            _vsys_t = _trfit_results[n_ring_params_free]
            _vsys_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _vsys_t = _vsys
            _vsys_et = 999

        if _params['pa_fitting'] == 'free':
            _pa_t = _trfit_results[n_ring_params_free]
            _pa_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _pa_t = _pa
            _pa_et = 999

        if _params['incl_fitting'] == 'free':
            _incl_t = _trfit_results[n_ring_params_free]
            _incl_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _incl_t = _incl
            _incl_et = 999

        if _params['vrot_fitting'] == 'free':
            _vrot_t = _trfit_results[n_ring_params_free]
            _vrot_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _vrot_t = _vrot
            _vrot_et = 999

        if _params['vrad_fitting'] == 'free':
            _vrad_t = _trfit_results[n_ring_params_free]
            _vrad_et = _trfit_results[ndim+n_ring_params_free]
            n_ring_params_free += 1
        else:
            _vrad_t = _vrad
            _vrad_et = 999

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




def set_params_fit_option(_params, sigma, sigma_fitoption, sigma_init, \
                          xpos, xpos_fitoption, xpos_init, \
                          ypos, ypos_fitoption, ypos_init, \
                          vsys, vsys_fitoption, vsys_init, \
                          pa, pa_fitoption, pa_init, \
                          incl, incl_fitoption, incl_init, \
                          vrot, vrot_fitoption, vrot_init, \
                          vrad, vrad_fitoption, vrad_init, \
                          nrings_reliable, first_run):

    if sigma_fitoption == 'free':
        _params['sigma_fitting'] = 'free'
    else:
        _params['sigma_fitting'] = 'fixed'
    _params['_sigma_init'] = sigma_init

    if xpos_fitoption == 'free':
        _params['xpos_fitting'] = 'free'
    else:
        _params['xpos_fitting'] = 'fixed'
    _params['_xpos_init'] = xpos_init

    if ypos_fitoption == 'free':
        _params['ypos_fitting'] = 'free'
    else:
        _params['ypos_fitting'] = 'fixed'
    _params['_ypos_init'] = ypos_init

    if vsys_fitoption == 'free':
        _params['vsys_fitting'] = 'free'
    else:
        _params['vsys_fitting'] = 'fixed'
    _params['_vsys_init'] = vsys_init

    if pa_fitoption == 'free':
        _params['pa_fitting'] = 'free'
    else:
        _params['pa_fitting'] = 'fixed'
    _params['_pa_init'] = pa_init

    if incl_fitoption == 'free':
        _params['incl_fitting'] = 'free'
    else:
        _params['incl_fitting'] = 'fixed'
    _params['_incl_init'] = incl_init

    if vrot_fitoption == 'free':
        _params['vrot_fitting'] = 'free'
    else:
        _params['vrot_fitting'] = 'fixed'
    _params['_vrot_init'] = vrot_init

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




    fit_opt = np.zeros(8, dtype=np.int32)

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

    tr_params_priors_init = np.zeros(2*8, dtype=np.float32)

    if first_run == 'True':
        xpos0 = _params['_xpos_el'] - _params['xpos_bounds_width'] 
        xpos1 = _params['_xpos_el'] + _params['xpos_bounds_width'] 
        if xpos0 < 0: xpos0 = 0
        if xpos1 > _params['naxis1']-1: xpos1 = _params['naxis1'] - 2

        ypos0 = _params['_ypos_el'] - _params['ypos_bounds_width'] 
        ypos1 = _params['_ypos_el'] + _params['ypos_bounds_width'] 
        if ypos0 < 0: ypos0 = 0
        if ypos1 > _params['naxis2']-1: ypos1 = _params['naxis2'] - 2

        vsys0 = _params['_vsys_el'] - _params['vsys_bounds_width'] 
        vsys1 = _params['_vsys_el'] + _params['vsys_bounds_width'] 

        pa0 = _params['_pa_el'] - _params['pa_bounds_width'] 
        pa1 = _params['_pa_el'] + _params['pa_bounds_width'] 
        if pa0 < 0: pa0 = 0
        if pa1 > 360: pa1 = 360

        incl0 = _params['_incl_el'] - _params['incl_bounds_width'] 
        incl1 = _params['_incl_el'] + _params['incl_bounds_width'] 
        if incl0 <= 0: incl0 = 1
        if incl1 >= 90: incl1 = 89

        sigma0 = 0
        sigma1 = 100

        vrot0 = 0
        vrot1 = 200

    else:
        xpos0 = _params['_xpos_init'] - _params['ring_w'] * 0.5
        xpos1 = _params['_xpos_init'] + _params['ring_w'] * 0.5
        if xpos0 < 0: xpos0 = 0
        if xpos1 > _params['naxis1']-1: xpos1 = _params['naxis1'] - 2

        ypos0 = _params['_ypos_init'] - _params['ring_w'] * 0.5
        ypos1 = _params['_ypos_init'] + _params['ring_w'] * 0.5
        if ypos0 < 0: ypos0 = 0
        if ypos1 > _params['naxis2']-1: ypos1 = _params['naxis2'] - 2

        vsys0 = _params['_vsys_init'] - _params['vsys_bounds_width'] 
        vsys1 = _params['_vsys_init'] + _params['vsys_bounds_width'] 

        pa0 = _params['_pa_init'] - _params['pa_bounds_width'] 
        pa1 = _params['_pa_init'] + _params['pa_bounds_width'] 

        incl0 = _params['_incl_init'] - _params['incl_bounds_width'] 
        incl1 = _params['_incl_init'] + _params['incl_bounds_width'] 

        if incl0 <= 0: incl0 = 1
        if incl1 >= 90: incl1 = 89

        sigma0 = 0
        sigma1 = _params['_sigma_init'] + _params['sigma_bounds_width'] 
        if sigma0 < 0: sigma0 = 0

        vrot0 = 0
        vrot1 = 200
        if vrot1 < 0: vrot1 = 500

    tr_params_priors_init = [sigma0, xpos0, ypos0, vsys0, pa0, incl0, vrot0, -999, \
                             sigma1, xpos1, ypos1, vsys1, pa1, incl1, vrot1, 999]


    return fit_opt, ndim, tr_params_priors_init



def trfit_2d(_input_vf, combined_vfe_res, \
            _ri, _ro, \
            _tr_model_vf, _input_int_w, _input_vdisp, _wt_2d_geo, \
            _params, tr_params_bounds, _r_galaxy_plane_2d, \
            tck_vrot_bs_init_from_trfit, tck_pa_bs_init_from_trfit, tck_incl_bs_init_from_trfit, _2dbat_run_i, region):


    _sigma = _params['_sigma_init']
    _xpos = _params['_xpos_init']
    _ypos = _params['_ypos_init']
    _pa = _params['_pa_init']
    _incl = _params['_incl_init']
    _vsys = _params['_vsys_init']
    _vrot = _params['_vrot_init']
    _vrad = _params['_vrad_init']
    side = 0


    npoints_total = 0 
    npoints_in_current_ring = 0
    npoints_in_a_ring_total_including_blanks, npoints_in_ring, _ij_aring, _wt_2d = define_tilted_ring(_input_vf, _xpos, _ypos, _pa, _incl, _ri, _ro, side, _params)

    

    n_ij_aring = _ij_aring.shape[0]
    r_galaxy_plane = np.zeros(n_ij_aring, dtype=np.float64)
    print('-'*50)
    print("+ | r_galaxy_plane_e_geo: %.2f" % _params['r_galaxy_plane_e_geo'])
    print("+ | N-points to fit [%.2f ~ %.2f]:" % (_ri, _ro), n_ij_aring)
    print("+ | xpos_init: %.2f" % _xpos)
    print("+ | ypos_init: %.2f" % _ypos)
    print("+ | vsys_init: %.2f" % _vsys)
    print("+ | pa_init: %.2f" % _pa)
    print("+ | incl_init: %.2f" % _incl)
    print('-'*50)
    print(tr_params_bounds)
    print('-'*50)

    _trfit_results, _n_dim, fit_opt_2d, std_resample_run = run_nested_sampler_trfit_2d(_input_vf, combined_vfe_res, \
                                                                                       _tr_model_vf, \
                                                                                       _input_int_w, \
                                                                                       _input_vdisp, \
                                                                                       _wt_2d_geo, \
                                                                                       _ij_aring, \
                                                                                       _params, \
                                                                                       tr_params_bounds, \
                                                                                       _r_galaxy_plane_2d, \
                                                                                       r_galaxy_plane, \
                                                                                       tck_vrot_bs_init_from_trfit, \
                                                                                       tck_pa_bs_init_from_trfit, \
                                                                                       tck_incl_bs_init_from_trfit, \
                                                                                       _2dbat_run_i, \
                                                                                       region)

    print("")
    for p in range(0, _n_dim):
        print("p-", p, "=", _trfit_results[p], "+/-", _trfit_results[p+_n_dim])

    return _ij_aring, _trfit_results, _n_dim, fit_opt_2d, std_resample_run



    fit_opt = np.zeros(8, dtype=np.int32)

    n_coeffs_vrot_bs, tck_vrot_bs = bspline_ncoeffs_tck(_params, 'vrot', _params['nrings_intp'])
    n_coeffs_pa_bs, tck_pa_bs = bspline_ncoeffs_tck(_params, 'pa', _params['nrings_intp'])
    n_coeffs_incl_bs, tck_incl_bs = bspline_ncoeffs_tck(_params, 'incl', _params['nrings_intp'])


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














def estimate_init_values(_params, sigma, sigma_init, \
                          xpos, xpos_init, \
                          ypos, ypos_init, \
                          vsys, vsys_init, \
                          pa, pa_init, \
                          incl, incl_init, \
                          vrot, vrot_init, \
                          vrad, vrad_init):
    try:
        if _params['sigma_fitting'] == 'free':
            _params['_sigma_init'] = sigma_init
        if _params['xpos_fitting'] == 'free':
            _params['_xpos_init'] = xpos_init
        if _params['ypos_fitting'] == 'free':
            _params['_ypos_init'] = ypos_init
        if _params['vsys_fitting'] == 'free':
            _params['_vsys_init'] = vsys_init
        if _params['pa_fitting'] == 'free':
            _params['_pa_init'] = pa_init
        if _params['incl_fitting'] == 'free':
            _params['_incl_init'] = incl_init
        if _params['vrot_fitting'] == 'free':
            _params['_vrot_init'] = vrot_init
        if _params['vrad_fitting'] == 'free':
            _params['_vrad_init'] = vrad_init
    except:
        pass

    print("initialized...")



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
    _input_vf_vlos_LU_masked = np.where((_input_vf > _params['_vlos_lower']), _input_vf, -1E10)
    _input_vf_vlos_LU_masked = np.where((_input_vf < _params['_vlos_upper']), _input_vf_vlos_LU_masked, -1E10)

    return _input_vf_vlos_LU_masked, _tr_model_vf


def gaussian(x, mean, sigma):
    return np.exp(-np.power(x - mean, 2.) / (2 * np.power(sigma, 2.)))


def sigma_clipping(data, n_sigma=3, max_iter=1000):
    for _i in range(max_iter):
        old_size = data.size
        mean, sigma = norm.fit(data)
        filtered_data = data[(data > mean - n_sigma * sigma) & (data < mean + n_sigma * sigma)]
        if filtered_data.size == old_size:
            break  # 변화가 없으면 종료
        data = filtered_data
    return norm.fit(data)  # 최종 평균과 표준편차 반환



def find_area_tofit(_params, _2dbat_run_i, _2dbat_run_i_pre):
    global _tr_model_vf


    with fits.open(_params['wdir'] + '/' + _params['input_int'], 'update') as hdu:
        try:
            _input_int = hdu[0].data
        except Exception as e:
            print(f"Error: {e}")

    with fits.open(_params['wdir'] + '/' + _params['input_vf'], 'update') as hdu:
        try:
            _input_vf = hdu[0].data
            _naxis1 = hdu[0].header['NAXIS1']
            _naxis2 = hdu[0].header['NAXIS2']
        except Exception as e:
            print(f"Error: {e}")


    with fits.open(_params['wdir'] + '/' + _params['input_vdisp'], 'update') as hdu:
        try:
            _input_vdisp = hdu[0].data
        except Exception as e:
            print(f"Error: {e}")


    x_grid_tr, y_grid_tr = _params['x_grid_tr'], _params['y_grid_tr']
    x_grid_2d, y_grid_2d = _params['x_grid_2d'], _params['y_grid_2d']
    x_indices_2d = np.arange(0, _naxis1, x_grid_2d)
    y_indices_2d = np.arange(0, _naxis2, y_grid_2d)

    if _params['input_vf_e'] != 999:
        with fits.open(_params['wdir'] + '/' + _params['input_vf_e'], 'update') as hdu:
            try:
                _input_vf_e = hdu[0].data
            except Exception as e:
                print(f"Error: {e}")


        _input_vf_e_flatten = _input_vf_e[~np.isnan(_input_vf_e)].flatten()



        mean_vf_e, sigma_vf_e = sigma_clipping(_input_vf_e_flatten)

        clip_l_vf_e = mean_vf_e - 3*sigma_vf_e
        clip_u_vf_e = mean_vf_e + 3*sigma_vf_e
        if clip_l_vf_e < 0: clip_l_vf_e= 0

        condition_2d_e = (_input_vf_e[y_indices_2d[:, None], x_indices_2d] > clip_l_vf_e) \
                     & (_input_vf_e[y_indices_2d[:, None], x_indices_2d] < clip_u_vf_e) \
                     & (_input_vf[y_indices_2d[:, None], x_indices_2d] > _params['_vlos_lower']) \
                     & (_input_vf[y_indices_2d[:, None], x_indices_2d] < _params['_vlos_upper'])

        _input_vf_e_clipped_rs = np.full((_naxis2, _naxis1), fill_value=np.nan, dtype=np.float64)
        _input_vf_e_clipped_rs[y_indices_2d[:, None], x_indices_2d] = np.where(condition_2d_e, _input_vf_e[y_indices_2d[:, None], x_indices_2d], np.nan)

        median_val = np.nanmedian(_input_vf_e_clipped_rs)
        q1 = np.nanpercentile(_input_vf_e_clipped_rs, 25)
        q3 = np.nanpercentile(_input_vf_e_clipped_rs, 75)
        iqr = q3 - q1

        scaled_data_vf_e = (_input_vf_e_clipped_rs - median_val) / iqr

        scaled_data_positive_vf_e = np.abs(scaled_data_vf_e)
        _input_vf_e_clipped_rs[y_indices_2d[:, None], x_indices_2d] = np.where(condition_2d_e & (scaled_data_positive_vf_e[y_indices_2d[:, None], x_indices_2d] > 0.0001), scaled_data_positive_vf_e[y_indices_2d[:, None], x_indices_2d], np.nan)
        write_fits_images(_params, _input_vf_e_clipped_rs, _2dbat_run_i, '_input_vf_e_clipped_rs.fits')

    else:
        condition_2d_e = (_input_vf[y_indices_2d[:, None], x_indices_2d] > _params['_vlos_lower']) \
                       & (_input_vf[y_indices_2d[:, None], x_indices_2d] < _params['_vlos_upper'])

        _input_vf_e_clipped_rs = np.full((_naxis2, _naxis1), fill_value=np.nan, dtype=np.float64)
        _input_vf_e_clipped_rs[y_indices_2d[:, None], x_indices_2d] = np.where(condition_2d_e, 1., np.nan)
        write_fits_images(_params, _input_vf_e_clipped_rs, _2dbat_run_i, '_input_vf_e_clipped_rs.fits')


    if _2dbat_run_i_pre != 999:
        with fits.open(_params['wdir'] + '/' + _params['_2dbatdir'] + ".%d" % _2dbat_run_i_pre + '/' + _params['input_vf_m_bsfit'], 'update') as hdu:
            try:
                _input_vf_m_bsfit = hdu[0].data
            except Exception as e:
                print(f"Error: {e}")
        
        _input_vf_m_bsfit_flatten = _input_vf_m_bsfit[~np.isnan(_input_vf_m_bsfit)].flatten()


        mean_res, sigma_res = sigma_clipping(_input_vf_m_bsfit_flatten)

        clip_l_res = mean_res - 3*sigma_res
        clip_u_res = mean_res + 3*sigma_res


        print(mean_res, sigma_res)
        print(clip_l_res, clip_u_res)

        condition_2d_res = (_input_vf_m_bsfit[y_indices_2d[:, None], x_indices_2d] > clip_l_res) \
                     & (_input_vf_m_bsfit[y_indices_2d[:, None], x_indices_2d] < clip_u_res) \
                     & (_input_vf[y_indices_2d[:, None], x_indices_2d] > _params['_vlos_lower']) \
                     & (_input_vf[y_indices_2d[:, None], x_indices_2d] < _params['_vlos_upper'])

        _input_vf_m_bsfit_clipped_rs = np.full((_naxis2, _naxis1), fill_value=np.nan, dtype=np.float64)
        _input_vf_m_bsfit_clipped_rs[y_indices_2d[:, None], x_indices_2d] = np.where(condition_2d_res, _input_vf_m_bsfit[y_indices_2d[:, None], x_indices_2d], np.nan)

        median_val = np.nanmedian(_input_vf_m_bsfit_clipped_rs)
        q1 = np.nanpercentile(_input_vf_m_bsfit_clipped_rs, 25)
        q3 = np.nanpercentile(_input_vf_m_bsfit_clipped_rs, 75)
        iqr = q3 - q1

        print(median_val, iqr)

        scaled_data_res = (_input_vf_m_bsfit_clipped_rs - median_val) / iqr

        scaled_data_positive_res = np.abs(scaled_data_res)

        print(_input_vf_m_bsfit_clipped_rs.shape)
        print(scaled_data_res.shape)
        _input_vf_m_bsfit_clipped_rs[y_indices_2d[:, None], x_indices_2d] = np.where(condition_2d_res & (scaled_data_positive_res[y_indices_2d[:, None], x_indices_2d] > 0.0001), scaled_data_positive_res[y_indices_2d[:, None], x_indices_2d], np.nan)

        write_fits_images(_params, _input_vf_m_bsfit_clipped_rs, _2dbat_run_i, '_input_vf_m_bsfit_cliiped_rs.fits')

    else:
        condition_2d_res = (_input_vf[y_indices_2d[:, None], x_indices_2d] > _params['_vlos_lower']) \
                       & (_input_vf[y_indices_2d[:, None], x_indices_2d] < _params['_vlos_upper'])

        _input_vf_m_bsfit_clipped_rs = np.full((_naxis2, _naxis1), fill_value=np.nan, dtype=np.float64)
        _input_vf_m_bsfit_clipped_rs[y_indices_2d[:, None], x_indices_2d] = np.where(condition_2d_res, 1., np.nan)
        write_fits_images(_params, _input_vf_m_bsfit_clipped_rs, _2dbat_run_i, '_input_vf_m_bsfit_cliiped_rs.fits')


    _params['naxis1'] = _naxis1
    _params['naxis2'] = _naxis2
    _input_vf = find_the_largest_blob(_input_vf,  _params['_vlos_lower'])
    _input_int = find_the_largest_blob(_input_int,  _params['_int_lower'])


    _tr_model_vf = np.full((_naxis2, _naxis1), fill_value=np.nan, dtype=np.float64)
    _input_vf_vlos_LU_masked_2d = np.full((_naxis2, _naxis1), fill_value=np.nan, dtype=np.float64)
    _input_vf_vlos_LU_masked_tr = np.full((_naxis2, _naxis1), fill_value=np.nan, dtype=np.float64)
    _input_vdisp_LU_masked_2d = np.full((_naxis2, _naxis1), fill_value=np.nan, dtype=np.float64)
    

    _input_int_LU_masked_nogrid = np.where((_input_int > _params['_int_lower']), _input_int, np.nan)
    _input_int_LU_masked_nogrid = np.where((_input_int_LU_masked_nogrid < _params['_int_upper']), _input_int_LU_masked_nogrid, np.nan)

    filtered_data_int = _input_int_LU_masked_nogrid[~np.isnan(_input_int_LU_masked_nogrid)].flatten()

    mean_int, sigma_int = norm.fit(filtered_data_int)

    _input_int_w = np.where(_input_int >= mean_int, 1, gaussian(_input_int, mean_int, sigma_int))


    write_fits_images(_params, _input_int_w, _2dbat_run_i, '_input_int_w.fits')



    _input_vf_vlos_LU_masked_nogrid = np.where((_input_vf > _params['_vlos_lower']), _input_vf, np.nan)
    _input_vf_vlos_LU_masked_nogrid = np.where((_input_vf_vlos_LU_masked_nogrid < _params['_vlos_upper']), _input_vf_vlos_LU_masked_nogrid, np.nan)






    _input_vdisp_LU_masked_nogrid = np.where((_input_vdisp > _params['_vdisp_lower']), _input_vdisp, np.nan)
    _input_vdisp_LU_masked_nogrid = np.where((_input_vdisp_LU_masked_nogrid < _params['_vdisp_upper']), _input_vdisp_LU_masked_nogrid, np.nan)



    _input_vdisp = _input_vdisp_LU_masked_nogrid / np.nansum(_input_vdisp_LU_masked_nogrid)

    write_fits_images(_params, _input_vdisp, _2dbat_run_i, '_input_vdisp_norm.fits')

    combined_int_vdisp = _input_int_w*_input_vdisp
    filtered_data_int_vdisp = combined_int_vdisp[~np.isnan(combined_int_vdisp)].flatten()

    value_50percentile = np.percentile(filtered_data_int_vdisp, 50)

    _input_int_vdisp_w_elfit = np.where(combined_int_vdisp <= value_50percentile, np.nan, combined_int_vdisp)

    write_fits_images(_params, _input_int_w*_input_vdisp, _2dbat_run_i, '_input_int_vdisp_w.fits')
    write_fits_images(_params, _input_int_vdisp_w_elfit, _2dbat_run_i, '_input_int_vdisp_w_elfit.fits')


    combined_vfe_res = _input_vf_m_bsfit_clipped_rs*_input_vf_e_clipped_rs


    if _params['input_vf_e'] != 999 or _2dbat_run_i_pre != 999:

        combined_vfe_res_flatten = combined_vfe_res[~np.isnan(combined_vfe_res)].flatten()

        mean_combined, sigma_combined = sigma_clipping(combined_vfe_res_flatten)

        clip_l_combined = mean_combined - 3*sigma_combined
        clip_u_combined = mean_combined + 3*sigma_combined


        print(mean_combined, sigma_combined)
        print(clip_l_combined, clip_u_combined)

        condition_2d_combined = (combined_vfe_res[y_indices_2d[:, None], x_indices_2d] > clip_l_combined) \
                     & (combined_vfe_res[y_indices_2d[:, None], x_indices_2d] < clip_u_combined)

        combined_vfe_res_clipped_rs = np.full((_naxis2, _naxis1), fill_value=np.nan, dtype=np.float64)
        combined_vfe_res_clipped_rs[y_indices_2d[:, None], x_indices_2d] = np.where(condition_2d_combined, combined_vfe_res[y_indices_2d[:, None], x_indices_2d], np.nan)

        median_val = np.nanmedian(combined_vfe_res_clipped_rs)
        q1 = np.nanpercentile(combined_vfe_res_clipped_rs, 25)
        q3 = np.nanpercentile(combined_vfe_res_clipped_rs, 75)
        iqr = q3 - q1

        print(median_val, iqr)

        scaled_data_combined = (combined_vfe_res_clipped_rs - median_val) / iqr

        scaled_data_positive_combined = np.abs(scaled_data_combined)

        combined_vfe_res_clipped_rs[y_indices_2d[:, None], x_indices_2d] = np.where(condition_2d_combined & (scaled_data_positive_combined[y_indices_2d[:, None], x_indices_2d] > 0.0001), scaled_data_positive_combined[y_indices_2d[:, None], x_indices_2d], np.nan)

        write_fits_images(_params, combined_vfe_res_clipped_rs, _2dbat_run_i, 'combined_vfe_res_clipped_rs.fits')

    else: # 999 and 999

        condition_2d_combined = (_input_vf[y_indices_2d[:, None], x_indices_2d] > _params['_vlos_lower']) \
                       & (_input_vf[y_indices_2d[:, None], x_indices_2d] < _params['_vlos_upper'])

        combined_vfe_res_clipped_rs = np.full((_naxis2, _naxis1), fill_value=np.nan, dtype=np.float64)
        combined_vfe_res_clipped_rs[y_indices_2d[:, None], x_indices_2d] = np.where(condition_2d_combined, 1., np.nan)
        write_fits_images(_params, combined_vfe_res_clipped_rs, _2dbat_run_i, 'combined_vfe_res_clipped_rs.fits')






    if _2dbat_run_i_pre != 999:
        condition_2d = (_input_vf[y_indices_2d[:, None], x_indices_2d] > _params['_vlos_lower']) \
                     & (_input_vf[y_indices_2d[:, None], x_indices_2d] < _params['_vlos_upper'])
        
    else:
        condition_2d = (_input_vf[y_indices_2d[:, None], x_indices_2d] > _params['_vlos_lower']) & (_input_vf[y_indices_2d[:, None], x_indices_2d] < _params['_vlos_upper'])

    condition_2d_vdisp = (_input_vdisp[y_indices_2d[:, None], x_indices_2d] > _params['_vdisp_lower']) & (_input_vdisp[y_indices_2d[:, None], x_indices_2d] < _params['_vdisp_upper'])

    _input_vf_vlos_LU_masked_2d[y_indices_2d[:, None], x_indices_2d] = np.where(condition_2d, _input_vf[y_indices_2d[:, None], x_indices_2d], np.nan)
    _input_vdisp_LU_masked_2d[y_indices_2d[:, None], x_indices_2d] = np.where(condition_2d_vdisp, _input_vdisp[y_indices_2d[:, None], x_indices_2d], np.nan)

    write_fits_images(_params, _input_vf_vlos_LU_masked_2d, _2dbat_run_i, 'input_vf_filtered.fits')

    x_indices_tr = np.arange(0, _naxis1, x_grid_tr)
    y_indices_tr = np.arange(0, _naxis2, y_grid_tr)

    condition_tr = (_input_vf[y_indices_tr[:, None], x_indices_tr] > _params['_vlos_lower']) & (_input_vf[y_indices_tr[:, None], x_indices_tr] < _params['_vlos_upper'])

    _input_vf_vlos_LU_masked_tr[y_indices_tr[:, None], x_indices_tr] = np.where(condition_tr, _input_vf[y_indices_tr[:, None], x_indices_tr], np.nan)

    return _input_vf_vlos_LU_masked_nogrid, _input_vf_vlos_LU_masked_tr, _input_vf_vlos_LU_masked_2d, _tr_model_vf, \
           _input_int_w, _input_vdisp, _input_int_w*_input_vdisp, _input_int_vdisp_w_elfit, _input_int_LU_masked_nogrid, combined_vfe_res_clipped_rs




def bspline_fit_to_1d(_cube_mask_2d, _params, ring_param):

    with fits.open(_params['wdir'] + '/' + _params['input_datacube'], 'update') as hdu:
        _naxis1 = hdu[0].header['NAXIS1']
        try:
            hdu[0].header['RESTFREQ'] = hdu[0].header['FREQ0'] # THIS IS NEEDED WHEN INPUTING FITS PROCESSED WITH GIPSY
        except:
            pass



def derive_rad_profiles(_cube_mask_2d, _params, ring_param):

    with fits.open(_params['wdir'] + '/' + _params['input_datacube'], 'update') as hdu:
        _naxis1 = hdu[0].header['NAXIS1']
        try:
            hdu[0].header['RESTFREQ'] = hdu[0].header['FREQ0'] # THIS IS NEEDED WHEN INPUTING FITS PROCESSED WITH GIPSY
        except:
            pass

    print("ok")






















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

    _x = np.linspace(0, 1, _naxis3, dtype=np.float32)
    _vel_min = cube.spectral_axis.min().value
    _vel_max = cube.spectral_axis.max().value
    _params['vel_min'] = _vel_min   
    _params['vel_max'] = _vel_max  

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

    _inputDataCube = fitsio.read(_params['wdir'] + '/' + _params['input_datacube'])
    return _inputDataCube, _x



def update_header_cube_to_2d(_hdulist_nparray, _hdu_cube):

    _hdulist_nparray[0].header.insert('NAXIS2', ('CDELT1', _hdu_cube[0].header['CDELT1']), after=True)
    _hdulist_nparray[0].header.insert('CDELT1', ('CRPIX1', _hdu_cube[0].header['CRPIX1']), after=True)
    _hdulist_nparray[0].header.insert('CRPIX1', ('CRVAL1', _hdu_cube[0].header['CRVAL1']), after=True)
    _hdulist_nparray[0].header.insert('CRVAL1', ('CTYPE1', _hdu_cube[0].header['CTYPE1']), after=True)
    try:
        _hdulist_nparray[0].header.insert('CTYPE1', ('CUNIT1', _hdu_cube[0].header['CUNIT1']), after=True)
    except:
        _hdulist_nparray[0].header.insert('CTYPE1', ('CUNIT1', 'deg'), after=True)


    _hdulist_nparray[0].header.insert('CUNIT1', ('CDELT2', _hdu_cube[0].header['CDELT2']), after=True)
    _hdulist_nparray[0].header.insert('CDELT2', ('CRPIX2', _hdu_cube[0].header['CRVAL2']), after=True)
    _hdulist_nparray[0].header.insert('CRPIX2', ('CRVAL2', _hdu_cube[0].header['CRVAL2']), after=True)
    _hdulist_nparray[0].header.insert('CRVAL2', ('CTYPE2', _hdu_cube[0].header['CTYPE2']), after=True)

    try:
        _hdulist_nparray[0].header.insert('CTYPE2', ('CUNIT2', _hdu_cube[0].header['CUNIT2']), after=True)
    except:
        _hdulist_nparray[0].header.insert('CTYPE2', ('CUNIT2', 'deg'), after=True)



def write_fits_seg(_segarray, _segfitsfile):
    hdu = fits.PrimaryHDU(data=_segarray)
    hdu.writeto(_segfitsfile, overwrite=True)




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


    _input_cube = SpectralCube.read(_params['wdir'] + '/' + _params['input_datacube']).with_spectral_unit(u.km/u.s, velocity_convention='radio')


    _input_cube.beam_threshold = 0.1

   
    _flux_threshold = _params['mom0_nrms_limit']*_params['_rms_med'] + _params['_bg_med']
    peak_sn_mask = _input_cube > _flux_threshold*u.Jy/u.beam

    _input_cube_peak_sn_masked = _input_cube.with_mask(peak_sn_mask)

    mom0 = _input_cube_peak_sn_masked.with_spectral_unit(u.km/u.s).moment(order=0, axis=0)

    if _input_cube.beam_threshold > 0.09: # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
        _N_masked = np.where(np.isnan(_input_cube_peak_sn_masked.hdulist[0].data), -1E5, _input_cube_peak_sn_masked.hdulist[0].data)


    _N_masked = np.where(np.isnan(_input_cube_peak_sn_masked.hdu.data), -1E5, _input_cube_peak_sn_masked.hdu.data)

    _N_masked = np.where(np.isinf(_N_masked), -1E5, _N_masked)
    _N_masked = np.where(np.isinf(-1*_N_masked), -1E5, _N_masked)
    _N = (_N_masked > -1E5).sum(axis=0)

    _rms_int = np.sqrt(_N) * _params['_rms_med'] * (cdelt3/1000.)


    _sn_int_map = mom0 / _rms_int


    _sn_int_map_nparray = _sn_int_map.hdu.data
    _sn_int_map_nparray = np.where(np.isnan(_sn_int_map_nparray), 0, _sn_int_map_nparray)
    _sn_int_map_nparray = np.where(np.isinf(_sn_int_map_nparray), 1E5, _sn_int_map_nparray)
    _sn_int_map_nparray = np.where(np.isinf(-1*_sn_int_map_nparray), -1E5, _sn_int_map_nparray)


    peak_flux_map = _input_cube.hdu.data.max(axis=0)
    peak_flux_map = np.where(np.isnan(peak_flux_map), -1E5, peak_flux_map)
    peak_flux_map = np.where(np.isinf(peak_flux_map), 1E5, peak_flux_map)
    peak_flux_map = np.where(np.isinf(-1*peak_flux_map), -1E5, peak_flux_map)

    peak_sn_map = (peak_flux_map - _params['_bg_med']) / _params['_rms_med']


    mom0.write('test1.mom0.fits', overwrite=True)

    _sn_int_map.hdu.header['BUNIT'] = 's/n'
    _sn_int_map.write('test1.sn_int.fits', overwrite=True)

    return peak_sn_map, _sn_int_map_nparray










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


    _input_cube = SpectralCube.read(_params['wdir'] + '/' + _params['input_datacube']).with_spectral_unit(u.km/u.s, velocity_convention='radio')


    _input_cube.beam_threshold = 0.1

   
    _flux_threshold = _params['mom0_nrms_limit']*_params['_rms_med'] + _params['_bg_med']
    peak_sn_mask = _input_cube > _flux_threshold*u.Jy/u.beam

    _input_cube_peak_sn_masked = _input_cube.with_mask(peak_sn_mask)

    mom0 = _input_cube_peak_sn_masked.with_spectral_unit(u.km/u.s).moment(order=0, axis=0)

    if _input_cube.beam_threshold > 0.09: # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
        _N_masked = np.where(np.isnan(_input_cube_peak_sn_masked.hdulist[0].data), -1E5, _input_cube_peak_sn_masked.hdulist[0].data)


    _N_masked = np.where(np.isnan(_input_cube_peak_sn_masked.hdu.data), -1E5, _input_cube_peak_sn_masked.hdu.data)

    _N_masked = np.where(np.isinf(_N_masked), -1E5, _N_masked)
    _N_masked = np.where(np.isinf(-1*_N_masked), -1E5, _N_masked)
    _N = (_N_masked > -1E5).sum(axis=0)

    _rms_int = np.sqrt(_N) * _params['_rms_med'] * (cdelt3/1000.)


    _sn_int_map = mom0 / _rms_int


    _sn_int_map_nparray = _sn_int_map.hdu.data
    _sn_int_map_nparray = np.where(np.isnan(_sn_int_map_nparray), 0, _sn_int_map_nparray)
    _sn_int_map_nparray = np.where(np.isinf(_sn_int_map_nparray), 1E5, _sn_int_map_nparray)
    _sn_int_map_nparray = np.where(np.isinf(-1*_sn_int_map_nparray), -1E5, _sn_int_map_nparray)


    peak_flux_map = _input_cube.hdu.data.max(axis=0)
    peak_flux_map = np.where(np.isnan(peak_flux_map), -1E5, peak_flux_map)
    peak_flux_map = np.where(np.isinf(peak_flux_map), 1E5, peak_flux_map)
    peak_flux_map = np.where(np.isinf(-1*peak_flux_map), -1E5, peak_flux_map)

    peak_sn_map = (peak_flux_map - _params['_bg_med']) / _params['_rms_med']


    mom0.write('test1.mom0.fits', overwrite=True)

    _sn_int_map.hdu.header['BUNIT'] = 's/n'
    _sn_int_map.write('test1.sn_int.fits', overwrite=True)

    return peak_sn_map, _sn_int_map_nparray



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
    
    
    cubedata = fitsio.read(_params['wdir'] + '/' + _params['input_datacube'])
   
    _chan_linefree1 = np.mean(fits.open(_params['wdir'] + '/' + _params['input_datacube'])[0].data[0:int(_naxis3*0.05):1, :, :], axis=0) # first 5% channels
    _chan_linefree2 = np.mean(fits.open(_params['wdir'] + '/' + _params['input_datacube'])[0].data[int(_naxis3*0.95):_naxis3-1:1, :, :], axis=0) # last 5% channels
    _chan_linefree = (_chan_linefree1 + _chan_linefree2)/2.

    _chan_linefree = np.where(np.isnan(_chan_linefree), -1E5, _chan_linefree)
    _chan_linefree = np.where(np.isinf(_chan_linefree), 1E5, _chan_linefree)
    _chan_linefree = np.where(np.isinf(-1*_chan_linefree), -1E5, _chan_linefree)
    _mean_bg, _median_bg, _std_bg = sigma_clipped_stats(_chan_linefree, sigma=3.0)

    _flux_threshold = _params['mom0_nrms_limit']*_params['_rms_med'] + _median_bg
    _input_cube = SpectralCube.read(_params['wdir'] + '/' + _params['input_datacube']).with_spectral_unit(u.km/u.s, velocity_convention='radio')

    peak_sn_mask = _input_cube > _flux_threshold*u.Jy/u.beam
    _input_cube_peak_sn_masked = _input_cube.with_mask(peak_sn_mask)
    mom0 = _input_cube_peak_sn_masked.with_spectral_unit(u.km/u.s).moment(order=0, axis=0)

    peak_flux_map = _input_cube.hdu.data.max(axis=0)
    peak_flux_map = np.where(np.isnan(peak_flux_map), -1E5, peak_flux_map)
    peak_flux_map = np.where(np.isinf(peak_flux_map), 1E5, peak_flux_map)
    peak_flux_map = np.where(np.isinf(-1*peak_flux_map), -1E5, peak_flux_map)


    peak_sn_map = (peak_flux_map - _median_bg) / _params['_rms_med']

    _N_masked = np.where(np.isnan(_input_cube_peak_sn_masked.hdu.data), -1E5, _input_cube_peak_sn_masked.hdu.data)
    _N_masked = np.where(np.isinf(_N_masked), -1E5, _N_masked)
    _N_masked = np.where(np.isinf(-1*_N_masked), -1E5, _N_masked)

    _N = (_N_masked > -1E5).sum(axis=0)
    _rms_int = np.sqrt(_N) * _params['_rms_med'] * (_params['cdelt3']/1000.)

    print(mom0)
    print(_rms_int)
    _sn_int_map = mom0 / _rms_int

    _sn_int_map_nparray = _sn_int_map.hdu.data
    _sn_int_map_nparray = np.where(np.isnan(_sn_int_map_nparray), 0, _sn_int_map_nparray)
    _sn_int_map_nparray = np.where(np.isinf(_sn_int_map_nparray), 1E5, _sn_int_map_nparray)
    _sn_int_map_nparray = np.where(np.isinf(-1*_sn_int_map_nparray), -1E5, _sn_int_map_nparray)

    mom0.write('test1.mom0.fits', overwrite=True)
    _sn_int_map.hdu.header['BUNIT'] = 's/n'
    _sn_int_map.write('test1.sn_int.fits', overwrite=True)

    return peak_sn_map, _sn_int_map_nparray

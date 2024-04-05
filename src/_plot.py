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



import matplotlib
from mpl_toolkits.mplot3d import Axes3D


from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.patches import Ellipse
import pandas as pd
import numpy as np
import sys
import math

from astropy.io import fits

import os

from _dirs_files import make_dirs






def align_data_to_header(df, column_names):
    formatted_data = df.to_string(header=False, index=False)
    return formatted_data


def format_header_string(column_names):
    numbered_header = [f'! {i+1}: {name}' for i, name in enumerate(column_names)]
    header_string = '\n'.join(numbered_header)
    return header_string

def add_comment_to_first_line(data_string):
    lines = data_string.split('\n')
    lines[0] = '!' + lines[0]
    return '\n'.join(lines)


def plot_write_2dbat_all(_2dbat_trfit_txt, input_vf, bsfit_vf, trfit_vf, output_filename, _params, _2dbat_run_i, output_txt):

    _dir_2dbat_PI_output = _params['wdir'] + '/' + _params['_2dbatdir'] + ".%d" % _2dbat_run_i
    with open(f'{_dir_2dbat_PI_output}/%s' % _2dbat_trfit_txt, 'r') as file:
        column_names = [
            'radius [pix]', 'sigma [km/s]', 'sigma-e [km/s]', 'xpos [pix]', 'xpos-e [pix]',
            'ypos [pix]', 'ypos-e [pix]', 'vsys [km/s]', 'vsys-e [km/s]', 'pa [deg]',
            'pa-e [deg]', 'incl [deg]', 'incl-e [deg]', 'vrot_bs [km/s]', 'vrot_bs-e [km/s]', 'vrot_b [km/s]', 'vrot_b-e [km/s]',
            'vrot_a [km/s]', 'vrot_a-e [km/s]', 'vrot_r [km/s]', 'vrot_r-e [km/s]',
            'vrad [km/s]', 'vrad-e [km/s]', 'npoints_total-b [pixs]', 'npoints_available-b [pixs]',
            'npoints_total-a [pixs]', 'npoints_available-a [pixs]', 'npoints_total-r [pixs]',
            'npoints_available-r [pixs]'
        ]

        df = pd.read_csv(
            file,
            sep='\s+',
            comment='!',
            header=None,
            names=column_names,
            skiprows=1  # skip the first raw 
        )


    with open(_dir_2dbat_PI_output + '/' + output_txt, 'w') as file:
        header_string = format_header_string(column_names)
        file.write(header_string + '\n')

        aligned_data = df.to_string(index=False, justify='right')
        aligned_data_with_comment = add_comment_to_first_line(aligned_data)
        file.write(aligned_data_with_comment)


    fig = plt.figure(figsize=(11.69, 8.27))

    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[3, 2])

    gs_left_top = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[0, 0], hspace=0.3)
    gs_left_bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 0])

    gs_right_top = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 1])
    gs_right_bottom = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[1, 1], hspace=1.5)



    vmin_res = -20
    vmax_res = 20
    contour_levels_res = np.linspace(vmin_res, vmax_res, 5)

    naxis1 = _params['naxis1']
    naxis2 = _params['naxis2']

    ax_f = float(naxis2/naxis1)

    x_min = int(df['xpos [pix]'][0] - df['radius [pix]'].iloc[-1] - df['radius [pix]'][0])
    x_max = int(df['xpos [pix]'][0] + df['radius [pix]'].iloc[-1] + df['radius [pix]'][0])

    y_min = int((df['ypos [pix]'][0] - df['radius [pix]'].iloc[-1] - df['radius [pix]'][0])*ax_f)
    y_max = int((df['ypos [pix]'][0] + df['radius [pix]'].iloc[-1] + df['radius [pix]'][0])*ax_f)

    if x_min < 0: x_min = 0
    if x_max > naxis1: x_max = naxis1
    if y_min < 0: y_min = 0
    if y_max > naxis2: y_max = naxis2


    vmin = np.nanmin(bsfit_vf)
    vmax = np.nanmax(bsfit_vf)

    vmin = np.nanpercentile(bsfit_vf, 5)
    vmax = np.nanpercentile(bsfit_vf, 95)

    vmin = math.floor(vmin / 10) * 10
    vmax = math.ceil(vmax / 10) * 10
    steps = int((vmax - vmin)*0.1)
    contour_levels = np.arange(vmin, vmax, steps)

    input_m_bsfit = input_vf[y_min:y_max, x_min:x_max] - bsfit_vf[y_min:y_max, x_min:x_max]

    vmin_res = np.nanpercentile(input_m_bsfit, 5)
    vmax_res = np.nanpercentile(input_m_bsfit, 95)
    vmin_res = math.floor(vmin_res / 10) * 10
    vmax_res = math.ceil(vmax_res / 10) * 10
    steps_res = int((vmax_res - vmin_res)*0.1)
    if steps_res == 0: steps_res = 1
    contour_levels_res = np.arange(vmin_res, vmax_res, steps_res)


    xc, yc = _params['_xpos_el'], _params['_ypos_el']  # 타원의 중심 좌표
    incl = _params['_incl_el']  # 타원의 기울기 각도(도)
    theta = _params['_pa_el'] + 90  # 타원의 회전 각도(도)
    _r_e = _params['r_galaxy_plane_e']
    r_ellipse = (_r_e, _r_e*np.cos(np.radians(incl)))  # 타원의 반경(가로, 세로)







    i, j = 0, 0
    ax = plt.subplot(gs_left_top[i, j])
    ellipse1 = Ellipse(xy=(xc, yc), width=2*r_ellipse[0], height=2*r_ellipse[1]*ax_f, angle=theta, edgecolor='grey', fc='None', lw=0.8)
    zoomed_input = input_vf[y_min:y_max, x_min:x_max]
    cax = ax.imshow(zoomed_input, cmap='rainbow', vmin=vmin, vmax=vmax, extent=[x_min, x_max, y_max, y_min])
    plt.contour(zoomed_input, levels=contour_levels, colors='black', linewidths=0.5, extent=[x_min, x_max, y_min, y_max] )
    ax.add_patch(ellipse1)  # 타원 추가
    ax.invert_yaxis()
    ax.tick_params(axis='both', labelsize=8)
    plt.grid(True)
    ax.set_title(f'Input VF', fontsize=8)

    i, j = 0, 1
    ax = plt.subplot(gs_left_top[i, j])
    ellipse2 = Ellipse(xy=(xc, yc), width=2*r_ellipse[0], height=2*r_ellipse[1]*ax_f, angle=theta, edgecolor='grey', fc='None', lw=0.8)
    zoomed_bsfit = bsfit_vf[y_min:y_max, x_min:x_max]
    cax2 = ax.imshow(zoomed_bsfit, cmap='rainbow', vmin=vmin, vmax=vmax, extent=[x_min, x_max, y_max, y_min])
    plt.contour(zoomed_bsfit, levels=contour_levels, colors='black', linewidths=0.5, extent=[x_min, x_max, y_min, y_max])
    ax.add_patch(ellipse2)  # 타원 추가
    ax.invert_yaxis()
    ax.tick_params(axis='both', labelsize=8)
    ax.set_yticklabels([])
    plt.grid(True)
    ax.set_title(f'BSfit VF', fontsize=8)

    i, j = 0, 2
    ax = plt.subplot(gs_left_top[i, j])
    ellipse3 = Ellipse(xy=(xc, yc), width=2*r_ellipse[0], height=2*r_ellipse[1]*ax_f, angle=theta, edgecolor='grey', fc='None', lw=0.8)
    zoomed_bsfit = bsfit_vf[y_min:y_max, x_min:x_max]
    zoomed_trfit = trfit_vf[y_min:y_max, x_min:x_max]
    cax3 = ax.imshow(zoomed_trfit, cmap='rainbow', vmin=vmin, vmax=vmax, extent=[x_min, x_max, y_max, y_min])
    plt.contour(zoomed_trfit, levels=contour_levels, colors='black', linewidths=0.5, extent=[x_min, x_max, y_min, y_max])
    ax.add_patch(ellipse3)  # 타원 추가
    ax.invert_yaxis()
    ax.tick_params(axis='both', labelsize=8)
    ax.set_yticklabels([])
    plt.grid(True)
    ax.set_title(f'TRfit VF', fontsize=8)

    i, j = 1, 1
    ax = plt.subplot(gs_left_top[i, j])
    ellipse4 = Ellipse(xy=(xc, yc), width=2*r_ellipse[0], height=2*r_ellipse[1]*ax_f, angle=theta, edgecolor='grey', fc='None', lw=0.8)
    cax4 = ax.imshow((zoomed_input - zoomed_bsfit), cmap='rainbow', vmin=vmin_res, vmax=vmax_res, extent=[x_min, x_max, y_max, y_min])
    plt.contour((zoomed_input - zoomed_bsfit), levels=contour_levels_res, colors='black', linewidths=0.5, extent=[x_min, x_max, y_min, y_max])
    ax.add_patch(ellipse4)  # 타원 추가
    ax.invert_yaxis()
    ax.tick_params(axis='both', labelsize=8)
    plt.grid(True)
    ax.set_title(f'res. (Input - BSfit)', fontsize=8)
    ax.set_xlabel('ra [pix]')
    ax.set_ylabel('dec [pix]')

    i, j = 1, 2
    ax = plt.subplot(gs_left_top[i, j])
    ellipse5 = Ellipse(xy=(xc, yc), width=2*r_ellipse[0], height=2*r_ellipse[1]*ax_f, angle=theta, edgecolor='grey', fc='None', lw=0.8)
    cax5 = ax.imshow(zoomed_input - zoomed_trfit, cmap='rainbow', vmin=vmin_res, vmax=vmax_res, extent=[x_min, x_max, y_max, y_min])
    plt.contour((zoomed_input - zoomed_trfit), levels=contour_levels_res, colors='black', linewidths=0.5, extent=[x_min, x_max, y_min, y_max])
    ax.add_patch(ellipse5)  # 타원 추가
    ax.invert_yaxis()
    ax.set_yticklabels([])
    ax.tick_params(axis='both', labelsize=8)
    plt.grid(True)
    ax.set_title(f'res. (Input - TRfit)', fontsize=8)


    gs_left_top_1_0 = gridspec.GridSpecFromSubplotSpec(1, 12, subplot_spec=gs_left_top[1, 0])

    cax6 = plt.subplot(gs_left_top_1_0[0, 0])
    cax6.axis('on')  # Turn off axis for the empty subplot
    ax.tick_params(axis='both', labelsize=8)
    cb1 = plt.colorbar(cax2, cax=cax6, orientation='vertical', format='%0.0f')
    cb1.ax.tick_params(labelsize=8)

    _min = vmin
    _max = vmax
    _mid = (_max + _min) * 0.5
    tick_values1 = [_min, _mid, _max]  # Customize these values
    cb1.set_ticks(tick_values1)

    cax7 = plt.subplot(gs_left_top_1_0[0, 4])
    cax7.axis('on')  # Turn off axis for the empty subplot
    ax.tick_params(axis='both', labelsize=8)
    cb2 = plt.colorbar(cax4, cax=cax7, orientation='vertical', format='%0.0f')
    cb2.ax.tick_params(labelsize=8)

    _min = vmin_res
    _max = vmax_res
    _mid = (_max + _min) * 0.5
    tick_values2 = [_min, _mid, _max]  # Customize these values
    cb2.set_ticks(tick_values2)


    j = 0
    ax = plt.subplot(gs_left_bottom[0, j])
    image_data = np.random.rand(10, 15)
    ax.imshow(image_data, cmap='viridis')
    ax.set_title(f'Major axis')
    ax.set_xlabel('offset position [pix]')
    ax.set_ylabel('$v_{los}$ - vsys [km/s]')

    j = 1
    ax = plt.subplot(gs_left_bottom[0, j])
    image_data = np.random.rand(10, 15)
    ax.imshow(image_data, cmap='viridis')
    ax.set_title(f'Minor axis')
    ax.set_xlabel('offset position [pix]')


    i, j = 0, 0
    ax1 = plt.subplot(gs_right_top[i, j])
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.set_ylim(np.min(df['pa [deg]'])-10, np.max(df['pa [deg]'])+10)
    ax1.errorbar(df['radius [pix]'], df['pa [deg]'], yerr=np.abs(df['pa-e [deg]']))
    ax1.scatter(df['radius [pix]'], df['pa [deg]'])
    ax1.plot(df['radius [pix]'], df['pa [deg]'])
    ax1.set_ylabel('pa [deg]')


    i, j = 0, 1
    ax2 = plt.subplot(gs_right_top[i, j])
    ax2.set_ylim(np.min(df['vsys [km/s]'])-10, np.max(df['vsys [km/s]'])+10)
    ax2.errorbar(df['radius [pix]'], df['vsys [km/s]'], yerr=np.abs(df['vsys-e [km/s]']))
    ax2.scatter(df['radius [pix]'], df['vsys [km/s]'])
    ax2.plot(df['radius [pix]'], df['vsys [km/s]'])
    ax2.set_yticklabels([])
    ax2 = ax2.twinx()
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.set_ylim(np.min(df['vsys [km/s]'])-10, np.max(df['vsys [km/s]'])+10)
    ax2.set_ylabel('vsys [km/s]', rotation=90, labelpad=15)

    i, j = 1, 0
    ax3 = plt.subplot(gs_right_top[i, j])
    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    ax3.yaxis.set_minor_locator(AutoMinorLocator())
    ax3.set_ylim(np.min(df['incl [deg]'])-10, np.max(df['incl [deg]'])+10)
    ax3.errorbar(df['radius [pix]'], df['incl [deg]'], yerr=np.abs(df['incl-e [deg]']))
    ax3.scatter(df['radius [pix]'], df['incl [deg]'])
    ax3.plot(df['radius [pix]'], df['incl [deg]'])
    ax3.set_xlabel('radius [pix]')
    ax3.set_ylabel('i [deg]')

    i, j = 1, 1
    ax4 = plt.subplot(gs_right_top[i, j])
    ax4.set_ylim(np.min(df['vrad [km/s]'])-10, np.max(df['vrad [km/s]'])+10)
    ax4.errorbar(df['radius [pix]'], df['vrad [km/s]'], yerr=np.abs(df['vrad-e [km/s]']))
    ax4.scatter(df['radius [pix]'], df['vrad [km/s]'])
    ax4.plot(df['radius [pix]'], df['vrad [km/s]'])
    ax4.set_yticklabels([])
    ax4.set_xlabel('radius [pix]')
    ax4 = ax4.twinx()
    ax4.xaxis.set_minor_locator(AutoMinorLocator())
    ax4.yaxis.set_minor_locator(AutoMinorLocator())
    ax4.set_ylim(np.min(df['vrad [km/s]'])-10, np.max(df['vrad [km/s]'])+10)
    ax4.set_ylabel('vrad [km/s]', rotation=90, labelpad=15)




    vrot_min = np.min(df['vrot_b [km/s]'])
    vrot_max = np.max(df['vrot_b [km/s]'])
    _gab = int((vrot_max - vrot_min)*0.2)
    vrot_min = -_gab*0.5
    vrot_max += 2*_gab

    r_min = np.min(df['radius [pix]'])
    r_max = np.max(df['radius [pix]'])
    _gab = int((r_max - r_min)*0.2)
    r_min = -_gab
    r_max += _gab

    ax = plt.subplot(gs_right_bottom[0, 0])
    ax.set_xlim(r_min, r_max)
    ax.set_ylim(vrot_min, vrot_max)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.errorbar(df['radius [pix]'], df['vrot_bs [km/s]'], yerr=np.abs(df['vrot_bs-e [km/s]']), color='black')
    ax.plot(df['radius [pix]'], df['vrot_bs [km/s]'], color='green', marker='s', fillstyle='none')

    ax.errorbar(df['radius [pix]'], df['vrot_b [km/s]'], yerr=np.abs(df['vrot_b-e [km/s]']), color='black')
    ax.scatter(df['radius [pix]'], df['vrot_b [km/s]'], color='black')
    ax.plot(df['radius [pix]'], df['vrot_b [km/s]'], color='black')

    ax.errorbar(df['radius [pix]'], df['vrot_a [km/s]'], yerr=np.abs(df['vrot_a-e [km/s]']), color='blue')
    ax.plot(df['radius [pix]'], df['vrot_a [km/s]'],  linestyle='-.', color='blue', marker='v', fillstyle='none')

    ax.errorbar(df['radius [pix]'], df['vrot_r [km/s]'], yerr=np.abs(df['vrot_r-e [km/s]']), color='red')
    ax.plot(df['radius [pix]'], df['vrot_r [km/s]'],  linestyle='-.', color='red', marker='^', fillstyle='none')

    ax.set_xlabel('radius [pix]')
    ax.set_ylabel('vrot [km/s]')

    plt.tight_layout()

    plt.savefig(output_filename, dpi=300, bbox_inches='tight')


def print_2dbat(_params, _2dbat_run_i, _2dbat_trfit_txt):


    _dir_2dbat_PI_output = _params['wdir'] + '/' + _params['_2dbatdir'] + ".%d" % _2dbat_run_i
    column_names = [
        'radius [pix]', 'sigma [km/s]', 'sigma-e [km/s]', 'xpos [pix]', 'xpos-e [pix]',
        'ypos [pix]', 'ypos-e [pix]', 'vsys [km/s]', 'vsys-e [km/s]', 'pa [deg]',
        'pa-e [deg]', 'incl [deg]', 'incl-e [deg]', 'vrot [km/s]', 'vrot-e [km/s]', 'vrad [km/s]', 'vrad-e [km/s]',
        'npoints_total [pixs]', 'npoints_available [pixs]'
    ]

    df = pd.read_csv(
        _dir_2dbat_PI_output + '/' + '%s' % _2dbat_trfit_txt,
        sep='\s+',
        comment='!',
        header=None,
        names=column_names,
        skiprows=1  # skip the first raw 
    )
    print(df)
    print("")



def write_2dbat_t1(_2dbat_trfit_final_b, nrings_reliable_b, _params, _2dbat_run_i, output_txt):

    _dir_2dbat_PI_output = _params['wdir'] + '/' + _params['_2dbatdir'] + ".%d" % _2dbat_run_i
    if not os.path.exists("%s" % _dir_2dbat_PI_output):
        make_dirs("%s" % _dir_2dbat_PI_output)

    with open(f'{_dir_2dbat_PI_output}/%s' % output_txt, 'w') as file:
        header = "! radius [pix]   sigma [km/s]     sigma-e [km/s]   xpos [pix]       xpos-e [pix]     ypos [pix]       ypos-e [pix]     vsys [km/s]      vsys-e [km/s]    pa [deg]         pa-e [deg]       incl [deg]       incl-e [deg]     \
          vrot [km/s]      vrot-e [km/s]    vrad [km/s]     vrad-e [km/s]      npoints_total [pixs]     npoints_available [pixs]\n"
        file.write(header)

        for n in range(nrings_reliable_b):
            line = "{:<16.2f} {:<16.2f} {:<16.2f} {:<16.2f} {:<16.2f} {:<16.2f} {:<16.2f} {:<16.2f} {:<16.2f} {:<16.2f} {:<16.2f} {:<16.2f} {:<16.2f} {:<16.2f} {:<16.2f} {:<16.2f} {:<16.2f} {:<16.0f} {:<16.0f}\n".format(
                _2dbat_trfit_final_b[0][n], \
                _2dbat_trfit_final_b[1][n], _2dbat_trfit_final_b[9][n], \
                _2dbat_trfit_final_b[2][n], _2dbat_trfit_final_b[10][n], \
                _2dbat_trfit_final_b[3][n], _2dbat_trfit_final_b[11][n], \
                _2dbat_trfit_final_b[4][n], _2dbat_trfit_final_b[12][n], \
                _2dbat_trfit_final_b[5][n], _2dbat_trfit_final_b[13][n], \

                _2dbat_trfit_final_b[6][n], _2dbat_trfit_final_b[14][n], \

                _2dbat_trfit_final_b[7][n], _2dbat_trfit_final_b[15][n], \
                _2dbat_trfit_final_b[8][n], _2dbat_trfit_final_b[16][n], \
                _2dbat_trfit_final_b[17][n], _2dbat_trfit_final_b[18][n])
            file.write(line)


def write_2dbat_t2(_vrot_bs, _vrot_bs_e, _2dbat_trfit_final_b, nrings_reliable_b, _2dbat_trfit_final_a, nrings_reliable_a, _2dbat_trfit_final_r, nrings_reliable_r, _params, _2dbat_run_i, output_txt):

    _dir_2dbat_PI_output = _params['wdir'] + '/' + _params['_2dbatdir'] + ".%d" % _2dbat_run_i
    if not os.path.exists("%s" % _dir_2dbat_PI_output):
        make_dirs("%s" % _dir_2dbat_PI_output)

    _nrings_max = max(nrings_reliable_b, nrings_reliable_a, nrings_reliable_r)

    if _nrings_max == nrings_reliable_b:
        _rings = _2dbat_trfit_final_b[0][:]
    elif _nrings_max == nrings_reliable_r:
        _rings = _2dbat_trfit_final_r[0][:]
    elif _nrings_max == nrings_reliable_a:
        _rings = _2dbat_trfit_final_a[0][:]


    with open(f'{_dir_2dbat_PI_output}/%s' % output_txt, 'w') as file:
        header = "! radius [pix]   sigma [km/s]     sigma-e [km/s]   xpos [pix]       xpos-e [pix]     ypos [pix]       ypos-e [pix]     vsys [km/s]      vsys-e [km/s]    pa [deg]         pa-e [deg]       incl [deg]       incl-e [deg]     \
               vrot_bs [km/s]    vrot_bs-e [km/s]     vrot_b [km/s]      vrot_b-e [km/s]    vrot_a [km/s]      vrot_a-e [km/s]     vrot_r [km/s]      vrot_r-e [km/s]      vrad [km/s]     vrad-e [km/s]      npoints_total-b [pixs]     npoints_available-b [pixs]    npoints_total-a [pixs]     npoints_available-a [pixs]     npoints_total-r [pixs]     npoints_available-r [pixs]\n"
        file.write(header)

        for n in range(1, _nrings_max):
            if n >= nrings_reliable_a:
                _npoints_total_a = 0
                _npoints_a = 0
                na = nrings_reliable_a - 1 
                _2dbat_trfit_final_a[7][na] = 999
                _2dbat_trfit_final_a[15][na] = 999
            else:
                _npoints_total_a = _2dbat_trfit_final_a[17][n]
                _npoints_a = _2dbat_trfit_final_a[18][n]
                na = n

            if n >= nrings_reliable_r:
                _npoints_total_r = 0
                _npoints_r = 0
                nr = nrings_reliable_r - 1 
                _2dbat_trfit_final_r[7][nr] = 999
                _2dbat_trfit_final_r[15][nr] = 999
            else:
                _npoints_total_r = _2dbat_trfit_final_r[17][n]
                _npoints_r = _2dbat_trfit_final_r[18][n]
                nr = n

            if n >= nrings_reliable_b:
                _npoints_total_b = 0
                _npoints_b = 0
                nb = nrings_reliable_b - 1 
                _2dbat_trfit_final_b[7][nb] = 999
                _2dbat_trfit_final_b[15][nb] = 999
            else:
                _npoints_total_b = _2dbat_trfit_final_b[17][n]
                _npoints_b = _2dbat_trfit_final_b[18][n]
                nb = n

            line = "{:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.0f} {:<15.0f} {:<15.0f} {:<15.0f} {:<15.0f} {:<15.0f}\n".format(
                _rings[n], \
                _2dbat_trfit_final_b[1][nb], _2dbat_trfit_final_b[9][nb], \
                _2dbat_trfit_final_b[2][nb], _2dbat_trfit_final_b[10][nb], \
                _2dbat_trfit_final_b[3][nb], _2dbat_trfit_final_b[11][nb], \
                _2dbat_trfit_final_b[4][nb], _2dbat_trfit_final_b[12][nb], \
                _2dbat_trfit_final_b[5][nb], _2dbat_trfit_final_b[13][nb], \
                _2dbat_trfit_final_b[6][nb], _2dbat_trfit_final_b[14][nb], \
                _vrot_bs[nb], _vrot_bs_e[nb], \
                _2dbat_trfit_final_b[7][nb], _2dbat_trfit_final_b[15][nb], \
                _2dbat_trfit_final_a[7][na], _2dbat_trfit_final_a[15][na], \
                _2dbat_trfit_final_r[7][nr], _2dbat_trfit_final_r[15][nr], \
                _2dbat_trfit_final_b[8][nb], _2dbat_trfit_final_b[16][nb], \
                _npoints_total_b, _npoints_b, \
                _npoints_total_a, _npoints_a, _npoints_total_r, _npoints_r)
            file.write(line)
            


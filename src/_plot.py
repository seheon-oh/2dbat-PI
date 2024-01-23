#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#|-----------------------------------------|
#| _plot.py
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
import pandas as pd
import numpy as np
import sys

from astropy.io import fits





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


def plot_2dbat(_2dbat_trfit_txt, input_vf, bsfit_vf, trfit_vf, output_filename, _params, _2dbat_run_i):

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
        _2dbat_trfit_txt,
        delim_whitespace=True,
        comment='!',
        header=None,
        names=column_names,
        skiprows=1  # skip the first raw 
    )


    print(df)
    _dir_2dbat_PI_output = _params['wdir'] + '/' + _params['_2dbatdir'] + ".%d" % _2dbat_run_i
    output_txt_filename = '2dbat_trfit_final.txt'
    with open(_dir_2dbat_PI_output + '/' + output_txt_filename, 'w') as file:
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

    vmin = np.nanmin(bsfit_vf)
    vmax = np.nanmax(bsfit_vf)
    contour_levels = np.linspace(vmin, vmax, 10)

    vmin_res = -20
    vmax_res = 20
    contour_levels_res = np.linspace(vmin_res, vmax_res, 5)


    i, j = 0, 0
    ax = plt.subplot(gs_left_top[i, j])
    cax = ax.imshow(input_vf, cmap='rainbow', vmin=vmin, vmax=vmax)
    plt.contour(input_vf, levels=contour_levels, colors='black', linewidths=0.5)
    ax.set_title(f'Input VF')

    i, j = 0, 1
    ax = plt.subplot(gs_left_top[i, j])
    cax2 = ax.imshow(bsfit_vf, cmap='rainbow', vmin=vmin, vmax=vmax)
    plt.contour(bsfit_vf, levels=contour_levels, colors='black', linewidths=0.5)
    ax.set_title(f'BSfit VF')

    i, j = 0, 2
    ax = plt.subplot(gs_left_top[i, j])
    cax3 = ax.imshow(trfit_vf, cmap='rainbow', vmin=vmin, vmax=vmax)
    plt.contour(trfit_vf, levels=contour_levels, colors='black', linewidths=0.5)
    ax.set_title(f'TRfit VF')

    i, j = 1, 1
    ax = plt.subplot(gs_left_top[i, j])
    cax4 = ax.imshow((input_vf - bsfit_vf), cmap='rainbow', vmin=vmin_res, vmax=vmax_res)
    plt.contour((input_vf - bsfit_vf), levels=contour_levels_res, colors='black', linewidths=0.5)
    ax.set_title(f'res. (Input - BSfit)')
    ax.set_xlabel('ra [pix]')
    ax.set_ylabel('dec [pix]')

    i, j = 1, 2
    ax = plt.subplot(gs_left_top[i, j])
    cax5 = ax.imshow(input_vf - trfit_vf, cmap='rainbow', vmin=vmin_res, vmax=vmax_res)
    plt.contour((input_vf - trfit_vf), levels=contour_levels_res, colors='black', linewidths=0.5)
    ax.set_title(f'res. (Input - TRfit)')


    gs_left_top_1_0 = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs_left_top[1, 0])

    cax6 = plt.subplot(gs_left_top_1_0[0, 0])
    cax6.axis('on')  # Turn off axis for the empty subplot
    cb1 = plt.colorbar(cax2, cax=cax6, orientation='vertical', format='%0.0f')

    _min = vmin
    _max = vmax
    _mid = (_max + _min) * 0.5
    tick_values1 = [_min, _mid, _max]  # Customize these values
    cb1.set_ticks(tick_values1)

    cax7 = plt.subplot(gs_left_top_1_0[0, 3])
    cax7.axis('on')  # Turn off axis for the empty subplot
    cb2 = plt.colorbar(cax4, cax=cax7, orientation='vertical', format='%0.0f')

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



    ax = plt.subplot(gs_right_bottom[0, 0])
    ax.set_ylim(-5, np.max(df['vrot_b [km/s]'])+20)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.errorbar(df['radius [pix]'], df['vrot_bs [km/s]'], yerr=np.abs(df['vrot_bs-e [km/s]']), color='black')
    ax.plot(df['radius [pix]'], df['vrot_bs [km/s]'], color='yellow', marker='s', fillstyle='none')

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

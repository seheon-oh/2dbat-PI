
#|-----------------------------------------|
# plotting
import matplotlib
from mpl_toolkits.mplot3d import Axes3D


from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import sys

from astropy.io import fits

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# re-defining plotting defaults


#-- END OF SUB-ROUTINE____________________________________________________________#


def align_data_to_header(df, column_names):
    # DataFrame을 문자열로 변환합니다. 이 때, 각 열의 최대 길이를 고려하여 정렬합니다.
    formatted_data = df.to_string(header=False, index=False)
    return formatted_data


def format_header_string(column_names):
    # put lien number and comment
    numbered_header = [f'! {i+1}: {name}' for i, name in enumerate(column_names)]
    # make individual lines 
    header_string = '\n'.join(numbered_header)
    #header_string += '\n! ' + '\t'.join(column_names)
    return header_string

def add_comment_to_first_line(data_string):
    # 데이터 문자열을 줄 단위로 분리합니다.
    lines = data_string.split('\n')
    # 첫 번째 줄에 주석을 추가하되, '!'와 헤더 키 사이의 공백을 없앱니다.
    lines[0] = '!' + lines[0]
    # 수정된 줄들을 다시 하나의 문자열로 합칩니다.
    return '\n'.join(lines)


def plot_2dbat(_2dbat_trfit_txt, input_vf, bsfit_vf, trfit_vf, output_filename, _params, _2dbat_run_i):

    # 열 이름을 수동으로 지정합니다.
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
    # output directory 
    _dir_2dbat_PI_output = _params['wdir'] + '/' + _params['_2dbatdir'] + ".%d" % _2dbat_run_i
    # write data in panda 
    output_txt_filename = '2dbat_trfit_final.txt'
    with open(_dir_2dbat_PI_output + '/' + output_txt_filename, 'w') as file:
        # header: lined keys
        header_string = format_header_string(column_names)
        file.write(header_string + '\n')

        # 데이터를 오른쪽 정렬하여 문자열로 변환합니다.
        aligned_data = df.to_string(index=False, justify='right')
        # 첫 번째 줄에 주석을 추가합니다.
        aligned_data_with_comment = add_comment_to_first_line(aligned_data)
        file.write(aligned_data_with_comment)

        # data
        #aligned_data = df.to_string(index=False, justify='right')
        #file.write(aligned_data)

    # A4 Landscape 크기로 설정합니다.
    fig = plt.figure(figsize=(11.69, 8.27))

    # 전체 그래프 영역을 왼쪽과 오른쪽으로 2등분하고, 하단 패널의 높이를 동일하게 맞춥니다.
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[3, 2])

    # 왼쪽 부분을 두 개의 그리드로 나눕니다: 상단 (2x3) 및 하단 (1x2).
    gs_left_top = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[0, 0], hspace=0.3)
    gs_left_bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 0])

    # 오른쪽 부분을 두 개의 그리드로 나눕니다: 상단 (2x2) 및 하단 (1x1).
    gs_right_top = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 1])
    gs_right_bottom = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[1, 1], hspace=1.5)

    vmin = np.nanmin(bsfit_vf)
    vmax = np.nanmax(bsfit_vf)
    contour_levels = np.linspace(vmin, vmax, 10)

    vmin_res = -20
    vmax_res = 20
    contour_levels_res = np.linspace(vmin_res, vmax_res, 5)


    # input vf
    i, j = 0, 0
    ax = plt.subplot(gs_left_top[i, j])
    cax = ax.imshow(input_vf, cmap='rainbow', vmin=vmin, vmax=vmax)
    plt.contour(input_vf, levels=contour_levels, colors='black', linewidths=0.5)
    ax.set_title(f'Input VF')
    #plt.colorbar(cax, ax=ax)  # colorbar 추가

    # bsfit vf
    i, j = 0, 1
    ax = plt.subplot(gs_left_top[i, j])
    cax2 = ax.imshow(bsfit_vf, cmap='rainbow', vmin=vmin, vmax=vmax)
    plt.contour(bsfit_vf, levels=contour_levels, colors='black', linewidths=0.5)
    ax.set_title(f'BSfit VF')

    # trfit vf
    i, j = 0, 2
    ax = plt.subplot(gs_left_top[i, j])
    cax3 = ax.imshow(trfit_vf, cmap='rainbow', vmin=vmin, vmax=vmax)
    plt.contour(trfit_vf, levels=contour_levels, colors='black', linewidths=0.5)
    ax.set_title(f'TRfit VF')

    # res vf (Input - BSfit)
    i, j = 1, 1
    ax = plt.subplot(gs_left_top[i, j])
    cax4 = ax.imshow((input_vf - bsfit_vf), cmap='rainbow', vmin=vmin_res, vmax=vmax_res)
    plt.contour((input_vf - bsfit_vf), levels=contour_levels_res, colors='black', linewidths=0.5)
    ax.set_title(f'res. (Input - BSfit)')
    ax.set_xlabel('ra [pix]')
    ax.set_ylabel('dec [pix]')

    # res vf (Input - TRfit)
    i, j = 1, 2
    ax = plt.subplot(gs_left_top[i, j])
    cax5 = ax.imshow(input_vf - trfit_vf, cmap='rainbow', vmin=vmin_res, vmax=vmax_res)
    plt.contour((input_vf - trfit_vf), levels=contour_levels_res, colors='black', linewidths=0.5)
    ax.set_title(f'res. (Input - TRfit)')


    # Create a subplot spec for the (1, 0) space, divided into 6 subplots
    gs_left_top_1_0 = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs_left_top[1, 0])

    # Create the first colorbar for the top left subplot
    cax6 = plt.subplot(gs_left_top_1_0[0, 0])
    cax6.axis('on')  # Turn off axis for the empty subplot
    cb1 = plt.colorbar(cax2, cax=cax6, orientation='vertical', format='%0.0f')
    #cb1.set_label('Colorbar 1')

    # Customize the ticks and labels for Colorbar 1
    _min = vmin
    _max = vmax
    _mid = (_max + _min) * 0.5
    tick_values1 = [_min, _mid, _max]  # Customize these values
    #tick_labels1 = ['Min', 'Mid', 'Max']  # Customize these labels
    cb1.set_ticks(tick_values1)
    #cb1.set_ticklabels(tick_labels1)

    # Create the second colorbar for the top right subplot
    cax7 = plt.subplot(gs_left_top_1_0[0, 3])
    cax7.axis('on')  # Turn off axis for the empty subplot
    cb2 = plt.colorbar(cax4, cax=cax7, orientation='vertical', format='%0.0f')
    #cb2.set_label('Colorbar 2')

    # Customize the ticks and labels for Colorbar 2
    _min = vmin_res
    _max = vmax_res
    _mid = (_max + _min) * 0.5
    tick_values2 = [_min, _mid, _max]  # Customize these values
    #tick_labels2 = ['Min', 'Mid', 'Max']  # Customize these labels
    cb2.set_ticks(tick_values2)
    #cb2.set_ticklabels(tick_labels2)


    # major axis
    j = 0
    ax = plt.subplot(gs_left_bottom[0, j])
    image_data = np.random.rand(10, 15)
    ax.imshow(image_data, cmap='viridis')
    ax.set_title(f'Major axis')
    ax.set_xlabel('offset position [pix]')
    ax.set_ylabel('$v_{los}$ - vsys [km/s]')

    # minor axis
    j = 1
    ax = plt.subplot(gs_left_bottom[0, j])
    image_data = np.random.rand(10, 15)
    ax.imshow(image_data, cmap='viridis')
    ax.set_title(f'Minor axis')
    ax.set_xlabel('offset position [pix]')


    #ax1.set_title(f'Graph Top {i * 2 + j + 1}')
    i, j = 0, 0
    ax1 = plt.subplot(gs_right_top[i, j])
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.set_ylim(np.min(df['pa [deg]'])-10, np.max(df['pa [deg]'])+10)
    ax1.errorbar(df['radius [pix]'], df['pa [deg]'], yerr=np.abs(df['pa-e [deg]']))
    ax1.scatter(df['radius [pix]'], df['pa [deg]'])
    ax1.plot(df['radius [pix]'], df['pa [deg]'])
    ax1.set_ylabel('pa [deg]')
    # sub-tick


    #ax2.set_title(f'Graph Top {0 * i + j + 1}')
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

    #ax3.set_title(f'Graph Top {0 * i + j + 1}')
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

    #ax4.set_title(f'Graph Top {0 * i + j + 1}')
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



    #ax.set_title(f'Graph Bottom')
    ax = plt.subplot(gs_right_bottom[0, 0])
    ax.set_ylim(-5, np.max(df['vrot_b [km/s]'])+20)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.errorbar(df['radius [pix]'], df['vrot_bs [km/s]'], yerr=np.abs(df['vrot_bs-e [km/s]']), color='black')
    #ax.scatter(df['radius [pix]'], df['vrot_bs [km/s]'], color='black')
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

    # 패널 간의 여백을 조정합니다.
    plt.tight_layout()

    #plt.show()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
"""Compare the different methods to create phase maps."""


import time
import pdb
import traceback
import sys
import os

import numpy as np
from numpy import pi

import shelve

import pyramid.magcreator as mc
import pyramid.projector as pj
import pyramid.phasemapper as pm
import pyramid.analytic as an
from pyramid.magdata import MagData
from pyramid.phasemap import PhaseMap

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.cm import RdBu


def run():

    print '\nACCESS SHELVE'
    # Create / Open databank:
    directory = '../../output/paper 1'
    if not os.path.exists(directory):
        os.makedirs(directory)
    data_shelve = shelve.open(directory + '/paper_1_shelve')

    ###############################################################################################
    print 'CH5-2 PHASE DIFFERENCES FOURIER SPACE'

    # Input parameters:
    res = 1.0  # in nm
    phi = pi/2
    dim = (16, 128, 128)  # in px (z, y, x)
    center = (dim[0]/2-0.5, dim[1]/2.-0.5, dim[2]/2.-0.5)  # in px (z, y, x) index starts with 0!
    radius = dim[1]/4  # in px
    height = dim[0]/2  # in px

    key = 'ch5-1-phase_diff_mag_dist'
    if key in data_shelve:
        print '--LOAD MAGNETIC DISTRIBUTIONS'
        (mag_data_disc, mag_data_vort) = data_shelve[key]
    else:
        print '--CREATE MAGNETIC DISTRIBUTIONS'
        # Create magnetic shape (4 times the size):
        res_big = res / 2
        dim_big = (dim[0]*2, dim[1]*2, dim[2]*2)
        center_big = (dim_big[0]/2-0.5, dim_big[1]/2.-0.5, dim_big[2]/2.-0.5)
        radius_big = dim_big[1]/4  # in px
        height_big = dim_big[0]/2  # in px
        mag_shape = mc.Shapes.disc(dim_big, center_big, radius_big, height_big)
        # Create MagData (4 times the size):
        mag_data_disc = MagData(res_big, mc.create_mag_dist(mag_shape, phi))
        mag_data_vort = MagData(res_big, mc.create_mag_dist_vortex(mag_shape, center_big))
        # Scale mag_data, resolution and dimensions:
        mag_data_disc.scale_down()
        mag_data_vort.scale_down()
        print '--SAVE MAGNETIC DISTRIBUTIONS'
        # Shelve magnetic distributions:
        data_shelve[key] = (mag_data_disc, mag_data_vort)

    print '--PLOT/SAVE PHASE DIFFERENCES'
    # Create projections along z-axis:
    projection_disc = pj.simple_axis_projection(mag_data_disc)
    projection_vort = pj.simple_axis_projection(mag_data_vort)
    # Get analytic solutions:
    phase_ana_disc = an.phase_mag_disc(dim, res, phi, center, radius, height)
    phase_ana_vort = an.phase_mag_vortex(dim, res, center, radius, height)
    # Create norm for the plots:
    bounds = np.array([-100, -50, -25, -5, 0, 5, 25, 50, 100])
    norm = BoundaryNorm(bounds, RdBu.N)
    # Disc:
    phase_num_disc = pm.phase_mag_fourier(res, projection_disc, padding=0)
    phase_diff_disc = PhaseMap(res, (phase_num_disc-phase_ana_disc)*1E3)  # in mrad -> *1000)
    RMS_disc = np.sqrt(np.mean(phase_diff_disc.phase**2))
    phase_diff_disc.display('Deviation (homog. magn. disc), RMS = {:3.2f} mrad'.format(RMS_disc),
                            labels=('x-axis [nm]', 'y-axis [nm]', 
                                    '$\Delta$phase [mrad] (padding = 0)'),
                            limit=np.max(bounds), norm=norm)
    plt.savefig(directory + '/ch5-2-disc_phase_diff_no_padding.png', bbox_inches='tight')
    # Vortex:
    phase_num_vort = pm.phase_mag_fourier(res, projection_vort, padding=0)
    phase_diff_vort = PhaseMap(res, (phase_num_vort-phase_ana_vort)*1E3)  # in mrad -> *1000
    RMS_vort = np.sqrt(np.mean(phase_diff_vort.phase**2))
    phase_diff_vort.display('Deviation (vortex state disc), RMS = {:3.2f} mrad'.format(RMS_vort),
                            labels=('x-axis [nm]', 'y-axis [nm]', 
                                    '$\Delta$phase [mrad] (padding = 0)'),
                            limit=np.max(bounds), norm=norm)
    plt.savefig(directory + '/ch5-2-vortex_phase_diff_no_padding.png', bbox_inches='tight')

    # Create norm for the plots:
    bounds = np.array([-3, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 3])
    norm = BoundaryNorm(bounds, RdBu.N)
    # Disc:
    phase_num_disc = pm.phase_mag_fourier(res, projection_disc, padding=10)
    phase_diff_disc = PhaseMap(res, (phase_num_disc-phase_ana_disc)*1E3)  # in mrad -> *1000)
    RMS_disc = np.sqrt(np.mean(phase_diff_disc.phase**2))
    phase_diff_disc.display('Deviation (homog. magn. disc), RMS = {:3.2f} mrad'.format(RMS_disc),
                            labels=('x-axis [nm]', 'y-axis [nm]', 
                                    '$\Delta$phase [mrad] (padding = 10)'),
                            limit=np.max(bounds), norm=norm)
    plt.savefig(directory + '/ch5-2-disc_phase_diff_padding_10.png', bbox_inches='tight')
    # Vortex:
    phase_num_vort = pm.phase_mag_fourier(res, projection_vort, padding=10)
    phase_diff_vort = PhaseMap(res, (phase_num_vort-phase_ana_vort)*1E3)  # in mrad -> *1000
    RMS_vort = np.sqrt(np.mean(phase_diff_vort.phase**2))
    phase_diff_vort.display('Deviation (vortex state disc), RMS = {:3.2f} mrad'.format(RMS_vort),
                            labels=('x-axis [nm]', 'y-axis [nm]', 
                                    '$\Delta$phase [mrad] (padding = 10)'),
                            limit=np.max(bounds), norm=norm)
    plt.savefig(directory + '/ch5-2-vortex_phase_diff_padding_10.png', bbox_inches='tight')

    ###############################################################################################
    print 'CH5-2 FOURIER PADDING VARIATION'

    # Input parameters:
    res = 1.0  # in nm
    phi = pi/2
    dim = (16, 128, 128)  # in px (z, y, x)
    center = (dim[0]/2-0.5, dim[1]/2.-0.5, dim[2]/2.-0.5)  # in px (z, y, x) index starts with 0!
    radius = dim[1]/4  # in px
    height = dim[0]/2  # in px
    
    key = 'ch5-2-fourier_padding_mag_dist'
    if key in data_shelve:
        print '--LOAD MAGNETIC DISTRIBUTIONS'
        (mag_data_disc, mag_data_vort) = data_shelve[key]
    else:
        print '--CREATE MAGNETIC DISTRIBUTIONS'
        # Create magnetic shape (4 times the size):
        res_big = res / 2
        dim_big = (dim[0]*2, dim[1]*2, dim[2]*2)
        center_big = (dim_big[0]/2-0.5, dim_big[1]/2.-0.5, dim_big[2]/2.-0.5)
        radius_big = dim_big[1]/4  # in px
        height_big = dim_big[0]/2  # in px
        mag_shape = mc.Shapes.disc(dim_big, center_big, radius_big, height_big)
        # Create MagData (4 times the size):
        mag_data_disc = MagData(res_big, mc.create_mag_dist(mag_shape, phi))
        mag_data_vort = MagData(res_big, mc.create_mag_dist_vortex(mag_shape, center_big))
        # Scale mag_data, resolution and dimensions:
        mag_data_disc.scale_down()
        mag_data_vort.scale_down()
        print '--SAVE MAGNETIC DISTRIBUTIONS'
        # Shelve magnetic distributions:
        data_shelve[key] = (mag_data_disc, mag_data_vort)

    # Create projections along z-axis:
    projection_disc = pj.simple_axis_projection(mag_data_disc)
    projection_vort = pj.simple_axis_projection(mag_data_vort)
    # Get analytic solutions:
    phase_ana_disc = an.phase_mag_disc(dim, res, phi, center, radius, height)
    phase_ana_vort = an.phase_mag_vortex(dim, res, center, radius, height)

    # List of applied paddings:
    padding_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    print '--LOAD/CREATE PADDING SERIES OF HOMOG. MAGN. DISC'
    data_disc = np.zeros((3, len(padding_list)))
    data_disc[0, :] = padding_list
    for (i, padding) in enumerate(padding_list):
        key = ', '.join(['Padding->RMS|duration', 'Fourier', 'padding={}'.format(padding_list[i]),
                        'res={}'.format(res), 'dim={}'.format(dim), 'phi={}'.format(phi), 'disc'])
        if key in data_shelve:
            data_disc[:, i] = data_shelve[key]
        else:
            print '----calculate and save padding =', padding_list[i]
            start_time = time.time()
            phase_num_disc = pm.phase_mag_fourier(res, projection_disc, padding=padding_list[i])
            data_disc[2, i] = time.time() - start_time
            phase_diff = (phase_num_disc-phase_ana_disc) * 1E3  # in mrad -> *1000)
            phase_map_diff = PhaseMap(res, phase_diff)
            phase_map_diff.display(labels=('x-axis [nm]', 'y-axis [nm]', 'phase [mrad]'))
            data_disc[1, i] = np.sqrt(np.mean(phase_diff**2))
            data_shelve[key] = data_disc[:, i]

    print '--PLOT/SAVE PADDING SERIES OF HOMOG. MAGN. DISC'
    # Plot RMS against padding:
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.axhline(y=0.18, linestyle='--', color='g', label='RMS [mrad] (real space)')
    axis.plot(data_disc[0], data_disc[1], 'go-', label='RMS [mrad] (Fourier space)')
    axis.set_title('Variation of the Padding (homog. magn. disc)', fontsize=18)
    axis.set_xlabel('padding', fontsize=15)
    axis.set_ylabel('RMS [mrad]', fontsize=15)
    axis.set_xlim(-0.5, 16.5)
    axis.set_ylim(-5, 45)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=10, integer= True))
    axis.legend()
    plt.tick_params(axis='both', which='major', labelsize=14)
    # Plot zoom inset:
    ins_axis = plt.axes([0.3, 0.3, 0.55, 0.4])
    ins_axis.axhline(y=0.18, linestyle='--', color='g')
    ins_axis.plot(data_disc[0], data_disc[1], 'go-')
    ins_axis.set_yscale('log')
    ins_axis.set_xlim(5.5, 16.5)
    ins_axis.set_ylim(0.1, 1.1)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig(directory + '/ch5-2-disc_padding_RMS.png', bbox_inches='tight')
    # Plot duration against padding:
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(data_disc[0], data_disc[2], 'bo-')
    axis.set_title('Variation of the Padding (homog. magn. disc)', fontsize=18)
    axis.set_xlabel('padding', fontsize=15)
    axis.set_ylabel('duration [s]', fontsize=15)
    axis.set_xlim(-0.5, 16.5)
    axis.set_ylim(-0.05, 1.5)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=10, integer= True))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=10))
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig(directory + '/ch5-2-disc_padding_duration.png', bbox_inches='tight')


    print '--LOAD/CREATE PADDING SERIES OF VORTEX STATE DISC'
    data_vort = np.zeros((3, len(padding_list)))
    data_vort[0, :] = padding_list
    for (i, padding) in enumerate(padding_list):
        key = ', '.join(['Padding->RMS|duration', 'Fourier', 'padding={}'.format(padding_list[i]),
                        'res={}'.format(res), 'dim={}'.format(dim), 'phi={}'.format(phi), 'vort'])
        if key in data_shelve:
            data_vort[:, i] = data_shelve[key]
        else:
            print '----calculate and save padding =', padding_list[i]
            start_time = time.time()
            phase_num_vort = pm.phase_mag_fourier(res, projection_vort, padding=padding_list[i])
            data_vort[2, i] = time.time() - start_time
            phase_diff = (phase_num_vort-phase_ana_vort) * 1E3  # in mrad -> *1000)
            phase_map_diff = PhaseMap(res, phase_diff)
            phase_map_diff.display(labels=('x-axis [nm]', 'y-axis [nm]', 'phase [mrad]'))
            data_vort[1, i] = np.sqrt(np.mean(phase_diff**2))
            data_shelve[key] = data_vort[:, i]

    print '--PLOT/SAVE PADDING SERIES OF VORTEX STATE DISC'
    # Plot RMS against padding:
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.axhline(y=0.22, linestyle='--', color='g', label='RMS [mrad] (real space)')
    axis.plot(data_vort[0], data_vort[1], 'go-', label='RMS [mrad] (Fourier space)')
    axis.set_title('Variation of the Padding (vortex state disc)', fontsize=18)
    axis.set_xlabel('padding', fontsize=15)
    axis.set_ylabel('RMS [mrad]', fontsize=15)
    axis.set_xlim(-0.5, 16.5)
    axis.set_ylim(-5, 45)
    plt.tick_params(axis='both', which='major', labelsize=14)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=10, integer= True))
    axis.legend()
    # Plot zoom inset:
    ins_axis = plt.axes([0.3, 0.3, 0.55, 0.4])
    ins_axis.axhline(y=0.22, linestyle='--', color='g')
    ins_axis.plot(data_vort[0], data_vort[1], 'go-')
    ins_axis.set_yscale('log')
    ins_axis.set_xlim(5.5, 16.5)
    ins_axis.set_ylim(0.1, 1.1)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig(directory + '/ch5-2-vortex_padding_RMS.png', bbox_inches='tight')
    # Plot duration against padding:
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(data_vort[0], data_vort[2], 'bo-')
    axis.set_title('Variation of the Padding (vortex state disc)', fontsize=18)
    axis.set_xlabel('padding', fontsize=15)
    axis.set_ylabel('duration [s]', fontsize=15)
    axis.set_xlim(-0.5, 16.5)
    axis.set_ylim(-0.05, 1.5)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig(directory + '/ch5-2-vortex_padding_duration.png', bbox_inches='tight')

    ###############################################################################################
    print 'CLOSING SHELVE\n'
    # Close shelve:
    data_shelve.close()
    
    ###############################################################################################


if __name__ == "__main__":
    try:
        run()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

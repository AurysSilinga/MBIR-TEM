#! python
# -*- coding: utf-8 -*-
"""Compare the different methods to create phase maps."""


import time
import pdb
import traceback
import sys
import os
import numpy as np
from numpy import pi

import matplotlib.pyplot as plt
from matplotlib.ticker import IndexLocator

import pyramid.magcreator as mc
import pyramid.projector as pj
import pyramid.phasemapper as pm
import pyramid.analytic as an
from pyramid.magdata import MagData
import shelve


def run():

    print '\nACCESS SHELVE'
    # Create / Open databank:
    directory = '../../output/paper 1'
    if not os.path.exists(directory):
        os.makedirs(directory)
    data_shelve = shelve.open(directory + '/paper_1_shelve')

    ###############################################################################################
    print 'CH5-3 METHOD COMPARISON'
    
    key = 'ch5-3-compare_method_data'
    if key in data_shelve:
        print '--LOAD METHOD DATA'
        (data_disc_fourier0, data_vort_fourier0,
         data_disc_fourier1, data_vort_fourier1,
         data_disc_fourier10, data_vort_fourier10,
         data_disc_real_s, data_vort_real_s,
         data_disc_real_d, data_vort_real_d) = data_shelve[key]
    else:
        # Input parameters:
        steps = 6 
        res = 0.25  # in nm
        phi = pi/2
        dim = (64, 512, 512)  # in px (z, y, x)
        center = (dim[0]/2-0.5, dim[1]/2.-0.5, dim[2]/2.-0.5)  # in px (z, y, x) index starts with 0!
        radius = dim[1]/4  # in px
        height = dim[0]/2  # in px
        
        print '--CREATE MAGNETIC SHAPE'
        mag_shape = mc.Shapes.disc(dim, center, radius, height)
        # Create MagData (4 times the size):
        print '--CREATE MAG. DIST. HOMOG. MAGN. DISC'
        mag_data_disc = MagData(res, mc.create_mag_dist(mag_shape, phi))
        print '--CREATE MAG. DIST. VORTEX STATE DISC'
        mag_data_vort = MagData(res, mc.create_mag_dist_vortex(mag_shape, center))
    
        # Create Data Arrays
        res_list = [res*2**i for i in np.linspace(1, steps, steps)]
        data_disc_fourier0 = np.vstack((res_list, np.zeros((2, steps))))
        data_vort_fourier0 = np.vstack((res_list, np.zeros((2, steps))))
        data_disc_fourier1 = np.vstack((res_list, np.zeros((2, steps))))
        data_vort_fourier1 = np.vstack((res_list, np.zeros((2, steps))))
        data_disc_fourier10 = np.vstack((res_list, np.zeros((2, steps))))
        data_vort_fourier10 = np.vstack((res_list, np.zeros((2, steps))))
        data_disc_real_s = np.vstack((res_list, np.zeros((2, steps))))
        data_vort_real_s = np.vstack((res_list, np.zeros((2, steps))))
        data_disc_real_d = np.vstack((res_list, np.zeros((2, steps))))
        data_vort_real_d = np.vstack((res_list, np.zeros((2, steps))))

        for i in range(steps):
            # Scale mag_data, resolution and dimensions:
            mag_data_disc.scale_down()
            mag_data_vort.scale_down()
            dim = mag_data_disc.dim
            res = mag_data_disc.res
            center = (dim[0]/2-0.5, dim[1]/2.-0.5, dim[2]/2.-0.5)  # in px (z, y, x) index starts with 0!
            radius = dim[1]/4  # in px
            height = dim[0]/2  # in px
            
            print '--res =', res, 'nm', 'dim =', dim
            
            print '----CALCULATE RMS/DURATION HOMOG. MAGN. DISC'
            # Create projections along z-axis:
            projection_disc = pj.simple_axis_projection(mag_data_disc)
            # Analytic solution:
            phase_ana_disc = an.phase_mag_disc(dim, res, phi, center, radius, height)
            # Fourier unpadded:
            padding = 0
            start_time = time.clock()
            phase_num_disc = pm.phase_mag_fourier(res, projection_disc, padding=padding)
            data_disc_fourier0[2, i] = time.clock() - start_time
            print '------time (disc, fourier0) =', data_disc_fourier0[2, i]
            phase_diff_disc = (phase_ana_disc-phase_num_disc) * 1E3  # in mrad -> *1000
            data_disc_fourier0[1, i] = np.sqrt(np.mean(phase_diff_disc**2))
            # Fourier padding 1:
            padding = 1
            start_time = time.clock()
            phase_num_disc = pm.phase_mag_fourier(res, projection_disc, padding=padding)
            data_disc_fourier1[2, i] = time.clock() - start_time
            print '------time (disc, fourier1) =', data_disc_fourier1[2, i]
            phase_diff_disc = (phase_ana_disc-phase_num_disc) * 1E3  # in mrad -> *1000
            data_disc_fourier1[1, i] = np.sqrt(np.mean(phase_diff_disc**2))
            # Fourier padding 10:
            padding = 10
            start_time = time.clock()
            phase_num_disc = pm.phase_mag_fourier(res, projection_disc, padding=padding)
            data_disc_fourier10[2, i] = time.clock() - start_time
            print '------time (disc, fourier10) =', data_disc_fourier10[2, i]
            phase_diff_disc = (phase_ana_disc-phase_num_disc) * 1E3  # in mrad -> *1000
            data_disc_fourier10[1, i] = np.sqrt(np.mean(phase_diff_disc**2))
            # Real space slab:
            start_time = time.clock()
            phase_num_disc = pm.phase_mag_real(res, projection_disc, 'slab')
            data_disc_real_s[2, i] = time.clock() - start_time
            print '------time (disc, real slab) =', data_disc_real_s[2, i]
            phase_diff_disc = (phase_ana_disc-phase_num_disc) * 1E3  # in mrad -> *1000
            data_disc_real_s[1, i] = np.sqrt(np.mean(phase_diff_disc**2))
            # Real space disc:
            start_time = time.clock()
            phase_num_disc = pm.phase_mag_real(res, projection_disc, 'disc')
            data_disc_real_d[2, i] = time.clock() - start_time
            print '------time (disc, real disc) =', data_disc_real_d[2, i]
            phase_diff_disc = (phase_ana_disc-phase_num_disc) * 1E3  # in mrad -> *1000
            data_disc_real_d[1, i] = np.sqrt(np.mean(phase_diff_disc**2))

            print '----CALCULATE RMS/DURATION HOMOG. MAGN. DISC'
            # Create projections along z-axis:
            projection_vort = pj.simple_axis_projection(mag_data_vort)
            # Analytic solution:
            phase_ana_vort = an.phase_mag_vortex(dim, res, center, radius, height)
            # Fourier unpadded:
            padding = 0
            start_time = time.clock()
            phase_num_vort = pm.phase_mag_fourier(res, projection_vort, padding=padding)
            data_vort_fourier0[2, i] = time.clock() - start_time
            print '------time (vortex, fourier0) =', data_vort_fourier0[2, i]
            phase_diff_vort = (phase_ana_vort-phase_num_vort) * 1E3  # in mrad -> *1000
            data_vort_fourier0[1, i] = np.sqrt(np.mean(phase_diff_vort**2))
            # Fourier padding 1:
            padding = 1
            start_time = time.clock()
            phase_num_vort = pm.phase_mag_fourier(res, projection_vort, padding=padding)
            data_vort_fourier1[2, i] = time.clock() - start_time
            print '------time (vortex, fourier1) =', data_vort_fourier1[2, i]
            phase_diff_vort = (phase_ana_vort-phase_num_vort) * 1E3  # in mrad -> *1000
            data_vort_fourier1[1, i] = np.sqrt(np.mean(phase_diff_vort**2))
            # Fourier padding 10:
            padding = 10
            start_time = time.clock()
            phase_num_vort = pm.phase_mag_fourier(res, projection_vort, padding=padding)
            data_vort_fourier10[2, i] = time.clock() - start_time
            print '------time (vortex, fourier10) =', data_vort_fourier10[2, i]
            phase_diff_vort = (phase_ana_vort-phase_num_vort) * 1E3  # in mrad -> *1000
            data_vort_fourier10[1, i] = np.sqrt(np.mean(phase_diff_vort**2))
            # Real space slab:
            start_time = time.clock()
            phase_num_vort = pm.phase_mag_real(res, projection_vort, 'slab')
            data_vort_real_s[2, i] = time.clock() - start_time
            print '------time (vortex, real slab) =', data_vort_real_s[2, i]
            phase_diff_vort = (phase_ana_vort-phase_num_vort) * 1E3  # in mrad -> *1000
            data_vort_real_s[1, i] = np.sqrt(np.mean(phase_diff_vort**2))
            # Real space disc:
            start_time = time.clock()
            phase_num_vort = pm.phase_mag_real(res, projection_vort, 'disc')
            data_vort_real_d[2, i] = time.clock() - start_time
            print '------time (vortex, real disc) =', data_vort_real_d[2, i]
            phase_diff_vort = (phase_ana_vort-phase_num_vort) * 1E3  # in mrad -> *1000
            data_vort_real_d[1, i] = np.sqrt(np.mean(phase_diff_vort**2))
            
        print '--SHELVE METHOD DATA'
        data_shelve[key] = (data_disc_fourier0, data_vort_fourier0,
                            data_disc_fourier1, data_vort_fourier1,
                            data_disc_fourier10, data_vort_fourier10,
                            data_disc_real_s, data_vort_real_s,
                            data_disc_real_d, data_vort_real_d)
    
    print '--PLOT/SAVE METHOD DATA'
    
    # Plot using shared rows and colums:
    fig, axes = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(12, 8))
    fig.tight_layout(rect=(0.05, 0.05, 0.95, 0.95))
    fig.suptitle('Method Comparison', fontsize=20)

    # Plot duration against res (disc) [top/left]:
    axes[1, 0].set_yscale('log')
    axes[1, 0].plot(data_disc_fourier0[0], data_disc_fourier0[1], ':bs')
    axes[1, 0].plot(data_disc_fourier1[0], data_disc_fourier1[1], ':bo')
    axes[1, 0].plot(data_disc_fourier10[0], data_disc_fourier10[1], ':b^')
    axes[1, 0].plot(data_disc_real_s[0], data_disc_real_s[1], '--rs')
    axes[1, 0].plot(data_disc_real_d[0], data_disc_real_d[1], '--ro')
    axes[1, 0].set_xlabel('res [nm]', fontsize=15)
    axes[1, 0].set_ylabel('RMS [mrad]', fontsize=15)
    axes[1, 0].set_xlim(-0.5, 16.5)
    axes[1, 0].tick_params(axis='both', which='major', labelsize=14)
    axes[1, 0].xaxis.set_major_locator(IndexLocator(base=4, offset=-0.5))

    # Plot RMS against res (disc) [bottom/left]:
    plt.tick_params(axis='both', which='major', labelsize=14)
    axes[0, 0].set_yscale('log')
    axes[0, 0].plot(data_disc_fourier0[0], data_disc_fourier0[2], ':bs')
    axes[0, 0].plot(data_disc_fourier1[0], data_disc_fourier1[2], ':bo')
    axes[0, 0].plot(data_disc_fourier10[0], data_disc_fourier10[2], ':b^')
    axes[0, 0].plot(data_disc_real_s[0], data_disc_real_s[2], '--rs')
    axes[0, 0].plot(data_disc_real_d[0], data_disc_real_d[2], '--ro')
    axes[0, 0].set_title('Homog. magn. disc', fontsize=18)
    axes[0, 0].set_ylabel('duration [s]', fontsize=15)
    axes[0, 0].set_xlim(-0.5, 16.5)
    axes[0, 0].tick_params(axis='both', which='major', labelsize=14)
    axes[0, 0].xaxis.set_major_locator(IndexLocator(base=4, offset=-0.5))

    # Plot duration against res (vortex) [top/right]:
    axes[1, 1].set_yscale('log')
    axes[1, 1].plot(data_vort_fourier0[0], data_vort_fourier0[1], ':bs')
    axes[1, 1].plot(data_vort_fourier1[0], data_vort_fourier1[1], ':bo')
    axes[1, 1].plot(data_vort_fourier10[0], data_vort_fourier10[1], ':b^')
    axes[1, 1].plot(data_vort_real_s[0], data_vort_real_s[1], '--rs')
    axes[1, 1].plot(data_vort_real_d[0], data_vort_real_d[1], '--ro')
    axes[1, 1].set_xlabel('res [nm]', fontsize=15)
    axes[1, 1].set_xlim(-0.5, 16.5)
    axes[1, 1].tick_params(axis='both', which='major', labelsize=14)
    axes[1, 1].xaxis.set_major_locator(IndexLocator(base=4, offset=-0.5))

    # Plot RMS against res (vortex) [bottom/right]:
    axes[0, 1].set_yscale('log')
    axes[0, 1].plot(data_vort_fourier0[0], data_vort_fourier0[2], ':bs',
                    label='Fourier padding=0')
    axes[0, 1].plot(data_vort_fourier1[0], data_vort_fourier1[2], ':bo',
                    label='Fourier padding=1')
    axes[0, 1].plot(data_vort_fourier10[0], data_vort_fourier10[2], ':b^',
                    label='Fourier padding=10')
    axes[0, 1].plot(data_vort_real_s[0], data_vort_real_s[2], '--rs',
                    label='Real space (slab)')
    axes[0, 1].plot(data_vort_real_d[0], data_vort_real_d[2], '--ro',
                    label='Real space (disc)')
    axes[0, 1].set_title('Vortex state disc', fontsize=18)
    axes[0, 1].set_xlim(-0.5, 16.5)
    axes[0, 1].tick_params(axis='both', which='major', labelsize=14)
    axes[0, 1].xaxis.set_major_locator(IndexLocator(base=4, offset=-0.5))
    axes[0, 1].legend(loc=1)
    
    # Save figure as .png:
    plt.show()
    plt.figtext(0.45, 0.85, 'a)', fontsize=30)
    plt.figtext(0.57, 0.85, 'b)', fontsize=30)
    plt.figtext(0.45, 0.15, 'c)', fontsize=30)
    plt.figtext(0.57, 0.15, 'd)', fontsize=30)
    plt.savefig(directory + '/ch5-3-method comparison.png', bbox_inches='tight')

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

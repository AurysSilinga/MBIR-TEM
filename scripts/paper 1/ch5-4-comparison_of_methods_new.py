#! python
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:37:30 2013

@author: Jan
"""


import time
import os

import numpy as np
from numpy import pi

import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, LogLocator, LogFormatter

import pyramid.magcreator as mc
import pyramid.analytic as an
from pyramid.magdata import MagData
from pyramid.projector import SimpleProjector
from pyramid.phasemapper import PMConvolve, PMFourier

import shelve


force_calculation = True


print '\nACCESS SHELVE'
# Create / Open databank:
directory = '../../output/paper 1'
if not os.path.exists(directory):
    os.makedirs(directory)
data_shelve = shelve.open(directory + '/paper_1_shelve')

###############################################################################################
print 'CH5-4 METHOD COMPARISON'

key = 'ch5-4-method_comparison_new'
if key in data_shelve and not force_calculation:
    print '--LOAD METHOD DATA'
    (data_disc_fourier0, data_vort_fourier0,
     data_disc_fourier3, data_vort_fourier3,
     data_disc_fourier10, data_vort_fourier10,
     data_disc_real_s, data_vort_real_s,
     data_disc_real_d, data_vort_real_d) = data_shelve[key]
else:
    # Input parameters:
    steps = 4
    a = 0.5  # in nm
    phi = pi/2
    dim = (32, 256, 256)  # in px (z, y, x)
    center = (dim[0]/2-0.5, dim[1]/2.-0.5, dim[2]/2.-0.5)  # in px (z, y, x) index starts at 0!
    radius = dim[1]/4  # in px
    height = dim[0]/2  # in px

    print '--CREATE MAGNETIC SHAPE'
    mag_shape = mc.Shapes.disc(dim, center, radius, height)
    # Create MagData (4 times the size):
    print '--CREATE MAG. DIST. HOMOG. MAGN. DISC'
    mag_data_disc = MagData(a, mc.create_mag_dist_homog(mag_shape, phi))
    print '--CREATE MAG. DIST. VORTEX STATE DISC'
    mag_data_vort = MagData(a, mc.create_mag_dist_vortex(mag_shape, center))

    # Create Data Arrays:
    dim_list = [dim[2]/2**i for i in range(steps)]
    data_disc_fourier0 = np.vstack((dim_list, np.zeros((2, steps))))
    data_vort_fourier0 = np.vstack((dim_list, np.zeros((2, steps))))
    data_disc_fourier3 = np.vstack((dim_list, np.zeros((2, steps))))
    data_vort_fourier3 = np.vstack((dim_list, np.zeros((2, steps))))
    data_disc_fourier10 = np.vstack((dim_list, np.zeros((2, steps))))
    data_vort_fourier10 = np.vstack((dim_list, np.zeros((2, steps))))
    data_disc_real_s = np.vstack((dim_list, np.zeros((2, steps))))
    data_vort_real_s = np.vstack((dim_list, np.zeros((2, steps))))
    data_disc_real_d = np.vstack((dim_list, np.zeros((2, steps))))
    data_vort_real_d = np.vstack((dim_list, np.zeros((2, steps))))

    for i in range(steps):
        # Scale mag_data, grid spacing and dimensions:
        dim = mag_data_disc.dim
        a = mag_data_disc.a
        center = (dim[0]/2-0.5, dim[1]/2.-0.5, dim[2]/2.-0.5)  # in px, index starts at 0!
        radius = dim[1]/4  # in px
        height = dim[0]/2  # in px

        print '--a =', a, 'nm', 'dim =', dim
        
        # Create projector along z-axis and phasemapper:
        projector = SimpleProjector(dim)
        pm_fourier0 = PMFourier(a, projector, padding=0)
        pm_fourier3 = PMFourier(a, projector, padding=3)
        pm_fourier10 = PMFourier(a, projector, padding=10)
        pm_slab = PMConvolve(a, projector, geometry='slab')
        pm_disc = PMConvolve(a, projector, geometry='disc')

        print '----CALCULATE RMS/DURATION HOMOG. MAGN. DISC'
        # Analytic solution:
        phase_ana_disc = an.phase_mag_disc(dim, a, phi, center, radius, height)
        # Fourier unpadded:
        start_time = time.clock()
        phase_num_disc = pm_fourier0(mag_data_disc)
        data_disc_fourier0[2, i] = time.clock() - start_time
        print '------time (disc, fourier0) =', data_disc_fourier0[2, i]
        phase_diff_disc = (phase_ana_disc-phase_num_disc) * 1E3  # in mrad -> *1000
        data_disc_fourier0[1, i] = np.sqrt(np.mean(phase_diff_disc.phase**2))
        # Fourier padding 3:
        start_time = time.clock()
        phase_num_disc = pm_fourier3(mag_data_disc)
        data_disc_fourier3[2, i] = time.clock() - start_time
        print '------time (disc, fourier3) =', data_disc_fourier3[2, i]
        phase_diff_disc = (phase_ana_disc-phase_num_disc) * 1E3  # in mrad -> *1000
        data_disc_fourier3[1, i] = np.sqrt(np.mean(phase_diff_disc.phase**2))
        # Fourier padding 10:
        start_time = time.clock()
        phase_num_disc = pm_fourier10(mag_data_disc)
        data_disc_fourier10[2, i] = time.clock() - start_time
        print '------time (disc, fourier10) =', data_disc_fourier10[2, i]
        phase_diff_disc = (phase_ana_disc-phase_num_disc) * 1E3  # in mrad -> *1000
        data_disc_fourier10[1, i] = np.sqrt(np.mean(phase_diff_disc.phase**2))
        # Real space slab:
        start_time = time.clock()
        phase_num_disc = pm_slab(mag_data_disc)
        data_disc_real_s[2, i] = time.clock() - start_time
        print '------time (disc, real slab) =', data_disc_real_s[2, i]
        phase_diff_disc = (phase_ana_disc-phase_num_disc) * 1E3  # in mrad -> *1000
        data_disc_real_s[1, i] = np.sqrt(np.mean(phase_diff_disc.phase**2))
        # Real space disc:
        start_time = time.clock()
        phase_num_disc = pm_disc(mag_data_disc)
        data_disc_real_d[2, i] = time.clock() - start_time
        print '------time (disc, real disc) =', data_disc_real_d[2, i]
        phase_diff_disc = (phase_ana_disc-phase_num_disc) * 1E3  # in mrad -> *1000
        data_disc_real_d[1, i] = np.sqrt(np.mean(phase_diff_disc.phase**2))
        print 'TIME:', data_disc_real_d[2, i]
        print 'RMS%:', np.sqrt(np.mean(((phase_ana_disc-phase_num_disc).phase/phase_ana_disc.phase)**2))*100, '%'

        print '----CALCULATE RMS/DURATION HOMOG. MAGN. DISC'
        # Analytic solution:
        phase_ana_vort = an.phase_mag_vortex(dim, a, center, radius, height)
        # Fourier unpadded:
        start_time = time.clock()
        phase_num_vort = pm_fourier0(mag_data_vort)
        data_vort_fourier0[2, i] = time.clock() - start_time
        print '------time (vortex, fourier0) =', data_vort_fourier0[2, i]
        phase_diff_vort = (phase_ana_vort-phase_num_vort) * 1E3  # in mrad -> *1000
        data_vort_fourier0[1, i] = np.sqrt(np.mean(phase_diff_vort.phase**2))
        # Fourier padding 3:
        start_time = time.clock()
        phase_num_vort = pm_fourier3(mag_data_vort)
        data_vort_fourier3[2, i] = time.clock() - start_time
        print '------time (vortex, fourier3) =', data_vort_fourier3[2, i]
        phase_diff_vort = (phase_ana_vort-phase_num_vort) * 1E3  # in mrad -> *1000
        data_vort_fourier3[1, i] = np.sqrt(np.mean(phase_diff_vort.phase**2))
        # Fourier padding 10:
        start_time = time.clock()
        phase_num_vort = pm_fourier10(mag_data_vort)
        data_vort_fourier10[2, i] = time.clock() - start_time
        print '------time (vortex, fourier10) =', data_vort_fourier10[2, i]
        phase_diff_vort = (phase_ana_vort-phase_num_vort) * 1E3  # in mrad -> *1000
        data_vort_fourier10[1, i] = np.sqrt(np.mean(phase_diff_vort.phase**2))
        # Real space slab:
        start_time = time.clock()
        phase_num_vort = pm_slab(mag_data_vort)
        data_vort_real_s[2, i] = time.clock() - start_time
        print '------time (vortex, real slab) =', data_vort_real_s[2, i]
        phase_diff_vort = (phase_ana_vort-phase_num_vort) * 1E3  # in mrad -> *1000
        data_vort_real_s[1, i] = np.sqrt(np.mean(phase_diff_vort.phase**2))
        # Real space disc:
        start_time = time.clock()
        phase_num_vort = pm_disc(mag_data_vort)
        data_vort_real_d[2, i] = time.clock() - start_time
        print '------time (vortex, real disc) =', data_vort_real_d[2, i]
        phase_diff_vort = (phase_ana_vort-phase_num_vort) * 1E3  # in mrad -> *1000
        data_vort_real_d[1, i] = np.sqrt(np.mean(phase_diff_vort.phase**2))

        # Scale down for next iteration:
        mag_data_disc.scale_down()
        mag_data_vort.scale_down()

    print '--SHELVE METHOD DATA'
    data_shelve[key] = (data_disc_fourier0, data_vort_fourier0,
                        data_disc_fourier3, data_vort_fourier3,
                        data_disc_fourier10, data_vort_fourier10,
                        data_disc_real_s, data_vort_real_s,
                        data_disc_real_d, data_vort_real_d)

print '--PLOT/SAVE METHOD DATA'

# Plot using shared rows and colums:
fig, axes = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(12, 8))
fig.tight_layout(rect=(0.05, 0.05, 0.95, 0.95))
fig.suptitle('Method Comparison', fontsize=20)

# Plot duration against a (disc) [top/left]:
axes[1, 0].set_xscale('log')
axes[1, 0].set_yscale('log')
axes[1, 0].plot(data_disc_fourier0[0], data_disc_fourier0[1], ':bs')
axes[1, 0].plot(data_disc_fourier3[0], data_disc_fourier3[1], ':bo')
axes[1, 0].plot(data_disc_fourier10[0], data_disc_fourier10[1], ':b^')
axes[1, 0].plot(data_disc_real_s[0], data_disc_real_s[1], '--rs')
axes[1, 0].plot(data_disc_real_d[0], data_disc_real_d[1], '--ro')
axes[1, 0].set_xlabel('grid size [px]', fontsize=15)
axes[1, 0].set_ylabel('RMS [mrad]', fontsize=15)
axes[1, 0].set_xlim(25, 350)
axes[1, 0].tick_params(axis='both', which='major', labelsize=14)
axes[1, 0].xaxis.set_major_locator(LogLocator(base=2))
axes[1, 0].xaxis.set_major_formatter(LogFormatter(base=2))
axes[1, 0].xaxis.set_minor_locator(NullLocator())
axes[1, 0].grid()

# Plot RMS against a (disc) [bottom/left]:
plt.tick_params(axis='both', which='major', labelsize=14)
axes[0, 0].set_xscale('log')
axes[0, 0].set_yscale('log')
axes[0, 0].plot(data_disc_fourier0[0], data_disc_fourier0[2], ':bs')
axes[0, 0].plot(data_disc_fourier3[0], data_disc_fourier3[2], ':bo')
axes[0, 0].plot(data_disc_fourier10[0], data_disc_fourier10[2], ':b^')
axes[0, 0].plot(data_disc_real_s[0], data_disc_real_s[2], '--rs')
axes[0, 0].plot(data_disc_real_d[0], data_disc_real_d[2], '--ro')
axes[0, 0].set_title('Homog. magn. disc', fontsize=18)
axes[0, 0].set_ylabel('duration [s]', fontsize=15)
axes[0, 0].set_xlim(25, 350)
axes[0, 0].tick_params(axis='both', which='major', labelsize=14)
axes[0, 0].xaxis.set_major_locator(LogLocator(base=2))
axes[0, 0].xaxis.set_major_formatter(LogFormatter(base=2))
axes[0, 0].xaxis.set_minor_locator(NullLocator())
axes[0, 0].grid()

# Plot duration against a (vortex) [top/right]:
axes[1, 1].set_xscale('log')
axes[1, 1].set_yscale('log')
axes[1, 1].plot(data_vort_fourier0[0], data_vort_fourier0[1], ':bs',
                label='Fourier padding=0')
axes[1, 1].plot(data_vort_fourier3[0], data_vort_fourier3[1], ':bo',
                label='Fourier padding=3')
axes[1, 1].plot(data_vort_fourier10[0], data_vort_fourier10[1], ':b^',
                label='Fourier padding=10')
axes[1, 1].plot(data_vort_real_s[0], data_vort_real_s[1], '--rs',
                label='Real space (slab)')
axes[1, 1].plot(data_vort_real_d[0], data_vort_real_d[1], '--ro',
                label='Real space (disc)')
axes[1, 1].set_xlabel('grid size [px]', fontsize=15)
axes[1, 1].set_xlim(25, 350)
axes[1, 1].tick_params(axis='both', which='major', labelsize=14)
axes[1, 1].xaxis.set_major_locator(LogLocator(base=2))
axes[1, 1].xaxis.set_major_formatter(LogFormatter(base=2))
axes[1, 1].xaxis.set_minor_locator(NullLocator())
axes[1, 1].grid()

# Plot RMS against a (vortex) [bottom/right]:
axes[0, 1].set_xscale('log')
axes[0, 1].set_yscale('log')
axes[0, 1].plot(data_vort_fourier0[0], data_vort_fourier0[2], ':bs',
                label='Fourier padding=0')
axes[0, 1].plot(data_vort_fourier3[0], data_vort_fourier3[2], ':bo',
                label='Fourier padding=3')
axes[0, 1].plot(data_vort_fourier10[0], data_vort_fourier10[2], ':b^',
                label='Fourier padding=10')
axes[0, 1].plot(data_vort_real_s[0], data_vort_real_s[2], '--rs',
                label='Real space (slab)')
axes[0, 1].plot(data_vort_real_d[0], data_vort_real_d[2], '--ro',
                label='Real space (disc)')
axes[0, 1].set_title('Vortex state disc', fontsize=18)
axes[0, 1].set_xlim(25, 350)
axes[0, 1].tick_params(axis='both', which='major', labelsize=14)
axes[0, 1].xaxis.set_major_locator(LogLocator(base=2))
axes[0, 1].xaxis.set_major_formatter(LogFormatter(base=2))
axes[0, 1].xaxis.set_minor_locator(NullLocator())
axes[0, 1].grid()

# Add legend:
axes[1, 1].legend(bbox_to_anchor=(0, 0, 0.955, 0.615), bbox_transform=fig.transFigure,
                  prop={'size':12})

# Save figure as .png:
plt.show()
plt.figtext(0.12, 0.85, 'a)', fontsize=30)
plt.figtext(0.57, 0.85, 'b)', fontsize=30)
plt.figtext(0.12, 0.15, 'c)', fontsize=30)
plt.figtext(0.57, 0.15, 'd)', fontsize=30)
plt.savefig(directory + '/ch5-3-method comparison.png', bbox_inches='tight')

###############################################################################################
print 'CLOSING SHELVE\n'
# Close shelve:
data_shelve.close()

###############################################################################################

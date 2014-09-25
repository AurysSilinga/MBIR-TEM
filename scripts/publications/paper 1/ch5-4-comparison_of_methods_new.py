# -*- coding: utf-8 -*-
"""Created on Fri Jul 26 14:37:30 2013 @author: Jan"""


import os

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, LogLocator, LogFormatter

import pyramid
import pyramid.magcreator as mc
import pyramid.analytic as an
from pyramid.magdata import MagData
from pyramid.projector import SimpleProjector
from pyramid.phasemapper import PMConvolve, PMFourier

import time
import shelve

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')


logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)

force_calculation = True

n = 1


def get_time(pm, mag_data, n):
    t = []
    for i in range(n):
        start = time.clock()
        pm(mag_data)
        t.append(time.clock()-start)
    t, dt = np.mean(t), np.std(t)
    phase_map = pm(mag_data)
    return phase_map, t, dt


print '\nACCESS SHELVE'
# Create / Open databank:
directory = '../../../output/paper 1'
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
     data_disc_real, data_vort_real) = data_shelve[key]
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
    data_disc_fourier0 = np.vstack((dim_list, np.zeros((3, steps))))
    data_vort_fourier0 = np.vstack((dim_list, np.zeros((3, steps))))
    data_disc_fourier3 = np.vstack((dim_list, np.zeros((3, steps))))
    data_vort_fourier3 = np.vstack((dim_list, np.zeros((3, steps))))
    data_disc_fourier10 = np.vstack((dim_list, np.zeros((3, steps))))
    data_vort_fourier10 = np.vstack((dim_list, np.zeros((3, steps))))
    data_disc_real = np.vstack((dim_list, np.zeros((3, steps))))
    data_vort_real = np.vstack((dim_list, np.zeros((3, steps))))

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
        pm_fourier10 = PMFourier(a, projector, padding=7)
        pm_real = PMConvolve(a, projector)

        print '----CALCULATE RMS/DURATION HOMOG. MAGN. DISC'
        # Analytic solution:
        phase_ana_disc = an.phase_mag_disc(dim, a, phi, center, radius, height)
        # Fourier unpadded:
        phase_num_disc, t, dt = get_time(pm_fourier0, mag_data_disc, n)
        data_disc_fourier0[2, i], data_disc_fourier0[3, i] = t, dt
        print '------time (disc, fourier0) =', data_disc_fourier0[2, i]
        phase_diff_disc = (phase_ana_disc-phase_num_disc) * 1E3  # in mrad -> *1000
        print 'phase mean:', np.mean(phase_num_disc.phase)
        data_disc_fourier0[1, i] = np.sqrt(np.mean(phase_diff_disc.phase**2))
        # Fourier padding 3:
        phase_num_disc, t, dt = get_time(pm_fourier3, mag_data_disc, n)
        data_disc_fourier3[2, i], data_disc_fourier3[3, i] = t, dt
        print '------time (disc, fourier3) =', data_disc_fourier3[2, i]
        phase_diff_disc = (phase_ana_disc-phase_num_disc) * 1E3  # in mrad -> *1000
        print 'phase mean:', np.mean(phase_num_disc.phase)
        data_disc_fourier3[1, i] = np.sqrt(np.mean(phase_diff_disc.phase**2))
        # Fourier padding 10:
        phase_num_disc, t, dt = get_time(pm_fourier10, mag_data_disc, n)
        data_disc_fourier10[2, i], data_disc_fourier10[3, i] = t, dt
        print '------time (disc, fourier10) =', data_disc_fourier10[2, i]
        phase_diff_disc = (phase_ana_disc-phase_num_disc) * 1E3  # in mrad -> *1000
        print 'phase mean:', np.mean(phase_num_disc.phase)
        data_disc_fourier10[1, i] = np.sqrt(np.mean(phase_diff_disc.phase**2))
        # Real space disc:
        phase_num_disc, t, dt = get_time(pm_real, mag_data_disc, n)
        data_disc_real[2, i], data_disc_real[3, i] = t, dt
        print '------time (disc, real space) =', data_disc_real[2, i]
        phase_diff_disc = (phase_ana_disc-phase_num_disc) * 1E3  # in mrad -> *1000
        print 'phase mean:', np.mean(phase_num_disc.phase)
        data_disc_real[1, i] = np.sqrt(np.mean(phase_diff_disc.phase**2))
        print 'TIME:', data_disc_real[2, i]
        print 'RMS%:', np.sqrt(np.mean(((phase_ana_disc-phase_num_disc).phase /
                                        phase_ana_disc.phase)**2))*100, '%'

        print '----CALCULATE RMS/DURATION VORTEX STATE DISC'
        # Analytic solution:
        phase_ana_vort = an.phase_mag_vortex(dim, a, center, radius, height)
        # Fourier unpadded:
        phase_num_vort, t, dt = get_time(pm_fourier0, mag_data_vort, n)
        phase_fft0 = phase_num_vort
        data_vort_fourier0[2, i], data_vort_fourier0[3, i] = t, dt
        print '------time (vortex, fourier0) =', data_vort_fourier0[2, i]
        phase_diff_vort = (phase_ana_vort-phase_num_vort) * 1E3  # in mrad -> *1000
        print 'phase mean:', np.mean(phase_num_vort.phase)
        phase_diff_vort -= np.mean(phase_diff_vort.phase)
        data_vort_fourier0[1, i] = np.sqrt(np.mean(phase_diff_vort.phase**2))
        # Fourier padding 3:
        phase_num_vort, t, dt = get_time(pm_fourier3, mag_data_vort, n)
        phase_fft3 = phase_num_vort
        data_vort_fourier3[2, i], data_vort_fourier3[3, i] = t, dt
        print '------time (vortex, fourier3) =', data_vort_fourier3[2, i]
        phase_diff_vort = (phase_ana_vort-phase_num_vort) * 1E3  # in mrad -> *1000
        print 'phase mean:', np.mean(phase_num_vort.phase)
        phase_diff_vort -= np.mean(phase_diff_vort.phase)
        data_vort_fourier3[1, i] = np.sqrt(np.mean(phase_diff_vort.phase**2))
        # Fourier padding 10:
        phase_num_vort, t, dt = get_time(pm_fourier10, mag_data_vort, n)
        phase_fft10 = phase_num_vort
        data_vort_fourier10[2, i], data_vort_fourier10[3, i] = t, dt
        print '------time (vortex, fourier10) =', data_vort_fourier10[2, i]
        phase_diff_vort = (phase_ana_vort-phase_num_vort) * 1E3  # in mrad -> *1000
        print 'phase mean:', np.mean(phase_num_vort.phase)
        phase_diff_vort -= np.mean(phase_diff_vort.phase)
        data_vort_fourier10[1, i] = np.sqrt(np.mean(phase_diff_vort.phase**2))
        # Real space disc:
        phase_num_vort, t, dt = get_time(pm_real, mag_data_vort, n)
        phase_real = phase_num_vort
        data_vort_real[2, i], data_vort_real[3, i] = t, dt
        print '------time (vortex, real space) =', data_vort_real[2, i]
        phase_diff_vort = (phase_ana_vort-phase_num_vort) * 1E3  # in mrad -> *1000
        print 'phase mean:', np.mean(phase_num_vort.phase)
#        phase_diff_vort -= np.mean(phase_diff_vort.phase)
        data_vort_real[1, i] = np.sqrt(np.mean(phase_diff_vort.phase**2))
        print 'TIME:', data_disc_real[2, i]
        print 'RMS%:', np.sqrt(np.mean(((phase_ana_vort-phase_num_vort).phase /
                                        phase_ana_vort.phase)**2))*100, '%'

        # Scale down for next iteration:
        mag_data_disc.scale_down()
        mag_data_vort.scale_down()

    print '--SHELVE METHOD DATA'
    data_shelve[key] = (data_disc_fourier0, data_vort_fourier0,
                        data_disc_fourier3, data_vort_fourier3,
                        data_disc_fourier10, data_vort_fourier10,
                        data_disc_real, data_vort_real)

print '--PLOT/SAVE METHOD DATA'

# Plot using shared rows and colums:
fig, axes = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(12, 8))
fig.tight_layout(rect=(0.05, 0.05, 0.95, 0.95))
fig.suptitle('Method Comparison', fontsize=20)

# Plot RMS against a (disc) [top/left]:
axes[1, 0].set_xscale('log')
axes[1, 0].set_yscale('log')
axes[1, 0].plot(data_disc_fourier0[0], data_disc_fourier0[1], ':bs')
axes[1, 0].plot(data_disc_fourier3[0], data_disc_fourier3[1], ':bo')
axes[1, 0].plot(data_disc_fourier10[0], data_disc_fourier10[1], ':b^')
axes[1, 0].plot(data_disc_real[0], data_disc_real[1], '--ro')
axes[1, 0].set_xlabel('grid size [px]', fontsize=15)
axes[1, 0].set_ylabel('RMS [mrad]', fontsize=15)
axes[1, 0].set_xlim(25, 350)
axes[1, 0].tick_params(axis='both', which='major', labelsize=14)
axes[1, 0].xaxis.set_major_locator(LogLocator(base=2))
axes[1, 0].xaxis.set_major_formatter(LogFormatter(base=2))
axes[1, 0].xaxis.set_minor_locator(NullLocator())
axes[1, 0].grid()

# Plot duration against a (disc) [bottom/left]:
plt.tick_params(axis='both', which='major', labelsize=14)
axes[0, 0].set_xscale('log')
axes[0, 0].set_yscale('log')
axes[0, 0].errorbar(data_disc_fourier0[0], data_disc_fourier0[2],
                    yerr=0*data_vort_fourier3[3], fmt=':bs')
axes[0, 0].errorbar(data_disc_fourier3[0], data_disc_fourier3[2],
                    yerr=0*data_vort_fourier3[3], fmt=':bo')
axes[0, 0].errorbar(data_disc_fourier10[0], data_disc_fourier10[2],
                    yerr=0*data_vort_fourier3[3], fmt=':b^')
axes[0, 0].errorbar(data_disc_real[0], data_disc_real[2],
                    yerr=0*data_vort_fourier3[3], fmt='--ro')
axes[0, 0].set_title('Homog. magn. disc', fontsize=18)
axes[0, 0].set_ylabel('duration [s]', fontsize=15)
axes[0, 0].set_xlim(25, 350)
axes[0, 0].set_ylim(1E-4, 1E1)
axes[0, 0].tick_params(axis='both', which='major', labelsize=14)
axes[0, 0].xaxis.set_major_locator(LogLocator(base=2))
axes[0, 0].xaxis.set_major_formatter(LogFormatter(base=2))
axes[0, 0].xaxis.set_minor_locator(NullLocator())
axes[0, 0].grid()

# Plot RMS against a (vortex) [top/right]:
axes[1, 1].set_xscale('log')
axes[1, 1].set_yscale('log')
axes[1, 1].plot(data_vort_fourier0[0], data_vort_fourier0[1], ':bs',
                label='Fourier padding=0')
axes[1, 1].plot(data_vort_fourier3[0], data_vort_fourier3[1], ':bo',
                label='Fourier padding=3')
axes[1, 1].plot(data_vort_fourier10[0], data_vort_fourier10[1], ':b^',
                label='Fourier padding=10')
axes[1, 1].plot(data_vort_real[0], data_vort_real[1], '--ro',
                label='Real space')
axes[1, 1].set_xlabel('grid size [px]', fontsize=15)
axes[1, 1].set_xlim(25, 350)
axes[1, 1].tick_params(axis='both', which='major', labelsize=14)
axes[1, 1].xaxis.set_major_locator(LogLocator(base=2))
axes[1, 1].xaxis.set_major_formatter(LogFormatter(base=2))
axes[1, 1].xaxis.set_minor_locator(NullLocator())
axes[1, 1].grid()

# Plot duration against a (vortex) [bottom/right]:
axes[0, 1].set_xscale('log')
axes[0, 1].set_yscale('log')
axes[0, 1].errorbar(data_vort_fourier0[0], data_vort_fourier0[2], yerr=0*data_vort_fourier0[3],
                    fmt=':bs', label='Fourier padding=0')
axes[0, 1].errorbar(data_vort_fourier3[0], data_vort_fourier3[2], yerr=0*data_vort_fourier3[3],
                    fmt=':bo', label='Fourier padding=3')
axes[0, 1].errorbar(data_vort_fourier10[0], data_vort_fourier10[2], yerr=0*data_vort_fourier3[3],
                    fmt=':b^', label='Fourier padding=10')
axes[0, 1].errorbar(data_vort_real[0], data_vort_real[2], yerr=0*data_vort_fourier3[3],
                    fmt='--ro', label='Real space')
axes[0, 1].set_title('Vortex state disc', fontsize=18)
axes[0, 1].set_xlim(25, 350)
axes[0, 1].set_ylim(1E-4, 1E1)
axes[0, 1].tick_params(axis='both', which='major', labelsize=14)
axes[0, 1].xaxis.set_major_locator(LogLocator(base=2))
axes[0, 1].xaxis.set_major_formatter(LogFormatter(base=2))
axes[0, 1].xaxis.set_minor_locator(NullLocator())
axes[0, 1].grid()

# Add legend:
axes[1, 1].legend(bbox_to_anchor=(0, 0, 0.955, 0.615), bbox_transform=fig.transFigure,
                  prop={'size': 12})

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

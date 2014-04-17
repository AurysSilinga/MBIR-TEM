#! python
# -*- coding: utf-8 -*-
"""Compare the different methods to create phase maps."""

import os

import time

import numpy as np
from numpy import pi

import pyramid
import pyramid.magcreator as mc
import pyramid.analytic as an
from pyramid.projector import SimpleProjector
from pyramid.phasemapper import PMAdapterFM, PMReal, PMConvolve, PMFourier
from pyramid.magdata import MagData

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')


logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)

# Input parameters:
b_0 = 1.0    # in T
a = 1.0  # in nm
phi = pi/4
numcore = False
padding = 5
density = 10
dim = (128, 128, 128)  # in px (z, y, x)

# Create magnetic shape:
geometry = 'sphere'
if geometry == 'slab':
    center = (dim[0]/2.-0.5, dim[1]/2.-0.5, dim[2]/2.-0.5)  # in px (z,y,x) index starts at 0!
    width = (dim[0]/2, dim[1]/2, dim[2]/2)  # in px (z, y, x)
    mag_shape = mc.Shapes.slab(dim, center, width)
    phase_map_ana = an.phase_mag_slab(dim, a, phi, center, width, b_0)
elif geometry == 'disc':
    center = (dim[0]/2.-0.5, dim[1]/2.-0.5, dim[2]/2.-0.5)  # in px (z,y,x) index starts at 0!
    radius = dim[1]/4  # in px
    height = dim[0]/2  # in px
    mag_shape = mc.Shapes.disc(dim, center, radius, height)
    phase_map_ana = an.phase_mag_disc(dim, a, phi, center, radius, height, b_0)
elif geometry == 'sphere':
    center = (dim[0]/2.-0.5, dim[1]/2.-0.5, dim[2]/2.-0.50)  # in px (z, y, x) index starts with 0!
    radius = dim[1]/4  # in px
    mag_shape = mc.Shapes.sphere(dim, center, radius)
    phase_map_ana = an.phase_mag_sphere(dim, a, phi, center, radius, b_0)

# Create MagData object and projector:
mag_data = MagData(a, mc.create_mag_dist_homog(mag_shape, phi))
mag_data.quiver_plot()
projector = SimpleProjector(dim)
# Construct PhaseMapper objects:
pm_adap = PMAdapterFM(a, projector, b_0)
pm_real = PMReal(a, projector, b_0, numcore=numcore)
pm_conv = PMConvolve(a, projector, b_0)
pm_four = PMFourier(a, projector, b_0, padding=padding)

# Get times for different approaches:
start_time = time.time()
phase_map_adap = pm_adap(mag_data)
print 'Time for PMAdapterFM:  ', time.time() - start_time
start_time = time.time()
phase_map_real = pm_real(mag_data)
print 'Time for PMReal:       ', time.time() - start_time
start_time = time.time()
phase_map_conv = pm_conv(mag_data)
print 'Time for PMConvolve:   ', time.time() - start_time
start_time = time.time()
phase_map_four = pm_four(mag_data)
print 'Time for PMFourier:    ', time.time() - start_time

# Display the combinated plots with phasemap and holography image:
phase_map_ana.display_combined('Analytic Solution', density=density)
phase_map_adap.display_combined('PMAdapterFM', density=density)
phase_map_real.display_combined('PMReal', density=density)
phase_map_conv.display_combined('PMConvolve', density=density)
phase_map_four.display_combined('PMFourier', density=density)

# Plot differences to the analytic solution:
phase_map_diff_adap = phase_map_adap - phase_map_ana
phase_map_diff_real = phase_map_real - phase_map_ana
phase_map_diff_conv = phase_map_conv - phase_map_ana
phase_map_diff_four = phase_map_four - phase_map_ana
RMS_adap = np.std(phase_map_diff_adap.phase)
RMS_real = np.std(phase_map_diff_real.phase)
RMS_conv = np.std(phase_map_diff_conv.phase)
RMS_four = np.std(phase_map_diff_four.phase)
phase_map_diff_adap.display_phase('PMAdapterFM difference (RMS = {:3.2e})'.format(RMS_adap))
phase_map_diff_real.display_phase('PMReal difference (RMS = {:3.2e})'.format(RMS_real))
phase_map_diff_conv.display_phase('PMConvolve difference (RMS = {:3.2e})'.format(RMS_conv))
phase_map_diff_four.display_phase('PMFourier difference (RMS = {:3.2e})'.format(RMS_four))

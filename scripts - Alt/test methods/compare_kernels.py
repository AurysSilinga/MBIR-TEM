#! python
# -*- coding: utf-8 -*-
"""Compare the phase map of one pixel for different approaches."""


import os

import numpy as np

import pyramid
import pyramid.magcreator as mc
from pyramid.projector import SimpleProjector
from pyramid.phasemapper import PhaseMapperRDFC, PhaseMapperFDFC
from pyramid.kernel import Kernel
from pyramid.magdata import MagData
from pyramid.phasemap import PhaseMap

import matplotlib.pyplot as plt
from matplotlib.ticker import IndexLocator

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')


logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)
directory = '../../output/magnetic distributions'
if not os.path.exists(directory):
    os.makedirs(directory)














# Input parameters:
a = 1.0  # in nm
phi = 0  # in rad
dim = (1, 16, 16)
pixel = (0, int(dim[1]/2), int(dim[2]/2))
limit = 0.35


def get_fourier_kernel():
    PHI_0 = 2067.83    # magnetic flux in T*nm²
    b_0 = 1
    coeff = - 1j * b_0 * a**2 / (2*PHI_0)
    nyq = 1 / a  # nyquist frequency
    f_u = np.linspace(0, nyq/2, dim[2]/2.+1)
    f_v = np.linspace(-nyq/2, nyq/2, dim[1], endpoint=False)
    f_uu, f_vv = np.meshgrid(f_u, f_v)
    phase_fft = coeff * f_vv / (f_uu**2 + f_vv**2 + 1e-30)
    # Transform to real space and revert padding:
    phase_fft = np.fft.ifftshift(phase_fft, axes=0)
    phase_fft_kernel = np.fft.fftshift(np.fft.irfft2(phase_fft), axes=(0, 1))
    return phase_fft_kernel


# Create magnetic data and projector:
mag_data = MagData(a, mc.create_mag_dist_homog(mc.Shapes.pixel(dim, pixel), phi))
mag_data.save_to_llg(directory + '/mag_dist_single_pixel.txt')
projector = SimpleProjector(dim)
# Kernel of a disc in real space:
phase_map_disc = PhaseMapperRDFC(Kernel(a, projector.dim_uv, geometry='disc'))(mag_data)
phase_map_disc.unit = 'mrad'
phase_map_disc.display_phase('Phase of one Pixel (Disc)', limit=limit)
## Kernel of a slab in real space:
#phase_map_slab = PhaseMapperRDFC(Kernel(a, projector.dim_uv, geometry='slab'))(mag_data)
#phase_map_slab.unit = 'mrad'
#phase_map_slab.display_phase('Phase of one Pixel (Slab)', limit=limit)
# Kernel of the Fourier method:
phase_map_fft = PhaseMapperFDFC(a, projector.dim_uv, padding=0)(mag_data)
phase_map_fft.unit = 'mrad'
phase_map_fft.display_phase('Phase of one Pixel (Fourier)', limit=limit)
# Kernel of the Fourier method, calculated directly:
phase_map_fft_kernel = PhaseMap(a, get_fourier_kernel(), unit='mrad')
phase_map_fft_kernel.display_phase('Phase of one Pixel (Fourier Kernel)', limit=limit)
# Kernel differences:
print 'Fourier Kernel, direct and indirect method are identical:', \
      np.all(phase_map_fft_kernel.phase - phase_map_fft.phase) == 0
(phase_map_disc-phase_map_fft).display_phase('Phase difference of one Pixel (Disc - Fourier)')

# Cross section plots of real space kernels:
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1)
x = np.linspace(-dim[1]/a/2, dim[1]/a/2-1, dim[1])
y_ft = phase_map_fft.phase[:, dim[1]/2]
y_re = phase_map_disc.phase[:, dim[1]/2]
axis.axhline(0, color='k')
axis.plot(x, y_re, label='Real space method')
axis.plot(x, y_ft, label='Fourier method')
axis.grid()
axis.legend()
axis.set_title('Real Space Kernel')
axis.set_xlim(-dim[1]/2, dim[1]/2-1)
axis.xaxis.set_major_locator(IndexLocator(base=dim[1]/8, offset=0))

# Cross section plots of Fourier space kernels:
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1)
x = range(dim[1])
k_re = np.abs(np.fft.ifftshift(np.fft.rfft2(phase_map_disc.phase), axes=0))[:, 0]**2
k_ft = np.abs(np.fft.ifftshift(np.fft.rfft2(phase_map_fft.phase), axes=0))[:, 0]**2
axis.axhline(0, color='k')
axis.plot(x, k_re, label='Real space method')
axis.plot(x, k_ft, label='Fourier method')
axis.grid()
axis.legend()
axis.set_title('Fourier Space Kernel')
axis.set_xlim(0, dim[1]-1)
axis.xaxis.set_major_locator(IndexLocator(base=dim[1]/8, offset=0))

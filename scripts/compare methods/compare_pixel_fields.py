#! python
# -*- coding: utf-8 -*-
"""Compare the phase map of one pixel for different real space approaches."""


import pdb
import traceback
import sys
import os

import numpy as np

import pyramid.magcreator as mc
import pyramid.projector as pj
import pyramid.phasemapper as pm
from pyramid.magdata import MagData
from pyramid.phasemap import PhaseMap

from scipy.signal import fftconvolve


def compare_pixel_fields():
    '''Calculate and display the phase map for different real space approaches.
    Arguments:
        None
    Returns:
        None

    '''
    directory = '../../output/magnetic distributions'
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Input parameters:
    res = 1.0  # in nm
    phi = 0  # in rad
    dim = (1, 64, 64)
    pixel = (0,  int(dim[1]/2),  int(dim[2]/2))
    limit = 0.35

    def get_fourier_kernel():
        PHI_0 = 2067.83    # magnetic flux in T*nm²
        b_0 = 1
        coeff = - 1j * b_0 * res**2 / (2*PHI_0)
        nyq = 1 / res  # nyquist frequency
        f_u = np.linspace(0, nyq/2, dim[2]/2.+1)
        f_v = np.linspace(-nyq/2, nyq/2, dim[1], endpoint=False)
        f_uu, f_vv = np.meshgrid(f_u, f_v)
        phase_fft = coeff * f_vv / (f_uu**2 + f_vv**2 + 1e-30) #* (8*(res/2)**3)*np.sinc(res/2*f_uu)*np.sinc(res/2*f_vv)
        # Transform to real space and revert padding:
        phase_fft = np.fft.ifftshift(phase_fft, axes=0)
        phase_fft_kernel = np.fft.fftshift(np.fft.irfft2(phase_fft), axes=(0, 1))
        return phase_fft_kernel

    # Create magnetic data, project it, get the phase map and display the holography image:
    mag_data = MagData(res, mc.create_mag_dist_homog(mc.Shapes.pixel(dim, pixel), phi))
    mag_data.save_to_llg(directory + '/mag_dist_single_pixel.txt')
    projection = pj.simple_axis_projection(mag_data)
    # Kernel of a disc in real space:
    phase_map_disc = PhaseMap(res, pm.phase_mag_real(res, projection, 'disc'), 'mrad')
#    phase_map_disc.display('Phase of one Pixel (Disc)', limit=limit)
    # Kernel of a slab in real space:
    phase_map_slab = PhaseMap(res, pm.phase_mag_real(res, projection, 'slab'), 'mrad')
#    phase_map_slab.display('Phase of one Pixel (Slab)', limit=limit)
    # Kernel of the Fourier method:
    phase_map_fft = PhaseMap(res, pm.phase_mag_fourier(res, projection, padding=0), 'mrad')
    phase_map_fft.display('Phase of one Pixel (Fourier)', limit=limit)
    
    # Kernel of the Fourier method, calculated directly:
    phase_map_fft_kernel = PhaseMap(res, get_fourier_kernel(), 'mrad')
#    phase_map_fft_kernel.display('Phase of one Pixel (Fourier Kernel)', limit=limit)
    # Kernel differences:
    print 'Fourier Kernel, direct and indirect method are identical:', \
          np.all(phase_map_fft_kernel.phase - phase_map_fft.phase) == 0
    phase_map_diff = PhaseMap(res, phase_map_disc.phase - phase_map_fft.phase, 'mrad')
#    phase_map_diff.display('Phase difference of one Pixel (Disc - Fourier)')

    phase_inv_fft = np.abs(np.fft.ifftshift(np.fft.rfft2(phase_map_fft.phase), axes=0))**2
    phase_map_inv_fft = PhaseMap(res, phase_inv_fft)
#    phase_map_inv_fft.display('FT of the Phase of one Pixel (Fourier, Power)')

    phase_inv_disc = np.abs(np.fft.ifftshift(np.fft.rfft2(phase_map_disc.phase), axes=0))**2
    phase_map_inv_disc = PhaseMap(res, phase_inv_disc)
#    phase_map_inv_disc.display('FT of the Phase of one Pixel (Disc, Power)')
    
    import matplotlib.pyplot as plt
    from matplotlib.ticker import IndexLocator 
    
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    x = np.linspace(-dim[1]/res/2, dim[1]/res/2-1, dim[1])
    y_ft = phase_map_fft.phase[:, dim[1]/2]
    y_re = phase_map_disc.phase[:, dim[1]/2]
    axis.axhline(0, color='k')
    axis.plot(x, y_re, label='Real space method')
    axis.plot(x, y_ft, label='Fourier method')
#    axis.plot(x, y_re-y_ft, label='Difference')
    axis.grid()
    axis.legend()
    axis.set_title('Real Space Kernel')
    axis.set_xlim(-dim[1]/2, dim[1]/2-1)
    axis.xaxis.set_major_locator(IndexLocator(base=dim[1]/8, offset=0))

    
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    x = range(dim[1])
    k_re = np.abs(np.fft.ifftshift(np.fft.rfft2(phase_map_disc.phase), axes=0))[:, 0]**2
    k_ft = np.abs(np.fft.ifftshift(np.fft.rfft2(phase_map_fft.phase), axes=0))[:, 0]**2



    phase = np.fft.ifftshift(np.fft.rfft2(phase_map_fft.phase), axes=0)
    nyq = 0.5 / res
    f_u = np.linspace(0, nyq, dim[2]/2.+1)
    f_v = np.linspace(-nyq, nyq, dim[1], endpoint=False)
    f_uu, f_vv = np.meshgrid(f_u, f_v)
    kern = 4*(res/2)**2*np.sinc(f_uu*res/2)*np.sinc(f_vv*res/2)
    phase_mod = fftconvolve(phase, kern, 'same')
    k_ft_mod = np.abs(phase_mod)[:, 0]**2
#    sincy = np.sinc((np.asarray(x)-(dim[1]/2-1))*dim[1])
#    quark



    axis.axhline(0, color='k')
    axis.plot(x, k_re, label='Real space method')
    axis.plot(x, k_ft, label='Fourier method')
#    axis.plot(x, k_ft_mod, label='Fourier modified')
    axis.grid()
    axis.legend()
    axis.set_title('Fourier Space Kernel')
    axis.set_xlim(0, dim[1]-1)
    axis.xaxis.set_major_locator(IndexLocator(base=dim[1]/8, offset=0))
    
#    # Convolve:
#    phase_fft = fftconvolve(phase_fft, np.sinc(f_uu/(nyq/2))*np.sinc(f_vv/nyq), 'same')


if __name__ == "__main__":
    try:
        compare_pixel_fields()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

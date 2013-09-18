# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 13:22:43 2013

@author: Jan
"""

from pylab import *


f = 1
a, b = f * 32, f * 64
f_u = np.linspace(0, 1./3, a)
f_v = np.linspace(-1./6., 1./6., b, endpoint=False)
f_uu, f_vv = np.meshgrid(f_u, f_v)
phase_fft = 1j * f_vv / (f_uu**2 + f_vv**2 + 1e-30)
# Transform to real space and revert padding:
phase_fft = np.fft.ifftshift(phase_fft, axes=0)
ss = np.fft.irfft2(phase_fft)
print ss
pcolormesh(np.fft.fftshift(np.fft.fftshift(ss, axes=1), axes=0), cmap='RdBu')
colorbar()
show()

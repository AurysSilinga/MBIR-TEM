# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:35:46 2013

@author: Jan
"""

import numpy as np
from numpy import pi

import matplotlib.pyplot as plt
from matplotlib.ticker import IndexLocator 

N = 16
T = 1
fs = 1. / T
fn = fs / 2.

x = np.linspace(-N/2, N/2, N, endpoint=False)
y = np.abs(x) / (x + 1E-30)**3
y[N/2] = 0
Y_ft = np.fft.fftshift(np.fft.rfft(y))

f = np.linspace(0, fn, N/2+1)
Y = -4 * pi * 1j / (f + 1E-30)
Y[0] = 0
y_ft = np.fft.fftshift(np.fft.irfft(Y))

#np.abs(np.fft.ifftshift(np.fft.rfft2(phase_map_disc.phase), axes=0))[:, 0]**2


fig = plt.figure()
axis = fig.add_subplot(1, 1, 1)
axis.axhline(0, color='k')
axis.plot(x, y, label='Real space method')
axis.plot(x, y_ft, label='Real space method')
axis.grid()
axis.legend()
axis.set_title('Real Space Kernel')
axis.set_xlim(-N/2, N/2)
axis.xaxis.set_major_locator(IndexLocator(base=N/8, offset=0))

fig = plt.figure()
axis = fig.add_subplot(1, 1, 1)
axis.axhline(0, color='k')
#axis.plot(f, np.abs(Y)**2, label='Real space method')
axis.plot(f, np.abs(Y_ft)**2, label='Real space method')
axis.grid()
axis.legend()
axis.set_title('Fourier Space Kernel')
#axis.set_xlim(0, dim[1]-1)
#axis.xaxis.set_major_locator(IndexLocator(base=dim[1]/8, offset=0))
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 02 10:01:27 2015

@author: Jan
"""

import numpy as np
from scipy import signal

dim = (16, 16)
image = np.ones(dim)
image = np.zeros(dim)
image[5:10, 7:12] = 1

yy, xx = np.indices(dim)

diam = 5
R = diam/2.
yk, xk = np.indices((diam, diam)) - (diam-1)/2.
r = np.sqrt(xk**2 + yk**2)
kernel = np.where(r <= R, True, False)

convolution = signal.fftconvolve(image, kernel, mode='same')

peak_coords = [(7, 9)]

means = np.zeros(len(peak_coords))

for i, peak in enumerate(peak_coords):
    means[i] = convolution[peak] / np.count_nonzero(kernel)

print means

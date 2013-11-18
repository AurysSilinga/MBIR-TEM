# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:05:39 2013

@author: Jan
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

w = 8
l = 108
x = np.linspace(0, 20*l, 20*l)


# Kastenfunktion:
y = np.where(x <= l, 1, 0)

fig = plt.figure()
axis = fig.add_subplot(1, 1, 1)
axis.plot(x, y)
axis.set_xlim(0, 2*l)
axis.set_ylim(0, 1.2)
axis.set_title('Kastenfunction --> FFT --> Sinc')

y_fft = np.fft.fft(y)

fig = plt.figure()
axis = fig.add_subplot(1, 1, 1)
axis.plot(x, np.abs(y_fft), 'k', label='abs')
axis.plot(x, np.real(y_fft), 'r', label='real')
axis.plot(x, np.imag(y_fft), 'b', label='imag')
axis.set_xlim(0, 200)
axis.legend()
axis.set_title('Kastenfunction --> FFT --> Sinc')


# Kastenfunktion mit Cos-"Softness":
y = np.where(x <= l, 1, np.where(x >= l+w, 0, 0.5*(1+np.cos(pi*(x-l)/w))))

fig = plt.figure()
axis = fig.add_subplot(1, 1, 1)
axis.plot(x, y)
axis.set_xlim(0, 2*l)
axis.set_ylim(0, 1.2)
axis.set_title('Kastenfunction mit Cos-Kanten')

y_fft = np.fft.fft(y)

fig = plt.figure()
axis = fig.add_subplot(1, 1, 1)
axis.plot(x, np.abs(y_fft), 'k', label='abs')
axis.plot(x, np.real(y_fft), 'r', label='real')
axis.plot(x, np.imag(y_fft), 'b', label='imag')
axis.set_xlim(0, 200)
axis.legend()
axis.set_title('Kastenfunction mit Cos-Kanten')
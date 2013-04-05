# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:24:40 2013

@author: Jan
"""

''' QUESTIONS
Array(zDim,yDim,xDim)? Transpose?
Does Bo only define the absolute magnitude?
Do the read in data only describe the direction and relative magnitude (normed?)

'''

import numpy as np
from numpy import pi, e
import matplotlib.pyplot as pp
import math
#from pylab import *

'''VARIABLES'''
b0 = 1
v0 = 0
Vacc = 30000
scale=1E-7
'''CONSTANTS'''
phi0 = 2067.83
filename = 'ATHLETICS Test.txt'

'''READ DATA'''
data = np.genfromtxt(filename, skip_header=2)
xDim, yDim, zDim = np.genfromtxt(filename, dtype=int, skip_header=1, skip_footer=len(data[:,0]))

res = (data[1,0] - data[0,0]) / scale
xLen, yLen, zLen = [data[-1,i]/scale+res/2 for i in range(3)]
xMag, yMag, zMag = [data[:,i].reshape(zDim,yDim,xDim).mean(0) for i in range(3,6)] #.T because x varies first
#Reshape in Python and Igor is different, Python fills rows first, Igor columns!

#TODO interpolation (redefine xDim,yDim,zDim) thickness ramp

xBigMag = np.zeros((2*xDim,2*yDim))
yBigMag = np.zeros((2*xDim,2*yDim))
xBigMag[xDim/2:3*xDim/2,yDim/2:3*yDim/2] = xMag * zLen
yBigMag[xDim/2:3*xDim/2,yDim/2:3*yDim/2] = yMag * zLen

'''COMPUTATION (FOURIER)'''
print xMag * zLen

xMagFFT = np.fft.fftshift(np.fft.rfft2(xBigMag.T), axes=0) #TODO Why transponed?
yMagFFT = np.fft.fftshift(np.fft.rfft2(yBigMag.T), axes=0)

sx2, sy2 = np.meshgrid(np.linspace(0, 0.05, xMagFFT.shape[1]), #TODO Why?
                       np.linspace(-0.05, 0.05, xMagFFT.shape[0]))

Ak = -(1j * b0) / (2 * phi0) * (xMagFFT * sy2 - yMagFFT * sx2) / (sx2 ** 2 + sy2 ** 2 + 1e-18) #TODO Why -?

Ar = np.fft.irfft2(np.fft.ifftshift(Ak, axes=0))
#phase = np.array([[math.atan(Ak[i,j].real/(Ak[i,j]+10**-18).imag) 
#        for i in range(2*xDim)] 
#    for j in range(2*xDim)])
pp.pcolormesh(Ar, cmap='Greys')
pp.colorbar()
pp.xlabel('x-axis')
pp.ylabel('y-axis')
pp.show()


#TODO Electrostatic Component

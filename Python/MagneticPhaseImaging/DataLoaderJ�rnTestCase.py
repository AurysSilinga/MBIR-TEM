# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:24:40 2013

@author: Jan
"""

'''IMPORT'''
import numpy as np
import matplotlib.pyplot as pp

'''CONSTANTS'''
phi0 = 2067.83

'''INPUT VARIABLES'''
b0 = 1
v0 = 0
Vacc = 30000
scale=1E-7
padFactor = 2
filename = '2D_hom_Filament.txt'

'''READ DATA'''
data = np.genfromtxt(filename, skip_header=2)
xDim, yDim, zDim = np.genfromtxt(filename, dtype=int, skip_header=1, skip_footer=len(data[:,0]))
res = (data[1,0] - data[0,0]) / scale
xLen, yLen, zLen = [data[-1,i]/scale+res/2 for i in range(3)]
xMag, yMag, zMag = [data[:,i].reshape(zDim,yDim,xDim).mean(0)*zLen for i in range(3,6)]
#Reshape in Python and Igor is different, Python fills rows first, Igor columns!
#TODO interpolation (redefine xDim,yDim,zDim) thickness ramp
xPad = xDim/2*(padFactor-1)
yPad = yDim/2*(padFactor-1)
xBigMag = np.zeros((padFactor*yDim,padFactor*xDim))
yBigMag = np.zeros((padFactor*yDim,padFactor*xDim))
xBigMag[yPad:yPad+yDim,xPad:xPad+xDim] = xMag
yBigMag[yPad:yPad+yDim,xPad:xPad+xDim] = yMag

'''COMPUTATION MAGNETIC PHASE SHIFT (FOURIER)'''
xMagFFT = np.fft.fftshift(np.fft.rfft2(xBigMag), axes=0)
yMagFFT = np.fft.fftshift(np.fft.rfft2(yBigMag), axes=0)
sx2, sy2 = np.meshgrid(np.linspace(0, 0.05, xMagFFT.shape[1]), #TODO Why?
                       np.linspace(-0.05, 0.05, xMagFFT.shape[0]))
Ak = -(1j * b0) / (2 * phi0) * (xMagFFT * sy2 - yMagFFT * sx2) / (sx2 ** 2 + sy2 ** 2 + 1e-18) #TODO Why -?
Ar = np.fft.irfft2(np.fft.ifftshift(Ak, axes=0))

#TODO Electrostatic Component

'''PLOT'''
pp.pcolormesh(Ar, cmap='Greys')
pp.colorbar()
pp.xlabel('x-axis')
pp.ylabel('y-axis')
pp.axis('equal')
pp.show()
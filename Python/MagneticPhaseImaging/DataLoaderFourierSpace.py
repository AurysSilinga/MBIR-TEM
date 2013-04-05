# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:24:40 2013

@author: Jan
c"""

'''IMPORT'''
import numpy as np
import matplotlib.pyplot as plt
import math as ma

'''CONSTANTS'''
phi0 = 2067.83          #magnetic flux in T*nm²

'''INPUT VARIABLES'''
b0 = 1.0                #in T
v0 = 0                  #TODO in ???
Vacc = 30000            #in V
scale= 1.0E-9 / 1.0E-2  #from cm to nm
padFactor = 1
filename = '2D_hom_Test1.txt'

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
Ny = 1 / res    #nyquist frequency
x =           np.linspace( 0, Ny/2, xMagFFT.shape[1])
y = np.linspace( -Ny/2, Ny/2,   xMagFFT.shape[0]+1)[:-1]  #TODO Why?
xx, yy = np.meshgrid(x, y)
                     
Ak = -(1j * b0) / (2 * phi0) * (xMagFFT * yy - yMagFFT * xx) / (xx ** 2 + yy ** 2 + 1e-18) #TODO Why -?
Ar = np.fft.irfft2(np.fft.ifftshift(Ak, axes=0))

#TODO Electrostatic Component

CosPhase = np.cos(Ar)

'''PLOT'''
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
plt.pcolormesh(Ar, cmap='Greys')

ticks = ax.get_xticks()*res
ax.set_xticklabels(ticks.astype(int))

ticks = ax.get_yticks()*res
ax.set_yticklabels(ticks.astype(int))

ax.set_title('Fourier Space Approach')
ax.set_xlabel('x-axis [nm]')
ax.set_ylabel('y-axis [nm]')
plt.colorbar()
plt.show()
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
import matplotlib.pyplot as pp
#from pylab import *

'''VARIABLES'''
b0 = 1
v0 = 0
Vacc = 30000
scale=1E-7
'''CONSTANTS'''
e = 2.71828182846
pi = 3.14159265359
phi0 = 2067.83
filename = 'ATHLETICS Test.txt'

'''READ DATA'''
data = np.genfromtxt(filename, skip_header=2)
xDim, yDim, zDim = np.genfromtxt(filename, dtype=int, skip_header=1, skip_footer=len(data[:,0]))

res = (data[1,0]-data[0,0])/scale
xLen, yLen, zLen = [data[-1,i]/scale+res/2 for i in range(0,3)]
xMag, yMag, zMag = [data[:,i].reshape(zDim,yDim,xDim).mean(0).T for i in range(3,6)] #.T because x varies first

#TODO interpolation (redefine xDim,yDim,zDim) thickness ramp

xBigMag = np.zeros((2*xDim,2*yDim))
yBigMag = np.zeros((2*xDim,2*yDim))
xBigMag[xDim/2:3*xDim/2,yDim/2:3*yDim/2] = xMag*zLen
yBigMag[xDim/2:3*xDim/2,yDim/2:3*yDim/2] = yMag*zLen

# UNTIL HERE EVERYTHING IS OKAY IN COMPARISON WITH IGOR

'''COMPUTATION (FOURIER)'''
xMagFFT = np.fft.rfft2(xBigMag)
yMagFFT = np.fft.rfft2(yBigMag)

#TODO define meshgrid and omit [kx,ky] in the following: x,y = meshgrid(arange(), arange())
Ak = np.array([[1j*b0/phi0 * (xMagFFT[kx,ky]*ky-yMagFFT[kx,ky]*kx) / (kx**2+ky**2+10**-18)
        for kx in range(xDim)] #TODO for kx, x in enumerate(arange(x0,x1,xd))]
    for ky in range(yDim)])
AkShifted = abs(np.fft.fftshift(Ak)) #for debugging only, shifts zero frequ to the center
Ar = np.fft.irfft2(Ak)
#phase = np.array([[math.atan(Ak[i,j].real/(Ak[i,j]+10**-18).imag) 
#        for i in range(2*xDim)] 
#    for j in range(2*xDim)])
pp.pcolormesh(Ar, cmap='Greys')
pp.colorbar()
pp.show()

A, B = [], []
for x,y in zip(range(len(B)), B):
    print x, y



#TODO Electrostatic Component
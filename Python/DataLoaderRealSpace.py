# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:24:40 2013

@author: Jan
"""

'''IMPORT'''
import numpy as np
from numpy import pi, e
import matplotlib.pyplot as plt
import math as ma

'''CONSTANTS'''
PHI_0 = 2067.83

'''INPUT VARIABLES'''
b0 = 1.0
v0 = 0
Vacc = 30000
scale=1E-7
padFactor = 2
filename = 'output.txt'

'''READ DATA'''
data = np.genfromtxt(filename, skip_header=2)
xDim, yDim, zDim = np.genfromtxt(filename, dtype=int, skip_header=1, skip_footer=len(data[:,0]))
res = (data[1,0] - data[0,0]) / scale
xLen, yLen, zLen = [data[-1,i]/scale+res/2 for i in range(3)]
xMag, yMag, zMag = [data[:,i].reshape(zDim,yDim,xDim).mean(0)*zLen for i in range(3,6)]
#Reshape in Python and Igor is different, Python fills rows first, Igor columns!
#TODO interpolation (redefine xDim,yDim,zDim) thickness ramp

'''COMPUTATION MAGNETIC PHASE SHIFT (REAL SPACE) SLAB'''
#    def F0(x,y):
#        a = np.log( x**2 + y**2 )
#        b = np.arctan( x / y )
#        return x*a - 2*x + 2*y*b   
#      
#    coeff = b0 * res / ( 4 * PHI_0 )
#      
#    def phiMag(x,y):
#        return coeff * ( - np.cos(beta) * ( F0(x-x0-Lx/2,y-y0-Ly/2)
#                                           -F0(x-x0+Lx/2,y-y0-Ly/2)
#                                           -F0(x-x0-Lx/2,y-y0+Ly/2)
#                                           +F0(x-x0+Lx/2,y-y0+Ly/2) )
#                         + np.sin(beta) * ( F0(y-y0-Ly/2,x-x0-Lx/2)
#                                           -F0(y-y0+Ly/2,x-x0-Lx/2)
#                                           -F0(y-y0-Ly/2,x-x0+Lx/2)
#                                           +F0(y-y0+Ly/2,x-x0+Lx/2) ) )
#    
#    return phiMag(xx, yy)






def F(n,m):
    a = np.log( res**2/4 * ( (2*n-1)**2 + (2*m-1)**2 ) )
    b = np.arctan( (2*n-1) / (2*m-1) )
    return res * ( (n-1/2)*a - (2*n-1) + (2*m-1)*b )
    
coeff = b0 / ( 4 * PHI_0 )

beta = np.arctan2(yMag, xMag)

absMag = np.sqrt(xMag**2 + yMag**2)

def phi_ij(i,j,p,q):
    return absMag[j,i] * ( - np.cos(beta[j,i]) * (F(p-i,q-j)-F(p-i+1,q-j)
                                                 -F(p-i,q-j+1)+F(p-i+1,q-j+1))
                           + np.sin(beta[j,i]) * (F(q-j,p-i)-F(q-j+1,p-i)
                                                 -F(q-j,p-i+1)+F(q-j+1,p-i+1)) )

phaseMag = np.zeros((yDim,xDim))



for q in range(yDim):
    for p in range(xDim):
        for j in range(yDim):
            for i in range(xDim):
                phaseMag[q,p] += phi_ij(i,j,p,q)
                
phaseMag *= coeff

#TODO Electrostatic Component

'''PLOT'''
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
plt.pcolormesh(phaseMag, cmap='gray')

ticks = ax.get_xticks()*res
ax.set_xticklabels(ticks.astype(int))

ticks = ax.get_yticks()*res
ax.set_yticklabels(ticks.astype(int))

ax.set_title('Real Space Approach')
ax.set_xlabel('x-axis [nm]')
ax.set_ylabel('y-axis [nm]')
plt.colorbar()
plt.show()
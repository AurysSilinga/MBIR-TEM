# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:24:40 2013

@author: Jan
"""

'''IMPORT'''
from pylab import *
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
#from matplotlib.ticker import FormatStrFormatter

'''CONSTANTS'''
phi0 = 2.06783E-15 #in Wb=Tm²

'''INPUT VARIABLES'''
b0 = 1.0 #in T
v0 = 0
Vacc = 30000
xDim, yDim = 20, 30
res = 10E-9 #in nm
beta = 300./360.*2*pi
x0 = res*xDim*0.5
y0 = res*yDim*0.5
Lx = res*xDim*0.5
Ly = res*yDim*0.5

'''CREATE COORDINATE GRIDS'''
x = np.linspace(res/2,xDim*res-res/2,num=xDim)
y = np.linspace(res/2,yDim*res-res/2,num=yDim)
#xx, yy = np.meshgrid(x,y)

'''COMPUTATION MAGNETIC PHASE SHIFT (REAL SPACE)'''
def F0(x,y):
    a = np.log( x**2 + y**2 )
    b = np.arctan2( x, y )              #atan or atan2?
    return x*a - 2*x + 2*y*b   
  
coeff = b0 * res / ( 4 * phi0 )
  
def phiMag(x,y):
    return coeff * ( - np.cos(beta) * ( F0(x-x0-Lx/2,y-y0-Ly/2)-F0(x-x0+Lx/2,y-y0-Ly/2)
                                       -F0(x-x0-Lx/2,y-y0+Ly/2)+F0(x-x0+Lx/2,y-y0+Ly/2) )
                     + np.sin(beta) * ( F0(y-y0-Ly/2,x-x0-Lx/2)-F0(y-y0+Ly/2,x-x0-Lx/2)
                                       -F0(y-y0-Ly/2,x-x0+Lx/2)+F0(y-y0+Ly/2,x-x0+Lx/2) ) )
                                       
def Test(x,y):
    return ( abs(x-x0)<Lx/2 and abs(y-y0)<Ly/2 )
    
xx, yy = np.meshgrid(x,y)

phaseMag = phiMag(xx, yy)

CosPhase = np.cos(phaseMag)
    
'''PLOT'''
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
plt.pcolormesh(CosPhase, cmap='Greys')
#xticks = ax.get_xticks()*res
#ax.set_xticklabels(xticks)
#yticks = ax.get_yticks()*res
#ax.set_yticklabels(yticks)
ax.set_title('Analytical Solution')
ax.set_xlabel('x-axis [nm]')
ax.set_ylabel('y-axis [nm]')
plt.colorbar()
plt.show()
print 'blabla'
print('blabla')
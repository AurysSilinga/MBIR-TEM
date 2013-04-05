# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:40:01 2013

@author: Jan

Create simple LLG Files which describe magnetization in 2D (z-Dim=1)
"""

'''IMPORT'''
import numpy as np
import math as m
from numpy import pi

'''VARIABLES'''
xDim, yDim = 10, 12
res = 1.0
beta = pi/4
xWidth = xDim*0.8
yWidth = yDim*0.8
r = xDim*0.4
description = 'homog_mag_in_x'

'''FUNCTIONS'''
def inBound(xPos,yPos):
    return True

'''CREATE COORDINATE GRIDS'''
x = np.linspace(res/2,xDim*res-res/2,num=xDim)
y = np.linspace(res/2,yDim*res-res/2,num=yDim)
xx, yy = np.meshgrid(x,y)

'''CREATE MAGNETISATION GRID'''
#Shape of the magnetization
xCenter = res*xDim/2
yCenter = res*yDim/2
shapeMag = np.array([[m.sqrt((xi-xCenter)**2+(yi-yCenter)**2)<r 
                        for yi in y]
                    for xi in x])
#Direction of the magnetization
xMag = np.array(np.ones((xDim,yDim)))*m.cos(beta)*shapeMag
yMag = np.array(np.ones((xDim,yDim)))*m.sin(beta)*shapeMag





#xMag = 1 * inBound(x,y)
               
#'''RESHAPE INTO VECTORS'''
#x = np.reshape(xx,(-1))
#y = np.reshape(yy,(-1))
#z = np.array(np.ones(xDim*yDim)*res/2)       
#xMag   = np.reshape(xMag,(-1))
#yMag   = np.reshape(yMag,(-1))
#zMag   = np.array(np.zeros(xDim*yDim))   
#
#'''SAVE DATA TO FILE'''       
#data = np.array([x, y, z, xMag, yMag, zMag]).T
#header = 'LLGFileCreator2D: '+description+'\n' \
#          +'    '+str(xDim)+'    '+str(yDim)+'    '+str(1)
#with open(description+'.txt','w') as f:
#    print>>f, str(header)
#    print>>f, '\n'.join('   '.join('{:7.6e}'.format(cell) for cell in row) for row in data)
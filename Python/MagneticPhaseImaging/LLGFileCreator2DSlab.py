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
xDim, yDim = 10, 10
res = 1.0E-6
beta = pi/4
xCenter = res*xDim/2
yCenter = res*yDim/2
xWidth = res*xDim/2
yWidth = res*yDim/2
description = '2D_hom_Slab'

'''CREATE COORDINATE GRIDS'''
x = np.linspace(res/2,xDim*res-res/2,num=xDim)
y = np.linspace(res/2,yDim*res-res/2,num=yDim)
xx, yy = np.meshgrid(x,y)

'''CREATE MAGNETISATION GRID'''
#Shape of the magnetization
shapeMag = np.array([[abs(xi-xCenter)<xWidth/2 and abs(yi-yCenter)<yWidth/2
                        for xi in x]
                    for yi in y])
#Direction of the magnetization
xMag = np.array(np.ones((yDim,xDim)))*m.cos(beta)*shapeMag
yMag = np.array(np.ones((yDim,xDim)))*m.sin(beta)*shapeMag
               
'''RESHAPE INTO VECTORS'''
xx = np.reshape(xx,(-1))
yy = np.reshape(yy,(-1))
zz = np.array(np.ones(xDim*yDim)*res/2)
xMag   = np.reshape(xMag,(-1))
yMag   = np.reshape(yMag,(-1))
zMag   = np.array(np.zeros(xDim*yDim))   

'''SAVE DATA TO FILE'''       
data = np.array([xx, yy, zz, xMag, yMag, zMag]).T
header = 'LLGFileCreator2D: '+description+'\n' \
          +'    '+str(xDim)+'    '+str(yDim)+'    '+str(1)
with open(description+'.txt','w') as f:
    print>>f, str(header)
    print>>f, '\n'.join('   '.join('{:7.6e}'.format(cell) for cell in row) for row in data)
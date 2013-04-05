# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:40:01 2013

@author: Jan

Create simple LLG Files which describe magnetization in 2D (z-Dim=1)
"""

import numpy as np

xDim, yDim = 5,4
res = 1E-6

dataMag, dataCoord = np.array(np.zeros((xDim,yDim,zDim))), np.array(np.zeros((xDim,yDim,zDim)))

x, y, z = np.meshgrid(np.linspace(0, 0.05, xMagFFT.shape[1]), #TODO Why?
                       np.linspace(-0.05, 0.05, xMagFFT.shape[0]))

dataCoord = np.arange(7)








#'''VARIABLES'''
#b0 = 1
#v0 = 0
#Vacc = 30000
#scale=1E-7
#'''CONSTANTS'''
#phi0 = 2067.83
#filename = 'ATHLETICS Test.txt'
#
#'''READ DATA'''
#data = np.genfromtxt(filename, skip_header=2)
#xDim, yDim, zDim = np.genfromtxt(filename, dtype=int, skip_header=1, skip_footer=len(data[:,0]))
#
#res = (data[1,0] - data[0,0]) / scale
#xLen, yLen, zLen = [data[-1,i]/scale+res/2 for i in range(3)]
#xMag, yMag, zMag = [data[:,i].reshape(zDim,yDim,xDim).mean(0) for i in range(3,6)] #.T because x varies first
##Reshape in Python and Igor is different, Python fills rows first, Igor columns!
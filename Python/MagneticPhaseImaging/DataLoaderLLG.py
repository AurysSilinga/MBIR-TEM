# -*- coding: utf-8 -*-
"""
Created on Wed Apr 03 11:14:03 2013

@author: Jan
"""


class MagnetizationDataLLG:
    '''An object storing magnetization data loaded from a LLG-file'''
    
    def __init__(self, filename):
        '''load magnetization in LLG-file format'''
        import numpy as np
        scale = 1.0E-9 / 1.0E-2  #from cm to nm
        data = np.genfromtxt(filename, skip_header=2)
        x_dim, y_dim, z_dim = np.genfromtxt(filename, dtype=int, 
                                            skip_header=1, 
                                            skip_footer=len(data[:, 0]))
        res = (data[1, 0] - data[0, 0]) / scale
        x_len, y_len, z_len = [data[-1, i]/scale+res/2 for i in range(3)]
        x_mag, y_mag, z_mag = [data[:, i].reshape(z_dim, y_dim, x_dim).mean(0) 
                               *z_len for i in range(3,6)]
        #Reshape in Python and Igor is different, 
        #Python fills rows first, Igor columns!
        self.filename = filename
        self.res = res
        self.dim = (x_dim, y_dim, z_dim)
        self.length = (x_len, y_len, z_len)
        self.magnitude = (x_mag, y_mag, z_mag)
    
    def __str__(self):
        '''return filename'''
        return self.filename
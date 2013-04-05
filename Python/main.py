# -*- coding: utf-8 -*-
"""
Created on Wed Apr 03 11:15:38 2013

@author: Jan
"""


import magneticimaging.magcreator as mc
import magneticimaging.dataloader as dl
import magneticimaging.phasemap as pm
from numpy import pi
    

def phase_from_mag():
    '''Calculate and display the phase map from a given magnetization.
    Arguments:
        None
    Returns:
        None
    
    '''
    # TODO: Input via GUI
    filename = 'output.txt'
    b_0 = 1.0  # in T
    v_0 = 0  # TODO: units?
    v_acc = 3000  # in V
    padding = 0
    
    dim = (30, 30)  # in px (y,x)
    res = 10  # in nm
    beta = pi/4
    
    shape_fun = mc.slab
    # Slab:
    center = (15, 15)  # in px (y,x)
    width = (15, 15)  # in px (y,x)
    # Disc:    
    radius = 15
    # Filament:
    pos = 5
    # Alternating Filaments:
    spacing = 5
    # Single Pixel
    pixel = (5, 5)
    
    
    # create magnetic distribution:
    mc.create_hom_mag(dim, res, beta, filename, shape_fun, center, width)
    
    # load magnetic distribution:
    mag_data = dl.MagDataLLG(filename)

    # calculate phase map:
    phase = pm.fourier_space(mag_data, b_0, padding)    
    
    # display phase map:
    pm.display(phase, mag_data.res, 'Fourier Space Approach')
  
  
if __name__ == "__main__":
    phase_from_mag()
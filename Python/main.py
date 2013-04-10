# -*- coding: utf-8 -*-
"""
Created on Wed Apr 03 11:15:38 2013

@author: Jan
"""


import magneticimaging.magcreator as mc
import magneticimaging.dataloader as dl
import magneticimaging.phasemap as pm
import magneticimaging.holoimage as hi
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
    v_acc = 30000  # in V
    padding = 0
    density = 1
    
    dim = (500, 500)  # in px (y,x)
    res = 10  # in nm
    beta = pi/4
    
    shape_fun = mc.disc
    # Slab:
    center = (250, 250)  # in px (y,x)
    width = (200, 200)  # in px (y,x)
    # Disc:    
    radius = 100
    # Filament:
    pos = 5
    # Alternating Filaments:
    spacing = 5
    # Single Pixel
    pixel = (5, 5)
    
    plot_mag_density = False
    
    # create magnetic distribution:
    mc.create_hom_mag(dim, res, beta, filename, plot_mag_density, 
                      shape_fun, center, radius)
    
    # load magnetic distribution:
    mag_data = dl.MagDataLLG(filename)

    # calculate phase map:
    phase = pm.fourier_space(mag_data, b_0, padding)    
    
    # display phase map:
    pm.display(phase, mag_data.res, 'Fourier Space Approach')
    
    # display cosine of the phase map:
    pm.display_cos(phase, mag_data.res, density, 'Fourier Space Approach')
  
    hi.holo_image(phase, mag_data.res, density, 'Fourier Space Approach')
    
    hi.make_color_wheel()
  
if __name__ == "__main__":
    phase_from_mag()
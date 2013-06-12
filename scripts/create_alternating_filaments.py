# -*- coding: utf-8 -*-
"""Create magnetic distribution of alternating filaments"""


import pdb, traceback, sys
import numpy as np
from numpy import pi

import pyramid.magcreator as mc
from pyramid.magdata import MagData


def create_alternating_filaments():
    '''Calculate, display and save the magnetic distribution of alternating filaments to file.
    Arguments:
        None
    Returns:
        None
    
    '''
    # Input parameters:
    filename = '../output/mag_dist_alt_filaments.txt'    
    dim = (1, 21, 21)  # in px (z, y, x)
    res = 10.0  # in nm
    phi = pi/2       
    spacing = 5    
    # Create lists for magnetic objects:
    count = int((dim[1]-1) / spacing) + 1
    mag_shape_list = np.zeros((count,) + dim)
    phi_list      = np.zeros(count)
    for i in range(count):  
        pos = i * spacing
        mag_shape_list[i] = mc.Shapes.filament(dim, (0, pos))
        phi_list[i] = (1-2*(int(pos/spacing)%2)) * phi
    # Create magnetic distribution
    magnitude = mc.create_mag_dist_comb(mag_shape_list, phi_list) 
    mag_data = MagData(res, magnitude)
    mag_data.quiver_plot()
    mag_data.save_to_llg(filename)    
    

if __name__ == "__main__":
    try:
        create_alternating_filaments()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
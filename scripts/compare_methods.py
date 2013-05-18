# -*- coding: utf-8 -*-
"""Compare the different methods to create phase maps."""


import time
import pdb, traceback, sys
import numpy as np
from numpy import pi

import pyramid.magcreator  as mc
import pyramid.projector   as pj
import pyramid.phasemapper as pm
import pyramid.holoimage   as hi
import pyramid.analytic    as an
from pyramid.magdata  import MagData
from pyramid.phasemap import PhaseMap


def phase_from_mag():
    '''Calculate and display the phase map from a given magnetization.
    Arguments:
        None
    Returns:
        None
    
    '''
    # Input parameters:
    b_0     =  1    # in T
    res     = 10.0  # in nm
    beta    = pi/4
    padding = 20
    density = 10
    dim = (1, 50, 50)  # in px (z, y, x)    
    # Create magnetic shape:
    geometry = 'slab'        
    if geometry == 'slab':
        center = (0, 24, 24)  # in px (z, y, x) index starts with 0!
        width  = (1, 25, 25)  # in px (z, y, x)
        mag_shape = mc.Shapes.slab(dim, center, width)
        phase_ana = an.phasemap_slab(dim, res, beta, center, width, b_0)
    elif geometry == 'disc':
        radius = 12.5  # in px 
        height =  1    # in px
        mag_shape = mc.Shapes.disc(dim, center, radius, height)
        phase_ana = an.phasemap_disc(dim, res, beta, center, radius, b_0)
    # Project the magnetization data:    
    mag_data = MagData(res, mc.create_mag_dist(mag_shape, beta))    
    projection = pj.simple_axis_projection(mag_data)
    # Construct phase maps:
    phase_map_ana  = PhaseMap(res, phase_ana)
    phase_map_fft  = PhaseMap(res, pm.phase_mag_fourier(res, projection, b_0, padding))
    phase_map_slab = PhaseMap(res, pm.phase_mag_real(res, projection, 'slab', b_0))
    phase_map_disc = PhaseMap(res, pm.phase_mag_real(res, projection, 'disc', b_0))
    # Display the combinated plots with phasemap and holography image:
    hi.display_combined(phase_map_ana,  density, 'Analytic Solution')
    hi.display_combined(phase_map_fft,  density, 'Fourier Space')
    hi.display_combined(phase_map_slab, density, 'Real Space (Slab)')
    hi.display_combined(phase_map_disc, density, 'Real Space (Disc)')
    # Display all phase maps:
    phase_map_ana.display('Analytic Solution')
    phase_map_fft.display('Fourier Space')
    phase_map_slab.display('Real Space (Slab)')
    phase_map_disc.display('Real Space (Disc)')
    # Plot differences to the analytic solution:
    phase_map_diff_fft  = PhaseMap(res, phase_map_ana.phase-phase_map_fft.phase)
    phase_map_diff_slab = PhaseMap(res, phase_map_ana.phase-phase_map_slab.phase)
    phase_map_diff_disc = PhaseMap(res, phase_map_ana.phase-phase_map_disc.phase)
    RMS_fft  = phase_map_diff_fft.phase
    RMS_slab = phase_map_diff_slab.phase
    RMS_disc = phase_map_diff_disc.phase
    phase_map_diff_fft.display('Fourier Space (RMS = {})'.format(np.std(RMS_fft)))
    phase_map_diff_slab.display('Real Space (Slab) (RMS = {})'.format(np.std(RMS_slab)))
    phase_map_diff_disc.display('Real Space (Disc) (RMS = {})'.format(np.std(RMS_disc)))
    
    
if __name__ == "__main__":
    try:
        phase_from_mag()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
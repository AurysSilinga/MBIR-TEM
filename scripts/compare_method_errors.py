# -*- coding: utf-8 -*-
"""Compare the different methods to create phase maps."""


import time
import pdb, traceback, sys
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

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
    betas    = pi/4#np.linspace(0, 2*pi, endpoint=False, num=72)
    paddings = range(26)
    dim = (1, 100, 100)  # in px (z, y, x)    
    # Create magnetic shape:
    geometry = 'slab'        
    if geometry == 'slab':
        center = (0, 49, 49)  # in px (z, y, x) index starts with 0!
        width  = (1, 50, 50)  # in px (z, y, x)
        mag_shape = mc.Shapes.slab(dim, center, width)
        phase_ana = an.phase_mag_slab(dim, res, beta, center, width, b_0)
    elif geometry == 'disc':
        center = (0, 49, 49)  # in px (z, y, x) index starts with 0!
        radius = 25  # in px 
        height =  1  # in px
        mag_shape = mc.Shapes.disc(dim, center, radius, height)
        phase_ana = an.phase_mag_disc(dim, res, beta, center, radius, b_0)
    # Project the magnetization data:    
    mag_data = MagData(res, mc.create_mag_dist(mag_shape, beta))    
    projection = pj.simple_axis_projection(mag_data)
   
    '''FOURIER'''
    #padding
    RMS = np.zeros(len(paddings))
    duration = np.zeros(len(paddings))
    for (i, padding) in enumerate(paddings):
        print 'padding =', padding[i]
        start_time = time.time()
        phase_num  = pm.phase_mag_fourier(res, projection, b_0, padding[i])
        duration[i] = time.time() - start_time
        phase_diff = phase_ana - phase_num
        RMS[i]  = np.std(phase_diff)
    
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(padding, RMS)
    
    fig = plt.figure()
    fig.add_subplot(111)    
    plt.plot(padding, duration)
    
    
    '''REAL SLAB '''
    beta    = np.linspace(0, 2*pi, endpoint=False, num=72)
    
    RMS = np.zeros(len(beta))
    for i in range(len(beta)):
        print 'beta =', round(360*beta[i]/(2*pi))
        mag_data = MagData(res, mc.create_mag_dist(mag_shape, beta[i]))
        projection = pj.simple_axis_projection(mag_data)
        phase_num  = pm.phase_mag_real(res, projection, 'slab', b_0)
        phase_ana  = an.phase_mag_slab(dim, res, beta[i], center, width, b_0)
        phase_diff = phase_ana - phase_num
        RMS[i]  = np.std(phase_diff)
    
    fig = plt.figure()
    fig.add_subplot(111)    
    plt.plot(round(360*beta/(2*pi)), RMS)
    
      
#    phase_map_slab = PhaseMap(res, pm.phase_mag_real(res, projection, 'slab', b_0))
#    phase_map_disc = PhaseMap(res, pm.phase_mag_real(res, projection, 'disc', b_0))
#    # Display the combinated plots with phasemap and holography image:
#    hi.display_combined(phase_map_ana,  density, 'Analytic Solution')
#    hi.display_combined(phase_map_fft,  density, 'Fourier Space')
#    hi.display_combined(phase_map_slab, density, 'Real Space (Slab)')
#    hi.display_combined(phase_map_disc, density, 'Real Space (Disc)')
#
#    # Plot differences to the analytic solution:
#    
#    phase_map_diff_slab = PhaseMap(res, phase_map_ana.phase-phase_map_slab.phase)
#    phase_map_diff_disc = PhaseMap(res, phase_map_ana.phase-phase_map_disc.phase)
#    
#    RMS_slab = phase_map_diff_slab.phase
#    RMS_disc = phase_map_diff_disc.phase

    
    
if __name__ == "__main__":
    try:
        phase_from_mag()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
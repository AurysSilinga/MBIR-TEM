# -*- coding: utf-8 -*-
"""
Created on Wed Apr 03 11:15:38 2013

@author: Jan
"""

import numpy as np
import matplotlib.pyplot as plt
import pyramid.magcreator as mc
import pyramid.dataloader as dl
import pyramid.phasemap as pm
import pyramid.holoimage as hi
import pyramid.analytic as an
import time
import pdb, traceback, sys
from numpy import pi


def phase_from_mag():
    '''Calculate and display the phase map from a given magnetization.
    Arguments:
        None
    Returns:
        None
    
    '''
    # TODO: Input via GUI
    b_0 = 1.0  # in T
    v_0 = 0  # TODO: units?
    v_acc = 30000  # in V
    padding = 20
    density = 10
    
    dim = (50, 50)  # in px (y,x)
    res = 10.0  # in nm
    beta = pi/4
    
    center = (24, 24)  # in px (y,x) index starts with 0!
    width  = (25, 25)  # in px (y,x)
    radius = 12.5
    
    filename = '../output/output.txt'
    geometry = 'slab'        
    
    if geometry == 'slab':
        mag_shape = mc.slab(dim, center, width)
        phase_ana = an.phasemap_slab(dim, res, beta, center, width, b_0)
    elif geometry == 'slab':
        mag_shape = mc.disc(dim, center, radius)
        phase_ana = an.phasemap_disc(dim, res, beta, center, radius, b_0)
    
    holo_ana  = hi.holo_image(phase_ana, res, density)
    display_combined(phase_ana, res, holo_ana, 'Analytic Solution')
    
    '''CREATE MAGNETIC DISTRIBUTION'''
    mc.create_hom_mag(dim, res, beta, mag_shape, filename)
    
    '''LOAD MAGNETIC DISTRIBUTION'''
    mag_data = dl.MagDataLLG(filename)
    
    # TODO: get it to work:
    phase_el = pm.phase_elec(mag_data, v_0, v_acc)    
    
    '''NUMERICAL SOLUTION'''
    # numerical solution Fourier Space:
    tic = time.clock()
    phase_fft = pm.fourier_space(mag_data, b_0, padding)   
    toc = time.clock()
    print 'Time for Fourier Space Approach:     ' + str(toc - tic)
    holo_fft = hi.holo_image(phase_fft, mag_data.res, density)
    display_combined(phase_fft, mag_data.res, holo_fft, 'Fourier Space Approach')
    
    # numerical solution Real Space (Slab):
    tic = time.clock()
    phase_slab = pm.real_space(mag_data, 'slab', b_0)
    toc = time.clock()
    print 'Time for Real Space Approach (Slab): ' + str(toc - tic)
    holo_slab = hi.holo_image(phase_slab, mag_data.res, density)
    display_combined(phase_slab, mag_data.res, holo_slab, 'Real Space Approach (Slab)')
    
    # numerical solution Real Space (Disc):
    tic = time.clock()
    phase_disc = pm.real_space(mag_data, 'disc', b_0)
    toc = time.clock()
    print 'Time for Real Space Approach (Disc): ' + str(toc - tic)
    holo_disc = hi.holo_image(phase_disc, mag_data.res, density)
    display_combined(phase_disc, mag_data.res, holo_disc, 'Real Space Approach (Disc)')
    
    '''DIFFERENCES'''
    diff_fft  = phase_fft  - phase_ana 
    diff_slab = phase_slab - phase_ana
    diff_disc = phase_disc - phase_ana
    rms_fft  = np.sqrt((diff_fft**2).mean())
    rms_slab = np.sqrt((diff_slab**2).mean())
    rms_disc = np.sqrt((diff_disc**2).mean())
    pm.display_phase(diff_fft,  res, 'FFT - Analytic (RMS = ' + '{:3.2e}'.format(rms_fft) + ')')
    pm.display_phase(diff_slab, res, 'Slab - Analytic (RMS = ' +'{:3.2e}'.format(rms_slab) + ')')
    pm.display_phase(diff_disc, res, 'Disc - Analytic (RMS = ' + '{:3.2e}'.format(rms_disc) + ')')
 
    
def display_combined(phase, res, holo, title):
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(title, fontsize=20)
    
    holo_axis = fig.add_subplot(1,2,1, aspect='equal')
    hi.display_holo(holo, 'Holography Image', holo_axis)
    
    phase_axis = fig.add_subplot(1,2,2, aspect='equal')
    pm.display_phase(phase, res, 'Phasemap', phase_axis)
    
    
if __name__ == "__main__":
    try:
        phase_from_mag()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
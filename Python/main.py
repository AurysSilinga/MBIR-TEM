# -*- coding: utf-8 -*-
"""
Created on Wed Apr 03 11:15:38 2013

@author: Jan
"""

import matplotlib.pyplot as plt
import magneticimaging.magcreator as mc
import magneticimaging.dataloader as dl
import magneticimaging.phasemap as pm
import magneticimaging.holoimage as hi
import magneticimaging.analytic as an
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
    
    '''INPUT'''
    # TODO: Input via GUI
    filename = 'output.txt'
    b_0 = 1.0  # in T
    v_0 = 0  # TODO: units?
    v_acc = 30000  # in V
    padding = 20
    density = 10
    
    dim = (50, 50)  # in px (y,x)
    res = 10.0  # in nm
    beta = pi/4
    
    plot_mag_distr = True
    
    # Slab:
    shape_fun = mc.slab
    center = (24, 24)  # in px (y,x) index starts with 0!
    width  = (25, 25)  # in px (y,x)
    params = (center, width)
#    # Disc:
#    shape_fun = mc.disc
#    center = (4, 4)  # in px (y,x)
#    radius = 2
#    params = (center, radius)
#    # Filament:
#    shape_fun = mc.filament
#    pos = 5
#    x_or_y = 'y'
#    params = (pos, x_or_y)
#    # Alternating Filaments:
#    shape_fun = mc.alternating_filaments
#    spacing = 5
#    x_or_y = 'y'
#    params = (spacing, x_or_y)
#    # Single Pixel:
#    shape_fun = mc.single_pixel
#    pixel = (5, 5)
#    params = pixel
    
    '''CREATE MAGNETIC DISTRIBUTION'''
    mc.create_hom_mag(dim, res, beta, shape_fun, params,
                      filename, plot_mag_distr)
    
    '''LOAD MAGNETIC DISTRIBUTION'''
    mag_data = dl.MagDataLLG(filename)
    
    '''COLOR WHEEL'''
    hi.make_color_wheel()

    '''NUMERICAL SOLUTION'''
    # numerical solution Fourier Space:
    ticf = time.clock()
    phase_fft = pm.fourier_space(mag_data, b_0, padding)   
    tocf = time.clock()
    print 'Time for Fourier Space Approach: ' + str(tocf - ticf)
    holo_fft = hi.holo_image(phase_fft, mag_data.res, density)
    display_combined(phase_fft, mag_data.res, holo_fft, 
                     'Fourier Space Approach')
    # numerical solution Real Space:
    ticr = time.clock()
    phase_real = pm.real_space(mag_data, b_0)
    tocr = time.clock()
    print 'Time for Real Space Approach:    ' + str(tocr - ticr)
    print 'Fourier Approach is ' + str((tocr-ticr) / (tocf-ticf)) + ' faster!'
    holo_real = hi.holo_image(phase_real, mag_data.res, density)
    display_combined(phase_real, mag_data.res, holo_real, 
                     'Real Space Approach')
    
    '''ANALYTIC SOLUTION'''
    # analytic solution slab:
    phase = an.phasemap_slab(dim, res, beta, center, width, b_0)
    holo  = hi.holo_image(phase, res, density)
    display_combined(phase, res, holo, 'Analytic Solution - Slab')
#    # analytic solution disc:
#    phase = an.phasemap_disc(dim, res, beta, center, radius, b_0)
#    holo  = hi.holo_image(phase, res, density)
#    display_combined(phase, res, holo, 'Analytic Solution - Disc')
#    # analytic solution sphere:
#    phase = an.phasemap_sphere(dim, res, beta, center, radius, b_0)
#    holo  = hi.holo_image(phase, res, density)
#    display_combined(phase, res, holo, 'Analytic Solution - Sphere')

    
    '''DIFFERENCES'''
    diff_real_to_ana = phase_real - phase
    diff_fft_to_ana  = phase_fft  - phase 
    diff_real_to_fft = phase_fft - phase_real
    pm.display_phase(diff_real_to_ana, res, 'Difference: Analytic - Real')
    pm.display_phase(diff_fft_to_ana,  res, 'Difference: Analytic - Fourier')
    pm.display_phase(diff_real_to_fft, res, 'Difference: Fourier - Real') 
 
    
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
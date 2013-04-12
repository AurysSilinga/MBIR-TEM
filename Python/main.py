# -*- coding: utf-8 -*-
"""
Created on Wed Apr 03 11:15:38 2013

@author: Jan
"""


import magneticimaging.magcreator as mc
import magneticimaging.dataloader as dl
import magneticimaging.phasemap as pm
import magneticimaging.holoimage as hi
import magneticimaging.analytic as an
import time
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
    
    dim = (40, 40)  # in px (y,x)
    res = 10.0  # in nm
    beta = pi/4
    
    plot_mag_distr = False
    
    # Slab:
    shape_fun = mc.slab
    center = (20, 20)  # in px (y,x)
    width = (20, 20)  # in px (y,x)
    params = (center, width)
#    # Disc:
#    shape_fun = mc.disc
#    center = (30, 30)  # in px (y,x)
#    radius = 15
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
    pm.display(phase_fft, mag_data.res, 'Fourier Space Approach')
    hi.holo_image(phase_fft, mag_data.res, density, 'Fourier Space Approach')
    # numerical solution Real Space:
    ticr = time.clock()
    phase_real = pm.real_space(mag_data, b_0)
    tocr = time.clock()
    print 'Time for Real Space Approach:    ' + str(tocr - ticr)
    print 'Fourier Approach is ' + str((tocr-ticr) / (tocf-ticf)) + ' faster!'
    pm.display(phase_real, mag_data.res, 'Real Space Approach')
    hi.holo_image(phase_real, mag_data.res, density, 'Real Space Approach')
    
    '''ANALYTIC SOLUTION'''
    # analytic solution slab:
    phase = an.phasemap_slab(dim, res, beta, center, width, b_0)
    pm.display(phase, res, 'Analytic Solution - Slab')
    hi.holo_image(phase, res, density, 'Analytic Solution - Slab')
#    # analytic solution disc:
#    phase = an.phasemap_disc(dim, res, beta, center, radius, b_0)
#    pm.display(phase, res, 'Analytic Solution - Disc')
#    hi.holo_image(phase, res, density, 'Analytic Solution - Disc')
#    # analytic solution sphere:
#    phase = an.phasemap_sphere(dim, res, beta, center, radius, b_0)
#    pm.display(phase, res, 'Analytic Solution - Sphere')
#    hi.holo_image(phase, res, density, 'Analytic Solution - Sphere')
    
    '''DIFFERENCES'''
    diff_real_to_ana = phase_real - phase
    diff_fft_to_ana  = phase_fft  - phase 
    diff_real_to_fft = phase_real - phase_fft
    pm.display(diff_real_to_ana, res, 'Difference: Analytic - Real')
    pm.display(diff_fft_to_ana,  res, 'Difference: Analytic - Fourier')
    pm.display(diff_real_to_fft, res, 'Difference: Real - Fourier')
    
    pass    
    
if __name__ == "__main__":
    phase_from_mag()
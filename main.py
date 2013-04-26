# -*- coding: utf-8 -*-
"""
Created on Wed Apr 03 11:15:38 2013

@author: Jan
"""

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
    
    '''INPUT'''
    # TODO: Input via GUI
    filename = 'output.txt'
    b_0 = 10.0  # in T
    v_0 = 0  # TODO: units?
    v_acc = 30000  # in V
    padding = 20
    density = 100
    
    dim = (100, 100)  # in px (y,x)
    res = 1.0  # in nm
    beta = pi/2
    
    plot_mag_distr = True
    
    # Slab:
    shape_fun = mc.slab
    center = (49, 49)  # in px (y,x) index starts with 0!
    width  = (50, 50)  # in px (y,x)
    params = (center, width)
#    # Disc:
#    shape_fun = mc.disc
#    center = (4, 4)  # in px (y,x)
#    radius = 2.5
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
    
    '''CREATE LOGO'''    
    mc.create_logo(128, res, beta, filename, plot_mag_distr)
    mag_data = dl.MagDataLLG(filename)
    phase, pixel_stub = pm.real_space_slab(mag_data, b_0)  
    holo = hi.holo_image(phase, mag_data.res, density)
    hi.display_holo(holo, '')
    
    '''CREATE MAGNETIC DISTRIBUTION'''
    mc.create_hom_mag(dim, res, beta, shape_fun, params,
                      filename, plot_mag_distr)
    
    '''LOAD MAGNETIC DISTRIBUTION'''
    mag_data = dl.MagDataLLG(filename)
    
    
    # TODO: get it to work:
    phase_el = pm.phase_elec(mag_data, v_0=1, v_acc=200000)    
    
    
    
    '''COLOR WHEEL'''
    hi.make_color_wheel()
    
    
    phase_stub, phi_cos_real_slab = pm.real_space_slab(mag_data, b_0)
    phase_stub, phi_cos_real_disc = pm.real_space_disc(mag_data, b_0) 
    phi_cos_diff = phi_cos_real_slab - phi_cos_real_disc
    pm.display_phase(phi_cos_diff, mag_data.res, 'Difference: One Pixel Slab - Disc')
    
    
    '''NUMERICAL SOLUTION'''
    # numerical solution Fourier Space:
    tic = time.clock()
    phase_fft = pm.fourier_space(mag_data, b_0, padding)   
    toc = time.clock()
    print 'Time for Fourier Space Approach:     ' + str(toc - tic)
    holo_fft = hi.holo_image(phase_fft, mag_data.res, density)
    display_combined(phase_fft, mag_data.res, holo_fft, 
                     'Fourier Space Approach')
    # numerical solution Real Space (Slab):
    tic = time.clock()
    phase_real_slab, pixel_stub = pm.real_space_slab(mag_data, b_0)
    toc = time.clock()
    print 'Time for Real Space Approach (Slab): ' + str(toc - tic)
    holo_real_slab = hi.holo_image(phase_real_slab, mag_data.res, density)
    display_combined(phase_real_slab, mag_data.res, holo_real_slab, 
                     'Real Space Approach (Slab)')
    # numerical solution Real Space (Disc):
    tic = time.clock()
    phase_real_disc, pixel_stub = pm.real_space_disc(mag_data, b_0)
    toc = time.clock()
    print 'Time for Real Space Approach (Disc): ' + str(toc - tic)
    holo_real_disc = hi.holo_image(phase_real_disc, mag_data.res, density)
    display_combined(phase_real_disc, mag_data.res, holo_real_disc,
                     'Real Space Approach (Disc)')
    
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
    diff_fft_to_ana       = phase_fft       - phase 
    diff_real_slab_to_ana = phase_real_slab - phase
    diff_real_disc_to_ana = phase_real_disc - phase
    diff_slab_to_disc     = phase_real_disc - phase_real_slab
    pm.display_phase(diff_fft_to_ana,       res, 'Difference: FFT - Analytic')
    pm.display_phase(diff_real_slab_to_ana, res, 'Difference: Slab - Analytic')
    pm.display_phase(diff_real_disc_to_ana, res, 'Difference: Disc - Analytic')
    pm.display_phase(diff_slab_to_disc,     res, 'Difference: Disc - Slab')
    
    # TODO: Delete
#    import pdb; pdb.set_trace()
 
    
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
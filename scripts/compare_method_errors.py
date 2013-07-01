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
import pyramid.analytic    as an
from pyramid.magdata  import MagData
from pyramid.phasemap import PhaseMap
import shelve


def phase_from_mag():
    '''Calculate and display the phase map from a given magnetization.
    Arguments:
        None
    Returns:
        None
    
    '''
    # Create / Open databank:
    data_shelve = shelve.open('../output/method_errors_shelve')
    
    '''FOURIER PADDING->RMS|DURATION'''
    # Parameters:
    b_0 =  1    # in T
    res = 10.0  # in nm
    dim = (1, 128, 128)    
    phi = -pi/4
    padding_list = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4,5, 6,7, 8,9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    geometry = 'disc'
    # Create magnetic shape:
    if geometry == 'slab':
        center = (0, dim[1]/2-0.5, dim[2]/2-0.5)  # in px (z, y, x) index starts with 0!
        width  = (1, dim[1]/2, dim[2]/2)  # in px (z, y, x)
        mag_shape = mc.Shapes.slab(dim, center, width)
        phase_ana = an.phase_mag_slab(dim, res, phi, center, width, b_0)
    elif geometry == 'disc':
        center = (0, dim[1]/2-0.5, dim[2]/2-0.5)  # in px (z, y, x) index starts with 0!
        radius = dim[1]/4  # in px 
        height =  1  # in px
        mag_shape = mc.Shapes.disc(dim, center, radius, height)
        phase_ana = an.phase_mag_disc(dim, res, phi, center, radius, b_0)
    # Project the magnetization data:    
    mag_data = MagData(res, mc.create_mag_dist(mag_shape, phi))    
    projection = pj.simple_axis_projection(mag_data)
    # Create data:
    data = np.zeros((3, len(padding_list)))
    data[0, :] = padding_list
    for (i, padding) in enumerate(padding_list):
        print 'padding =', padding_list[i]
        # Key:
        key = ', '.join(['Padding->RMS|duration', 'Fourier', 'padding={}'.format(padding_list[i]), 
                        'B0={}'.format(b_0), 'res={}'.format(res), 'dim={}'.format(dim),
                        'phi={}'.format(phi), 'geometry={}'.format(geometry)])
        if data_shelve.has_key(key):
            data[:, i] = data_shelve[key]
        else:
            start_time = time.time()
            phase_num = pm.phase_mag_fourier(res, projection, b_0, padding_list[i])
            data[2, i] = time.time() - start_time
            phase_diff = phase_ana - phase_num
            PhaseMap(res, phase_diff).display()
            data[1, i] = np.std(phase_diff)
            data_shelve[key] = data[:, i]    
    # Plot duration against padding:
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_yscale('log')
    axis.plot(data[0], data[1])
    axis.set_title('Fourier Space Approach: Variation of the Padding')
    axis.set_xlabel('padding')
    axis.set_ylabel('RMS')
    # Plot RMS against padding:
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(data[0], data[2])
    axis.set_title('Fourier Space Approach: Variation of the Padding')
    axis.set_xlabel('padding')
    axis.set_ylabel('duration [s]')
    
    
    
    
    
    
    
    '''VARY DIMENSIONS FOR ALL APPROACHES'''
    
    b_0 =  1    # in T
    phi = -pi/4
    dim_list = [(1, 4, 4), (1, 8, 8), (1, 16, 16), (1, 32, 32), (1, 64, 64), (1, 128, 128)]
    res_list = [64., 32., 16., 8., 4., 2., 1.]  # in nm
    
    
    
    data_sl_p_fourier0 = np.zeros((3, len(res_list)))
    data_sl_w_fourier0 = np.zeros((3, len(res_list)))
    data_disc_fourier0 = np.zeros((3, len(res_list)))
    
    data_sl_p_fourier1 = np.zeros((3, len(res_list)))
    data_sl_w_fourier1 = np.zeros((3, len(res_list)))
    data_disc_fourier1 = np.zeros((3, len(res_list)))
    
    data_sl_p_fourier20 = np.zeros((3, len(res_list)))
    data_sl_w_fourier20 = np.zeros((3, len(res_list)))
    data_disc_fourier20 = np.zeros((3, len(res_list)))
    
    data_sl_p_real_s = np.zeros((3, len(res_list)))
    data_sl_w_real_s = np.zeros((3, len(res_list)))
    data_disc_real_s = np.zeros((3, len(res_list)))
    
    data_sl_p_real_d= np.zeros((3, len(res_list)))
    data_sl_w_real_d = np.zeros((3, len(res_list)))
    data_disc_real_d = np.zeros((3, len(res_list)))
    
    data_slab_perfect[0, :] = res_list
    data_slab_worst[0, :] = res_list
    data_disc[0, :] = res_list
    
    
    
    '''FOURIER UNPADDED'''
    
    for i, (dim, res) in enumerate(zip(dim_list, res_list)):
        
        print 'dim =', str(dim)        
        
        # ANALYTIC SOLUTIONS:
        # Slab (perfectly aligned):
        center = (0, dim[1]/2.-0.5, dim[2]/2.-0.5)  # in px (z, y, x) index starts with 0!
        width  = (1, dim[1], dim[2])  # in px (z, y, x)
        mag_shape_slab_perfect = mc.Shapes.slab(dim, center, width)
        phase_ana_slab_perfect = an.phase_mag_slab(dim, res, phi, center, width, b_0)
        mag_data_slab_perfect = MagData(res, mc.create_mag_dist(mag_shape_slab_perfect, phi))
        projection_slab_perfect = pj.simple_axis_projection(mag_data_slab_perfect)
        # Slab (worst case):
        center = (0, dim[1]/2, dim[2]/2)  # in px (z, y, x) index starts with 0!
        width  = (1, dim[1], dim[2])  # in px (z, y, x)
        mag_shape_slab_worst = mc.Shapes.slab(dim, center, width)
        phase_ana_slab_worst = an.phase_mag_slab(dim, res, phi, center, width, b_0)
        mag_data_slab_worst = MagData(res, mc.create_mag_dist(mag_shape_slab_worst, phi))
        projection_slab_worst = pj.simple_axis_projection(mag_data_slab_worst)
        # Disc:
        center = (0, dim[1]/2.-0.5, dim[2]/2.-0.5)  # in px (z, y, x) index starts with 0!
        radius = dim[1]/2  # in px 
        height = 1  # in px
        mag_shape_disc = mc.Shapes.disc(dim, center, radius, height)
        phase_ana_disc = an.phase_mag_disc(dim, res, phi, center, radius, height, b_0)
        mag_data_disc = MagData(res, mc.create_mag_dist(mag_shape_disc, phi))
        projection_disc = pj.simple_axis_projection(mag_data_disc)
        
        # NUMERICAL SOLUTIONS:
        # Slab (perfectly aligned):
        key = ', '.join(['Resolution->RMS|duration', 'Fourier', 'padding=0', 
                        'B0={}'.format(b_0), 'res={}'.format(res), 'dim={}'.format(dim),
                        'phi={}'.format(phi), 'geometry=slab_perfect'])
        if data_shelve.has_key(key):
            data_slab_perfect[:, i] = data_shelve[key]
        else:
            start_time = time.time()
            phase_num = pm.phase_mag_fourier(res, projection_slab_perfect, b_0, 0)
            data_slab_perfect[2, i] = time.time() - start_time
            phase_diff = phase_ana_slab_perfect - phase_num
            data_slab_perfect[1, i] = np.std(phase_diff)
            data_shelve[key] = data_slab_perfect[:, i]
        # Slab (worst case):
        key = ', '.join(['Resolution->RMS|duration', 'Fourier', 'padding=0', 
                        'B0={}'.format(b_0), 'res={}'.format(res), 'dim={}'.format(dim),
                        'phi={}'.format(phi), 'geometry=slab_worst'])
        if data_shelve.has_key(key):
            data_slab_worst[:, i] = data_shelve[key]
        else:
            start_time = time.time()
            phase_num = pm.phase_mag_fourier(res, projection_slab_worst, b_0, 0)
            data_slab_worst[2, i] = time.time() - start_time
            phase_diff = phase_ana_slab_worst - phase_num
            data_slab_worst[1, i] = np.std(phase_diff)
            data_shelve[key] = data_slab_worst[:, i]
        # Disc:
        key = ', '.join(['Resolution->RMS|duration', 'Fourier', 'padding=0', 
                        'B0={}'.format(b_0), 'res={}'.format(res), 'dim={}'.format(dim),
                        'phi={}'.format(phi), 'geometry=disc'])
        if data_shelve.has_key(key):
            data_disc[:, i] = data_shelve[key]
        else:
            start_time = time.time()
            phase_num = pm.phase_mag_fourier(res, projection_disc, b_0, 0)
            data_disc[2, i] = time.time() - start_time
            phase_diff = phase_ana_disc - phase_num
            data_disc[1, i] = np.std(phase_diff)
            data_shelve[key] = data_disc[:, i]    
            
    # Plot duration against res:
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_yscale('log')
    axis.plot(data_slab_perfect[0], data_slab_perfect[1],
              data_slab_worst[0], data_slab_worst[1],
              data_disc[0], data_disc[1])
    axis.set_title('Variation of the resolution (Fourier without padding)')
    axis.set_xlabel('res [nm]')
    axis.set_ylabel('RMS')
    # Plot RMS against res:
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(data_slab_perfect[0], data_slab_perfect[1],
              data_slab_worst[0], data_slab_worst[1],
              data_disc[0], data_disc[1])
    axis.set_title('Variation of the resolution (Fourier without padding)')
    axis.set_xlabel('res [nm]')
    axis.set_ylabel('duration [s]')
    
    
    
    
    
    
    
    
    
    data_shelve.close()

    
    
if __name__ == "__main__":
    try:
        phase_from_mag()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
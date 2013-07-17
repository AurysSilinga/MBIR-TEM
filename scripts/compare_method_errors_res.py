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
import shelve


def compare_method_errors_res():
    '''Calculate and display the phase map from a given magnetization.
    Arguments:
        None
    Returns:
        None
    
    '''
    # Create / Open databank:
    data_shelve = shelve.open('../output/method_errors_shelve')
    
    
    
    
    
    
    
    '''VARY DIMENSIONS FOR ALL APPROACHES'''
    
    b_0 =  1    # in T
    phi = -pi/4
    dim_list = [(1, 4, 4), (1, 8, 8), (1, 16, 16), (1, 32, 32), (1, 64, 64), 
                (1, 128, 128), (1, 256, 256), (1, 512, 512)]
    res_list = [64., 32., 16., 8., 4., 2., 1., 0.5, 0.25]  # in nm
    
    
    '''CREATE DATA ARRAYS'''
        
    data_sl_p_fourier0 = np.zeros((3, len(res_list)))
    data_sl_w_fourier0 = np.zeros((3, len(res_list)))
    data_disc_fourier0 = np.zeros((3, len(res_list)))
    data_vort_fourier0 = np.zeros((3, len(res_list)))

    data_sl_p_fourier1 = np.zeros((3, len(res_list)))
    data_sl_w_fourier1 = np.zeros((3, len(res_list)))
    data_disc_fourier1 = np.zeros((3, len(res_list)))
    data_vort_fourier1 = np.zeros((3, len(res_list)))

    data_sl_p_fourier20 = np.zeros((3, len(res_list)))
    data_sl_w_fourier20 = np.zeros((3, len(res_list)))
    data_disc_fourier20 = np.zeros((3, len(res_list)))
    data_vort_fourier20 = np.zeros((3, len(res_list)))

    data_sl_p_real_s = np.zeros((3, len(res_list)))
    data_sl_w_real_s = np.zeros((3, len(res_list)))
    data_disc_real_s = np.zeros((3, len(res_list)))
    data_vort_real_s = np.zeros((3, len(res_list)))

    data_sl_p_real_d= np.zeros((3, len(res_list)))
    data_sl_w_real_d = np.zeros((3, len(res_list)))
    data_disc_real_d = np.zeros((3, len(res_list)))
    data_vort_real_d = np.zeros((3, len(res_list)))

    
    '''CREATE DATA ARRAYS'''
        
    data_sl_p_fourier0[0, :] = res_list
    data_sl_w_fourier0[0, :] = res_list
    data_disc_fourier0[0, :] = res_list
    data_vort_fourier0[0, :] = res_list
    
    data_sl_p_fourier1[0, :] = res_list
    data_sl_w_fourier1[0, :] = res_list
    data_disc_fourier1[0, :] = res_list
    data_vort_fourier1[0, :] = res_list
    
    data_sl_p_fourier20[0, :] = res_list
    data_sl_w_fourier20[0, :] = res_list
    data_disc_fourier20[0, :] = res_list
    data_vort_fourier20[0, :] = res_list
    
    data_sl_p_real_s[0, :] = res_list
    data_sl_w_real_s[0, :] = res_list
    data_disc_real_s[0, :] = res_list
    data_vort_real_s[0, :] = res_list
    
    data_sl_p_real_d[0, :]= res_list
    data_sl_w_real_d[0, :] = res_list
    data_disc_real_d[0, :] = res_list
    data_vort_real_d[0, :] = res_list
        
    
    
    for i, (dim, res) in enumerate(zip(dim_list, res_list)):
        
        print 'i =', i, '   dim =', str(dim)        
        
        '''ANALYTIC SOLUTIONS'''
        
        # Slab (perfectly aligned):
        center = (0, dim[1]/2.-0.5, dim[2]/2.-0.5)  # in px (z, y, x) index starts with 0!
        width  = (1, dim[1]/2, dim[2]/2)  # in px (z, y, x)
        mag_shape_sl_p = mc.Shapes.slab(dim, center, width)
        phase_ana_sl_p = an.phase_mag_slab(dim, res, phi, center, width, b_0)
        mag_data_sl_p = MagData(res, mc.create_mag_dist(mag_shape_sl_p, phi))
        projection_sl_p = pj.simple_axis_projection(mag_data_sl_p)
        # Slab (worst case):
        center = (0, dim[1]/2, dim[2]/2)  # in px (z, y, x) index starts with 0!
        width  = (1, dim[1]/2, dim[2]/2)  # in px (z, y, x)
        mag_shape_sl_w = mc.Shapes.slab(dim, center, width)
        phase_ana_sl_w = an.phase_mag_slab(dim, res, phi, center, width, b_0)
        mag_data_sl_w = MagData(res, mc.create_mag_dist(mag_shape_sl_w, phi))
        projection_sl_w = pj.simple_axis_projection(mag_data_sl_w)
        # Disc:
        center = (0, dim[1]/2.-0.5, dim[2]/2.-0.5)  # in px (z, y, x) index starts with 0!
        radius = dim[1]/4  # in px 
        height = 1  # in px
        mag_shape_disc = mc.Shapes.disc(dim, center, radius, height)
        phase_ana_disc = an.phase_mag_disc(dim, res, phi, center, radius, height, b_0)
        mag_data_disc = MagData(res, mc.create_mag_dist(mag_shape_disc, phi))
        projection_disc = pj.simple_axis_projection(mag_data_disc)
        # Vortex:
        center_vortex = (center[1], center[2])
        
        
        '''FOURIER UNPADDED'''
        padding = 0
        # Slab (perfectly aligned):
        key = ', '.join(['Resolution->RMS|duration', 'Fourier', 'padding={}'.format(padding), 
                        'B0={}'.format(b_0), 'res={}'.format(res), 'dim={}'.format(dim),
                        'phi={}'.format(phi), 'geometry=sl_p'])
        if data_shelve.has_key(key):
            data_sl_p_fourier0[:, i] = data_shelve[key]
        else:
            start_time = time.time()
            phase_num = pm.phase_mag_fourier(res, projection_sl_p, b_0, padding)
            data_sl_p_fourier0[2, i] = time.time() - start_time
            phase_diff = phase_ana_sl_p - phase_num
            data_sl_p_fourier0[1, i] = np.std(phase_diff)
            data_shelve[key] = data_sl_p_fourier0[:, i]
        # Slab (worst case):
        key = ', '.join(['Resolution->RMS|duration', 'Fourier', 'padding={}'.format(padding), 
                        'B0={}'.format(b_0), 'res={}'.format(res), 'dim={}'.format(dim),
                        'phi={}'.format(phi), 'geometry=sl_w'])
        if data_shelve.has_key(key):
            data_sl_w_fourier0[:, i] = data_shelve[key]
        else:
            start_time = time.time()
            phase_num = pm.phase_mag_fourier(res, projection_sl_w, b_0, padding)
            data_sl_w_fourier0[2, i] = time.time() - start_time
            phase_diff = phase_ana_sl_w - phase_num
            data_sl_w_fourier0[1, i] = np.std(phase_diff)
            data_shelve[key] = data_sl_w_fourier0[:, i]
        # Disc:
        key = ', '.join(['Resolution->RMS|duration', 'Fourier', 'padding={}'.format(padding), 
                        'B0={}'.format(b_0), 'res={}'.format(res), 'dim={}'.format(dim),
                        'phi={}'.format(phi), 'geometry=disc'])
        if data_shelve.has_key(key):
            data_disc_fourier0[:, i] = data_shelve[key]
        else:
            start_time = time.time()
            phase_num = pm.phase_mag_fourier(res, projection_disc, b_0, padding)
            data_disc_fourier0[2, i] = time.time() - start_time
            phase_diff = phase_ana_disc - phase_num
            data_disc_fourier0[1, i] = np.std(phase_diff)
            data_shelve[key] = data_disc_fourier0[:, i]
            
            
        '''FOURIER PADDED ONCE'''
        padding = 1
        # Slab (perfectly aligned):
        key = ', '.join(['Resolution->RMS|duration', 'Fourier', 'padding={}'.format(padding), 
                        'B0={}'.format(b_0), 'res={}'.format(res), 'dim={}'.format(dim),
                        'phi={}'.format(phi), 'geometry=sl_p'])
        if data_shelve.has_key(key):
            data_sl_p_fourier1[:, i] = data_shelve[key]
        else:
            start_time = time.time()
            phase_num = pm.phase_mag_fourier(res, projection_sl_p, b_0, padding)
            data_sl_p_fourier1[2, i] = time.time() - start_time
            phase_diff = phase_ana_sl_p - phase_num
            data_sl_p_fourier1[1, i] = np.std(phase_diff)
            data_shelve[key] = data_sl_p_fourier1[:, i]
        # Slab (worst case):
        key = ', '.join(['Resolution->RMS|duration', 'Fourier', 'padding={}'.format(padding), 
                        'B0={}'.format(b_0), 'res={}'.format(res), 'dim={}'.format(dim),
                        'phi={}'.format(phi), 'geometry=sl_w'])
        if data_shelve.has_key(key):
            data_sl_w_fourier1[:, i] = data_shelve[key]
        else:
            start_time = time.time()
            phase_num = pm.phase_mag_fourier(res, projection_sl_w, b_0, padding)
            data_sl_w_fourier1[2, i] = time.time() - start_time
            phase_diff = phase_ana_sl_w - phase_num
            data_sl_w_fourier1[1, i] = np.std(phase_diff)
            data_shelve[key] = data_sl_w_fourier1[:, i]
        # Disc:
        key = ', '.join(['Resolution->RMS|duration', 'Fourier', 'padding={}'.format(padding), 
                        'B0={}'.format(b_0), 'res={}'.format(res), 'dim={}'.format(dim),
                        'phi={}'.format(phi), 'geometry=disc'])
        if data_shelve.has_key(key):
            data_disc_fourier1[:, i] = data_shelve[key]
        else:
            start_time = time.time()
            phase_num = pm.phase_mag_fourier(res, projection_disc, b_0, padding)
            data_disc_fourier1[2, i] = time.time() - start_time
            phase_diff = phase_ana_disc - phase_num
            data_disc_fourier1[1, i] = np.std(phase_diff)
            data_shelve[key] = data_disc_fourier1[:, i]    
            
            
        '''FOURIER PADDED 20'''
        if dim[1] <= 128:
            padding = 20
            # Slab (perfectly aligned):
            key = ', '.join(['Resolution->RMS|duration', 'Fourier', 'padding={}'.format(padding), 
                            'B0={}'.format(b_0), 'res={}'.format(res), 'dim={}'.format(dim),
                            'phi={}'.format(phi), 'geometry=sl_p'])
            if data_shelve.has_key(key):
                data_sl_p_fourier20[:, i] = data_shelve[key]
            else:
                start_time = time.time()
                phase_num = pm.phase_mag_fourier(res, projection_sl_p, b_0, padding)
                data_sl_p_fourier20[2, i] = time.time() - start_time
                phase_diff = phase_ana_sl_p - phase_num
                data_sl_p_fourier20[1, i] = np.std(phase_diff)
                data_shelve[key] = data_sl_p_fourier20[:, i]
            # Slab (worst case):
            key = ', '.join(['Resolution->RMS|duration', 'Fourier', 'padding={}'.format(padding), 
                            'B0={}'.format(b_0), 'res={}'.format(res), 'dim={}'.format(dim),
                            'phi={}'.format(phi), 'geometry=sl_w'])
            if data_shelve.has_key(key):
                data_sl_w_fourier20[:, i] = data_shelve[key]
            else:
                start_time = time.time()
                phase_num = pm.phase_mag_fourier(res, projection_sl_w, b_0, padding)
                data_sl_w_fourier20[2, i] = time.time() - start_time
                phase_diff = phase_ana_sl_w - phase_num
                data_sl_w_fourier20[1, i] = np.std(phase_diff)
                data_shelve[key] = data_sl_w_fourier20[:, i]
            # Disc:
            key = ', '.join(['Resolution->RMS|duration', 'Fourier', 'padding={}'.format(padding), 
                            'B0={}'.format(b_0), 'res={}'.format(res), 'dim={}'.format(dim),
                            'phi={}'.format(phi), 'geometry=disc'])
            if data_shelve.has_key(key):
                data_disc_fourier20[:, i] = data_shelve[key]
            else:
                start_time = time.time()
                phase_num = pm.phase_mag_fourier(res, projection_disc, b_0, padding)
                data_disc_fourier20[2, i] = time.time() - start_time
                phase_diff = phase_ana_disc - phase_num
                data_disc_fourier20[1, i] = np.std(phase_diff)
                data_shelve[key] = data_disc_fourier20[:, i]
            
            
        '''REAL SLAB'''
        method = 'slab'
        # Slab (perfectly aligned):
        key = ', '.join(['Resolution->RMS|duration', 'Real', 'method={}'.format(method), 
                        'B0={}'.format(b_0), 'res={}'.format(res), 'dim={}'.format(dim),
                        'phi={}'.format(phi), 'geometry=sl_p'])
        if data_shelve.has_key(key):
            data_sl_p_real_s[:, i] = data_shelve[key]
        else:
            start_time = time.time()
            phase_num = pm.phase_mag_real(res, projection_sl_p, method, b_0)
            data_sl_p_real_s[2, i] = time.time() - start_time
            phase_diff = phase_ana_sl_p - phase_num
            data_sl_p_real_s[1, i] = np.std(phase_diff)
            data_shelve[key] = data_sl_p_real_s[:, i]
        # Slab (worst case):
        key = ', '.join(['Resolution->RMS|duration', 'Real', 'method={}'.format(method), 
                        'B0={}'.format(b_0), 'res={}'.format(res), 'dim={}'.format(dim),
                        'phi={}'.format(phi), 'geometry=sl_w'])
        if data_shelve.has_key(key):
            data_sl_w_real_s[:, i] = data_shelve[key]
        else:
            start_time = time.time()
            phase_num = pm.phase_mag_real(res, projection_sl_w, method, b_0)
            data_sl_w_real_s[2, i] = time.time() - start_time
            phase_diff = phase_ana_sl_w - phase_num
            data_sl_w_real_s[1, i] = np.std(phase_diff)
            data_shelve[key] = data_sl_w_real_s[:, i]
        # Disc:
        key = ', '.join(['Resolution->RMS|duration', 'Real', 'method={}'.format(method), 
                        'B0={}'.format(b_0), 'res={}'.format(res), 'dim={}'.format(dim),
                        'phi={}'.format(phi), 'geometry=disc'])
        if data_shelve.has_key(key):
            data_disc_real_s[:, i] = data_shelve[key]
        else:
            start_time = time.time()
            phase_num = pm.phase_mag_real(res, projection_disc, method, b_0)
            data_disc_real_s[2, i] = time.time() - start_time
            phase_diff = phase_ana_disc - phase_num
            data_disc_real_s[1, i] = np.std(phase_diff)
            data_shelve[key] = data_disc_real_s[:, i]    
            
            
        '''REAL DISC'''
        method = 'disc'
        # Slab (perfectly aligned):
        key = ', '.join(['Resolution->RMS|duration', 'Real', 'method={}'.format(method), 
                        'B0={}'.format(b_0), 'res={}'.format(res), 'dim={}'.format(dim),
                        'phi={}'.format(phi), 'geometry=sl_p'])
        if data_shelve.has_key(key):
            data_sl_p_real_d[:, i] = data_shelve[key]
        else:
            start_time = time.time()
            phase_num = pm.phase_mag_real(res, projection_sl_p, method, b_0)
            data_sl_p_real_d[2, i] = time.time() - start_time
            phase_diff = phase_ana_sl_p - phase_num
            data_sl_p_real_d[1, i] = np.std(phase_diff)
            data_shelve[key] = data_sl_p_real_d[:, i]
        # Slab (worst case):
        key = ', '.join(['Resolution->RMS|duration', 'Real', 'method={}'.format(method), 
                        'B0={}'.format(b_0), 'res={}'.format(res), 'dim={}'.format(dim),
                        'phi={}'.format(phi), 'geometry=sl_w'])
        if data_shelve.has_key(key):
            data_sl_w_real_d[:, i] = data_shelve[key]
        else:
            start_time = time.time()
            phase_num = pm.phase_mag_real(res, projection_sl_w, method, b_0)
            data_sl_w_real_d[2, i] = time.time() - start_time
            phase_diff = phase_ana_sl_w - phase_num
            data_sl_w_real_d[1, i] = np.std(phase_diff)
            data_shelve[key] = data_sl_w_real_d[:, i]
        # Disc:
        key = ', '.join(['Resolution->RMS|duration', 'Real', 'method={}'.format(method), 
                        'B0={}'.format(b_0), 'res={}'.format(res), 'dim={}'.format(dim),
                        'phi={}'.format(phi), 'geometry=disc'])
        if data_shelve.has_key(key):
            data_disc_real_d[:, i] = data_shelve[key]
        else:
            start_time = time.time()
            phase_num = pm.phase_mag_real(res, projection_disc, method, b_0)
            data_disc_real_d[2, i] = time.time() - start_time
            phase_diff = phase_ana_disc - phase_num
            data_disc_real_d[1, i] = np.std(phase_diff)
            data_shelve[key] = data_disc_real_d[:, i]
       
      
    # Plot duration against res (perfect slab):
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_yscale('log')
    axis.plot(data_sl_p_fourier0[0], data_sl_p_fourier0[1], 'b+',
              data_sl_p_fourier1[0], data_sl_p_fourier1[1], 'bx',
              data_sl_p_fourier20[0], data_sl_p_fourier20[1], 'b*',
              data_sl_p_real_s[0], data_sl_p_real_s[1], 'rs',
              data_sl_p_real_d[0], data_sl_p_real_d[1], 'ro')
    axis.set_title('Variation of the resolution (perfectly adjusted slab)')
    axis.set_xlabel('res [nm]')
    axis.set_ylabel('RMS')
    # Plot RMS against res (perfect slab):
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_yscale('log')
    axis.plot(data_sl_p_fourier0[0], data_sl_p_fourier0[2], 'b+',
              data_sl_p_fourier1[0], data_sl_p_fourier1[2], 'bx',
              data_sl_p_fourier20[0], data_sl_p_fourier20[2], 'b*',
              data_sl_p_real_s[0], data_sl_p_real_s[2], 'rs',
              data_sl_p_real_d[0], data_sl_p_real_d[2], 'ro')
    axis.set_title('Variation of the resolution (perfectly adjusted slab)')
    axis.set_xlabel('res [nm]')
    axis.set_ylabel('duration [s]')
    
    # Plot duration against res (worst case slab):
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_yscale('log')
    axis.plot(data_sl_w_fourier0[0], data_sl_w_fourier0[1], 'b+',
              data_sl_w_fourier1[0], data_sl_w_fourier1[1], 'bx',
              data_sl_w_fourier20[0], data_sl_w_fourier20[1], 'b*',
              data_sl_w_real_s[0], data_sl_w_real_s[1], 'rs',
              data_sl_w_real_d[0], data_sl_w_real_d[1], 'ro')
    axis.set_title('Variation of the resolution (worst case slab)')
    axis.set_xlabel('res [nm]')
    axis.set_ylabel('RMS')
    # Plot RMS against res (worst case slab):
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_yscale('log')
    axis.plot(data_sl_w_fourier0[0], data_sl_w_fourier0[2], 'b+',
              data_sl_w_fourier1[0], data_sl_w_fourier1[2], 'bx',
              data_sl_w_fourier20[0], data_sl_w_fourier20[2], 'b*',
              data_sl_w_real_s[0], data_sl_w_real_s[2], 'rs',
              data_sl_w_real_d[0], data_sl_w_real_d[2], 'ro')
    axis.set_title('Variation of the resolution (worst case slab)')
    axis.set_xlabel('res [nm]')
    axis.set_ylabel('duration [s]')
    
    # Plot duration against res (disc<):
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_yscale('log')
    axis.plot(data_disc_fourier0[0], data_disc_fourier0[1], 'b+',
              data_disc_fourier1[0], data_disc_fourier1[1], 'bx',
              data_disc_fourier20[0], data_disc_fourier20[1], 'b*',
              data_disc_real_s[0], data_disc_real_s[1], 'rs',
              data_disc_real_d[0], data_disc_real_d[1], 'ro')
    axis.set_title('Variation of the resolution (disc)')
    axis.set_xlabel('res [nm]')
    axis.set_ylabel('RMS')
    # Plot RMS against res (disc):
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_yscale('log')
    axis.plot(data_disc_fourier0[0], data_disc_fourier0[2], 'b+',
              data_disc_fourier1[0], data_disc_fourier1[2], 'bx',
              data_disc_fourier20[0], data_disc_fourier20[2], 'b*',
              data_disc_real_s[0], data_disc_real_s[2], 'rs',
              data_disc_real_d[0], data_disc_real_d[2], 'ro')
    axis.set_title('Variation of the resolution (disc)')
    axis.set_xlabel('res [nm]')
    axis.set_ylabel('duration [s]')
    
    
    data_shelve.close()

    
    
if __name__ == "__main__":
    try:
        compare_method_errors_res()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
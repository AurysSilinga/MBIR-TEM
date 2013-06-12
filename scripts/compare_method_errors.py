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
    
    
#    fig = plt.figure()
#    axis = fig.add_subplot(1, 1, 1, aspect='equal')
#    im = plt.pcolormesh(self.phase, cmap=cmap)
#    # Set the axes ticks and labels:
#    ticks = axis.get_xticks()*self.res
#    axis.set_xticklabels(ticks)
#    ticks = axis.get_yticks()*self.res
#    axis.set_yticklabels(ticks)
#    axis.set_title(title)
#    axis.set_xlabel('x-axis [nm]')
#    axis.set_ylabel('y-axis [nm]')
#    # Plot the phase map:
#    fig = plt.gcf()
#    fig.subplots_adjust(right=0.85)
#    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
#    fig.colorbar(im, cax=cbar_ax)
#    plt.show()

#    '''REAL SLAB '''
#    phi = np.linspace(0, 2*pi, endpoint=False, num=72)
#    
#    RMS = np.zeros(len(phi))
#    for i in range(len(phi)):
#        print 'phi =', round(360*phi[i]/(2*pi))
#        mag_data = MagData(res, mc.create_mag_dist(mag_shape, phi[i]))
#        projection = pj.simple_axis_projection(mag_data)
#        phase_num  = pm.phase_mag_real(res, projection, 'slab', b_0)
#        phase_ana  = an.phase_mag_slab(dim, res, phi[i], center, width, b_0)
#        phase_diff = phase_ana - phase_num
#        RMS[i]  = np.std(phase_diff)
#    
#    fig = plt.figure()
#    fig.add_subplot(111)    
#    plt.plot(np.round(360*phi/(2*pi)), RMS)
    
      
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

    data_shelve.close()

    
    
if __name__ == "__main__":
    try:
        phase_from_mag()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
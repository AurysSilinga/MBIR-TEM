# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:25:53 2013

@author: Jan
"""

import sys
import traceback
import pdb
import os

import numpy as np
from numpy import pi

import shelve

import pyramid.phasemapper as pm
import pyramid.projector as pj
import pyramid.holoimage as hi
from pyramid.magdata import MagData
from pyramid.phasemap import PhaseMap

import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

from matplotlib.ticker import FixedFormatter, IndexLocator, LinearLocator


print '\nACCESS SHELVE'
# Create / Open databank:
directory = '../../output'
if not os.path.exists(directory):
    os.makedirs(directory)
data_shelve = shelve.open(directory + '/reconstruction/reconstruct_shelf')

key = 'reconstruct_core_shell'
if key in data_shelve:
    (projections, phase_maps, mag_data) = data_shelve[key]
else:
    mag_data = MagData.load_from_llg(directory + 
                                     '/magnetic distributions/mag_dist_core_shell_disc.txt')
    mag_data.quiver_plot()
    res = mag_data.res
    dim = mag_data.dim
    slices = 30
    angles = np.linspace(0, 2*pi, slices, endpoint=True)
    projections = []
    phase_maps = []
    for tilt_angle in angles:
        print '{:.2f} pi'.format(tilt_angle/pi)
        # Project along tilt_angle
        projection = pj.single_tilt_projection(mag_data, tilt_angle)
        projections.append(projection)
        # Construct phase maps:
        phase_map = PhaseMap(res, pm.phase_mag_real_conv(res, projection), unit='rad')
        phase_maps.append(phase_map)
        # Display the combinated plots with phasemap and holography image:
        title = 'Phase at {:.2f} pi'.format(tilt_angle/pi)
    #    phase_map.display(title, limit=None)
    #    hi.display_combined(phase_map, 1, title=title)
    #    plt.savefig(directory+'/reconstruction/tilt_series_comb_'+
    #                '{:.2f}'.format(tilt_angle/pi)+'pi.png')
    # Save data
    data_shelve[key] = (projections, phase_maps, mag_data)

# Close shelve:
data_shelve.close()

# Plot Animation:
fig = plt.figure()
limit = None
images_ph = []
for i in range(len(phase_maps)):
    images_ph.append([plt.imshow(phase_maps[i].phase, cmap='RdBu')])
ani_ph = ArtistAnimation(fig, images_ph, interval=50, blit=True)
plt.show()
fig_phase = plt.figure()
#fig_projection = plt.figure()
images_pr = []
for i in range(len(phase_maps)):
    images_pr.append([plt.quiver(projections[i][1], projections[i][0], pivot='middle',
                                 angles='xy', scale_units='xy', headwidth=6, headlength=7)])
ani_pr = ArtistAnimation(fig_phase, images_pr, interval=50, blit=True)
plt.show()

# RECONSTRUCTION:
mask = mag_data.get_mask()
dim = mag_data.dim
res = mag_data.res
mag_data_reconstruct = MagData(res, (np.zeros(dim),)*3)

mag_vector = mag_data_reconstruct.get_vector(mask)

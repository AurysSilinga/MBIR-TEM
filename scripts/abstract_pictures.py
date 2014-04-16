# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 14:28:10 2014

@author: Jan
"""


from numpy import pi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import pyramid.magcreator as mc
from pyramid.magdata import MagData
from pyramid.projector import SimpleProjector, YTiltProjector
from pyramid.phasemapper import PMConvolve


###################################################################################################
print('Jan')

dim = (128, 128, 128)
center = (int(dim[0]/2), int(dim[1]/2), int(dim[2]/2))
radius = dim[0]/4.
a = 1.

magnitude = mc.create_mag_dist_homog(mc.Shapes.sphere(dim, center, radius), pi/4)

mag_data = MagData(a, magnitude)

projector = SimpleProjector(dim)

phase_map = PMConvolve(a, projector)(mag_data)

axis = phase_map.display_phase()
axis.tick_params(axis='both', which='major', labelsize=18)
axis.set_title('Phase map', fontsize=24)
axis.set_xlabel('x [nm]', fontsize=18)
axis.set_ylabel('y [nm]', fontsize=18)

axis = phase_map.display_holo(density=20, interpolation='bilinear')
axis.tick_params(axis='both', which='major', labelsize=18)
axis.set_title('Magnetic induction map', fontsize=24)
axis.set_xlabel('x [nm]', fontsize=18)
axis.set_ylabel('y [nm]', fontsize=18)

mag_data.scale_down(2)

mag_data.quiver_plot()
axis = plt.gca()
axis.tick_params(axis='both', which='major', labelsize=18)
axis.set_title('Magnetization distribution', fontsize=24)
axis.set_xlabel('x [nm]', fontsize=18)
axis.set_ylabel('y [nm]', fontsize=18)

phase_map.make_color_wheel()

shape_vort = mc.Shapes.disc((64, 64, 64), (31.5, 31.5, 31.5), 24, 10)

magnitude_vort = mc.create_mag_dist_vortex(shape_vort)

mag_vort = MagData(a, magnitude_vort)

mag_vort.scale_down(2)

mag_vort.quiver_plot()


###################################################################################################
print('Patrick')

a = 10.0  # nm
b_0 = 3  # T

dim = (128, 128, 128)
center = (int(dim[0]/2), int(dim[1]/2), int(dim[2]/2))  # in px (z, y, x), index starts with 0!

mag_data = MagData(a, mc.create_mag_dist_homog(mc.Shapes.ellipse(dim, center, (20., 60.), 5.), 0))

tilts = np.array([0., 60.])/180.*pi

projectors = [YTiltProjector(mag_data.dim, tilt) for tilt in tilts]
phasemappers = [PMConvolve(mag_data.a, proj, b_0) for proj in projectors]

phase_maps = [pm(mag_data) for pm in phasemappers]

axis = phase_maps[0].display_holo(density=1, interpolation='bilinear')
axis.tick_params(axis='both', which='major', labelsize=18)
axis.set_title('Magnetic induction map', fontsize=24)
axis.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:g}'.format(x*a)))
axis.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:g}'.format(x*a)))
axis.set_xlabel('x [nm]', fontsize=18)
axis.set_ylabel('y [nm]', fontsize=18)

axis = phase_maps[0].display_phase(limit=17)
axis.tick_params(axis='both', which='major', labelsize=18)
axis.set_title('Phase map', fontsize=24)
axis.set_xlabel('x [nm]', fontsize=18)
axis.set_ylabel('y [nm]', fontsize=18)

axis = phase_maps[1].display_holo(density=1, interpolation='bilinear')
axis.tick_params(axis='both', which='major', labelsize=18)
axis.set_title('Magnetic induction map', fontsize=24)
axis.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:g}'.format(x*a)))
axis.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:g}'.format(x*a)))
axis.set_xlabel('x [nm]', fontsize=18)
axis.set_ylabel('y [nm]', fontsize=18)

axis = phase_maps[1].display_phase(limit=17)
axis.tick_params(axis='both', which='major', labelsize=18)
axis.set_title('Phase map', fontsize=24)
axis.set_xlabel('x [nm]', fontsize=18)
axis.set_ylabel('y [nm]', fontsize=18)


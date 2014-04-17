# -*- coding: utf-8 -*-
"""
Created on Thu Jan 09 20:18:56 2014

@author: Jan
"""


from numpy import pi

import pyramid.magcreator as mc
from pyramid.magdata import MagData
from pyramid.projector import SimpleProjector
from pyramid.phasemapper import PMAdapterFM
from pyramid.kernel import Kernel


magnitude = mc.create_mag_dist_vortex(mc.Shapes.disc((32,64,128),(32,32,64),16,8))
mag_data = MagData(1, magnitude)
#mag_data = MagData.load_from_llg('../output/magnetic distributions/mag_dist_disc.txt')
#mag_data.quiver_plot()
phase_map = PMAdapterFM(mag_data.a, SimpleProjector(mag_data.dim), geometry='disc')(mag_data)
phase_map.display_combined(30)

#phase_map.make_color_wheel()

# TODO: use for compare pixel fields!
k_d = Kernel(1, (5,5), geometry='disc')
k_s = Kernel(1, (5,5), geometry='slab')
u_d, v_d = k_d.u, k_d.v
u_s, v_s = k_s.u, k_s.v

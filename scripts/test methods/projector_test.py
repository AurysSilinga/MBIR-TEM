# -*- coding: utf-8 -*-
"""Created on Fri Aug 15 08:09:10 2014 @author: Jan"""


import numpy as np

import pyramid.magcreator as mc
from pyramid.magdata import MagData
from pyramid.projector import SimpleProjector, XTiltProjector, YTiltProjector
from pyramid.phasemapper import PMConvolve


dim = (16, 24, 32)
center = (int(dim[0]/2), int(dim[1]/2), int(dim[2]/2))
width = (2, 8, 16)
dim_uv = (32, 48)
a = 2
b_0 = 1.5

shape = mc.Shapes.slab(dim, center, width)
mag_data = MagData(a, mc.create_mag_dist_homog(shape, phi=np.pi/6, theta=np.pi/3))

# PROJECTOR TESTING
PMConvolve(a, SimpleProjector(dim, 'z', dim_uv), b_0)(mag_data).display_phase('z simple')
PMConvolve(a, XTiltProjector(dim, 0, dim_uv), b_0)(mag_data).display_phase('z (x-tilt)')
PMConvolve(a, SimpleProjector(dim, 'x', dim_uv), b_0)(mag_data).display_phase('x simple')
PMConvolve(a, YTiltProjector(dim, np.pi/2, dim_uv), b_0)(mag_data).display_phase('x (y-tilt)')
PMConvolve(a, XTiltProjector(dim, np.pi/2, dim_uv), b_0)(mag_data).display_phase('y (x-tilt)')
PMConvolve(a, SimpleProjector(dim, 'y', dim_uv), b_0)(mag_data).display_phase('y simple')

# -*- coding: utf-8 -*-
"""Created on Fri Aug 15 08:09:10 2014 @author: Jan"""


import numpy as np

import pyramid.magcreator as mc
from pyramid.magdata import MagData
from pyramid.projector import SimpleProjector, XTiltProjector, YTiltProjector, RotTiltProjector
from pyramid.phasemapper import PhaseMapperRDFC
from pyramid.kernel import Kernel


dim = (16, 24, 32)
center = (int(dim[0]/2), int(dim[1]/2), int(dim[2]/2))
width = (2, 8, 16)
dim_uv = (32, 32)
a = 2
b_0 = 1.5

shape = mc.Shapes.slab(dim, center, width)
mag_data = MagData(a, mc.create_mag_dist_homog(shape, phi=np.pi/6, theta=np.pi/3))
phase_mapper = PhaseMapperRDFC(Kernel(a, dim_uv, b_0))

# PROJECTOR TESTING
phase_mapper(SimpleProjector(dim, 'z', dim_uv)(mag_data)).display_phase('z simple')
phase_mapper(XTiltProjector(dim, 0, dim_uv)(mag_data)).display_phase('z (x-tilt)')
phase_mapper(RotTiltProjector(dim, 0, 0, dim_uv)(mag_data)).display_phase('z (arb.)')
phase_mapper(SimpleProjector(dim, 'x', dim_uv)(mag_data)).display_phase('x simple')
phase_mapper(YTiltProjector(dim, np.pi/2, dim_uv)(mag_data)).display_phase('x (y-tilt)')
phase_mapper(RotTiltProjector(dim, 0, np.pi/2, dim_uv)(mag_data)).display_phase('x (arb.)')
phase_mapper(SimpleProjector(dim, 'y', dim_uv)(mag_data)).display_phase('y simple')
phase_mapper(XTiltProjector(dim, np.pi/2, dim_uv)(mag_data)).display_phase('y (x-tilt)')
phase_mapper(RotTiltProjector(dim, np.pi/2, np.pi/2, dim_uv)(mag_data)).display_phase('y (Arb.)')
mag_data.quiver_plot3d()

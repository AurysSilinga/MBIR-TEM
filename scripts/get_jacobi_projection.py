# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 16:45:13 2013

@author: Jan
"""

import pyramid.magcreator as mc
from pyramid.magdata import MagData
import pyramid.projector as pj
from pyramid.projector import Projection


a = 1.0
dim = (2, 2, 2)
px = (0, 0, 0)
mag_data = MagData(a, mc.create_mag_dist_homog(mc.Shapes.pixel(dim, px), 0))

proj = pj.simple_axis_projection(mag_data)


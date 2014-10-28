# -*- coding: utf-8 -*-
"""Created on Thu Oct 02 11:53:25 2014 @author: Jan"""


import os

import numpy as np
import matplotlib.pyplot as plt

import pyramid
from pyramid.magdata import MagData
from pyramid.phasemapper import pm

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')
PATH = '../../output/bielefeld/'


def load_from_llg(filename):
    '''Custom for Bielefeld files'''
    SCALE = 1  # .0E-9 / 1.0E-2  # From cm to nm
    data = np.genfromtxt(filename, skip_header=2)
    dim = tuple(np.genfromtxt(filename, dtype=int, skip_header=1, skip_footer=len(data[:, 0])))
    a = (data[1, 0] - data[0, 0]) / SCALE
    magnitude = data[:, 3:6].T.reshape((3,)+dim)
#    import pdb; pdb.set_trace()
    return MagData(a, magnitude)


logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)
## 2DCoFe:
mag_data_2DCoFe = MagData.load_from_llg(PATH+'magnetic_distribution_2DCoFe.txt')
mag_data_2DCoFe.quiver_plot(proj_axis='x')
plt.savefig(PATH+'magnetic_distribution_2DCoFe.png')
mag_data_2DCoFe.quiver_plot3d()
phase_map_2DCoFe = pm(mag_data_2DCoFe, axis='x')
phase_map_2DCoFe.display_combined(gain='auto')
# AH21FIN:
mag_data_AH21FIN = MagData.load_from_llg(PATH+'magnetic_distribution_AH21FIN.txt')
mag_data_AH21FIN.quiver_plot(proj_axis='x')
plt.savefig(PATH+'magnetic_distribution_AH21FIN.png')
mag_data_AH21FIN.quiver_plot3d()
phase_map_AH21FIN = pm(mag_data_AH21FIN, axis='x')
phase_map_AH21FIN.display_combined(gain='auto')
# AH41FIN:
mag_data_AH41FIN = MagData.load_from_llg(PATH+'magnetic_distribution_AH41FIN.txt')
mag_data_AH41FIN.quiver_plot(proj_axis='x')
plt.savefig(PATH+'magnetic_distribution_AH41FIN.png')
mag_data_AH41FIN.quiver_plot3d()
phase_map_AH41FIN = pm(mag_data_AH41FIN, axis='x')
phase_map_AH41FIN.display_combined(gain='auto')
# Chain:
mag_data_chain = load_from_llg(PATH+'magnetic_distribution_Chain.txt')
mag_data_chain.quiver_plot(proj_axis='x')
plt.savefig(PATH+'magnetic_distribution_Chain.png')
mag_data_chain.quiver_plot3d()
phase_map_chain = pm(mag_data_chain, axis='x')
phase_map_chain.display_combined(gain='auto')
# Cylinder:
mag_data_cyl = load_from_llg(PATH+'magnetic_distribution_Cylinder.txt')
mag_data_cyl.quiver_plot(proj_axis='z')
plt.savefig(PATH+'magnetic_distribution_Cylinder.png')
mag_data_cyl.quiver_plot3d()
phase_map_cyl = pm(mag_data_cyl, axis='z')
phase_map_cyl.display_combined(gain='auto')
# Ring:
mag_data_ring = load_from_llg(PATH+'magnetic_distribution_ring.txt')
mag_data_ring.quiver_plot(proj_axis='x')
plt.savefig(PATH+'magnetic_distribution_ring.png')
mag_data_ring.quiver_plot3d()
phase_map_ring = pm(mag_data_ring, axis='x')
phase_map_ring.display_combined(gain='auto')

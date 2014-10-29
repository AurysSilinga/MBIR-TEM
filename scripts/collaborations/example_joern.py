# -*- coding: utf-8 -*-
"""Created on Mon Aug 11 08:41:23 2014 @author: Jan"""


import os

import numpy as np

import pickle

import matplotlib.pyplot as plt

import pyramid
from pyramid.phasemap import PhaseMap
from pyramid.projector import SimpleProjector
from pyramid.phasemapper import pm
from pyramid.dataset import DataSet
from pyramid.regularisator import ZeroOrderRegularisator, FirstOrderRegularisator
import pyramid.reconstruction as rc

from time import clock

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')


logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)
logging.basicConfig(level=logging.INFO)
###################################################################################################
threshold = 1
a = 1.0  # in nm
gain = 5
b_0 = 1
inter = 'none'
dim = (1,) + (64, 64)
dim_small = (64, 64)
smoothed_pictures = True
lam = 1E-4
order = 1
log = True
PATH = './'
dirname = PATH
###################################################################################################
# Read in files:
phase_map = PhaseMap.load_from_netcdf4(PATH+'phase_map.nc')
#mask = np.genfromtxt('mask.txt', dtype=bool)
with open(PATH + 'mask.pickle') as pf:
    mask = pickle.load(pf)
# Setup:
data_set = DataSet(a, dim, b_0, mask=mask)
data_set.append(phase_map, SimpleProjector(dim))
regularisator = ZeroOrderRegularisator(lam)
regularisator = FirstOrderRegularisator(mask, lam)
print "OOO"

# Reconstruct the magnetic distribution:
tic = clock()
mag_data_rec = rc.optimize_linear(data_set, regularisator=regularisator)
#mag_data_rec = rc.optimize_nonlin(data_set, regularisator=regularisator)
#  .optimize_simple_leastsq(phase_map, mask, b_0, lam=lam, order=order)

print 'reconstruction time:', clock() - tic
# Display the reconstructed phase map and holography image:
phase_map_rec = pm(mag_data_rec)
phase_map_rec.display_combined('Reconstr. Distribution', gain=gain, interpolation=inter, show=False)
plt.savefig(dirname + "/reconstr.png")

# Plot the magnetization:
axis = (mag_data_rec*(1/mag_data_rec.magnitude.max())).quiver_plot(show=False)
axis.set_xlim(20, 45)
axis.set_ylim(20, 45)
plt.savefig(dirname + "/quiver.png")

# Display the Phase:
phase_diff = phase_map_rec-phase_map
phase_diff.display_phase('Difference', show=False)
plt.savefig(dirname + "/difference.png")

# Get the average difference from the experimental results:
print 'Average difference:', np.average(phase_diff.phase)
# Plot holographic contour maps with overlayed magnetic distributions:
axis = phase_map_rec.display_holo('Magnetization Overlay', gain=0.1, interpolation=inter, show=False)
mag_data_rec.quiver_plot(axis=axis, show=False)
axis = plt.gca()
axis.set_xlim(20, 45)
axis.set_ylim(20, 45)
plt.savefig(dirname + "/overlay_normal.png")

axis = phase_map_rec.display_holo('Magnetization Overlay', gain=0.1, interpolation=inter, show=False)
mag_data_rec.quiver_plot(axis=axis, log=log, show=False)
axis = plt.gca()
axis.set_xlim(20, 45)
axis.set_ylim(20, 45)
plt.savefig(dirname + "/overlay_log.png")

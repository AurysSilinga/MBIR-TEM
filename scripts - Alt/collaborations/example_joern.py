# -*- coding: utf-8 -*-
"""Created on Mon Aug 11 08:41:23 2014 @author: Jan"""


import os

import numpy as np

import pickle
import matplotlib
matplotlib.use("Agg")
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

if "PBS_ARRAYID" in os.environ:
    job_number = int(os.getenv("PBS_ARRAYID"))
    experiments = []
    for lam in [1e-9, 1e-8, 1e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1]:
        for order in [0, 1]:
            for p in [2, 1.5, 1.1, 1.01, 1.001]:
                for use_mask in [True, False]:
                    experiments.append(
                        (lam, order, p, use_mask))
    assert 0 <= job_number < len(experiments), \
        "job id too large, maximum={}".format(len(experiments) - 1)
    lam, order, p, use_mask = experiments[job_number]
    print experiments[job_number]
    dirname = "new-order_{}-p_{}-mask_{}-lambda_{:.9f}-job{}/".format(
        order, p, use_mask, lam, job_number)
    if os.path.exists(dirname):
        print dirname
#        exit()
    else:
        os.mkdir(dirname)
else:
    p = 2
    lam = 1e-8
    use_mask = True
    order = 1
    dirname = "./"


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')

logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)
#logging.basicConfig(level=logging.INFO)

###################################################################################################
threshold = 1
a = 1.0  # in nm
gain = 5
b_0 = 1
inter = 'none'
smoothed_pictures = True
lam = 1E-6
log = True
PATH = '../../output/joern/'
###################################################################################################
# Read in files:
phase_map = PhaseMap.load_from_netcdf4(PATH+'phase_map.nc')
with open(PATH + 'mask.pickle') as pf:
    mask = pickle.load(pf)
dim = mask.shape
# Setup:
if not use_mask:
    mask = np.ones_like(mask, dtype=bool)
data_set = DataSet(a, dim, b_0, mask=mask)
data_set.append(phase_map, SimpleProjector(dim))
if order == 0:
    regularisator = ZeroOrderRegularisator(mask, lam, p)
elif order == 1:
    regularisator = FirstOrderRegularisator(mask, lam, p)
else:
    raise NotImplementedError

# Reconstruct the magnetic distribution:
tic = clock()

if p == 2:
    mag_data_rec, cost = rc.optimize_linear(data_set, regularisator=regularisator, max_iter=50)
else:
    print regularisator.p
    mag_data_rec = rc.optimize_nonlin(data_set, regularisator=regularisator)

with open(dirname + "result.pickle", "wb") as pf:
    import cPickle
    cPickle.dump(mag_data_rec, pf)
#  .optimize_simple_leastsq(phase_map, mask, b_0, lam=lam, order=order)

print 'reconstruction time:', clock() - tic
# Display the reconstructed phase map and holography image:
phase_map_rec = pm(mag_data_rec)
phase_map_rec.display_combined('Reconstr. Distribution', gain=gain, interpolation=inter)
plt.savefig(dirname + "/reconstr.png")

# Plot the magnetization:
axis = (mag_data_rec*(1/mag_data_rec.magnitude.max())).quiver_plot()
axis.set_xlim(20./64*dim[1], 45./64*dim[2])
axis.set_ylim(20./64*dim[1], 45./64*dim[2])
plt.savefig(dirname + "/quiver.png")

# Display the Phase:
phase_diff = phase_map_rec-phase_map
phase_diff.display_phase('Difference')
plt.savefig(dirname + "/difference.png")

# Get the average difference from the experimental results:
print 'Average difference:', np.average(phase_diff.phase)
# Plot holographic contour maps with overlayed magnetic distributions:
axis = phase_map_rec.display_holo('Magnetization Overlay', gain=0.1, interpolation=inter)
mag_data_rec.quiver_plot(axis=axis)
axis = plt.gca()
axis.set_xlim(20./64*dim[1], 45./64*dim[2])
axis.set_ylim(20./64*dim[1], 45./64*dim[2])
plt.savefig(dirname + "/overlay_normal.png")

axis = phase_map_rec.display_holo('Magnetization Overlay', gain=0.1, interpolation=inter)
mag_data_rec.quiver_plot(axis=axis, log=log)
axis = plt.gca()
axis.set_xlim(20./64*dim[1], 45./64*dim[2])
axis.set_ylim(20./64*dim[1], 45./64*dim[2])
plt.savefig(dirname + "/overlay_log.png")

# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:19:33 2013

@author: Jan
"""  # TODO: Docstring

# TODO: Implement

import numpy as np
import pyramid.projector as pj
import pyramid.phasemapper as pm
from pyramid.magdata import MagData
from scipy.optimize import leastsq


def reconstruct_simple_lsqu(phase_map, mask, b_0):
    # TODO: Docstring!
    # Read in parameters:
    y_m = phase_map.phase.reshape(-1)  # Measured phase map as a vector
    res = phase_map.res  # Resolution
    dim = mask.shape  # Dimensions of the mag. distr.
    count = mask.sum()  # Number of pixels with magnetization
    lam = 1e-6  # Regularisation parameter
    # Create empty MagData object for the reconstruction:
    mag_data_rec = MagData(res, (np.zeros(dim), np.zeros(dim), np.zeros(dim)))
    ############################# FORWARD MODEL ###################################################
    # Function that returns the phase map for a magnetic configuration x:
    def F(x):
        mag_data_rec.set_vector(mask, x)
        phase = pm.phase_mag_real(res, pj.simple_axis_projection(mag_data_rec), 'slab', b_0)
        return phase.reshape(-1)
    ############################# FORWARD MODEL ###################################################
    ############################# RECONSTRUCTION ##################################################
    # Cost function which should be minimized:
    def J(x_i):
        y_i = F(x_i)
        term1 = (y_i - y_m)
        term2 = lam * x_i
        return np.concatenate([term1, term2])
    # Reconstruct the magnetization components:
    x_rec, _ = leastsq(J, np.zeros(3*count))
    ############################# RECONSTRUCTION ##################################################
    mag_data_rec.set_vector(mask, x_rec)
    return mag_data_rec
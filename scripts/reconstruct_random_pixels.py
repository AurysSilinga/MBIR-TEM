# -*- coding: utf-8 -*-
"""Create random magnetic distributions."""

import random as rnd
import pdb, traceback, sys
import numpy as np
from numpy import pi

import pyramid.magcreator as mc
from pyramid.magdata import MagData
import pyramid.phasemapper as pm
import pyramid.projector as pj
import pyramid.holoimage as hi
from pyramid.phasemap import PhaseMap
from scipy.optimize import leastsq


def reconstruct_random_distribution():
    '''Calculate and reconstruct a random magnetic distribution.
    Arguments:
        None
    Returns:
        None

    '''
    # Input parameters:
    n_pixel = 10
    dim = (32, 32, 32)
    b_0 = 1    # in T
    res = 10.0 # in nm
    rnd.seed(12)
    threshold = 0
    # Create lists for magnetic objects:
    mag_shape_list = np.zeros((n_pixel,) + dim)
    beta_list      = np.zeros(n_pixel)
    magnitude_list = np.zeros(n_pixel)
    for i in range(n_pixel):
        pixel = (rnd.randrange(dim[0]), rnd.randrange(dim[1]), rnd.randrange(dim[2]))
        mag_shape_list[i,...] = mc.Shapes.pixel(dim, pixel)
        beta_list[i] = 2*pi*rnd.random()
        magnitude_list[i] = rnd.random()
    # Create magnetic distribution:
    magnitude = mc.create_mag_dist_comb(mag_shape_list, beta_list, magnitude_list)
    mag_data = MagData(res, magnitude)
    # Display phase map and holography image:
    projection = pj.simple_axis_projection(mag_data)
    phase_map = PhaseMap(res, pm.phase_mag_real(res, projection, 'slab', b_0))
    hi.display_combined(phase_map, 10, 'Generated Distribution')
    # Get the locations of the magnetized pixels (mask):
    z_mag, y_mag, x_mag = mag_data.magnitude
    z_mask = abs(z_mag) > threshold
    x_mask = abs(x_mag) > threshold
    y_mask = abs(y_mag) > threshold
    mask = np.logical_or(np.logical_or(x_mask, y_mask), z_mask)
    # True values for the magnetisation informations, condensed into one vector:
    x_t = mag_data.get_vector(mask)
    # Create empty MagData object for the reconstruction:
    mag_data_rec = MagData(res, (np.zeros(dim), np.zeros(dim), np.zeros(dim)))
    ############################# FORWARD MODEL ###################################################
    # Function that returns the phase map for a magnetic configuration x:
    def F(x):
        mag_data_rec.set_vector(mask, x)
        phase = pm.phase_mag_real(res, pj.simple_axis_projection(mag_data_rec), 'slab', b_0)
        return phase.reshape(-1)
    ############################# FORWARD MODEL ###################################################
    # Get a vector containing the measured phase at the specified places:
    y_m = F(x_t)
    print "y_m", y_m
    ############################# RECONSTRUCTION ##################################################
    lam = 1e-6  # Regularisation parameter
    # Cost function which should be minimized:
    def J(x_i):
        y_i = F(x_i)
        term1 = (y_i - y_m)
        term2 = lam * x_i
        return np.concatenate([term1, term2])
    # Reconstruct the magnetization components:
    x_f, _ = leastsq(J, np.zeros(x_t.shape))
    ############################# RECONSTRUCTION ##################################################
    # Save the reconstructed values in the MagData object:
    y_f = F(x_f)
    print "y_f", y_f
    # Display the reconstructed phase map and holography image:
    projection_rec = pj.simple_axis_projection(mag_data_rec)
    phase_map_rec = PhaseMap(res, pm.phase_mag_real(res, projection_rec, 'slab', b_0))
    hi.display_combined(phase_map_rec, 10, 'Reconstructed Distribution')


if __name__ == "__main__":
    try:
        reconstruct_random_distribution()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

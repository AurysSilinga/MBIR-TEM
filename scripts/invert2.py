# -*- coding: utf-8 -*-
"""Create random magnetic distributions."""

import random as rnd
import pdb, traceback, sys
import numpy as np
from numpy import pi
import pylab
from copy import deepcopy

import pyramid.magcreator as mc
from pyramid.magdata import MagData
import pyramid.phasemapper as pm
import pyramid.projector as pj
import pyramid.holoimage as hi
from pyramid.phasemap import PhaseMap
from scipy.optimize import leastsq


def create_random_distribution():
    '''Calculate, display and save a random magnetic distribution to file.
    Arguments:
        None
    Returns:
        None

    '''
    # Input parameters:
    count = 200
    dim = (1, 32, 32)
    b_0 = 1    # in T
    res = 10 # in nm
    rnd.seed(12)
    # Create lists for magnetic objects:
    mag_shape_list = np.zeros((count,) + dim)
    beta_list      = np.zeros(count)
    magnitude_list = np.zeros(count)
    for i in range(count):
        pixel = (rnd.randrange(dim[0]), rnd.randrange(dim[1]), rnd.randrange(dim[2]))
        mag_shape_list[i,...] = mc.Shapes.pixel(dim, pixel)
        beta_list[i] = 2*pi*rnd.random()
        magnitude_list[i] = rnd.random()
    # Create magnetic distribution
    magnitude = mc.create_mag_dist_comb(mag_shape_list, beta_list, magnitude_list)
    mag_data = MagData(res, magnitude)
#    mag_data.quiver_plot()
#    mag_data.save_to_llg('output/mag_dist_random_pixel.txt')


    magx, magy, magz = mag_data.magnitude
    maskx, masky, maskz = magx != 0, magy != 0, magz !=0
    x_t = np.concatenate([magx[maskx], magy[masky], magz[maskz]])
    print "x_t", x_t

    def F(x):
        mag_data_temp = MagData(deepcopy(mag_data.res), deepcopy(mag_data.magnitude))
        magx, magy, magz = mag_data_temp.magnitude
        maskx, masky, maskz = magx != 0, magy != 0, magz !=0
#        print maskx.sum() + masky.sum() + maskz.sum()
        assert len(x) == maskx.sum() + masky.sum() + maskz.sum()
        magx[maskx] = x[:maskx.sum()]
        magy[masky] = x[maskx.sum():maskx.sum() + masky.sum()]
        magz[maskz] = x[maskx.sum() + masky.sum():]
        projection = pj.simple_axis_projection(mag_data_temp)
        phase_map_slab = PhaseMap(res, pm.phase_mag_real_ANGLE(res, projection, 'slab', b_0))
        return phase_map_slab.phase.reshape(-1)

    y_m = F(x_t)
    print "y_m", y_m

    lam = 1e-6

    def J(x_i):
#        print "x_i", x_i
        y_i = F(x_i)
#        dd1 = np.zeros(mx.shape)
#        dd2 = np.zeros(mx.shape)
        term1 = (y_i - y_m)
        term2 = lam * x_i
#        dd1[:, :-1, :] += np.diff(mx, axis=1)
#        dd1[:, -1, :] += np.diff(mx, axis=1)[:, -1, :]
#        dd1[:, :, :-1] += np.diff(mx, axis=2)
#        dd1[:, :, -1] += np.diff(mx, axis=2)[:, :, -1]

#        dd2[:, :-1, :] += np.diff(my, axis=1)
#        dd2[:, -1, :] += np.diff(my, axis=1)[:, -1, :]
#        dd2[:, :, :-1] += np.diff(my, axis=2)
#        dd2[:, :, -1] += np.diff(my, axis=2)[:, :, -1]

#        result = np.concatenate([term1, np.sqrt(abs(dd1.reshape(-1))), np.sqrt(abs(dd2.reshape(-1)))])
#        result = np.concatenate([term1, np.sqrt(abs(dd1.reshape(-1)))])
#        print result
        return np.concatenate([term1, term2])

    x_f, _ = leastsq(J, np.zeros(x_t.shape))
    y_f = F(x_f)
#    print "y_m", y_m
#    print "y_f", y_f
#    print "dy", y_f - y_m
#    print "x_t", x_t
#    print "x_f", x_f
#    print "dx", x_f - x_t
#    pylab.show()
    projection = pj.simple_axis_projection(mag_data)
    phase_map  = PhaseMap(res, pm.phase_mag_real(res, projection, 'slab'))
    hi.display(hi.holo_image(phase_map, 10))


if __name__ == "__main__":
    try:
        create_random_distribution()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

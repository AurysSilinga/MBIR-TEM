# -*- coding: utf-8 -*-
"""Reconstruct magnetic distributions from given phasemaps.

This module reconstructs 3-dimensional magnetic distributions (as :class:`~pyramid.magdata.MagData`
objects) from a given set of phase maps (represented by :class:`~pyramid.phasemap.PhaseMap`
objects) by using several model based reconstruction algorithms which use the forward model
provided by :mod:`~pyramid.projector` and :mod:`~pyramid.phasemapper` and a priori knowledge of
the distribution.
So far, only a simple least square algorithm for known pixel locations for 2-dimensional problems
is implemented (:func:`~.reconstruct_simple_leastsq`), but more complex solutions are planned.

"""# TODO: Docstrings!


# TODO: Delete
''' FRAGEN FÜR JÖRN:
1) Property und Python "Magic" ist wann sinnvoll?
2) Aktuelle Designstruktur, welche Designpatterns kommen vor und sind sinnvoll?
3) Macht logging Sinn?
4) Ist scipy.optimize.minimize(...) eine gute Wahl?
5) Wo besser Module, wo Klassen?
6) Arbitrary grid to cartesian grid
'''


import numpy as np

from scipy.optimize import minimize

from pyramid.kernel import Kernel
from pyramid.forwardmodel import ForwardModel
from pyramid.costfunction import Costfunction
from pyramid.magdata import MagData



def optimize_cg(self, data_collection, first_guess):
    # TODO: where should data_collection and first_guess go? here or __init__?
    data = data_collection
    mag_0 = first_guess
    x_0 = first_guess.mag_vec
    y = data.phase_vec
    kern = Kernel(data.dim, data.a, data.b_0)
    F = ForwardModel(data, kern)
    C = Costfunction(y, F)

    result = minimize(C, x_0, method='CG', jac=C.jac, hessp=C.hess_dot)

    x_opt = result.x

    mag_opt = MagData(mag_0.a, np.zeros((3,)+mag_0.dim)).mag_vec = x_opt

    return mag_opt



# TODO: Implement the following:

# -*- coding: utf-8 -*-
"""Reconstruct magnetic distributions from given phasemaps.

This module reconstructs 3-dimensional magnetic distributions (as :class:`~pyramid.magdata.MagData`
objects) from a given set of phase maps (represented by :class:`~pyramid.phasemap.PhaseMap`
objects) by using several model based reconstruction algorithms which use the forward model
provided by :mod:`~pyramid.projector` and :mod:`~pyramid.phasemapper` and a priori knowledge of
the distribution.
So far, only a simple least square algorithm for known pixel locations for 2-dimensional problems
is implemented (:func:`~.reconstruct_simple_leastsq`), but more complex solutions are planned.

"""



import numpy as np

from scipy.optimize import leastsq

import pyramid.projector as pj
import pyramid.phasemapper as pm
from pyramid.magdata import MagData
from pyramid.projector import Projection
from pyramid.kernel import Kernel


def reconstruct_simple_leastsq(phase_map, mask, b_0=1):
    '''Reconstruct a magnetic distribution for a 2-D problem with known pixel locations.

    Parameters
    ----------
        phase_map : :class:`~pyramid.phasemap.PhaseMap`
            A :class:`~pyramid.phasemap.PhaseMap` object, representing the phase from which to
            reconstruct the magnetic distribution.
        mask : :class:`~numpy.ndarray` (N=3)
            A boolean matrix (or a matrix consisting of ones and zeros), representing the
            positions of the magnetized voxels in 3 dimensions.
        b_0 : float, optional
            The magnetic induction corresponding to a magnetization `M`\ :sub:`0` in T.
            The default is 1.
    Returns
    -------
        mag_data : :class:`~pyramid.magdata.MagData`
            The reconstructed magnetic distribution as a
            :class:`~pyramid.magdata.MagData` object.

    Notes
    -----
    Only works for a single phase_map, if the positions of the magnetized voxels are known and
    for slice thickness of 1 (constraint for the `z`-dimension).

    '''
    # Read in parameters:
    y_m = phase_map.phase.reshape(-1)  # Measured phase map as a vector
    a = phase_map.a  # Grid spacing
    dim = mask.shape  # Dimensions of the mag. distr.
    count = mask.sum()  # Number of pixels with magnetization
    lam = 1e-6  # Regularisation parameter
    # Create empty MagData object for the reconstruction:
    mag_data_rec = MagData(a, (np.zeros(dim), np.zeros(dim), np.zeros(dim)))

    # Function that returns the phase map for a magnetic configuration x:
    def F(x):
        mag_data_rec.set_vector(mask, x)
        phase = pm.phase_mag_real(a, pj.simple_axis_projection(mag_data_rec), b_0)
        return phase.reshape(-1)

    # Cost function which should be minimized:
    def J(x_i):
        y_i = F(x_i)
        term1 = (y_i - y_m)
        term2 = lam * x_i
        return np.concatenate([term1, term2])

    # Reconstruct the magnetization components:
    x_rec, _ = leastsq(J, np.zeros(3*count))
    mag_data_rec.set_vector(mask, x_rec)
    return mag_data_rec

def reconstruct_test():
    product = (kernel.multiply_jacobi_T(projection.multiply_jacobi_T(x))
             * kernel.multiply_jacobi(projection.multiply_jacobi(x)))
# -*- coding: utf-8 -*-
"""Numerical core routines for the phase calculation using the real space approach.

Provides a helper function to speed up :func:`~pyramid.phasemapper.phase_mag_real` of module
:mod:`~pyramid.phasemapper`, by using C-speed for the for-loops and by omitting boundary and
wraparound checks.

"""


import numpy as np
import math

cimport cython
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def phase_mag_real_core(
        unsigned int v_dim, unsigned int u_dim,
        double[:, :] v_phi, double[:, :] u_phi,
        double[:, :] v_mag, double[:, :] u_mag,
        double[:, :] phase, float threshold):
    '''Numerical core routine for the phase calculation using the real space approach.

    Parameters
    ----------
    v_dim, u_dim : int
        Dimensions of the projection along the two major axes.
    v_phi, u_phi : :class:`~numpy.ndarray` (N=2)
        Lookup tables for the pixel fields oriented in `u`- and `v`-direction.
    v_mag, u_mag : :class:`~numpy.ndarray` (N=2)
        Magnetization components in `u`- and `v`-direction.
    phase : :class:`~numpy.ndarray` (N=2)
        Matrix in which the resulting magnetic phase map should be stored.
    threshold : float
        The `threshold` determines which pixels contribute to the magnetic phase.

    Returns
    -------
    None

    '''
    cdef unsigned int i, j, p, q, p_c, q_c
    cdef double u_m, v_m
    for j in range(v_dim):
        for i in range(u_dim):
            u_m = u_mag[j, i]
            v_m = v_mag[j, i]
            p_c = u_dim - 1 - i
            q_c = v_dim - 1 - j
            if abs(u_m) > threshold:
                for q in range(v_dim):
                    for p in range(u_dim):
                        phase[q, p] += u_m * u_phi[q_c + q, p_c + p]
            if abs(v_m) > threshold:
                for q in range(v_dim):
                    for p in range(u_dim):
                        phase[q, p] -= v_m * v_phi[q_c + q, p_c + p]

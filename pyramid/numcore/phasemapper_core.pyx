# -*- coding: utf-8 -*-
"""Numerical core routines for the :mod:`~.pyramid.phasemapper` module.

Provides helper functions to speed up phase mapping calculations in the
:mod:`~pyramid.phasemapper` module, by using C-speed for the for-loops and by omitting boundary
and wraparound checks.

"""


import numpy as np
import math

cimport cython
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def phasemapper_real_convolve(
        unsigned int v_dim, unsigned int u_dim,
        float[:, :] v_phi, float[:, :] u_phi,
        float[:, :] v_mag, float[:, :] u_mag,
        float[:, :] phase, float threshold):
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
    cdef float u_m, v_m
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


@cython.boundscheck(False)
@cython.wraparound(False)
def jac_dot_real_convolve(
        unsigned int v_dim, unsigned int u_dim,
        float[:, :] u_phi, float[:, :] v_phi,
        float[:] vector,
        float[:] result):
    '''Numerical core routine for the Jacobi matrix multiplication for the phase mapper.

    Parameters
    ----------
    v_dim, u_dim : int
        Dimensions of the projection along the two major axes.
    v_phi, u_phi : :class:`~numpy.ndarray` (N=2)
        Lookup tables for the pixel fields oriented in `u`- and `v`-direction.
    vector : :class:`~numpy.ndarray` (N=1)
        Input vector which should be multiplied by the Jacobi-matrix.
    result : :class:`~numpy.ndarray` (N=1)
        Vector in which the result of the multiplication should be stored.

    Returns
    -------
    None

    Notes
    -----
    The strategy involves iterating over the magnetization first and over the kernel in an inner
    iteration loop.

    '''
    cdef unsigned int s, i, j, u_min, u_max, v_min, v_max, r, u, v, size
    size = u_dim * v_dim  # Number of pixels
    s = 0  # Current contributing pixel (numbered consecutively)
    # Iterate over all contributingh pixels:
    for j in range(v_dim):
        for i in range(u_dim):
            u_min = (u_dim-1) - i  # u_max = u_min + u_dim
            v_min = (v_dim-1) - j  # v_max = v_min + v_dim
            r = 0  # Current result component / affected pixel (numbered consecutively)
            # Go over the current kernel cutout [v_min:v_max, u_min:u_max]:
            for v in range(v_min, v_min + v_dim):
                for u in range(u_min, u_min + u_dim):
                    result[r] += vector[s] * u_phi[v, u]
                    result[r] -= vector[s+size] * v_phi[v, u]
                    r += 1
            s += 1
    # TODO: linearize u and v, too?


@cython.boundscheck(False)
@cython.wraparound(False)
def jac_T_dot_real_convolve(
        unsigned int v_dim, unsigned int u_dim,
        float[:, :] u_phi, float[:, :] v_phi,
        float[:] vector,
        float[:] result):
    '''Core routine for the transposed Jacobi multiplication for the phase mapper.

    Parameters
    ----------
    v_dim, u_dim : int
        Dimensions of the projection along the two major axes.
    v_phi, u_phi : :class:`~numpy.ndarray` (N=2)
        Lookup tables for the pixel fields oriented in `u`- and `v`-direction.
    vector : :class:`~numpy.ndarray` (N=1)
        Input vector which should be multiplied by the transposed Jacobi-matrix.
    result : :class:`~numpy.ndarray` (N=1)
        Vector in which the result of the multiplication should be stored.

    Returns
    -------
    None

    Notes
    -----
    The strategy involves iterating over the magnetization first and over the kernel in an inner
    iteration loop.

    '''
    cdef unsigned int s, i, j, u_min, u_max, v_min, v_max, r, u, v, size
    size = u_dim * v_dim  # Number of pixels
    r = 0  # Current result component / contributing pixel (numbered consecutively)
    # Iterate over all contributingh pixels:
    for j in range(v_dim):
        for i in range(u_dim):
            u_min = (u_dim-1) - i  # u_max = u_min + u_dim
            v_min = (v_dim-1) - j  # v_max = v_min + v_dim
            s = 0  # Current affected pixel (numbered consecutively)
            # Go over the current kernel cutout [v_min:v_max, u_min:u_max]:
            for v in range(v_min, v_min + v_dim):
                for u in range(u_min, u_min + u_dim):
                    result[r] += vector[s] * u_phi[v, u]
                    result[r+size] -= vector[s] * v_phi[v, u]
                    s += 1
            r += 1
    # TODO: linearize u and v, too?

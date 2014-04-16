# -*- coding: utf-8 -*-
"""Numerical core routines for the :mod:`~.pyramid.kernel` module.

Provides helper functions to speed up kernel calculations in the :class:`~pyramid.kernel.Kernel`
class, by using C-speed for the for-loops and by omitting boundary and wraparound checks.

"""


import numpy as np
import math

cimport cython
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def multiply_jacobi_core(
        unsigned int v_dim, unsigned int u_dim,
        double[:, :] u_phi, double[:, :] v_phi,
        double[:] vector,
        double[:] result):
    '''Numerical core routine for the Jacobi matrix multiplication in the :class:`~.Kernel` class.

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
    cdef unsigned int s, i, j, u_min, u_max, v_min, v_max, ri, u, v, size
    size = u_dim * v_dim  # Number of pixels
    s = 0  # Current pixel (numbered consecutively)
    # Go over all pixels:
    for j in range(v_dim):
        for i in range(u_dim):
            u_min = (u_dim - 1) - i  # u_max = u_min + u_dim
            v_min = (v_dim - 1) - j  # v_max = v_min + v_dim
            ri = 0  # Current result component (numbered consecutively)
            # Go over the current kernel cutout [v_min:v_max, u_min:u_max]:
            for v in range(v_min, v_min + v_dim):
                for u in range(u_min, u_min + u_dim):
                    result[ri] += vector[s] * u_phi[v, u]
                    result[ri] -= vector[s+size] * v_phi[v, u]
                    ri += 1
            s += 1

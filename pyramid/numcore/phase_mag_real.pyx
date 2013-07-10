import numpy as np
import math
cimport cython
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def phase_mag_real_helper_1(
        unsigned int v_dim, unsigned int u_dim,
        double[:, :] phi_u,
        double[:, :] phi_v,
        double[:, :] u_mag,
        double[:, :] v_mag,
        double[:, :] phase, float threshold):
    cdef unsigned int i, j, ii, jj, iii, jjj
    cdef double u, v
    for j in range(v_dim):
        for i in range(u_dim):
            u = u_mag[j, i]
            v = v_mag[j, i]
            iii = u_dim - 1 - i
            jjj = v_dim - 1 - j
            if abs(u) > threshold:
               for jj in range(phase.shape[0]):
                    for ii in range(phase.shape[1]):
                        phase[jj, ii] += u * phi_u[jjj + jj, iii + ii]
            if abs(v) > threshold:
               for jj in range(phase.shape[0]):
                    for ii in range(phase.shape[1]):
                        phase[jj, ii] -= v * phi_v[jjj + jj, iii + ii]


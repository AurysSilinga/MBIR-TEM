import numpy as np
import math
cimport cython
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def phase_mag_real_helper(
        unsigned int v_dim, unsigned int u_dim,
        double[:, :] phi_u, double[:, :] phi_v,
        double[:, :] u_mag, double[:, :] v_mag,
        double[:, :] phase, float threshold):
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
                        phase[q, p] += u_m * phi_u[q_c + q, p_c + p]
            if abs(v_m) > threshold:
                for q in range(v_dim):
                    for p in range(u_dim):
                        phase[q, p] -= v_m * phi_v[q_c + q, p_c + p]

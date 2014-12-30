# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 17:20:27 2014

@author: Jan
"""


import numpy as np

import jutil

from pyramid import fft
from pyramid.magdata import MagData
from pyramid.phasemap import PhaseMap

class Diagnostics(object):

    # TODO: Docstrings and position of properties!

    def __init__(self, x_rec, cost, max_iter=100):
        self.x_rec = x_rec
        self.cost = cost
        self.max_iter = max_iter
        self.fwd_model = cost.fwd_model
        self.Se_inv = self.cost.Se_inv
        self.dim = self.cost.data_set.dim
        self.row_idx = None
        self.set_position(0)#(0, self.dim[0]//2, self.dim[1]//2, self.dim[2]//2))
        self._A = jutil.operator.CostFunctionOperator(self.cost, self.x_rec)
        self._P = jutil.preconditioner.CostFunctionPreconditioner(self.cost, self.x_rec)

    def set_position(self, pos):
        # TODO: Does not know about the mask... thus gives wrong results or errors
#        m, z, y, x = pos
#        row_idx = m*np.prod(self.dim) + z*self.dim[1]*self.dim[2] + y*self.dim[2] + x
        row_idx = pos
        if row_idx != self.row_idx:
            self.row_idx = row_idx
            self._updated_std = False
            self._updated_gain_row = False
            self._updated_avrg_kern_row = False
            self._updated_measure_contribution = False

    @property
    def std(self):
        if not self._updated_std:
            e_i = fft.zeros(self.cost.n, dtype=fft.FLOAT)
            e_i[self.row_idx] = 1
            row = jutil.cg.conj_grad_solve(self._A, e_i, P=self._P, max_iter=self.max_iter)
            self._m_inv_row = row
            self._std = np.sqrt(self._m_inv_row[self.row_idx])
            self._updated_std = True
        return self._std

    @property
    def gain_row(self):
        if not self._updated_gain_row:
            self.std  # evoke to update self._m_inv_row if necessary # TODO: make _m_inv_row checked!
            self._gain_row = self.Se_inv.dot(self.fwd_model.jac_dot(self.x_rec, self._m_inv_row))
            self._updated_gain_row = True
        return self._gain_row

    @property
    def avrg_kern_row(self):
        if not self._updated_avrg_kern_row:
            self._avrg_kern_row = self.fwd_model.jac_T_dot(self.x_rec, self.gain_row)
            self._updated_avrg_kern_row = True
        return self._avrg_kern_row

    @property
    def measure_contribution(self):
        if not self._updated_measure_contribution:
            cache = self.fwd_model.jac_dot(self.x_rec, fft.ones(self.cost.n, fft.FLOAT))
            cache = self.fwd_model.jac_T_dot(self.x_rec, self.Se_inv.dot(cache))
            mc = jutil.cg.conj_grad_solve(self._A, cache, P=self._P, max_iter=self.max_iter)
            self._measure_contribution = mc
            self._updated_measure_contribution = True
        return self._measure_contribution

    def get_avg_kernel(self, pos=None):
        if pos is not None:
            self.set_position(pos)
        mag_data_avg_kern = MagData(self.cost.data_set.a, fft.zeros((3,)+self.dim))
        mag_data_avg_kern.set_vector(self.avrg_kern_row, mask=self.cost.data_set.mask)
        return mag_data_avg_kern

    def get_gain_maps(self, pos=None):
        if pos is not None:
            self.set_position(pos)
        hp = self.cost.data_set.hook_points
        result = []
        for i, projector in enumerate(self.cost.data_set.projectors):
            gain = self.gain_row[hp[i]:hp[i+1]].reshape(projector.dim_uv)
            result.append(PhaseMap(self.cost.data_set.a, gain))
        return result

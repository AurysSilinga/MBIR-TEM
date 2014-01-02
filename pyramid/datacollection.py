# -*- coding: utf-8 -*-
"""Class for the collection of phase maps and additional data."""


import numpy as np
from numbers import Number

from pyramid.phasemap import PhaseMap
from pyramid.projector import Projector


class DataCollection(object):

    @property
    def a(self):
        return self._a
    
    @property
    def dim(self):
        return self._dim

    @property
    def phase_vec(self):
        return np.concatenate([p.phase_vec for p in self.phase_maps])

    @property
    def phase_maps(self):
        return [d[0] for d in self.data]

    @property
    def projectors(self):
        return [d[1] for d in self.data]

    def __init__(self, a, dim):
        # TODO: Docstring!
        assert isinstance(a, Number), 'Grid spacing has to be a number!'
        assert a >= 0, 'Grid spacing has to be a positive number!'
        self._a = a
        assert isinstance(dim, tuple) and len(dim)==2, 'Dimension has to be a tuple of length 2!'
        assert np.all([])
        self.dim = dim
        self.data = []

    def append(self, (phase_map, projector)):
        # TODO: Docstring!
        assert isinstance(phase_map, PhaseMap) and isinstance(projector, Projector),  \
            'Argument has to be a tuple of a PhaseMap and a Projector object!'
        assert phase_map.dim == self.dim, 'Added phasemap must have the same dimension!'
        assert (projector.dim_v, projector.dim_u) == self.dim, 'Projector dimensions must match!'
        self.data.append((phase_map, projector))

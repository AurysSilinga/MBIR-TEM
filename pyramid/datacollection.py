# -*- coding: utf-8 -*-
"""This module provides the :class:`~.DataCollection` class for the collection of phase maps
and additional data like corresponding projectors."""


import logging

import numpy as np
from numbers import Number

from pyramid.phasemap import PhaseMap
from pyramid.projector import Projector


class DataCollection(object):

    '''Class for collecting phase maps and corresponding projectors.

    Represents a collection of (e.g. experimentally derived) phase maps, stored as
    :class:`~.PhaseMap` objects and corresponding projectors stored as :class:`~.Projector`
    objects. At creation, the grid spacing `a` and the dimension `dim_uv` of the projected grid.
    Data can be added via the :func:`~.append` method, where a :class:`~.PhaseMap` and a
    :class:`~.Projector` have to be given as tuple argument.

    Attributes
    ----------
    a: float
        The grid spacing in nm.
    dim_uv: tuple (N=2)
        Dimensions of the projected grid.
    phase_maps:
        A list of all stored :class:`~.PhaseMap` objects.
    projectors:
        A list of all stored :class:`~.Projector` objects.
    phase_vec: :class:`~numpy.ndarray` (N=1)
        The concatenaded, vectorized phase of all ;class:`~.PhaseMap` objects.

    '''

    @property
    def a(self):
        return self._a
    
    @property
    def dim_uv(self):
        return self._dim_uv

    @property
    def phase_vec(self):
        return np.concatenate([p.phase_vec for p in self.phase_maps])

    @property
    def phase_maps(self):
        return [d[0] for d in self.data]

    @property
    def projectors(self):
        return [d[1] for d in self.data]

    def __init__(self, a, dim_uv, b_0):
        self.log = logging.getLogger(__name__)
        assert isinstance(a, Number), 'Grid spacing has to be a number!'
        assert a >= 0, 'Grid spacing has to be a positive number!'
        self._a = a
        assert isinstance(dim_uv, tuple) and len(dim_uv)==2, \
            'Dimension has to be a tuple of length 2!'
        assert np.all([])
        self._dim_uv = dim_uv
        self.b_0 = b_0
        self.data = []
        # TODO: make it work
#        self.log.info('Created:', str(self))

    def __repr__(self):
        self.log.info('Calling __repr__')
        return '%s(a=%r, dim_uv=%r)' % (self.__class__, self.a, self.dim_uv)

    def __str__(self):
        self.log.info('Calling __str__')
        return 'DataCollection(%s, a=%s, dim_uv=%s, data_count=%s)' % \
            (self.__class__, self.a, self.dim_uv, len(self.data))

    def append(self, (phase_map, projector)):
        '''Appends a data pair of phase map and projection infos to the data collection.`

        Parameters
        ----------
        (phase_map, projector): tuple (N=2)
            tuple which contains a :class:`~.PhaseMap` object and a :class:`~.Projector` object,
            which should be added to the data collection.

        Returns
        -------
        None

        '''
        self.log.info('Calling append')
        assert isinstance(phase_map, PhaseMap) and isinstance(projector, Projector),  \
            'Argument has to be a tuple of a PhaseMap and a Projector object!'
        assert phase_map.dim_uv == self.dim_uv, 'Added phasemap must have the same dimension!'
        assert projector.dim_uv == self.dim_uv, 'Projector dimensions must match!'
        self.data.append((phase_map, projector))

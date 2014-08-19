# -*- coding: utf-8 -*-
"""This module provides the :class:`~.DataSet` class for the collection of phase maps
and additional data like corresponding projectors."""


import numpy as np
from numbers import Number

import matplotlib.pyplot as plt

from pyramid.phasemap import PhaseMap
from pyramid.phasemapper import PMConvolve
from pyramid.projector import Projector

import logging


class DataSet(object):

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

    LOG = logging.getLogger(__name__+'.DataSet')

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
        self.LOG.debug('Calling __init__')
        assert isinstance(a, Number), 'Grid spacing has to be a number!'
        assert a >= 0, 'Grid spacing has to be a positive number!'
        self._a = a
        assert isinstance(dim_uv, tuple) and len(dim_uv) == 2, \
            'Dimension has to be a tuple of length 2!'
        self._dim_uv = dim_uv
        self.b_0 = b_0
        self.data = []
        self.LOG.debug('Created: '+str(self))

    def __repr__(self):
        self.LOG.debug('Calling __repr__')
        return '%s(a=%r, dim_uv=%r, b_0=%r)' % (self.__class__, self.a, self.dim_uv, self.b_0)

    def __str__(self):
        self.LOG.debug('Calling __str__')
        return 'DataSet(a=%s, dim_uv=%s, b_0=%s)' % (self.a, self.dim_uv, self.b_0)

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
        self.LOG.debug('Calling append')
        assert isinstance(phase_map, PhaseMap) and isinstance(projector, Projector),  \
            'Argument has to be a tuple of a PhaseMap and a Projector object!'
        assert phase_map.dim_uv == self.dim_uv, 'Added phasemap must have the same dimension!'
        assert projector.dim_uv == self.dim_uv, 'Projector dimensions must match!'
        self.data.append((phase_map, projector))

    def display_phase(self, phase_maps=None, title='Phase Map',
                      cmap='RdBu', limit=None, norm=None):
        '''Display all phasemaps saved in the :class:`~.DataSet` as a colormesh.

        Parameters
        ----------
        phase_maps : list of :class:`~.PhaseMap`, optional
            List of phase_maps to display with annotations from the projectors. If none are
            given, the phase_maps in the dataset are used (which is the default behaviour).
        title : string, optional
            The main part of the title of the plots. The default is 'Phase Map'. Additional
            projector info is appended to this.
        cmap : string, optional
            The :class:`~matplotlib.colors.Colormap` which is used for the plots as a string.
            The default is 'RdBu'.
        limit : float, optional
            Plotlimit for the phase in both negative and positive direction (symmetric around 0).
            If not specified, the maximum amplitude of the phase is used.
        norm : :class:`~matplotlib.colors.Normalize` or subclass, optional
            Norm, which is used to determine the colors to encode the phase information.
            If not specified, :class:`~matplotlib.colors.Normalize` is automatically used.

        Returns
        -------
        None

        '''
        self.LOG.debug('Calling display')
        if phase_maps is None:
            phase_maps = self.phase_maps
        [phase_map.display_phase('{} ({})'.format(title, self.projectors[i].get_info()),
                                 cmap, limit, norm)
            for (i, phase_map) in enumerate(phase_maps)]
        plt.show()

    def display_combined(self, phase_maps=None, title='Combined Plot', cmap='RdBu', limit=None,
                         norm=None, density=1, interpolation='none', grad_encode='bright'):
        '''Display all phasemaps and the resulting color coded holography images.

        Parameters
        ----------
        phase_maps : list of :class:`~.PhaseMap`, optional
            List of phase_maps to display with annotations from the projectors. If none are
            given, the phase_maps in the dataset are used (which is the default behaviour).
        title : string, optional
            The title of the plot. The default is 'Combined Plot'.
        cmap : string, optional
            The :class:`~matplotlib.colors.Colormap` which is used for the plot as a string.
            The default is 'RdBu'.
        limit : float, optional
            Plotlimit for the phase in both negative and positive direction (symmetric around 0).
            If not specified, the maximum amplitude of the phase is used.
        norm : :class:`~matplotlib.colors.Normalize` or subclass, optional
            Norm, which is used to determine the colors to encode the phase information.
            If not specified, :class:`~matplotlib.colors.Normalize` is automatically used.
        density : float, optional
            The gain factor for determining the number of contour lines in the holographic
            contour map. The default is 1.
        interpolation : {'none, 'bilinear', 'cubic', 'nearest'}, optional
            Defines the interpolation method for the holographic contour map.
            No interpolation is used in the default case.
        grad_encode: {'bright', 'dark', 'color', 'none'}, optional
            Encoding mode of the phase gradient. 'none' produces a black-white image, 'color' just
            encodes the direction (without gradient strength), 'dark' modulates the gradient
            strength with a factor between 0 and 1 and 'bright' (which is the default) encodes
            the gradient strength with color saturation.

        Returns
        -------
        None

        '''
        self.LOG.debug('Calling display_combined')
        if phase_maps is None:
            phase_maps = self.phase_maps
        [phase_map.display_combined('{} ({})'.format(title, self.projectors[i].get_info()),
                                    cmap, limit, norm, density, interpolation, grad_encode)
            for (i, phase_map) in enumerate(phase_maps)]
        plt.show()

    def create_phase_maps(self, mag_data):
        '''Create a list of phasemaps with the projectors in the dataset for a given
        :class:`~.MagData` object.

        Parameters
        ----------
        mag_data : :class:`~.MagData`
            Magnetic distribution to which the projectors of the dataset should be applied.

        Returns
        -------
        phase_maps : list of :class:`~.phasemap.PhaseMap`
            A list of the phase_maps resulting from the projections specified in the dataset.

        Notes
        -----
        For the phasemapping, the :class:`~.PMConvolve` class is used.

        '''
        return [PMConvolve(self.a, proj, self.b_0)(mag_data) for proj in self.projectors]

# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""This module provides the :class:`~.DataSet` class for the collection of phase maps
and additional data like corresponding projectors."""


import numpy as np
from numbers import Number
from scipy import sparse
import matplotlib.pyplot as plt

from pyramid.phasemap import PhaseMap
from pyramid.phasemapper import PhaseMapperRDFC
from pyramid.projector import Projector
from pyramid.kernel import Kernel

import logging


__all__ = ['DataSet']


class DataSet(object):

    '''Class for collecting phase maps and corresponding projectors.

    Represents a collection of (e.g. experimentally derived) phase maps, stored as
    :class:`~.PhaseMap` objects and corresponding projectors stored as :class:`~.Projector`
    objects. At creation, the grid spacing `a` and the dimension `dim` of the magnetization
    distribution have to be given. Data can be added via the :func:`~.append` method, where
    a :class:`~.PhaseMap`, a :class:`~.Projector` and additional info have to be given.

    Attributes
    ----------
    a: float
        The grid spacing in nm.
    dim: tuple (N=3)
        Dimensions of the 3D magnetization distribution.
    phase_maps:
        A list of all stored :class:`~.PhaseMap` objects.
    b_0: double
        The saturation induction in `T`.
    mask: :class:`~numpy.ndarray` (N=3), optional
        A boolean mask which defines the magnetized volume in 3D.
    projectors: list of :class:`~.Projector`
        A list of all stored :class:`~.Projector` objects.
    phase_maps: list of :class:`~.PhaseMap`
        A list of all stored :class:`~.PhaseMap` objects.
    phase_vec: :class:`~numpy.ndarray` (N=1)
        The concatenaded, vectorized phase of all ;class:`~.PhaseMap` objects.
    Se_inv : :class:`~numpy.ndarray` (N=2), optional
        Inverted covariance matrix of the measurement errors. The matrix has size `NxN` with N
        being the length of the targetvector y (vectorized phase map information).
    m: int
        Size of the image space.
    n: int
        Size of the input space.

    '''

    _log = logging.getLogger(__name__+'.DataSet')

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, a):
        assert isinstance(a, Number), 'Grid spacing has to be a number!'
        assert a >= 0, 'Grid spacing has to be a positive number!'
        self._a = float(a)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        if mask is not None:
            assert mask.shape == self.dim, 'Mask dimensions must match!'
        else:
            mask = np.ones(self.dim, dtype=bool)
        self._mask = mask.astype(np.bool)

    @property
    def m(self):
        return np.sum([len(p.phase_vec) for p in self.phase_maps])

    @property
    def n(self):
        return 3 * np.sum(self.mask)

    @property
    def count(self):
        return len(self.projectors)

    @property
    def phase_vec(self):
        return np.concatenate([p.phase_vec for p in self.phase_maps])

    @property
    def hook_points(self):
        result = [0]
        for i, phase_map in enumerate(self.phase_maps):
            result.append(result[i]+np.prod(phase_map.dim_uv))
        return result

    @property
    def phase_mappers(self):
        dim_uv_set = set([p.dim_uv for p in self.projectors])
        kernel_list = [Kernel(self.a, dim_uv) for dim_uv in dim_uv_set]
        return {kernel.dim_uv: PhaseMapperRDFC(kernel) for kernel in kernel_list}

    def __init__(self, a, dim, b_0=1, mask=None, Se_inv=None):
        self._log.debug('Calling __init__')
        assert isinstance(dim, tuple) and len(dim) == 3, \
            'Dimension has to be a tuple of length 3!'
        self.a = a
        self.dim = dim
        self.b_0 = b_0
        self.mask = mask
        self.Se_inv = Se_inv
        self.phase_maps = []
        self.projectors = []
        self._log.debug('Created: '+str(self))

    def __repr__(self):
        self._log.debug('Calling __repr__')
        return '%s(a=%r, dim=%r, b_0=%r, mask=%r, Se_inv=%r)' % (self.__class__, self.a, self.dim,
                                                                 self.b_0, self.mask, self.Se_inv)

    def __str__(self):
        self._log.debug('Calling __str__')
        return 'DataSet(a=%s, dim=%s, b_0=%s)' % (self.a, self.dim, self.b_0)

    def append(self, phase_map, projector):
        '''Appends a data pair of phase map and projection infos to the data collection.`

        Parameters
        ----------
        phase_map: :class:`~.PhaseMap`
            A :class:`~.PhaseMap` object which should be added to the data collection.
        projector: :class:`~.Projector`
            A :class:`~.Projector` object which should be added to the data collection.

        Returns
        -------
        None

        '''
        self._log.debug('Calling append')
        assert isinstance(phase_map, PhaseMap) and isinstance(projector, Projector),  \
            'Argument has to be a tuple of a PhaseMap and a Projector object!'
        assert projector.dim == self.dim, '3D dimensions must match!'
        assert phase_map.dim_uv == projector.dim_uv, 'Projection dimensions (dim_uv) must match!'
        self.phase_maps.append(phase_map)
        self.projectors.append(projector)

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
            A list of the phase maps resulting from the projections specified in the dataset.

        '''
        self._log.debug('Calling create_phase_maps')
        phase_maps = []
        for projector in self.projectors:
            mag_proj = projector(mag_data)
            phase_map = self.phase_mappers[projector.dim_uv](mag_proj)
            phase_map.mask = mag_proj.get_mask()[0, ...]
            phase_maps.append(phase_map)
        return phase_maps

    def set_Se_inv_block_diag(self, cov_list):
        '''Set the Se_inv matrix as a block diagonal matrix

        Parameters
        ----------
        cov_list: list of :class:`~numpy.ndarray`
            List of inverted covariance matrices (one for each projection).

        Returns
        -------
            None

        '''
        self._log.debug('Calling set_Se_inv_block_diag')
        assert len(cov_list) == len(self.phase_maps), 'Needs one covariance matrix per phase map!'
        self.Se_inv = sparse.block_diag(cov_list).tocsr()

    def set_Se_inv_diag_with_conf(self, conf_list=None):
        '''Set the Se_inv matrix as a block diagonal matrix from a list of confidence matrizes.

        Parameters
        ----------
        conf_list: list of :class:`~numpy.ndarray` (optional)
            List of 2D confidence matrizes (one for each projection) which define trust regions.
            If not given this uses the confidence matrizes of the phase maps.

        Returns
        -------
            None

        '''
        self._log.debug('Calling set_Se_inv_diag_with_conf')
        if conf_list is None:  # if no confidence matrizes are given, extract from the phase maps!
            conf_list = [phase_map.confidence for phase_map in self.phase_maps]
        cov_list = [sparse.diags(c.flatten().astype(np.float32), 0) for c in conf_list]
        self.set_Se_inv_block_diag(cov_list)

    def set_3d_mask(self, mask_list=None):
        '''Set the 3D mask from a list of 2D masks.

        Parameters
        ----------
        mask_list: list of :class:`~numpy.ndarray` (optional)
            List of 2D masks, which represent the projections of the 3D mask. If not given this
            uses the mask matrizes of the phase maps. If just one phase map is present, the
            according mask is simply expanded to 3D and used directly.

        Returns
        -------
            None

        '''
        self._log.debug('Calling set_3d_mask')
        if mask_list is None:  # if no masks are given, extract from phase maps:
            mask_list = [phase_map.mask for phase_map in self.phase_maps]
        if len(mask_list) == 1:  # just one phase_map --> 3D mask equals 2D mask
            self.mask = np.expand_dims(mask_list[0], axis=0)  # z-dim is set to 1!
        else:  # 3D mask has to be constructed from 2D masks:
            mask_3d_inv = np.zeros(self.dim)
            for i, projector in enumerate(self.projectors):
                mask_2d_inv = np.logical_not(self.phase_maps[i].mask.reshape(-1))  # inv. 2D mask
                # Add extrusion of inv. 2D mask:
                mask_3d_inv += projector.weight.T.dot(mask_2d_inv).reshape(self.dim)
            self.mask = np.where(mask_3d_inv == 0, True, False)

    def display_mask(self, ar_dens=1):
        '''If it exists, display the 3D mask of the magnetization distribution.

        Parameters
        ----------
        ar_dens: int (optional)
            Number defining the cell density which is plotted. A higher ar_dens number skips more
            arrows (a number of 2 plots every second arrow). Default is 1.

        Returns
        -------
            None

        '''
        self._log.debug('Calling display_mask')
        if self.mask is not None:
            from mayavi import mlab
            zz, yy, xx = np.indices(self.dim)
            ad = ar_dens
            zz = zz[::ad, ::ad, ::ad].flatten()
            yy = yy[::ad, ::ad, ::ad].flatten()
            xx = xx[::ad, ::ad, ::ad].flatten()
            mask_vec = self.mask[::ad, ::ad, ::ad].flatten().astype(dtype=np.int)
            mlab.figure(size=(750, 700))
            plot = plot = mlab.points3d(xx, yy, zz, mask_vec, opacity=0.5, mode='cube',
                                        scale_factor=ar_dens)
            mlab.outline(plot)
            mlab.axes(plot)
            return plot

    def display_phase(self, mag_data=None, title='Phase Map',
                      cmap='RdBu', limit=None, norm=None):
        '''Display all phasemaps saved in the :class:`~.DataSet` as a colormesh.

        Parameters
        ----------
        mag_data : :class:`~.MagData`, optional
            Magnetic distribution to which the projectors of the dataset should be applied. If not
            given, the phase_maps in the dataset are used.
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
        self._log.debug('Calling display_phase')
        if mag_data is not None:
            phase_maps = self.create_phase_maps(mag_data)
        else:
            phase_maps = self.phase_maps
        [phase_map.display_phase('{} ({})'.format(title, self.projectors[i].get_info()),
                                 cmap, limit, norm)
            for (i, phase_map) in enumerate(phase_maps)]
        plt.show()

    def display_combined(self, mag_data=None, title='Combined Plot', cmap='RdBu', limit=None,
                         norm=None, gain='auto', interpolation='none', grad_encode='bright'):
        '''Display all phasemaps and the resulting color coded holography images.

        Parameters
        ----------
        mag_data : :class:`~.MagData`, optional
            Magnetic distribution to which the projectors of the dataset should be applied. If not
            given, the phase_maps in the dataset are used.
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
        gain : float, optional
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
        self._log.debug('Calling display_combined')
        if mag_data is not None:
            phase_maps = self.create_phase_maps(mag_data)
        else:
            phase_maps = self.phase_maps
        for (i, phase_map) in enumerate(phase_maps):
            phase_map.display_combined('{} ({})'.format(title, self.projectors[i].get_info()),
                                       cmap, limit, norm, gain, interpolation, grad_encode)
        plt.show()

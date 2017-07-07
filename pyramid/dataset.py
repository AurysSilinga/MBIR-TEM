# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""This module provides the :class:`~.DataSet` class for the collection of phase maps
and additional data like corresponding projectors."""

import logging
from numbers import Number

from pyramid.kernel import Kernel
from pyramid.phasemap import PhaseMap
from pyramid.phasemapper import PhaseMapperRDFC
from pyramid.projector import Projector
from pyramid.fielddata import ScalarData
from pyramid.ramp import Ramp

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

__all__ = ['DataSet']


class DataSet(object):
    """Class for collecting phase maps and corresponding projectors.

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
    b_0: double
        The saturation induction in `T`.
    mask: :class:`~numpy.ndarray` (N=3), optional
        A boolean mask which defines the magnetized volume in 3D.
    Se_inv : :class:`~numpy.ndarray` (N=2), optional
        Inverted covariance matrix of the measurement errors. The matrix has size `NxN` with N
        being the length of the targetvector y (vectorized phase map information).
    projectors: list of :class:`~.Projector`
        A list of all stored :class:`~.Projector` objects.
    phasemaps: list of :class:`~.PhaseMap`
        A list of all stored :class:`~.PhaseMap` objects.
    phase_vec: :class:`~numpy.ndarray` (N=1)
        The concatenaded, vectorized phase of all :class:`~.PhaseMap` objects.
    count(self): int
        Number of phase maps and projectors in the dataset.
    hook_points(self): :class:`~numpy.ndarray` (N=1)
        Hook points which determine the start of values of a phase map in the `phase_vec`.
        The length is `count + 1`.

    """

    _log = logging.getLogger(__name__ + '.DataSet')

    @property
    def a(self):
        """The grid spacing in nm."""
        return self._a

    @a.setter
    def a(self, a):
        assert isinstance(a, Number), 'Grid spacing has to be a number!'
        assert a >= 0, 'Grid spacing has to be a positive number!'
        self._a = float(a)

    @property
    def mask(self):
        """A boolean mask which defines the magnetized volume in 3D."""
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
        """Size of the image space."""
        return np.sum([len(p.phase_vec) for p in self.phasemaps])

    @property
    def n(self):
        """Size of the input space."""
        return 3 * np.sum(self.mask)

    @property
    def count(self):
        """Number of phase maps and projectors in the dataset."""
        return len(self.projectors)

    @property
    def phase_vec(self):
        """The concatenaded, vectorized phase of all ;class:`~.PhaseMap` objects."""
        return np.concatenate([p.phase_vec for p in self.phasemaps])

    @property
    def hook_points(self):
        """Hook points which determine the start of values of a phase map in the `phase_vec`."""
        result = [0]
        for i, phasemap in enumerate(self.phasemaps):
            result.append(result[i] + np.prod(phasemap.dim_uv))
        return result

    @property
    def phasemaps(self):
        """List of all PhaseMaps in the DataSet."""
        return self._phasemaps

    @property
    def projectors(self):
        """List of all Projectors in the DataSet."""
        return self._projectors

    @property
    def phasemappers(self):
        # TODO: get rid, only use phasemapper_dict!!
        """List of all PhaseMappers in the DataSet."""
        return self._phasemappers

    @property
    def phasemapper_dict(self):
        """Dictionary of all PhaseMappers in the DataSet."""
        return self._phasemapper_dict

    def __init__(self, a, dim, b_0=1, mask=None, Se_inv=None):
        dim = tuple(dim)
        self._log.debug('Calling __init__')
        assert isinstance(dim, tuple) and len(dim) == 3, \
            'Dimension has to be a tuple of length 3!'
        self.a = a
        self.dim = dim
        self.b_0 = b_0
        self.mask = mask
        self.Se_inv = Se_inv
        self._phasemaps = []
        self._projectors = []
        self._phasemappers = []
        self._phasemapper_dict = {}
        self._log.debug('Created: ' + str(self))

    def __repr__(self):
        self._log.debug('Calling __repr__')
        return '%s(a=%r, dim=%r, b_0=%r, mask=%r, Se_inv=%r)' % (self.__class__, self.a, self.dim,
                                                                 self.b_0, self.mask, self.Se_inv)

    def __str__(self):
        self._log.debug('Calling __str__')
        return 'DataSet(a=%s, dim=%s, b_0=%s)' % (self.a, self.dim, self.b_0)

    def _append_single(self, phasemap, projector, phasemapper=None):
        self._log.debug('Calling _append')
        assert isinstance(phasemap, PhaseMap) and isinstance(projector, Projector), \
            'Argument has to be a tuple of a PhaseMap and a Projector object!'
        dim_uv = projector.dim_uv
        assert projector.dim == self.dim, '3D dimensions must match!'
        assert phasemap.dim_uv == dim_uv, 'Projection dimensions (dim_uv) must match!'
        assert phasemap.a == self.a, 'Grid spacing must match!'
        # Create lookup key:
        # TODO: Think again if phasemappers should be given as attribute (seems to be faulty
        # TODO: currently... Also not very expensive, so keep outside?
        if phasemapper is not None:
            key = dim_uv  # Create standard phasemapper, dim_uv is enough for identification!
        else:
            key = (dim_uv, str(phasemapper))  # Include string representation for identification!
        # Retrieve existing, use given or create new phasemapper:
        if key in self.phasemapper_dict:  # Retrieve existing phasemapper:
            phasemapper = self.phasemapper_dict[key]
        elif phasemapper is not None:  # Use given one (do nothing):
            pass
        else:  # Create new standard (RDFC) phasemapper:
            phasemapper = PhaseMapperRDFC(Kernel(self.a, dim_uv, self.b_0))
        self._phasemapper_dict[key] = phasemapper
        # Append everything to the lists (just contain pointers to objects!):
        self._phasemaps.append(phasemap)
        self._projectors.append(projector)
        self._phasemappers.append(phasemapper)

    def append(self, phasemap, projector, phasemapper=None):
        """Appends a data pair of phase map and projection infos to the data collection.`

        Parameters
        ----------
        phasemap: :class:`~.PhaseMap`
            A :class:`~.PhaseMap` object which should be added to the data collection.
        projector: :class:`~.Projector`
            A :class:`~.Projector` object which should be added to the data collection.
        phasemapper: :class:`~.PhaseMapper`, optional
            An optional :class:`~.PhaseMapper` object which should be added.

        Returns
        -------
        None

        """
        self._log.debug('Calling append')
        if type(phasemap) is not list:
            phasemap = [phasemap]
        if type(projector) is not list:
            projector = [projector]
        if type(phasemapper) is not list:
            phasemapper = [phasemapper] * len(phasemap)
        assert len(phasemap) == len(projector),\
            ('Phasemaps and projectors must have same' +
             'length(phasemaps: {}, projectors: {})!'.format(len(phasemap), len(projector)))
        for i in range(len(phasemap)):
            self._append_single(phasemap[i], projector[i], phasemapper[i])
        # Reset the Se_inv matrix from phasemaps confidence matrices:
        self.set_Se_inv_diag_with_conf()

    def create_phasemaps(self, magdata, difference=False, ramp=None):
        """Create a list of phasemaps with the projectors in the dataset for a given
        :class:`~.VectorData` object.

        Parameters
        ----------
        magdata : :class:`~.VectorData`
            Magnetic distribution to which the projectors of the dataset should be applied.
        difference : bool, optional
            If `True`, the phasemaps of the dataset are subtracted from the created ones to view
            difference images. Default is False.
        ramp : :class:`~.Ramp`
            A ramp object, which can be specified to add a ramp to the generated phasemaps.
            If `difference` is `True`, this can be interpreted as ramp correcting the phasemaps
            saved in the dataset.

        Returns
        -------
        phasemaps : list of :class:`~.phasemap.PhaseMap`
            A list of the phase maps resulting from the projections specified in the dataset.

        """
        self._log.debug('Calling create_phasemaps')
        phasemaps = []
        for i, projector in enumerate(self.projectors):
            mag_proj = projector(magdata)
            phasemap = self.phasemappers[i](mag_proj)
            if difference:
                phasemap -= self.phasemaps[i]
            if ramp is not None:
                assert type(ramp) == Ramp, 'ramp has to be a Ramp object!'
                phasemap += ramp(index=i)  # Full formula: phasemap -= phasemap_dataset - ramp
            phasemap.mask = mag_proj.get_mask()[0, ...]
            phasemaps.append(phasemap)
        return phasemaps

    def set_Se_inv_block_diag(self, cov_list):
        """Set the Se_inv matrix as a block diagonal matrix

        Parameters
        ----------
        cov_list: list of :class:`~numpy.ndarray`
            List of inverted covariance matrices (one for each projection).

        Returns
        -------
            None

        """
        self._log.debug('Calling set_Se_inv_block_diag')
        assert len(cov_list) == len(self.phasemaps), 'Needs one covariance matrix per phase map!'
        self.Se_inv = sparse.block_diag(cov_list).tocsr()

    def set_Se_inv_diag_with_conf(self, conf_list=None):
        """Set the Se_inv matrix as a block diagonal matrix from a list of confidence matrizes.

        Parameters
        ----------
        conf_list: list of :class:`~numpy.ndarray` (optional)
            List of 2D confidence matrizes (one for each projection) which define trust regions.
            If not given this uses the confidence matrizes of the phase maps.

        Returns
        -------
            None

        """
        self._log.debug('Calling set_Se_inv_diag_with_conf')
        if conf_list is None:  # if no confidence matrizes are given, extract from the phase maps!
            conf_list = [phasemap.confidence for phasemap in self.phasemaps]
        cov_list = [sparse.diags(c.ravel().astype(np.float32), 0) for c in conf_list]
        self.set_Se_inv_block_diag(cov_list)

    def set_3d_mask(self, mask_list=None, threshold=0.9):
        # TODO: This function should be in a separate module and not here (maybe?)!
        """Set the 3D mask from a list of 2D masks.

        Parameters
        ----------
        mask_list: list of :class:`~numpy.ndarray` (optional)
            List of 2D masks, which represent the projections of the 3D mask. If not given this
            uses the mask matrizes of the phase maps. If just one phase map is present, the
            according mask is simply expanded to 3D and used directly.
        threshold: float, optional
            The threshold, describing the minimal number of 2D masks which have to extrude to the
            point in 3D to be considered valid as containing magnetisation. `threshold` is a
            relative number in the range of [0, 1]. The default is 0.9. Choosing a value of 1 is
            the strictest possible setting (every 2D mask has to contain a 3D point to be valid).

        Returns
        -------
            None

        """
        self._log.debug('Calling set_3d_mask')
        if mask_list is None:  # if no masks are given, extract from phase maps:
            mask_list = [phasemap.mask for phasemap in self.phasemaps]
        if len(mask_list) == 1:  # just one phasemap --> 3D mask equals 2D mask
            self.mask = np.expand_dims(mask_list[0], axis=0)  # z-dim is set to 1!
        else:  # 3D mask has to be constructed from 2D masks:
            mask_3d = np.zeros(self.dim)
            for i, projector in enumerate(self.projectors):
                mask_2d = self.phasemaps[i].mask.reshape(-1)  # 2D mask
                # Add extrusion of 2D mask:
                mask_3d += projector.weight.T.dot(mask_2d).reshape(self.dim)
            self.mask = np.where(mask_3d >= threshold * self.count, True, False)

    def save(self, filename, overwrite=True):
        """Saves the dataset as a collection of HDF5 files.

        Parameters
        ----------
        filename: str
            Base name of the files which the dataset is saved into. HDF5 files are supported.
        overwrite: bool, optional
            If True (default), an existing file will be overwritten, if False, this
            (silently!) does nothing.
        """
        from .file_io.io_dataset import save_dataset
        save_dataset(self, filename, overwrite)

    def plot_mask(self):
        """If it exists, display the 3D mask of the magnetization distribution.

        Returns
        -------
            None

        """
        self._log.debug('Calling plot_mask')
        if self.mask is not None:
            return ScalarData(self.a, self.mask).plot_mask()

    def plot_phasemaps(self, magdata=None, title='Phase Map', difference=False, ramp=None,
                       **kwargs):
        """Display all phasemaps saved in the :class:`~.DataSet` as a colormesh.

        Parameters
        ----------
        magdata : :class:`~.VectorData`, optional
            Magnetic distribution to which the projectors of the dataset should be applied. If not
            given, the phasemaps in the dataset are used.
        title : string, optional
            The main part of the title of the plots. The default is 'Phase Map'. Additional
            projector info is appended to this.
        difference : bool, optional
            If `True`, the phasemaps of the dataset are subtracted from the created ones to view
            difference images. Default is False.
        ramp : :class:`~.Ramp`
            A ramp object, which can be specified to add a ramp to the generated phasemaps.
            If `magdata` is not given, this will instead just ramp correct the phasemaps saved
            in the dataset.

        Returns
        -------
        None

        """
        self._log.debug('Calling plot_phasemaps')
        if magdata is not None:  # Plot phasemaps of the given magnetisation distribution:
            phasemaps = self.create_phasemaps(magdata, difference=difference, ramp=ramp)
        else:  # Plot phasemaps saved in the DataSet (default):
            phasemaps = self.phasemaps
            if ramp is not None:
                for i, phasemap in enumerate(phasemaps):
                    assert type(ramp) == Ramp, 'ramp has to be a Ramp object!'
                    phasemap -= ramp(index=i)  # Ramp correction
        for (i, phasemap) in enumerate(phasemaps):
            phasemap.plot_phase(note='{} ({})'.format(title, self.projectors[i].get_info()),
                                **kwargs)

    def plot_phasemaps_combined(self, magdata=None, title='Combined Plot', difference=False,
                                ramp=None, **kwargs):
        """Display all phasemaps and the resulting color coded holography images.

        Parameters
        ----------
        magdata : :class:`~.VectorData`, optional
            Magnetic distribution to which the projectors of the dataset should be applied. If not
            given, the phasemaps in the dataset are used.
        title : string, optional
            The title of the plot. The default is 'Combined Plot'.
        difference : bool, optional
            If `True`, the phasemaps of the dataset are subtracted from the created ones to view
            difference images. Default is False.
        ramp : :class:`~.Ramp`
            A ramp object, which can be specified to add a ramp to the generated phasemaps.
            If `magdata` is not given, this will instead just ramp correct the phasemaps saved
            in the dataset.

        Returns
        -------
        None

        """
        self._log.debug('Calling plot_phasemaps_combined')
        if magdata is not None:
            phasemaps = self.create_phasemaps(magdata, difference=difference, ramp=ramp)
        else:
            phasemaps = self.phasemaps
            if ramp is not None:
                for i, phasemap in enumerate(phasemaps):
                    assert type(ramp) == Ramp, 'ramp has to be a Ramp object!'
                    phasemap -= ramp(index=i)  # Ramp correction
        for (i, phasemap) in enumerate(phasemaps):
            phasemap.plot_combined(note='{} ({})'.format(title, self.projectors[i].get_info()),
                                   **kwargs)
        plt.show()

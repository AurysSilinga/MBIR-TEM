# -*- coding: utf-8 -*-
# Copyright 2016 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""IO functionality for Projector objects."""

import logging

import os

from scipy.sparse import csr_matrix

import numpy as np

import h5py

from .. import projector

__all__ = ['load_projector']
_log = logging.getLogger(__name__)


def save_projector(projector, filename, overwrite=True):
    """%s"""
    _log.debug('Calling save_projector')
    name, extension = os.path.splitext(filename)
    assert extension in ['.hdf5', ''], 'For now only HDF5 format is supported!'
    filename = name + '.hdf5'  # In case no extension is provided, set to HDF5!
    if not os.path.isfile(filename) or overwrite:  # Write if file does not exist or if forced:
        with h5py.File(filename, 'w') as f:
            class_name = projector.__class__.__name__
            f.attrs['class'] = class_name
            if class_name == 'SimpleProjector':
                f.attrs['axis'] = projector.axis
            else:
                f.attrs['tilt'] = projector.tilt
                if class_name == 'RotTiltProjector':
                    f.attrs['rotation'] = projector.rotation
            f.attrs['dim'] = projector.dim
            f.attrs['dim_uv'] = projector.dim_uv
            f.create_dataset('data', data=projector.weight.data)
            f.create_dataset('indptr', data=projector.weight.indptr)
            f.create_dataset('indices', data=projector.weight.indices)
            f.create_dataset('coeff', data=projector.coeff)
save_projector.__doc__ %= projector.Projector.save.__doc__


def load_projector(filename):
    """Load HDF5 file into a :class:`~pyramid.projector.Projector` instance (or a subclass).

    Parameters
    ----------
    filename:  str
        The filename to be loaded.

    Returns
    -------
    projector : :class:`~.Projector`
        A :class:`~.Projector` object containing the loaded data.

    """
    _log.debug('Calling load_projector')
    name, extension = os.path.splitext(filename)
    assert extension in ['.hdf5', ''], 'For now only HDF5 format is supported!'
    filename = name + '.hdf5'  # In case no extension is provided, set to HDF5!
    with h5py.File(filename, 'r') as f:
        # Retrieve dimensions:
        dim = f.attrs.get('dim')
        dim_uv = f.attrs.get('dim_uv')
        size_2d, size_3d = np.prod(dim_uv), np.prod(dim)
        # Retrieve weight matrix:
        data = f.get('data')
        indptr = f.get('indptr')
        indices = f.get('indices')
        weight = csr_matrix((data, indices, indptr), shape=(size_2d, size_3d))
        # Retrieve coefficients:
        coeff = np.copy(f.get('coeff'))
        # Construct projector:
        result = projector.Projector(dim, dim_uv, weight, coeff)
        # Specify projector type:
        class_name = f.attrs.get('class')
        result.__class__ = getattr(projector, class_name)
        if class_name == 'SimpleProjector':
            result.axis = f.attrs.get('axis')
        else:
            result.tilt = f.attrs.get('tilt')
            if class_name == 'RotTiltProjector':
                result.rotation = f.attrs.get('rotation')
        # Return projector object:
        return result

# -*- coding: utf-8 -*-
# Copyright 2016 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""IO functionality for DataSet objects."""

import logging

import os

import h5py

import numpy as np

from ..dataset import DataSet
from ..file_io.io_projector import load_projector
from ..file_io.io_phasemap import load_phasemap

__all__ = ['load_projector']
_log = logging.getLogger(__name__)


def save_dataset(dataset, filename, overwrite=True):
    """%s"""
    _log.debug('Calling save_dataset')
    path, filename = os.path.split(filename)
    name, extension = os.path.splitext(filename)
    assert extension in ['.hdf5', ''], 'For now only HDF5 format is supported!'
    filename = name + '.hdf5'  # In case no extension is provided, set to HDF5!
    # Header file:
    header_name = os.path.join(path, 'dataset_{}'.format(filename))
    if not os.path.isfile(header_name) or overwrite:  # Write if file does not exist or if forced:
        with h5py.File(header_name, 'w') as f:
            f.attrs['a'] = dataset.a
            f.attrs['dim'] = dataset.dim
            f.attrs['b_0'] = dataset.b_0
            if dataset.mask is not None:
                f.create_dataset('mask', data=dataset.mask)
            if dataset.Se_inv is not None:
                f.create_dataset('Se_inv', data=dataset.Se_inv)
    # PhaseMaps and Projectors:
    for i, projector in enumerate(dataset.projectors):
        projector_name = 'projector_{}_{}_{}{}'.format(name, i, projector.get_info(), extension)
        projector.save(os.path.join(path, projector_name), overwrite=overwrite)
        phasemap_name = 'phasemap_{}_{}_{}{}'.format(name, i, projector.get_info(), extension)
        dataset.phasemaps[i].save(os.path.join(path, phasemap_name), overwrite=overwrite)
save_dataset.__doc__ %= DataSet.save.__doc__


def load_dataset(filename):
    """Load HDF5 file into a :class:`~pyramid.dataset.DataSet` instance.

    Parameters
    ----------
    filename:  str
        The filename to be loaded.

    Returns
    -------
    projector : :class:`~.Projector`
        A :class:`~.Projector` object containing the loaded data.

    Notes
    -----
    This loads a header file and all matching HDF5 files which can be found. The filename
    conventions have to be strictly followed for the process to be successful!

    """
    _log.debug('Calling load_dataset')
    path, filename = os.path.split(filename)
    if path == '':
        path = '.'  # Make sure this can be used later!
    name, extension = os.path.splitext(filename)
    assert extension in ['.hdf5', ''], 'For now only HDF5 format is supported!'
    if name.startswith('dataset_'):
        name = name.split('dataset_')[1]
    filename = name + '.hdf5'  # In case no extension is provided, set to HDF5!
    # Header file:
    header_name = os.path.join(path, 'dataset_{}'.format(filename))
    with h5py.File(header_name, 'r') as f:
        a = f.attrs.get('a')
        dim = f.attrs.get('dim')
        b_0 = f.attrs.get('b_0')
        mask = np.copy(f.get('mask', None))
        Se_inv = np.copy(f.get('Se_inv', None))
        dataset = DataSet(a, dim, b_0, mask, Se_inv)
    # Projectors:
    projectors = []
    for f in os.listdir(path):
        if f.startswith('projector') and f.endswith('.hdf5'):
            projector_name, i = f.split('_')[1:3]
            if projector_name == name:
                projector = load_projector(os.path.join(path, f))
                projectors.append((int(i), projector))
    projectors = [p[1] for p in sorted(projectors, key=lambda x: x[0])]
    dataset.projectors = projectors
    # PhaseMaps:
    phasemaps = []
    for f in os.listdir(path):
        if f.startswith('phasemap') and f.endswith('.hdf5'):
            phasemap_name, i = f.split('_')[1:3]
            if phasemap_name == name:
                phasemap = load_phasemap(os.path.join(path, f))
                phasemaps.append((int(i), phasemap))
    phasemaps = [p[1] for p in sorted(phasemaps, key=lambda x: x[0])]
    dataset.phasemaps = phasemaps
    # Return DataSet:
    return dataset

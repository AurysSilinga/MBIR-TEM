# -*- coding: utf-8 -*-
# Copyright 2016 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""IO functionality for ScalarData objects."""

import logging

import os

import numpy as np

from ..fielddata import ScalarData

__all__ = ['load_scalardata']
_log = logging.getLogger(__name__)


def load_scalardata(filename, a=None, **kwargs):
    """Load supported file into a :class:`~pyramid.fielddata.ScalarData` instance.

    The function loads the file according to the extension:
        - hdf5 for HDF5.
        - EMD Electron Microscopy Dataset format (also HDF5).
        - npy or npz for numpy formats.

    Any extra keyword is passed to the corresponsing reader. For
    available options see their individual documentation.

    Parameters
    ----------
    filename:  str
        The filename to be loaded.
    a: float or None, optional
        If the grid spacing is not None, it will override a possibly loaded value from the files.

    Returns
    -------
    scalardata : :class:`~.ScalarData`
        A :class:`~.ScalarData` object containing the loaded data.

    """
    _log.debug('Calling load_scalardata')
    extension = os.path.splitext(filename)[1]
    # Load from npy-files:
    if extension in ['.npy', '.npz']:
        return _load_from_npy(filename, a, **kwargs)
    # Load with HyperSpy:
    else:
        if extension == '':
            filename = '{}.hdf5'.format(filename)  # Default!
        return _load_from_hs(filename, a, **kwargs)


def _load_from_npy(filename, a, **kwargs):
    _log.debug('Calling load_from_npy')
    if a is None:
        a = 1.  # Use default!
    return ScalarData(a, np.load(filename, **kwargs))


def _load_from_hs(filename, a, **kwargs):
    _log.debug('Calling load_from_hs')
    try:
        import hyperspy.api as hs
    except ImportError:
        _log.error('This method recquires the hyperspy package!')
        return
    scalardata = ScalarData.from_signal(hs.load(filename, **kwargs))
    if a is not None:
        scalardata.a = a
    return scalardata


def save_scalardata(scalardata, filename, **kwargs):
    """%s"""
    _log.debug('Calling save_scalardata')
    extension = os.path.splitext(filename)[1]
    if extension in ['.npy', '.npz']:  # Save to npy-files:
        _save_to_npy(scalardata, filename, **kwargs)
    else:  # Try HyperSpy:
        _save_to_hs(scalardata, filename, **kwargs)
save_scalardata.__doc__ %= ScalarData.save.__doc__


def _save_to_npy(scalardata, filename, **kwargs):
    _log.debug('Calling save_to_npy')
    np.save(filename, scalardata.field, **kwargs)


def _save_to_hs(scalardata, filename, **kwargs):
    _log.debug('Calling save_to_hyperspy')
    scalardata.to_signal().save(filename, **kwargs)

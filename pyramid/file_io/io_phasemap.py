# -*- coding: utf-8 -*-
# Copyright 2016 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""IO functionality for FieldData objects."""

import logging

import os

import numpy as np

from PIL import Image

from ..phasemap import PhaseMap

__all__ = ['load_phasemap']
_log = logging.getLogger(__name__)


def load_phasemap(filename, mask=None, confidence=None, a=None, **kwargs):
    """Load supported file into a :class:`~pyramid.phasemap.PhaseMap` instance.

    The function loads the file according to the extension:
        - hdf5 for HDF5.
        - rpl for Ripple (useful to export to Digital Micrograph).
        - dm3 and dm4 for Digital Micrograph files.
        - unf for SEMPER unf binary format.
        - txt format.
        - npy or npz for numpy formats.
        - Many image formats such as png, tiff, jpeg...

    Any extra keyword is passed to the corresponsing reader. For
    available options see their individual documentation.

    Parameters
    ----------
    filename:  str
        The filename to be loaded.
    mask: str or tuple of str and dict, optional
        If this is a filename, a mask will be loaded from an appropriate file. If additional
        keywords should be provided, use a tuple of the filename and an according dictionary.
        If this is `None`, no mask will be loaded.
        default is False.
    confidence: str or tuple of str and dict, optional
        If this is a filename, a confidence matrix will be loaded from an appropriate file. If
        additional keywords should be provided, use a tuple of the filename and an according
        dictionary. If this is `None`, no mask will be loaded.
    a: float or None, optional
        If the grid spacing is not None, it will override a possibly loaded value from the files.

    Returns
    -------
    phasemap : :class:`~.PhaseMap`
        A :class:`~.PhaseMap` object containing the loaded data.

    """
    _log.debug('Calling load_phasemap')
    phasemap = _load(filename, as_phasemap=True, a=a, **kwargs)
    if mask is not None:
        filemask, kwargs_mask = _parse_add_param(mask)
        phasemap.mask = _load(filemask, **kwargs_mask)
    if confidence is not None:
        fileconf, kwargs_conf = _parse_add_param(confidence)
        phasemap.confidence = _load(fileconf, **kwargs_conf)
    return phasemap


def _load(filename, as_phasemap=False, a=1., **kwargs):
    _log.debug('Calling _load')
    extension = os.path.splitext(filename)[1]
    # Load from txt-files:
    if extension == '.txt':
        return _load_from_txt(filename, as_phasemap, a, **kwargs)
    # Load from npy-files:
    elif extension in ['.npy', '.npz']:
        return _load_from_npy(filename, as_phasemap, a, **kwargs)
    elif extension in ['.jpeg', '.jpg', '.png', '.bmp', '.tif']:
        return _load_from_img(filename, as_phasemap, a, **kwargs)
    # Load with HyperSpy:
    else:
        if extension == '':
            filename = '{}.hdf5'.format(filename)  # Default!
        return _load_from_hs(filename, as_phasemap, a, **kwargs)


def _parse_add_param(param):
    if param is None:
        return None, {}
    elif isinstance(param, str):
        return param, {}
    elif isinstance(param, (list, tuple)) and len(param) == 2:
        return param
    else:
        raise ValueError('Parameter can be a string or a tuple/list of a string and a dict!')


def _load_from_txt(filename, as_phasemap, a, **kwargs):

    def _load_arr(filename, **kwargs):
        with open(filename, 'r') as phase_file:
            if phase_file.readline().startswith('PYRAMID'):  # File has pyramid structure:
                return np.loadtxt(filename, delimiter='\t', skiprows=2)
            else:  # Try default args:
                return np.loadtxt(filename, **kwargs)

    result = _load_arr(filename, **kwargs)
    if as_phasemap:
        if a is None:
            a = 1.  # Default!
            with open(filename, 'r') as phase_file:
                header = phase_file.readline()
                if header.startswith('PYRAMID'):  # File has pyramid structure:
                    a = float(phase_file.readline()[15:-4])
        return PhaseMap(a, result)
    else:
        return result


def _load_from_npy(filename, as_phasemap, a, **kwargs):

    result = np.load(filename, **kwargs)
    if as_phasemap:
        if a is None:
            a = 1.  # Use default!
        return PhaseMap(a, result)
    else:
        return result


def _load_from_img(filename, as_phasemap, a, **kwargs):

    result = np.asarray(Image.open(filename, **kwargs).convert('L'))
    if as_phasemap:
        if a is None:
            a = 1.  # Use default!
        return PhaseMap(a, result)
    else:
        return result


def _load_from_hs(filename, as_phasemap, a, **kwargs):
    try:
        import hyperspy.api as hs
    except ImportError:
        _log.error('This method recquires the hyperspy package!')
        return
    phasemap = PhaseMap.from_signal(hs.load(filename, **kwargs))
    if as_phasemap:
        if a is not None:
            phasemap.a = a
        return phasemap
    else:
        return phasemap.phase


def save_phasemap(phasemap, filename, save_mask, save_conf, pyramid_format, **kwargs):
    """%s"""
    _log.debug('Calling save_phasemap')
    extension = os.path.splitext(filename)[1]
    if extension == '.txt':  # Save to txt-files:
        _save_to_txt(phasemap, filename, pyramid_format, save_mask, save_conf, **kwargs)
    elif extension in ['.npy', '.npz']:  # Save to npy-files:
        _save_to_npy(phasemap, filename, save_mask, save_conf, **kwargs)
    else:  # Try HyperSpy:
        _save_to_hs(phasemap, filename, save_mask, save_conf, **kwargs)
save_phasemap.__doc__ %= PhaseMap.save.__doc__


def _save_to_txt(phasemap, filename, pyramid_format, save_mask, save_conf, **kwargs):

    def _save_arr(filename, array, tag, **kwargs):
        if pyramid_format:
            with open(filename, 'w') as phase_file:
                name = os.path.splitext(os.path.split(filename)[1])[0]
                phase_file.write('PYRAMID-{}: {}\n'.format(tag, name))
                phase_file.write('grid spacing = {} nm\n'.format(phasemap.a))
            save_kwargs = {'fmt': '%7.6e', 'delimiter': '\t'}
        else:
            save_kwargs = kwargs
        with open(filename, 'ba') as phase_file:
            np.savetxt(phase_file, array, **save_kwargs)

    name, extension = os.path.splitext(filename)
    _save_arr('{}{}'.format(name, extension), phasemap.phase, 'PHASEMAP', **kwargs)
    if save_mask:
        _save_arr('{}_mask{}'.format(name, extension), phasemap.mask, 'MASK', **kwargs)
    if save_conf:
        _save_arr('{}_conf{}'.format(name, extension), phasemap.confidence, 'CONFIDENCE', **kwargs)


def _save_to_npy(phasemap, filename, save_mask, save_conf, **kwargs):
    name, extension = os.path.splitext(filename)
    np.save('{}{}'.format(name, extension), phasemap.phase, **kwargs)
    if save_mask:
        np.save('{}_mask{}'.format(name, extension), phasemap.mask, **kwargs)
    if save_conf:
        np.save('{}_conf{}'.format(name, extension), phasemap.confidence, **kwargs)


def _save_to_hs(phasemap, filename, save_mask, save_conf, **kwargs):
    name, extension = os.path.splitext(filename)
    phasemap.to_signal().save(filename, **kwargs)
    if extension not in ['.hdf5', '.HDF5', '']:  # mask and confidence are saved separately:
        import hyperspy.api as hs
        if save_mask:
            mask_name = '{}_mask{}'.format(name, extension)
            hs.signals.Signal2D(phasemap.mask, **kwargs).save(mask_name)
        if save_conf:
            conf_name = '{}_conf{}'.format(name, extension)
            hs.signals.Signal2D(phasemap.confidence, **kwargs).save(conf_name)

# -*- coding: utf-8 -*-
# Copyright 2016 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""IO functionality for VectorData objects."""

import logging

import os

import numpy as np

from ..fielddata import VectorData

__all__ = ['load_vectordata']
_log = logging.getLogger(__name__)


def load_vectordata(filename, a=None, **kwargs):
    """Load supported file into a :class:`~pyramid.fielddata.VectorData` instance.

    The function loads the file according to the extension:
        - hdf5 for HDF5.
        - EMD Electron Microscopy Dataset format (also HDF5).
        - llg format.
        - ovf format.
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
    vectordata : :class:`~.VectorData`
        A :class:`~.VectorData` object containing the loaded data.

    """
    _log.debug('Calling load_vectordata')
    extension = os.path.splitext(filename)[1]
    # Load from llg-files:
    if extension in ['.llg', '.txt']:
        return _load_from_llg(filename, a)
    # Load from ovf-files:
    if extension in ['.ovf', '.omf', '.ohf', 'obf']:
        return _load_from_ovf(filename, a)
    # Load from npy-files:
    elif extension in ['.npy', '.npz']:
        return _load_from_npy(filename, a, **kwargs)
    # Load with HyperSpy:
    else:
        if extension == '':
            filename = '{}.hdf5'.format(filename)  # Default!
        return _load_from_hs(filename, a, **kwargs)


def _load_from_llg(filename, a):
    _log.debug('Calling load_from_llg')
    SCALE = 1.0E-9 / 1.0E-2  # From cm to nm
    data = np.genfromtxt(filename, skip_header=2)
    dim = tuple(np.genfromtxt(filename, dtype=int, skip_header=1, skip_footer=len(data[:, 0])))
    if a is None:
        a = (data[1, 0] - data[0, 0]) / SCALE
    field = data[:, 3:6].T.reshape((3,) + dim)
    return VectorData(a, field)


def _load_from_ovf(filename, a):
    _log.debug('Calling load_from_ovf')
    with open(filename, 'rb') as mag_file:
        assert mag_file.readline().startswith(b'# OOMMF')  # Make sure file has .ovf-format!
        read_header, read_data = False, False
        header = {}
        x_mag, y_mag, z_mag = [], [], []
        data_mode = ''
        for line in mag_file:
            # Read in additional info:
            if not read_header and not read_data:
                if line.startswith(b'# Segment count'):
                    assert int(line.split()[3]) == 1, 'Only one vector field can be read at time!'
                elif line.startswith(b'# Begin: Header'):
                    read_header = True
                elif line.startswith(b'# Begin: Data'):
                    read_data = True
                    data_mode = ' '.join(line.decode('utf-8').split()[3:])
                    assert data_mode in ['text', 'Text', 'Binary 4', 'Binary 8'], \
                        'Data mode {} is currently not supported by this reader!'.format(data_mode)
                    assert header.get('meshtype') == 'rectangular', \
                        'Only rectangular grids can be currently read!'
            # Read in header:
            elif read_header:  # Read header:
                if line.startswith(b'# End: Header'):  # Header is done:
                    read_header = False
                    continue
                line = line.decode('utf-8')  # Decode to use strings here!
                line_list = line.split()
                if '##' in line_list:  # Strip trailing comments:
                    del line_list[line_list.index('##'):]
                if len(line_list) <= 1:  # Just '#' or empty line:
                    continue
                key, value = line_list[1].strip(':'), ' '.join(line_list[2:])
                if key not in header:  # Add new key, value pair:
                    header[key] = value
                elif key == 'Desc':  # Can go over several lines:
                    header['Desc'] = ' '.join([header['Desc'], value])
            # Read in data:
            # TODO: Make it work for both text and binary! Put into HyperSpy?
            # TODO: http://math.nist.gov/oommf/doc/userguide11b2/userguide/vectorfieldformat.html
            elif read_data:  # Currently in a data block:
                if data_mode in ['text', 'Text']:  # Read data as text:
                    if line.startswith(b'# End: Data'):  # Header is done:
                        read_data = False
                    else:
                        x, y, z = [float(i) for i in line.split()]
                        x_mag.append(x)
                        y_mag.append(y)
                        z_mag.append(z)
                elif 'Binary' in data_mode:
                    count = int(data_mode.split()[-1])
                    dtype = '>f{}'.format(count)
                    dim = (int(header['znodes']), int(header['ynodes']), int(header['xnodes']))
                    test = np.fromfile(mag_file, dtype='<f4', count=count*2+1)
                    if count == 4:  # Binary 4:
                        assert test == '123456789.0', 'Wrong test bytes!'
                    if count == 8:  # Binary 4:
                        assert test == '123456789012345.0', 'Wrong test bytes!'
                    data = np.fromfile(mag_file, dtype=dtype, count=3*np.prod(dim))
                    data.reshape((3,) + dim)
                    x_mag, y_mag, z_mag = data
                    read_data = False  # Stop reading data and search for new Segments (if any).
        # Format after reading:
        dim = (int(header['znodes']), int(header['ynodes']), int(header['xnodes']))
        x_mag = np.asarray(x_mag).reshape(dim)
        y_mag = np.asarray(y_mag).reshape(dim)
        z_mag = np.asarray(z_mag).reshape(dim)
        field = np.asarray((x_mag, y_mag, z_mag)) * float(header.get('valuemultiplier', 1))
        if a is None:
            assert header.get('xstepsize') == header.get('ystepsize') == header.get('zstepsize'), \
                'Grid spacing is not equidistant!'
            a = float(header.get('xstepsize', 1.))
            meshunit = header.get('meshunit', 'nm')
            a *= {'m': 1e9, 'mm': 1e6, 'µm': 1e3, 'nm': 1}[meshunit]  # Conversion to nm
        return VectorData(a, field)


def _load_from_npy(filename, a, **kwargs):
    _log.debug('Calling load_from_npy')
    if a is None:
        a = 1.  # Use default!
    return VectorData(a, np.load(filename, **kwargs))


def _load_from_hs(filename, a, **kwargs):
    _log.debug('Calling load_from_hs')
    try:
        import hyperspy.api as hs
    except ImportError:
        _log.error('This method recquires the hyperspy package!')
        return
    vectordata = VectorData.from_signal(hs.load(filename, **kwargs))
    if a is not None:
        vectordata.a = a
    return vectordata


def save_vectordata(vectordata, filename, **kwargs):
    """%s"""
    _log.debug('Calling save_vectordata')
    extension = os.path.splitext(filename)[1]
    if extension == '.llg':  # Save to llg-files:
        _save_to_llg(vectordata, filename)
    elif extension == '.ovf':  # Save to ovf-files:
        _save_to_ovf(vectordata, filename, **kwargs)
    elif extension in ['.npy', '.npz']:  # Save to npy-files:
        _save_to_npy(vectordata, filename, **kwargs)
    else:  # Try HyperSpy:
        _save_to_hs(vectordata, filename, **kwargs)
save_vectordata.__doc__ %= VectorData.save.__doc__


def _save_to_llg(vectordata, filename):
    _log.debug('Calling save_to_llg')
    dim = vectordata.dim
    SCALE = 1.0E-9 / 1.0E-2  # from nm to cm
    # Create 3D meshgrid and reshape it and the field into a list where x varies first:
    zz, yy, xx = vectordata.a * SCALE * (np.indices(dim) + 0.5).reshape(3, -1)
    x_vec, y_vec, z_vec = vectordata.field.reshape(3, -1)
    data = np.array([xx, yy, zz, x_vec, y_vec, z_vec]).T
    # Save data to file:
    with open(filename, 'w') as mag_file:
        mag_file.write('LLGFileCreator: {:s}\n'.format(filename))
        mag_file.write('    {:d}    {:d}    {:d}\n'.format(*dim))
        mag_file.writelines('\n'.join('   '.join('{:7.6e}'.format(cell)
                                                 for cell in row) for row in data))


def _save_to_ovf(vectordata, filename):
    _log.debug('Calling save_to_ovf')
    with open(filename, 'w') as mag_file:
        mag_file.write('# OOMMF OVF 2.0\n')
        mag_file.write('# Segment count: 1\n')
        mag_file.write('# Begin: Segment\n')
        # Write Header:
        mag_file.write('# Begin: Header\n')
        name = os.path.split(os.path.split(filename)[1])
        mag_file.write('# Title: PYRAMID-VECTORDATA {}\n'.format(name))
        mag_file.write('# meshtype: rectangular\n')
        mag_file.write('# meshunit: nm\n')
        mag_file.write('# valueunit: A/m\n')
        mag_file.write('# valuemultiplier: 1.\n')
        mag_file.write('# xmin: 0.\n')
        mag_file.write('# ymin: 0.\n')
        mag_file.write('# zmin: 0.\n')
        mag_file.write('# xmax: {}\n'.format(vectordata.a * vectordata.dim[2]))
        mag_file.write('# ymax: {}\n'.format(vectordata.a * vectordata.dim[1]))
        mag_file.write('# zmax: {}\n'.format(vectordata.a * vectordata.dim[0]))
        mag_file.write('# xbase: 0.\n')
        mag_file.write('# ybase: 0.\n')
        mag_file.write('# zbase: 0.\n')
        mag_file.write('# xstepsize: {}\n'.format(vectordata.a))
        mag_file.write('# ystepsize: {}\n'.format(vectordata.a))
        mag_file.write('# zstepsize: {}\n'.format(vectordata.a))
        mag_file.write('# xnodes: {}\n'.format(vectordata.dim[2]))
        mag_file.write('# ynodes: {}\n'.format(vectordata.dim[1]))
        mag_file.write('# znodes: {}\n'.format(vectordata.dim[0]))
        mag_file.write('# End: Header\n')
        # Write data:
        mag_file.write('# Begin: Data Text\n')
        x_mag, y_mag, z_mag = vectordata.field
        x_mag = x_mag.ravel()
        y_mag = y_mag.ravel()
        z_mag = z_mag.ravel()
        for i in range(np.prod(vectordata.dim)):
            mag_file.write('{:g} {:g} {:g}\n'.format(x_mag[i], y_mag[i], z_mag[i]))
        mag_file.write('# End: Data Text\n')
        mag_file.write('# End: Segment\n')


def _save_to_npy(vectordata, filename, **kwargs):
    _log.debug('Calling save_to_npy')
    np.save(filename, vectordata.field, **kwargs)


def _save_to_hs(vectordata, filename, **kwargs):
    _log.debug('Calling save_to_hyperspy')
    vectordata.to_signal().save(filename, **kwargs)

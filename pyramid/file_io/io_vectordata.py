# -*- coding: utf-8 -*-
# Copyright 2016 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""IO functionality for VectorData objects."""

import logging

import os

import re

import numpy as np

from ..fielddata import VectorData
from .. import colors

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
        return _load_from_llg(filename, a, **kwargs)
    # Load from ovf-files:
    if extension in ['.ovf', '.omf', '.ohf', 'obf']:
        return _load_from_ovf(filename, a, **kwargs)
    # Load from npy-files:
    elif extension in ['.npy', '.npz']:
        return _load_from_npy(filename, a, **kwargs)
    # Load from vtk-file:
    elif extension == '.vtk':
        return _load_from_vtk(filename, a, **kwargs)
    # Load from tec-file:
    elif extension == '.tec':
        return _load_from_tec(filename, a, **kwargs)
    # Load with HyperSpy:
    else:
        if extension == '':
            filename = '{}.hdf5'.format(filename)  # Default!
        return _load_from_hs(filename, a, **kwargs)


def _load_from_llg(filename, a):
    _log.debug('Calling _load_from_llg')
    SCALE = 1.0E-9 / 1.0E-2  # From cm to nm
    data = np.genfromtxt(filename, skip_header=2)
    dim = tuple(np.genfromtxt(filename, dtype=int, skip_header=1, skip_footer=len(data[:, 0])))
    if a is None:
        a = (data[1, 0] - data[0, 0]) / SCALE
    field = data[:, 3:6].T.reshape((3,) + dim)
    return VectorData(a, field)


def _load_from_ovf(filename, a=None, segment=None):
    _log.debug('Calling _load_from_ovf')
    with open(filename, 'rb') as mag_file:
        # TODO: Also handle OOMF 1.0? See later TODOs...
        line = mag_file.readline()
        assert line.startswith(b'# OOMMF')  # File has OVF format!
        read_header, read_data = False, False
        header = {'version': line.split()[-1].decode('utf-8')}
        x_mag, y_mag, z_mag = [], [], []
        data_mode = None
        while True:
            # --- READ START OF FILE OR IN BETWEEN SEGMENTS LINE BY LINE ---------------------------
            if not read_header and not read_data:  # Start of file or between segments!
                line = mag_file.readline()
                if line == b'':
                    break  # End of file is reached!
                if line.startswith(b'# Segment count'):
                    seg_count = int(line.split()[-1])  # Total number of segments (often just 1)!
                    seg_curr = 0  # Current segment (0: not in first segment, yet!)
                    if seg_count > 1:  # If multiple segments, check if "segment" was set correctly:
                        assert segment is not None, (f'Multiple ({seg_count}) segments were found! '
                                                     'Chose one via the segment parameter!')
                    elif segment is None:  # Only one segment AND parameter not set:
                        segment = 1  # Default to the first/only segment!
                    assert 0 < segment <= seg_count, (f'parameter segment={segment} out of bounds, '
                                                      f'Use value between 1 and {seg_count}!')
                    header['segment_count'] = seg_count
                    # TODO: navigation axis (segment count > 1) if implemented in HyperSpy reader!
                    # TODO: only works if all segments have the same dimensions! (So maybe not...)
                elif line.startswith(b'# Begin: Segment'):  # Segment start!
                    seg_curr += 1
                    if seg_curr > segment:
                        break  # Stop reading the file!
                elif line.startswith(b'# Begin: Header'):  # Header start!
                    read_header = True
                elif line.startswith(b'# Begin: Data'):  # Data start!
                    read_data = True
                    data_mode = ' '.join(line.decode('utf-8').split()[3:])
                    assert data_mode in ['text', 'Text', 'Binary 4', 'Binary 8'], \
                        'Data mode {} is currently not supported by this reader!'.format(data_mode)
                    assert header.get('meshtype') == 'rectangular', \
                        'Only rectangular grids can be currently read!'
            # --- READ HEADER LINE BY LINE ---------------------------------------------------------
            elif read_header:
                line = mag_file.readline()
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
                if key not in header:  # Add new key, value pair if not existant:
                    header[key] = value
                elif key == 'Desc':  # Can go over several lines:
                    header['Desc'] = ' '.join([header['Desc'], value])
            # --- READ DATA LINE BY LINE -----------------------------------------------------------
            elif read_data:  # Currently in a data block:
                if data_mode in ['text', 'Text']:  # Read data as text, line by line:
                    line = mag_file.readline()
                    if line.startswith(b'# End: Data'):
                        read_data = False  # Stop reading data and search for new segments!
                    elif seg_curr < segment:  # Do nothing with the line if wrong segment!
                        continue
                    else:
                        x, y, z = [float(i) for i in line.split()]
                        x_mag.append(x)
                        y_mag.append(y)
                        z_mag.append(z)
                # TODO: Make it work for both text and binary! Put into HyperSpy?
                # TODO: http://math.nist.gov/oommf/doc/userguide11b2/userguide/vectorfieldformat.html
                # TODO: http://math.nist.gov/oommf/doc/userguide12a5/userguide/OVF_2.0_format.html
                # TODO: 1.0 and 2.0 DIFFER (little and big endian in binary data -.-)
                elif 'Binary' in data_mode:  # Read data as binary, all bytes at the same time:
                    # TODO: currently every segment until the wanted one is processed. Necessary?
                    count = int(data_mode.split()[-1])
                    if header['version'] == '1.0':  # Big endian:
                        dtype = '>f{}'.format(count)
                    elif header['version'] == '2.0':  # Little endian:
                        dtype = '<f{}'.format(count)
                    test = np.fromfile(mag_file, dtype=dtype, count=1)
                    if count == 4:  # Binary 4:
                        assert test == 1234567.0, 'Wrong test bytes!'
                    elif count == 8:  # Binary 8:
                        assert test == 123456789012345.0, 'Wrong test bytes!'
                    dim = (int(header['znodes']), int(header['ynodes']), int(header['xnodes']))
                    data = np.fromfile(mag_file, dtype=dtype, count=3*np.prod(dim))
                    x_mag, y_mag, z_mag = data[0::3], data[1::3], data[2::3]
                    read_data = False  # Stop reading data and search for new segments (if any).
        # --- READING DONE -------------------------------------------------------------------------
        # Format after reading:
        dim = (int(header['znodes']), int(header['ynodes']), int(header['xnodes']))
        x_mag = np.asarray(x_mag).reshape(dim)
        y_mag = np.asarray(y_mag).reshape(dim)
        z_mag = np.asarray(z_mag).reshape(dim)
        field = np.asarray((x_mag, y_mag, z_mag)) * float(header.get('valuemultiplier', 1))
        if a is None:
            # TODO: If transferred to HyperSpy, this has to stay in Pyramid reader!
            xstep = float(header.get('xstepsize'))
            ystep = float(header.get('ystepsize'))
            zstep = float(header.get('zstepsize'))
            if not np.allclose(xstep, ystep) and np.allclose(xstep, zstep):
                _log.warning('Grid spacing is not equal in x, y and z (x will be used)!\n'
                             f'Found step sizes are x:{xstep}, y:{ystep}, z:{zstep} '
                             f'(all in {header.get("meshunit")})!')
            # Extract grid spacing from xstepsize and convert according to meshunit:
            unit = header.get('meshunit', 'nm')
            _log.info(f'unit: {unit}')
            if unit == 'unspecified':
                unit = 'nm'
            a = xstep * {'m': 1e9, 'mm': 1e6, 'µm': 1e3, 'nm': 1}[unit]
        return VectorData(a, field)


def _load_from_npy(filename, a, **kwargs):
    _log.debug('Calling _load_from_npy')
    if a is None:
        a = 1.  # Use default!
    return VectorData(a, np.load(filename, **kwargs))


def _load_from_hs(filename, a, **kwargs):
    _log.debug('Calling _load_from_hs')
    try:
        import hyperspy.api as hs
    except ImportError:
        _log.error('This method recquires the hyperspy package!')
        return
    vectordata = VectorData.from_signal(hs.load(filename, **kwargs))
    if a is not None:
        vectordata.a = a
    return vectordata


def _load_from_vtk(filename, a=None, **kwargs):
    # TODO: kwargs like mode, dim, origin, etc.!
    # TODO: Testing with examples: https://lorensen.github.io/VTKExamples/site/Python/#data-types
    from tvtk.api import tvtk
    # Infos about format: http://www.cacr.caltech.edu/~slombey/asci/vtk/vtk_formats.simple.html

    _log.debug('Calling _load_from_vtk')
    # Setting up reader:
    reader = tvtk.DataSetReader(file_name=filename, read_all_scalars=True, read_all_vectors=True)
    reader.update()
    # Getting output:
    output = reader.output
    assert output is not None, 'File reader could not find data or file "{}"!'.format(filename)
    # Reading points and vectors:
    # TODO: The following does not work for StructuredPoints!
    # TODO: However, they don't need interpolation and are easier to handle!
    # TODO: Therefore: check which structure the file uses and read from there!
    if isinstance(output, tvtk.StructuredPoints):
        _log.info('geometry: StructuredPoints')
        # Load relevant information from output (reverse to get typical Python order z,y,x):
        dim = output.dimensions[::-1]
        origin = output.origin[::-1]
        spacing = output.spacing[::-1]
        _log.info(f'dim: {dim}, origin: {origin}, spacing: {spacing}')
        assert len(dim) == 3, 'Data has to be three-dimensional!'
        assert spacing[0] == spacing[1] == spacing[2], \
            'The grid is not Euclidean (spacing: {})!'.format(spacing)
        # TODO: when the spacing is not equal in all directions: Interpolate (or just exclude case)
        # x = spacing[0] * (np.arange(dim[2]) - origin[2] + 0.5)
        # y = spacing[1] * (np.arange(dim[1]) - origin[1] + 0.5)
        # z = spacing[2] * (np.arange(dim[0]) - origin[0] + 0.5)
        # xi = np.asarray(list(itertools.product(z, y, x)))
        # point_array = np.fliplr(xi)  # fliplr for (x, y, z) order!
        if a is None:
            a = spacing[0]
        # Extract vector compontents and create magnitude array:
        vector_array = np.asarray(output.point_data.vectors, dtype=np.float)
        x_mag, y_mag, z_mag = vector_array.T
        magnitude = np.asarray((x_mag.reshape(dim), y_mag.reshape(dim), z_mag.reshape(dim)))
    elif isinstance(output, tvtk.RectilinearGrid):
        _log.info('geometry: RectilinearGrid')
        raise NotImplementedError('RectilinearGrid is currently not supported!')
        # TODO: Implement?
    elif isinstance(output, tvtk.StructuredGrid):
        _log.info('geometry: StructuredGrid')
        raise NotImplementedError('StructuredGrid is currently not supported!')
        # TODO: Implement?
    elif isinstance(output, tvtk.UnstructuredGrid):
        _log.info('geometry: UnstructuredGrid')
        if a is None:  # Set a default if none was set for the grid spacing:
            a = 1
        # Load relevant information from output:
        point_array = np.asarray(output.points, dtype=np.float)
        vector_array = np.asarray(output.point_data.vectors, dtype=np.float)
        magnitude = _interp_to_regular_grid(point_array, vector_array, a, **kwargs)
    elif isinstance(output, tvtk.PolyData):
        _log.info('geometry: PolyData')
        raise NotImplementedError('PolyData is currently not supported!')
        # TODO: Implement?
    else:
        raise TypeError('Data type of {} not understood!'.format(output))
    return VectorData(a, magnitude)  # TODO: a und dim eventuell nicht bekannt! NECESSARY!!!!
    # TODO: Copy a conversion from ovf reader!


def _load_from_tec(filename, a=None, **kwargs):
    _log.debug('Calling load_from_tec')
    with open(filename, 'r') as mag_file:
        # Read in lines:
        lines = mag_file.readlines()
        # Extract number of points from third line:
        match = re.search(R'N=(\d+)', lines[2])
        if match:
            n_points = int(match.group(1))
        else:
            raise IOError('File does not seem to match .tec format!')
    n_head, n_foot = 3, len(lines) - (3 + n_points)
    # Read in data:
    data = np.genfromtxt(filename, skip_header=n_head, skip_footer=n_foot)
    magnitude = _interp_to_regular_grid(data[:, :3], data[:, 3:], a, **kwargs)
    return VectorData(a, magnitude)  # TODO: a und dim eventuell nicht bekannt! NECESSARY!!!!
    # TODO: Copy a conversion from ovf reader!


def _interp_to_regular_grid(points, values, a, conversion=1, step=1, convex=True):
    # TODO: Docstring! Default: new grid centered around 0/0 origin of the point cloud (sensible?)
    # TODO: make extensible for scalarfield (4 cols) or general 3 cols coords and n columns?
    # TODO: Cleanup!
    from scipy.spatial import cKDTree, qhull
    from tqdm import tqdm
    import itertools
    from scipy.interpolate import LinearNDInterpolator
    from time import time
    _log.debug('Calling interpolate_to_regular_grid')
    z_uniq = np.unique(points[:, 2])
    _log.info(f'unique positions along z: {len(z_uniq)}')
    #  Local grid spacing can be in another unit (not nm), taken care of with `conversion`:
    a_local = a * conversion
    # Determine the size of the point cloud of irregular coordinates:
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    x_diff, y_diff, z_diff = np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2])
    _log.info(f'x-range: {x_min:.2g} <-> {x_max:.2g} ({x_diff:.2g})')
    _log.info(f'y-range: {y_min:.2g} <-> {y_max:.2g} ({y_diff:.2g})')
    _log.info(f'z-range: {z_min:.2g} <-> {z_max:.2g} ({z_diff:.2g})')
    # Determine dimensions from given grid spacing a:
    dim = np.round(np.asarray((z_diff, y_diff, x_diff)) / a_local).astype(int)
    x = x_min + a_local * (np.arange(dim[2]) + 0.5)  # +0.5: shift to pixel center!
    y = y_min + a_local * (np.arange(dim[1]) + 0.5)  # +0.5: shift to pixel center!
    z = z_min + a_local * (np.arange(dim[0]) + 0.5)  # +0.5: shift to pixel center!
    # Create points for new Euclidian grid; fliplr for (x, y, z) order:
    points_euc = np.fliplr(np.asarray(list(itertools.product(z, y, x))))
    # Make values 2D (if not already); double .T so that a new axis is added at the END (n, 1):
    values = np.atleast_2d(values.T).T
    # Prepare interpolated grid:
    interpolation = np.empty((values.shape[-1], *dim), dtype=np.float)
    _log.info(f'Dimensions of new grid: {(values.shape[-1], len(z), len(y), len(x))}')
    # Calculate the Delaunay triangulation (same for every component of multidim./vector fields):
    _log.info('Start Delaunay triangulation...')
    tick = time()
    triangulation = qhull.Delaunay(points[::step])
    tock = time()
    _log.info(f'Delaunay triangulation complete (took {tock-tick:.2f} s)!')
    # Perform the interpolation for each column of `values`:
    for i in tqdm(range(values.shape[-1])):
        # Create interpolator for the given triangulation and the values of the current column:
        interpolator = LinearNDInterpolator(triangulation, values[::step, i], fill_value=0)
        # Interpolate:
        interpolation[i, ...] = interpolator(points_euc).reshape(dim)
    # If NOT convex, we have to check for additional holes in the structure (EXPERIMENTAL):
    if not convex:  # Only necessary if the user expects holes in the (-> nonconvex) distribution:
        # Create k-dimensional tree for queries:
        tree = cKDTree(points)
        # Query the tree for nearest neighbors, x: points to query, k: number of neighbors, p: norm
        # to use (here: 2 - Euclidean), distance_upper_bound: maximum distance that is searched!
        data, leafsize = tree.query(x=points_euc, k=1, p=2, distance_upper_bound=2*a)
        # Create boolean mask that determines which interpolation points have no neighbor near enough:
        mask = np.isinf(data).reshape(dim)  # Points further away than upper bound were marked 'inf'!
        for i in tqdm(range(values.shape[-1])):  # TODO: tqdm? can take a looooong time...
            # interpolation[i, ...][mask] = 0  # TODO: Which one is correct?
            interpolation[i, ...].ravel()[mask.ravel()] = 0
        # TODO: Log how many points are added and such... DEBUG!!!
        # # Append interpolation points without close neighbors to list of original points:
        # points = np.vstack((points, xi[mask]))
        # # Accordingly append a list of fitting zeros to the values:
        # values = np.vstack((values, np.zeros_like(xi[mask])))
        # _log.info('Added {:d} zero points!'.format(np.sum(mask)))
    return np.squeeze(interpolation)


def save_vectordata(vectordata, filename, **kwargs):
    """%s"""
    _log.debug('Calling save_vectordata')
    extension = os.path.splitext(filename)[1]
    if extension == '.llg':  # Save to llg-files:
        _save_to_llg(vectordata, filename)
    elif extension == '.ovf':  # Save to ovf-files:
        _save_to_ovf(vectordata, filename)
    elif extension in ['.npy', '.npz']:  # Save to npy-files:
        _save_to_npy(vectordata, filename, **kwargs)
    elif extension == '.vtk':  # Save to vtk-files:
        _save_to_vtk(vectordata, filename)
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


def _save_to_vtk(vectordata, filename):
    # Infos about format: https://www.vtk.org/wp-content/uploads/2015/04/file-formats.pdf
    # See also: https://www.vtk.org/Wiki/VTK/Writing_VTK_files_using_python
    from tvtk.api import tvtk, write_data
    _log.debug('Calling save_to_vtk')
    # Put vector components in corresponding array:
    vectors = np.empty(vectordata.dim + (3,), dtype=float)
    vectors[..., 0] = vectordata.field[0, ...]
    vectors[..., 1] = vectordata.field[1, ...]
    vectors[..., 2] = vectordata.field[2, ...]
    vectors = vectors.reshape(-1, 3)  # TODO: copy in between necessary to make contiguous?
    # Calculate colors:
    x_mag, y_mag, z_mag = vectordata.field
    magvec = np.asarray((x_mag.ravel(), y_mag.ravel(), z_mag.ravel()))
    rgb = colors.CMAP_CIRCULAR_DEFAULT.rgb_from_vector(magvec)
    point_colors = tvtk.UnsignedIntArray()
    point_colors.number_of_components = 3
    point_colors.name = 'colors'
    point_colors.from_array(rgb)
    # Create the dataset:
    origin = (0, 0, 0)
    spacing = (vectordata.a,)*3
    dimensions = (vectordata.dim[2], vectordata.dim[1], vectordata.dim[0])
    sp = tvtk.StructuredPoints(origin=origin, spacing=spacing, dimensions=dimensions)
    sp.point_data.vectors = vectors
    sp.point_data.vectors.name = 'magnetisation'
    sp.point_data.scalars = point_colors
    sp.point_data.scalars.name = 'colors'
    # Write the data to file:
    write_data(sp, filename)


# TODO: Put load and save routines in format-specific files (ala hyperspy)

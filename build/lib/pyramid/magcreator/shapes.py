# -*- coding: utf-8 -*-
# Copyright 2016 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""Provide simple shapes.

This module is a collection of some methods that return a 3-dimensional
matrix that represents the field volume and consists of boolean values.
This matrix is used in the functions of the :mod:`~.magcreator` module to create
:class:`~pyramid.fielddata.VectorData` objects which store the field information.

"""

import logging

import numpy as np

__all__ = ['slab', 'disc', 'ellipse', 'ellipsoid', 'sphere', 'filament', 'pixel']
_log = logging.getLogger(__name__)


def slab(dim, center, width):
    """Create the shape of a slab.

    Parameters
    ----------
    dim : tuple (N=3)
        The dimensions of the grid `(z, y, x)`.
    center : tuple (N=3)
        The center of the slab in pixel coordinates `(z, y, x)`.
    width : tuple (N=3)
        The width of the slab in pixel coordinates `(z, y, x)`.

    Returns
    -------
    shape : :class:`~numpy.ndarray` (N=3)
        The shape as a 3D-array.

    """
    _log.debug('Calling slab')
    assert np.shape(dim) == (3,), 'Parameter dim has to be a tuple of length 3!'
    assert np.shape(center) == (3,), 'Parameter center has to be a tuple of length 3!'
    assert np.shape(width) == (3,), 'Parameter width has to be a tuple of length 3!'
    zz, yy, xx = np.indices(dim) + 0.5
    xx_shape = np.where(abs(xx - center[2]) <= width[2] / 2, True, False)
    yy_shape = np.where(abs(yy - center[1]) <= width[1] / 2, True, False)
    zz_shape = np.where(abs(zz - center[0]) <= width[0] / 2, True, False)
    return np.logical_and(np.logical_and(xx_shape, yy_shape), zz_shape)


def disc(dim, center, radius, height, axis='z'):
    """Create the shape of a cylindrical disc in x-, y-, or z-direction.

    Parameters
    ----------
    dim : tuple (N=3)
        The dimensions of the grid `(z, y, x)`.
    center : tuple (N=3)
        The center of the disc in pixel coordinates `(z, y, x)`.
    radius : float
        The radius of the disc in pixel coordinates.
    height : float
        The height of the disc in pixel coordinates.
    axis : {'z', 'y', 'x'}, optional
        The orientation of the disc axis. The default is 'z'.

    Returns
    -------
    shape : :class:`~numpy.ndarray` (N=3)
        The shape as a 3D-array.

    """
    _log.debug('Calling disc')
    assert np.shape(dim) == (3,), 'Parameter dim has to be a a tuple of length 3!'
    assert np.shape(center) == (3,), 'Parameter center has to be a a tuple of length 3!'
    assert radius > 0 and np.shape(radius) == (), 'Radius has to be a positive scalar value!'
    assert height > 0 and np.shape(height) == (), 'Height has to be a positive scalar value!'
    assert axis in {'z', 'y', 'x'}, 'Axis has to be x, y or z (as a string)!'
    zz, yy, xx = np.indices(dim) + 0.5
    xx -= center[2]
    yy -= center[1]
    zz -= center[0]
    if axis == 'z':
        uu, vv, ww = xx, yy, zz
    elif axis == 'y':
        uu, vv, ww = zz, xx, yy
    elif axis == 'x':
        uu, vv, ww = yy, zz, xx
    else:
        raise ValueError('{} is not a valid argument (use x, y or z)'.format(axis))
    return np.logical_and(np.where(np.hypot(uu, vv) <= radius, True, False),
                          np.where(abs(ww) <= height / 2, True, False))


def ellipse(dim, center, width, height, axis='z'):
    """Create the shape of an elliptical cylinder in x-, y-, or z-direction.

    Parameters
    ----------
    dim : tuple (N=3)
        The dimensions of the grid `(z, y, x)`.
    center : tuple (N=3)
        The center of the ellipse in pixel coordinates `(z, y, x)`.
    width : tuple (N=2)
        Length of the two axes of the ellipse.
    height : float
        The height of the ellipse in pixel coordinates.
    axis : {'z', 'y', 'x'}, optional
        The orientation of the ellipse axis. The default is 'z'.

    Returns
    -------
    shape : :class:`~numpy.ndarray` (N=3)
        The shape as a 3D-array.

    """
    _log.debug('Calling ellipse')
    assert np.shape(dim) == (3,), 'Parameter dim has to be a a tuple of length 3!'
    assert np.shape(center) == (3,), 'Parameter center has to be a a tuple of length 3!'
    assert np.shape(width) == (2,), 'Parameter width has to be a a tuple of length 2!'
    assert height > 0 and np.shape(height) == (), 'Height has to be a positive scalar value!'
    assert axis in {'z', 'y', 'x'}, 'Axis has to be x, y or z (as a string)!'
    zz, yy, xx = np.indices(dim) + 0.5
    xx -= center[2]
    yy -= center[1]
    zz -= center[0]
    if axis == 'z':
        uu, vv, ww = xx, yy, zz
    elif axis == 'y':
        uu, vv, ww = xx, zz, yy
    elif axis == 'x':
        uu, vv, ww = yy, zz, xx
    else:
        raise ValueError('{} is not a valid argument (use x, y or z)'.format(axis))
    distance = np.hypot(uu / (width[1] / 2), vv / (width[0] / 2))
    return np.logical_and(np.where(distance <= 1, True, False),
                          np.where(abs(ww) <= height / 2, True, False))


def ellipsoid(dim, center, width):
    """Create the shape of an ellipsoid.

    Parameters
    ----------
    dim : tuple (N=3)
        The dimensions of the grid `(z, y, x)`.
    center : tuple (N=3)
        The center of the ellipsoid in pixel coordinates `(z, y, x)`.
    width : tuple (N=3)
        The width of the ellipsoid `(z, y, x)`.

    Returns
    -------
    shape : :class:`~numpy.ndarray` (N=3)
        The shape as a 3D-array.

    """
    _log.debug('Calling ellipsoid')
    assert np.shape(dim) == (3,), 'Parameter dim has to be a a tuple of length 3!'
    assert np.shape(center) == (3,), 'Parameter center has to be a a tuple of length 3!'
    assert np.shape(width) == (3,), 'Parameter width has to be a a tuple of length 3!'
    zz, yy, xx = np.indices(dim) + 0.5
    distance = np.sqrt(((xx - center[2]) / (width[2] / 2)) ** 2
                       + ((yy - center[1]) / (width[1] / 2)) ** 2
                       + ((zz - center[0]) / (width[0] / 2)) ** 2)
    return np.where(distance <= 1, True, False)


def sphere(dim, center, radius):
    """Create the shape of a sphere.

    Parameters
    ----------
    dim : tuple (N=3)
        The dimensions of the grid `(z, y, x)`.
    center : tuple (N=3)
        The center of the sphere in pixel coordinates `(z, y, x)`.
    radius : float
        The radius of the sphere in pixel coordinates.

    Returns
    -------
    shape : :class:`~numpy.ndarray` (N=3)
        The shape as a 3D-array.

    """
    _log.debug('Calling sphere')
    assert np.shape(dim) == (3,), 'Parameter dim has to be a a tuple of length 3!'
    assert np.shape(center) == (3,), 'Parameter center has to be a a tuple of length 3!'
    assert radius > 0 and np.shape(radius) == (), 'Radius has to be a positive scalar value!'
    zz, yy, xx = np.indices(dim) + 0.5
    distance = np.sqrt((xx - center[2]) ** 2 + (yy - center[1]) ** 2 + (zz - center[0]) ** 2)
    return np.where(distance <= radius, True, False)


def filament(dim, pos, axis='y'):
    """Create the shape of a filament.

    Parameters
    ----------
    dim : tuple (N=3)
        The dimensions of the grid `(z, y, x)`.
    pos : tuple (N=2)
        The position of the filament in pixel coordinates `(coord1, coord2)`.
        `coord1` and `coord2` stand for the two axes, which are perpendicular to `axis`. For
        the default case (`axis = y`), it is `(coord1, coord2) = (z, x)`.
    axis :  {'y', 'x', 'z'}, optional
        The orientation of the filament axis. The default is 'y'.

    Returns
    -------
    shape : :class:`~numpy.ndarray` (N=3)
        The shape as a 3D-array.

    """
    _log.debug('Calling filament')
    assert np.shape(dim) == (3,), 'Parameter dim has to be a tuple of length 3!'
    assert np.shape(pos) == (2,), 'Parameter pos has to be a tuple of length 2!'
    assert axis in {'z', 'y', 'x'}, 'Axis has to be x, y or z (as a string)!'
    shape = np.zeros(dim, dtype=bool)
    if axis == 'z':
        shape[:, pos[0], pos[1]] = True
    elif axis == 'y':
        shape[pos[0], :, pos[1]] = True
    elif axis == 'x':
        shape[pos[0], pos[1], :] = True
    return shape


def pixel(dim, pixel):
    """Create the shape of a single pixel.

    Parameters
    ----------
    dim : tuple (N=3)
        The dimensions of the grid `(z, y, x)`.
    pixel : tuple (N=3)
        The coordinates of the pixel `(z, y, x)`.

    Returns
    -------
    shape : :class:`~numpy.ndarray` (N=3)
        The shape as a 3D-array.

    """
    _log.debug('Calling pixel')
    assert np.shape(dim) == (3,), 'Parameter dim has to be a tuple of length 3!'
    assert np.shape(pixel) == (3,), 'Parameter pixel has to be a tuple of length 3!'
    shape = np.zeros(dim, dtype=bool)
    shape[pixel] = True
    return shape

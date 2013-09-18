# -*- coding: utf-8 -*-
"""Create projections of a given magnetization distribution.

This module creates 2-dimensional projections from 3-dimensional magnetic distributions, which
are stored in :class:`~pyramid.magdata.MagData` objects. Either simple projections along the
major axes are possible (:func:`~.simple_axis_projection`), or projections along arbitrary
directions with the use of transfer functions (work in progress).

"""

import time

import numpy as np
from numpy import pi

import itertools

from pyramid.magdata import MagData


def simple_axis_projection(mag_data, axis='z', threshold=0):
    '''
    Project a magnetization distribution along one of the main axes of the 3D-grid.

    Parameters
    ----------
    mag_data : :class:`~pyramid.magdata.MagData`
        A :class:`~pyramid.magdata.MagData` object storing the magnetization distribution,
        which should be projected.
    axis : {'z', 'y', 'x'}, optional
        The projection direction as a string.
    threshold : float, optional
        A pixel only gets masked, if it lies above this threshold. The default is 0.

    Returns
    -------
    projection : tuple (N=3) of :class:`~numpy.ndarray` (N=2)
        The in-plane projection of the magnetization as a tuple, storing the `u`- and `v`-component
        of the magnetization and the thickness projection for the resulting 2D-grid. The latter
        has to be multiplied with the resolution for a value in nm.

    '''
    assert isinstance(mag_data, MagData), 'Parameter mag_data has to be a MagData object!'
    assert axis == 'z' or axis == 'y' or axis == 'x', 'Axis has to be x, y or z (as string)!'
    if axis == 'z':
        projection = (mag_data.magnitude[1].sum(0),  # y_mag -> v_mag
                      mag_data.magnitude[2].sum(0),  # x_mag -> u_mag
                      mag_data.get_mask(threshold).sum(0))  # thickness profile
    elif axis == 'y':
        projection = (mag_data.magnitude[0].sum(1),  # z_mag -> v_mag
                      mag_data.magnitude[2].sum(1),  # x_mag -> u_mag
                      mag_data.get_mask(threshold).sum(1))  # thickness profile
    elif axis == 'x':
        projection = (mag_data.magnitude[0].sum(2),  # z_mag -> v_mag
                      mag_data.magnitude[1].sum(2),  # y_mag -> u_mag
                      mag_data.get_mask(threshold).sum(2))  # thickness profile
    return projection


def single_tilt_projection(mag_data, tilt=0, threshold=0):
    # TODO: Docstring!!!
    assert isinstance(mag_data, MagData), 'Parameter mag_data has to be a MagData object!'
#    assert axis == 'z' or axis == 'y' or axis == 'x', 'Axis has to be x, y or z (as string)!'

    # Set starting variables:
    dim = (mag_data.dim[0], mag_data.dim[2])
    z_mag, y_mag, x_mag = mag_data.magnitude
    mask = mag_data.get_mask()
    projection = (np.zeros((mag_data.dim[1], mag_data.dim[2])),
                  np.zeros((mag_data.dim[1], mag_data.dim[2])),
                  np.zeros((mag_data.dim[1], mag_data.dim[2])))

    def get_position(p, m, b, size):
        x, y = np.array(p)[:, 0]+0.5, np.array(p)[:, 1]+0.5
        return (y-m*x-b)/np.sqrt(m**2+1) + size/2.

    def get_impact(pos, r, size):
        return [x for x in np.arange(np.floor(pos-r), np.floor(pos+r)+1, dtype=int)
                if 0 <= x < size]

    def get_weight(delta, rho):
        a, b = delta-rho, delta+rho
        if a >= 1 or b <= -1:  # TODO: Should not be necessary!
            print 'Outside of bounds:', delta
            return 0
        # Upper boundary:
        if b >= 1:
            w_b = 0.5
        else:
            w_b = (b*np.sqrt(1-b**2) + np.arctan(b/np.sqrt(1-b**2))) / pi
        # Lower boundary:
        if a <= -1:
            w_a = -0.5
        else:
            w_a = (a*np.sqrt(1-a**2) + np.arctan(a/np.sqrt(1-a**2))) / pi
        return w_b - w_a

    # Creating coordinate list of all voxels:
    xi = range(dim[0])
    yj = range(dim[1])
    ii, jj = np.meshgrid(xi, yj)
    voxels = list(itertools.product(yj, xi))

    # Calculate positions along the projected pixel coordinate system:
    direction = (-np.cos(tilt), np.sin(tilt))
    center = (dim[0]/2., dim[1]/2.)
    m = direction[0]/(direction[1]+1E-30)
    b = center[0] - m * center[1]
    positions = get_position(voxels, m, b, dim[0])

    # Calculate weights:
    r = 1/np.sqrt(np.pi)  # radius of the voxel circle
    rho = 0.5 / r  # TODO: ratio of radii
    weights = []
    for i, voxel in enumerate(voxels):
        voxel_weights = []
        impacts = get_impact(positions[i], r, dim[0])
        for impact in impacts:
            distance = np.abs(impact+0.5 - positions[i])
            delta = distance / r
            voxel_weights.append((impact, get_weight(delta, rho)))
        weights.append((voxel, voxel_weights))

    # Calculate projection with the calculated weights for the voxels:
    for i, weight in enumerate(weights):
        voxel = weights[i][0]
        voxel_weights = weights[i][1]
        for voxel_weight in voxel_weights:
            pixel, weight = voxel_weight
            # Component parallel to tilt axis (':' goes over all slices):
            projection[0][:, pixel] += weight * y_mag[voxel[0], :, voxel[1]]
            # Component perpendicular to tilt axis:
            projection[1][:, pixel] += weight * (x_mag[voxel[0], :, voxel[1]]*np.cos(tilt)
                                               + z_mag[voxel[0], :, voxel[1]]*np.sin(tilt))
            # Thickness profile:
            projection[2][:, pixel] += weight * mask[voxel[0], :, voxel[1]]

    return projection

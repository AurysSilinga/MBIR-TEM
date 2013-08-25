# -*- coding: utf-8 -*-
"""Create projections of a given magnetization distribution.

This module creates 2-dimensional projections from 3-dimensional magnetic distributions, which
are stored in :class:`~pyramid.magdata.MagData` objects. Either simple projections along the
major axes are possible (:func:`~.simple_axis_projection`), or projections along arbitrary
directions with the use of transfer functions (work in progress).

"""


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

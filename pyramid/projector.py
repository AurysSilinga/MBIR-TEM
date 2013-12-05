# -*- coding: utf-8 -*-
"""Create projections of a given magnetization distribution.

This module creates 2-dimensional projections from 3-dimensional magnetic distributions, which
are stored in :class:`~pyramid.magdata.MagData` objects. Either simple projections along the
major axes are possible (:func:`~.simple_axis_projection`), or projections with a tilt around
the y-axis. The thickness profile is also calculated and can be used for electric phase maps.

"""


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
        has to be multiplied with the grid spacing for a value in nm.

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


def single_tilt_projection(mag_data, tilt=0):
    '''
    Project a magnetization distribution which is tilted around the centered y-axis.

    Parameters
    ----------
    mag_data : :class:`~pyramid.magdata.MagData`
        A :class:`~pyramid.magdata.MagData` object storing the magnetization distribution,
        which should be projected.
    tilt : float, optional
        The counter-clockwise tilt angle around the y-axis. Default is 1.

    Returns
    -------
    projection : tuple (N=3) of :class:`~numpy.ndarray` (N=2)
        The in-plane projection of the magnetization as a tuple, storing the `u`- and `v`-component
        of the magnetization and the thickness projection for the resulting 2D-grid. The latter
        has to be multiplied with the grid spacing for a value in nm.

    '''
    assert isinstance(mag_data, MagData), 'Parameter mag_data has to be a MagData object!'

    def get_position(p, m, b, size):
        y, x = np.array(p)[:, 0]+0.5, np.array(p)[:, 1]+0.5
        return (y-m*x-b)/np.sqrt(m**2+1) + size/2.

    def get_impact(pos, r, size):
        return [x for x in np.arange(np.floor(pos-r), np.floor(pos+r)+1, dtype=int)
                if 0 <= x < size]

    def get_weight(delta, rho):  # use circles to represent the voxels
        lo, up = delta-rho, delta+rho
        # Upper boundary:
        if up >= 1:
            w_up = 0.5
        else:
            w_up = (up*np.sqrt(1-up**2) + np.arctan(up/np.sqrt(1-up**2))) / pi
        # Lower boundary:
        if lo <= -1:
            w_lo = -0.5
        else:
            w_lo = (lo*np.sqrt(1-lo**2) + np.arctan(lo/np.sqrt(1-lo**2))) / pi
        return w_up - w_lo

    # Set starting variables:
    # length along projection (proj, z), rotation (rot, y) and perpendicular (perp, x) axis:
    dim_proj, dim_rot, dim_perp = mag_data.dim
    z_mag, y_mag, x_mag = mag_data.magnitude
    mask = mag_data.get_mask()
    projection = (np.zeros((dim_rot, dim_perp)),
                  np.zeros((dim_rot, dim_perp)),
                  np.zeros((dim_rot, dim_perp)))
    # Creating coordinate list of all voxels:
    voxels = list(itertools.product(range(dim_proj), range(dim_perp)))

    # Calculate positions along the projected pixel coordinate system:
    center = (dim_proj/2., dim_perp/2.)
    m = np.where(tilt<=pi, -1/np.tan(tilt+1E-30), 1/np.tan(tilt+1E-30))
    b = center[0] - m * center[1]
    positions = get_position(voxels, m, b, dim_perp)

    # Calculate weights:
    r = 1/np.sqrt(np.pi)  # radius of the voxel circle
    rho = 0.5 / r
    weights = {}
    for i, voxel in enumerate(voxels):
        voxel_weights = []
        impacts = get_impact(positions[i], r, dim_perp)
        for impact in impacts:
            distance = np.abs(impact+0.5 - positions[i])
            delta = distance / r
            voxel_weights.append((impact, get_weight(delta, rho)))
        weights[voxel] = voxel_weights
    
    # Calculate projection with the calculated weights for the voxels (rotation around y-axis):
    for i, voxel in enumerate(weights):
        voxel_weights = weights[voxel]
        for pixel, weight in voxel_weights:
            # Component parallel to tilt axis (':' goes over all slices):
            projection[0][:, pixel] += weight * y_mag[voxel[0], :, voxel[1]]
            # Component perpendicular to tilt axis:
            projection[1][:, pixel] += weight * (x_mag[voxel[0], :, voxel[1]]*np.cos(tilt)
                                               + z_mag[voxel[0], :, voxel[1]]*np.sin(tilt))
            # Thickness profile:
            projection[2][:, pixel] += weight * mask[voxel[0], :, voxel[1]]
    
    return projection
    
class Projection:

    '''Class for calculating kernel matrices for the phase calculation.

    Represents the phase of a single magnetized pixel for two orthogonal directions (`u` and `v`),
    which can be accessed via the corresponding attributes. The default elementary geometry is
    `disc`, but can also be specified as the phase of a `slab` representation of a single
    magnetized pixel. During the construction, a few attributes are calculated that are used in
    the convolution during phase calculation.

    Attributes
    ----------
    dim : tuple (N=2)
        Dimensions of the projected magnetization grid.
    a : float
        The grid spacing in nm.
    geometry : {'disc', 'slab'}, optional
        The elementary geometry of the single magnetized pixel.
    b_0 : float, optional
        The saturation magnetic induction. Default is 1.
    u : :class:`~numpy.ndarray` (N=3)
        The phase contribution of one pixel magnetized in u-direction.
    v : :class:`~numpy.ndarray` (N=3)
        The phase contribution of one pixel magnetized in v-direction.
    u_fft : :class:`~numpy.ndarray` (N=3)
        The real FFT of the phase contribution of one pixel magnetized in u-direction.
    v_fft : :class:`~numpy.ndarray` (N=3)
        The real FFT of the phase contribution of one pixel magnetized in v-direction.
    dim_fft : tuple (N=2)
        Dimensions of the grid, which is used for the FFT. Calculated by adding the dimensions
        `dim` of the magnetization grid and the dimensions of the kernel (given by ``2*dim-1``)
        and increasing to the next multiple of 2 (for faster FFT).
    slice_fft : tuple (N=2) of :class:`slice`
        A tuple of :class:`slice` objects to extract the original field of view from the increased
        size (size_fft) of the grid for the FFT-convolution.

    ''' # TODO: Docstring!
    
    def __init__(self, dim_proj, dim_rot, dim_perp, v_mag, u_mag, thickness, weights, tilt):
        '''Constructor for a :class:`~.Kernel` object for representing a kernel matrix.

        Parameters
        ----------
        dim : tuple (N=2)
            Dimensions of the projected magnetization grid.
        a : float
            The grid spacing in nm.
        b_0 : float, optional
            The saturation magnetic induction. Default is 1.
        geometry : {'disc', 'slab'}, optional
            The elementary geometry of the single magnetized pixel.

        ''' # TODO: Docstring!
        self.dim_proj = dim_proj  # dimension along the projection axis
        self.dim_rot = dim_rot    # dimension along the rotation axis
        self.dim_perp = dim_perp  # dimension along the axis perpendicular to proj. and rotation
#        self.a = a
#        self.b_0 = b_0
        self.u = u_mag
        self.v = v_mag
        self.tilt = tilt
        self.weights = weights
        self.weights_inv = {}
        # TODO: Not necessary:
        for key, value_list in weights.iteritems():
            for new_key, value in value_list:
                self.weights_inv.setdefault(new_key, []).append((key, value))

    @classmethod
    def single_tilt_projection(cls, mag_data, tilt=0):
        '''
        Project a magnetization distribution which is tilted around the centered y-axis.
    
        Parameters
        ----------
        mag_data : :class:`~pyramid.magdata.MagData`
            A :class:`~pyramid.magdata.MagData` object storing the magnetization distribution,
            which should be projected.
        tilt : float, optional
            The counter-clockwise tilt angle around the y-axis. Default is 1.
    
        Returns
        -------
        projection : tuple (N=3) of :class:`~numpy.ndarray` (N=2)
            The in-plane projection of the magnetization as a tuple, storing the `u`- and `v`-component
            of the magnetization and the thickness projection for the resulting 2D-grid. The latter
            has to be multiplied with the grid spacing for a value in nm.
    
        '''
        assert isinstance(mag_data, MagData), 'Parameter mag_data has to be a MagData object!'
    
        def get_position(p, m, b, size):
            y, x = np.array(p)[:, 0]+0.5, np.array(p)[:, 1]+0.5
            return (y-m*x-b)/np.sqrt(m**2+1) + size/2.
    
        def get_impact(pos, r, size):
            return [x for x in np.arange(np.floor(pos-r), np.floor(pos+r)+1, dtype=int)
                    if 0 <= x < size]
    
        def get_weight(delta, rho):  # use circles to represent the voxels
            lo, up = delta-rho, delta+rho
            # Upper boundary:
            if up >= 1:
                w_up = 0.5
            else:
                w_up = (up*np.sqrt(1-up**2) + np.arctan(up/np.sqrt(1-up**2))) / pi
            # Lower boundary:
            if lo <= -1:
                w_lo = -0.5
            else:
                w_lo = (lo*np.sqrt(1-lo**2) + np.arctan(lo/np.sqrt(1-lo**2))) / pi
            return w_up - w_lo
    
        # Set starting variables:
        # length along projection (proj, z), rotation (rot, y) and perpendicular (perp, x) axis:
        dim_proj, dim_rot, dim_perp = mag_data.dim
        z_mag, y_mag, x_mag = mag_data.magnitude
        mask = mag_data.get_mask()
        projection = (np.zeros((dim_rot, dim_perp)),
                      np.zeros((dim_rot, dim_perp)),
                      np.zeros((dim_rot, dim_perp)))
        # Creating coordinate list of all voxels:
        voxels = list(itertools.product(range(dim_proj), range(dim_perp)))
    
        # Calculate positions along the projected pixel coordinate system:
        center = (dim_proj/2., dim_perp/2.)
        m = np.where(tilt<=pi, -1/np.tan(tilt+1E-30), 1/np.tan(tilt+1E-30))
        b = center[0] - m * center[1]
        positions = get_position(voxels, m, b, dim_perp)
    
        # Calculate weights:
        r = 1/np.sqrt(np.pi)  # radius of the voxel circle
        rho = 0.5 / r
        weights = {}
        for i, voxel in enumerate(voxels):
            voxel_weights = []
            impacts = get_impact(positions[i], r, dim_perp)
            for impact in impacts:
                distance = np.abs(impact+0.5 - positions[i])
                delta = distance / r
                voxel_weights.append((impact, get_weight(delta, rho)))
            weights[voxel] = voxel_weights
        
        # Calculate projection with the calculated weights for the voxels (rotation around y-axis):
        for i, voxel in enumerate(weights):
            voxel_weights = weights[voxel]
            for pixel, weight in voxel_weights:
                # Component parallel to tilt axis (':' goes over all slices):
                projection[0][:, pixel] += weight * y_mag[voxel[0], :, voxel[1]]
                # Component perpendicular to tilt axis:
                projection[1][:, pixel] += weight * (x_mag[voxel[0], :, voxel[1]]*np.cos(tilt)
                                                   + z_mag[voxel[0], :, voxel[1]]*np.sin(tilt))
                # Thickness profile:
                projection[2][:, pixel] += weight * mask[voxel[0], :, voxel[1]]
        
        return Projection(dim_proj, dim_rot, dim_perp, projection[0], projection[1], projection[2],
                          weights, tilt)

    def get_weight_matrix(self):
        dim_proj, dim_rot, dim_perp = self.dim_proj, self.dim_rot, self.dim_perp
        weights = self.weights
        size_2d = dim_rot * dim_perp
        size_3d = dim_rot * dim_perp * dim_proj
        weight_matrix = np.zeros((size_2d, size_3d))
        for voxel in weights:
            voxel_weights = weights[voxel]
            for pixel, weight in voxel_weights:
                for i in range(dim_rot):
                    pixel_pos = i*dim_rot + pixel
                    voxel_pos = voxel[0]*dim_proj*dim_rot + i*dim_rot + voxel[1]
                    weight_matrix[pixel_pos, voxel_pos] = weight
        return weight_matrix

    def multiply_weight_matrix(self, vector):
        dim_proj, dim_rot, dim_perp = self.dim_proj, self.dim_rot, self.dim_perp
        weights = self.weights
        size_2d = dim_rot * dim_perp
        result = np.zeros(size_2d)
        for voxel in weights:
            voxel_weights = weights[voxel]
            for pixel, weight in voxel_weights:
                for i in range(dim_rot):
                    pixel_pos = i*dim_rot + pixel
                    voxel_pos = voxel[0]*dim_proj*dim_rot + i*dim_rot + voxel[1]
                    result[pixel_pos] += weight * vector[voxel_pos]
        return result

    def multiply_weight_matrix_T(self, vector):
        dim_proj, dim_rot, dim_perp = self.dim_proj, self.dim_rot, self.dim_perp
        weights = self.weights
        size_3d = dim_proj * dim_rot * dim_perp
        result = np.zeros(size_3d)
        for voxel in weights:
            voxel_weights = weights[voxel]
            for pixel, weight in voxel_weights:
                for i in range(dim_rot):
                    pixel_pos = i*dim_rot + pixel
                    voxel_pos = voxel[0]*dim_proj*dim_rot + i*dim_rot + voxel[1]
                    result[voxel_pos] += weight * vector[pixel_pos]
        return result

    def get_jacobi(self):
        '''Calculate the Jacobi matrix for the phase calculation from a projected magnetization.

        Parameters
        ----------
        None
        
        Returns
        -------
        jacobi : :class:`~numpy.ndarray` (N=2)
            Jacobi matrix containing the derivatives of the phase at every pixel with respect to
            the projected magetization. Has `N` columns for the `u`-component of the magnetization
            and `N` columns for the `v`-component (from left to right) and ``N**2`` rows for the
            phase at every pixel.

        Notes
        -----
        Just use for small dimensions, Jacobi Matrix scales with order of ``N**4``.

        ''' # TODO: Docstring!
        dim_proj, dim_rot, dim_perp = self.dim_proj, self.dim_rot, self.dim_perp
        size_2d = dim_rot * dim_perp
        size_3d = dim_rot * dim_perp * dim_proj
        weights = self.weights
        tilt = self.tilt

        def get_weight_matrix():
            weight_matrix = np.zeros((size_2d, size_3d))
            for voxel in weights:
                voxel_weights = weights[voxel]
                for pixel, weight in voxel_weights:
                    for i in range(dim_rot):
                        pixel_pos = i*dim_rot + pixel
                        voxel_pos = voxel[0]*dim_proj*dim_rot + i*dim_rot + voxel[1]
                        weight_matrix[pixel_pos, voxel_pos] = weight
            return weight_matrix

        weight_matrix = get_weight_matrix()
        jacobi = np.zeros((2*size_2d, 3*size_3d))
        jacobi[:size_2d, :size_3d] = np.cos(tilt) * weight_matrix
        jacobi[:size_2d, 2*size_3d:] = np.sin(tilt) * weight_matrix
        jacobi[size_2d:, size_3d:2*size_3d] = weight_matrix
        return jacobi

    def multiply_jacobi(self, vector):
        '''Calculate the product of the Jacobi matrix with a given `vector`.

        Parameters
        ----------
        vector : :class:`~numpy.ndarray` (N=1)
            Vectorized form of the magnetization in `u`- and `v`-direction of every pixel
            (row-wise). The first ``N**2`` elements have to correspond to the `u`-, the next
            ``N**2`` elements to the `v`-component of the magnetization.
        
        Returns
        -------
        result : :class:`~numpy.ndarray` (N=1)
            Product of the Jacobi matrix (which is not explicitely calculated) with the vector.

        ''' # TODO: Docstring!
        dim_proj, dim_rot, dim_perp = self.dim_proj, self.dim_rot, self.dim_perp
        size_2d = dim_rot * dim_perp
        size_3d = dim_rot * dim_perp * dim_proj
        weights = self.weights
        tilt = self.tilt
        assert len(vector) == 3*size_3d

        def multiply_weight_matrix(vector):
            result = np.zeros(size_2d)
            for voxel in weights:
                voxel_weights = weights[voxel]
                for pixel, weight in voxel_weights:
                    for i in range(dim_rot):
                        pixel_pos = i*dim_rot + pixel
                        voxel_pos = voxel[0]*dim_proj*dim_rot + i*dim_rot + voxel[1]
                        result[pixel_pos] += weight * vector[voxel_pos]
            return result
        
        result = np.zeros(2*size_2d)
        result[:size_2d] = (np.cos(tilt) * multiply_weight_matrix(vector[:size_3d])
                          + np.sin(tilt) * multiply_weight_matrix(vector[2*size_3d:]))
        result[size_2d:] = multiply_weight_matrix(vector[size_3d:2*size_3d])
        return result

    def multiply_jacobi_T(self, vector):
        '''Calculate the product of the transposed Jacobi matrix with a given `vector`.

        Parameters
        ----------
        vector : :class:`~numpy.ndarray` (N=1)
            Vectorized form of the magnetization in `u`- and `v`-direction of every pixel
            (row-wise). The first ``N**2`` elements have to correspond to the `u`-, the next
            ``N**2`` elements to the `v`-component of the magnetization.
        
        Returns
        -------
        result : :class:`~numpy.ndarray` (N=1)
            Product of the transposed Jacobi matrix (which is not explicitely calculated) with
            the vector.

        ''' # TODO: Docstring!
        dim_proj, dim_rot, dim_perp = self.dim_proj, self.dim_rot, self.dim_perp
        size_2d = dim_rot * dim_perp
        size_3d = dim_rot * dim_perp * dim_proj
        weights = self.weights
        tilt = self.tilt
        assert len(vector) == 2*size_2d

        def multiply_weight_matrix_T(vector):
            result = np.zeros(size_3d)
            for voxel in weights:
                voxel_weights = weights[voxel]
                for pixel, weight in voxel_weights:
                    for i in range(dim_rot):
                        pixel_pos = i*dim_rot + pixel
                        voxel_pos = voxel[0]*dim_proj*dim_rot + i*dim_rot + voxel[1]
                        result[voxel_pos] += weight * vector[pixel_pos]
            return result

        result = np.zeros(3*size_3d)
        result[:size_3d] = np.cos(tilt) * multiply_weight_matrix_T(vector[:size_2d])
        result[size_3d:2*size_3d] = multiply_weight_matrix_T(vector[size_2d:])
        result[2*size_3d:] = np.sin(tilt) * multiply_weight_matrix_T(vector[:size_2d])
        return result

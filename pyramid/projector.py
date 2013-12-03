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
    
    def __init__(self, u_mag, v_mag, thickness, weights):
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
        self.dim = np.shape(u_mag)
#        self.a = a
#        self.b_0 = b_0
        self.u = u_mag
        self.v = v_mag
        self.weights = weights

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
        v_dim, u_dim = self.dim
        size = v_dim * u_dim
        jacobi = np.zeros((v_dim*u_dim, 2*v_dim*u_dim))  
        weight_matrix = np.zeros(dim)  # Check if correct (probably not, x and z instead of x and y)
        voxels = list(itertools.product(range(dim_proj), range(dim_perp)))
        for i, voxel in enumerate(self.weights):
            voxel_weights = self.weights[voxel]
            for pixel, weight in voxel_weights:
                # Component parallel to tilt axis (':' goes over all slices):
                projection[0][:, pixel] += weight * y_mag[voxel[0], :, voxel[1]]
                # Component perpendicular to tilt axis:
                projection[1][:, pixel] += weight * (x_mag[voxel[0], :, voxel[1]]*np.cos(tilt)
                                                   + z_mag[voxel[0], :, voxel[1]]*np.sin(tilt))
                # Thickness profile:
                projection[2][:, pixel] += weight * mask[voxel[0], :, voxel[1]]
        
        for i, vx in enumerate(self.weights):
            vx_pos = vx[1]*3 + vx[0]
            for px, weight in self.weights[voxel]:
                px_pos = (px[2]*3 + px[1])*3 + px[0]
                weight_matrix[vx_pos, px_pos] = weight
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        v_dim, u_dim = self.dim
        jacobi = np.zeros((v_dim*u_dim, 2*v_dim*u_dim))  
#       nc.get_jacobi_core(dim[0], dim[1], v_phi, u_phi, jacobi)
#       return jacobi
        for j in range(v_dim):
            for i in range(u_dim):
                u_column = i + u_dim*j
                v_column = i + u_dim*j + u_dim*v_dim
                u_min = (u_dim-1) - i
                u_max = (2*u_dim-1) - i
                v_min = (v_dim-1) - j
                v_max = (2*v_dim-1) - j
                # u_dim*v_dim columns for the u-component:
                jacobi[:, u_column] = self.u[v_min:v_max, u_min:u_max].reshape(-1)
                # u_dim*v_dim columns for the v-component (note the minus!):
                jacobi[:, v_column] = -self.v[v_min:v_max, u_min:u_max].reshape(-1)
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
        v_dim, u_dim = self.dim
        size = v_dim * u_dim
        assert len(vector) == 2*size, 'vector size not compatible!'
        result = np.zeros(size)
        for s in range(size):  # column-wise (two columns at a time, u- and v-component)
            i = s % u_dim
            j = int(s/u_dim)
            u_min = (u_dim-1) - i
            u_max = (2*u_dim-1) - i
            v_min = (v_dim-1) - j
            v_max = (2*v_dim-1) - j
            result += vector[s]*self.u[v_min:v_max, u_min:u_max].reshape(-1)  # u
            result += vector[s+size]*-self.v[v_min:v_max, u_min:u_max].reshape(-1)  # v        
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
        v_dim, u_dim = self.dim
        size = v_dim * u_dim
        assert len(vector) == size, 'vector size not compatible!'
        result = np.zeros(2*size)
        for s in range(size):  # row-wise (two rows at a time, u- and v-component)
            i = s % u_dim
            j = int(s/u_dim)
            u_min = (u_dim-1) - i
            u_max = (2*u_dim-1) - i
            v_min = (v_dim-1) - j
            v_max = (2*v_dim-1) - j
            result[s] = np.sum(vector*self.u[v_min:v_max, u_min:u_max].reshape(-1))  # u
            result[s+size] = np.sum(vector*-self.v[v_min:v_max, u_min:u_max].reshape(-1))  # v        
        return result

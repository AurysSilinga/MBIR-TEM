# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""This module provides the abstract base class :class:`~.Projector` and concrete subclasses for
projections of vector and scalar fields."""

import abc
import itertools
import logging

import numpy as np
from numpy import pi
from scipy.sparse import coo_matrix, csr_matrix

from pyramid.fielddata import VectorData, ScalarData
from pyramid.quaternion import Quaternion

__all__ = ['RotTiltProjector', 'XTiltProjector', 'YTiltProjector', 'SimpleProjector']


class Projector(object, metaclass=abc.ABCMeta):
    """Abstract base class representing a projection function.

    The :class:`~.Projector` class represents a projection function for a 3-dimensional
    vector- or scalar field onto a 2-dimensional grid. :class:`~.Projector` is an abstract base
    class and provides a unified interface which should be subclassed with a custom
    :func:`__init__` function, which should call the parent :func:`__init__` method. Concrete
    subclasses can be called as a function and take a `vector` as argument which contains the
    3-dimensional field. The output is the projected field, given as a `vector`. Depending on the
    length of the input and the given dimensions `dim` at construction time, vector or scalar
    projection is choosen intelligently.

    Attributes
    ----------
    dim : tuple (N=3)
        Dimensions (z, y, x) of the magnetization distribution.
    dim_uv : tuple (N=2)
        Dimensions (v, u) of the projected grid.
    size_3d : int
        Number of voxels of the 3-dimensional grid.
    size_2d : int
        Number of pixels of the 2-dimensional projected grid.
    weight : :class:`~scipy.sparse.csr_matrix` (N=2)
        The weight matrix containing the weighting coefficients for the 3D to 2D mapping.
    coeff : list (N=2)
        List containing the six weighting coefficients describing the influence of the 3 components
        of a 3-dimensional vector field on the 2 projected components.
    m: int
        Size of the image space.
    n: int
        Size of the input space.

    """

    _log = logging.getLogger(__name__ + '.Projector')

    @abc.abstractmethod
    def __init__(self, dim, dim_uv, weight, coeff):
        self._log.debug('Calling __init__')
        self.dim = dim
        self.dim_uv = dim_uv
        self.weight = weight
        self.coeff = coeff
        self.size_2d, self.size_3d = weight.shape
        self.n = 3 * np.prod(dim)
        self.m = 2 * np.prod(dim_uv)
        self._log.debug('Created ' + str(self))

    def __repr__(self):
        self._log.debug('Calling __repr__')
        return '%s(dim=%r, dim_uv=%r, weight=%r, coeff=%r)' % \
               (self.__class__, self.dim, self.dim_uv, self.weight, self.coeff)

    def __str__(self):
        self._log.debug('Calling __str__')
        return 'Projector(dim=%s, dim_uv=%s, coeff=%s)' % (self.dim, self.dim_uv, self.coeff)

    def __call__(self, field_data):
        if isinstance(field_data, VectorData):
            field_empty = np.zeros((3, 1) + self.dim_uv, dtype=field_data.field.dtype)
            field_data_proj = VectorData(field_data.a, field_empty)
            field_proj = self.jac_dot(field_data.field_vec).reshape((2,) + self.dim_uv)
            field_data_proj.field[0:2, 0, ...] = field_proj
        elif isinstance(field_data, ScalarData):
            field_empty = np.zeros((1,) + self.dim_uv, dtype=field_data.field.dtype)
            field_data_proj = ScalarData(field_data.a, field_empty)
            field_proj = self.jac_dot(field_data.field_vec).reshape(self.dim_uv)
            field_data_proj.field[0, ...] = field_proj
        else:
            raise TypeError('Input is neither of type VectorData or ScalarData')
        return field_data_proj

    def _vector_field_projection(self, vector):
        result = np.zeros(2 * self.size_2d, dtype=vector.dtype)
        # Go over all possible component projections (z, y, x) to (u, v):
        vec_x, vec_y, vec_z = np.split(vector, 3)
        vec_x_weighted = self.weight.dot(vec_x)
        vec_y_weighted = self.weight.dot(vec_y)
        vec_z_weighted = self.weight.dot(vec_z)
        slice_u = slice(0, self.size_2d)
        slice_v = slice(self.size_2d, 2 * self.size_2d)
        if self.coeff[0][0] != 0:  # x to u
            result[slice_u] += self.coeff[0][0] * vec_x_weighted
        if self.coeff[0][1] != 0:  # y to u
            result[slice_u] += self.coeff[0][1] * vec_y_weighted
        if self.coeff[0][2] != 0:  # z to u
            result[slice_u] += self.coeff[0][2] * vec_z_weighted
        if self.coeff[1][0] != 0:  # x to v
            result[slice_v] += self.coeff[1][0] * vec_x_weighted
        if self.coeff[1][1] != 0:  # y to v
            result[slice_v] += self.coeff[1][1] * vec_y_weighted
        if self.coeff[1][2] != 0:  # z to v
            result[slice_v] += self.coeff[1][2] * vec_z_weighted
        return result

    def _vector_field_projection_T(self, vector):
        result = np.zeros(3 * self.size_3d)
        # Go over all possible component projections (u, v) to (z, y, x):
        vec_u, vec_v = np.split(vector, 2)
        vec_u_weighted = self.weight.T.dot(vec_u)
        vec_v_weighted = self.weight.T.dot(vec_v)
        slice_x = slice(0, self.size_3d)
        slice_y = slice(self.size_3d, 2 * self.size_3d)
        slice_z = slice(2 * self.size_3d, 3 * self.size_3d)
        if self.coeff[0][0] != 0:  # u to x
            result[slice_x] += self.coeff[0][0] * vec_u_weighted
        if self.coeff[0][1] != 0:  # u to y
            result[slice_y] += self.coeff[0][1] * vec_u_weighted
        if self.coeff[0][2] != 0:  # u to z
            result[slice_z] += self.coeff[0][2] * vec_u_weighted
        if self.coeff[1][0] != 0:  # v to x
            result[slice_x] += self.coeff[1][0] * vec_v_weighted
        if self.coeff[1][1] != 0:  # v to y
            result[slice_y] += self.coeff[1][1] * vec_v_weighted
        if self.coeff[1][2] != 0:  # v to z
            result[slice_z] += self.coeff[1][2] * vec_v_weighted
        return result

    def _scalar_field_projection(self, vector):
        self._log.debug('Calling _scalar_field_projection')
        return np.array(self.weight.dot(vector))

    def _scalar_field_projection_T(self, vector):
        self._log.debug('Calling _scalar_field_projection_T')
        return np.array(self.weight.T.dot(vector))

    def jac_dot(self, vector):
        """Multiply a `vector` with the jacobi matrix of this :class:`~.Projector` object.

        Parameters
        ----------
        vector : :class:`~numpy.ndarray` (N=1)
            Vector containing the field which should be projected. Must have the same or 3 times
            the size of `size_3d` of the projector for  scalar and vector projection, respectively.

        Returns
        -------
        proj_vector : :class:`~numpy.ndarray` (N=1)
            Vector containing the projected field of the 2-dimensional grid. The length is
            always`size_2d`.

        """
        if len(vector) == 3 * self.size_3d:  # mode == 'vector'
            return self._vector_field_projection(vector)
        elif len(vector) == self.size_3d:  # mode == 'scalar'
            return self._scalar_field_projection(vector)
        else:
            raise AssertionError('Vector size has to be suited either for '
                                 'vector- or scalar-field-projection!')

    def jac_T_dot(self, vector):
        """Multiply a `vector` with the transp. jacobi matrix of this :class:`~.Projector` object.

        Parameters
        ----------
        vector : :class:`~numpy.ndarray` (N=1)
            Vector containing the field which should be projected. Must have the same or 2 times
            the size of `size_2d` of the projector for  scalar and vector projection, respectively.

        Returns
        -------
        proj_vector : :class:`~numpy.ndarray` (N=1)
            Vector containing the multiplication of the input with the transposed jacobi matrix
            of the :class:`~.Projector` object.

        """
        if len(vector) == 2 * self.size_2d:  # mode == 'vector'
            return self._vector_field_projection_T(vector)
        elif len(vector) == self.size_2d:  # mode == 'scalar'
            return self._scalar_field_projection_T(vector)
        else:
            raise AssertionError('Vector size has to be suited either for '
                                 'vector- or scalar-field-projection!')

    @abc.abstractmethod
    def get_info(self, verbose):
        """Get specific information about the projector as a string.

        Parameters
        ----------
        verbose: boolean, optional
            If this is true, the text looks prettier (maybe using latex). Default is False for the
            use in file names and such.

        Returns
        -------
        info : string
            Information about the projector as a string, e.g. for the use in plot titles.

        """
        raise NotImplementedError()


class RotTiltProjector(Projector):
    """Class representing a projection function with a rotation around z followed by tilt around x.

    The :class:`~.XTiltProjector` class represents a projection function for a 3-dimensional
    vector- or scalar field onto a 2-dimensional grid, which is a concrete subclass of
    :class:`~.Projector`.

    Attributes
    ----------
    dim : tuple (N=3)
        Dimensions (z, y, x) of the magnetization distribution.
    rotation : float
        Angle in `rad` describing the rotation around the z-axis before the tilt is happening.
    tilt : float
        Angle in `rad` describing the tilt of the beam direction relative to the x-axis.
    dim_uv : tuple (N=2), optional
        Dimensions (v, u) of the projection. If not set defaults to the (y, x)-dimensions.
    subcount : int (optional)
        Number of subpixels along one axis. This is used to create the lookup table which uses
        a discrete subgrid to estimate the impact point of a voxel onto a pixel and the weight on
        all surrounding pixels. Default is 11 (odd numbers provide a symmetric center).
    sparsity : float
        Measures the sparsity of the weighting (not the complete one!), 1 means completely sparse!

    """

    _log = logging.getLogger(__name__ + '.RotTiltProjector')

    def __init__(self, dim, rotation, tilt, dim_uv=None, subcount=11):
        self._log.debug('Calling __init__')
        self.rotation = rotation
        self.tilt = tilt
        # Determine dimensions:
        dim_z, dim_y, dim_x = dim
        center = (dim_z / 2., dim_y / 2., dim_x / 2.)
        if dim_uv is None:
            dim_v = max(dim_x, dim_y)  # first rotate around z-axis (take x and y into account)
            dim_u = max(dim_v, dim_z)  # then tilt around x-axis (now z matters, too)
            dim_uv = (dim_v, dim_u)
        dim_v, dim_u = dim_uv
        # Creating coordinate list of all voxels:
        voxels = list(itertools.product(range(dim_z), range(dim_y), range(dim_x)))
        # Calculate vectors to voxels relative to rotation center:
        voxel_vecs = (np.asarray(voxels) + 0.5 - np.asarray(center)).T
        # Create tilt, rotation and combined quaternion, careful: Quaternion(w,x,y,z), not (z,y,x):
        quat_x = Quaternion.from_axisangle((1, 0, 0), tilt)  # Tilt around x-axis
        quat_z = Quaternion.from_axisangle((0, 0, 1), rotation)  # Rotate around z-axis
        quat = quat_x * quat_z  # Combined quaternion (first rotate around z, then tilt around x)
        # Calculate impact positions on the projected pixel coordinate grid (flip because quat.):
        impacts = np.flipud(quat.matrix[:2, :].dot(np.flipud(voxel_vecs)))  # only care for x/y
        impacts[1, :] += dim_u / 2.  # Shift back to normal indices
        impacts[0, :] += dim_v / 2.  # Shift back to normal indices
        # Calculate equivalence radius:
        R = (3 / (4 * np.pi)) ** (1 / 3.)
        # Prepare weight matrix calculation:
        rows = []  # 2D projection
        columns = []  # 3D distribution
        data = []  # weights
        # Create 4D lookup table (1&2: which neighbour weight?, 3&4: which subpixel is hit?)
        weight_lookup = self._create_weight_lookup(subcount, R)
        # Go over all voxels:
        for i, voxel in enumerate(voxels):
            column_index = voxel[0] * dim_y * dim_x + voxel[1] * dim_x + voxel[2]
            remainder, impact = np.modf(impacts[:, i])  # split index of impact and remainder!
            sub_pixel = (remainder * subcount).astype(dtype=np.int)  # sub_pixel inside impact px.
            # Go over all influenced pixels (impact and neighbours, indices are [0, 1, 2]!):
            for px_ind in list(itertools.product(range(3), range(3))):
                # Pixel indices influenced by the impact (px_ind-1 to center them around impact):
                pixel = (impact + np.array(px_ind) - 1).astype(dtype=np.int)
                # Check if pixel is out of bound:
                if 0 <= pixel[0] < dim_uv[0] and 0 <= pixel[1] < dim_uv[1]:
                    # Lookup weight in 4-dimensional lookup table!
                    weight = weight_lookup[px_ind[0], px_ind[1], sub_pixel[0], sub_pixel[1]]
                    # Only write into sparse matrix if weight is not zero:
                    if weight != 0.:
                        row_index = pixel[0] * dim_u + pixel[1]
                        columns.append(column_index)
                        rows.append(row_index)
                        data.append(weight)
        # Calculate weight matrix and coefficients for jacobi matrix:
        shape = (np.prod(dim_uv), np.prod(dim))
        self.sparsity = 1. - len(data) / np.prod(shape, dtype=np.float)
        weights = csr_matrix(coo_matrix((data, (rows, columns)), shape=shape))
        # Calculate coefficients by rotating unity matrix (unit vectors, (x,y,z)):
        coeff = quat.matrix[:2, :].dot(np.eye(3))
        super().__init__(dim, dim_uv, weights, coeff)
        self._log.debug('Created ' + str(self))

    @staticmethod
    def _create_weight_lookup(subcount, R):
        s = subcount
        Rz = R * s  # Radius in subgrid units
        dim_zoom = (3 * s, 3 * s)  # Dimensions of the subgrid, (3, 3) because of neighbour count!
        cent_zoom = (np.asarray(dim_zoom) / 2.).astype(dtype=np.int)  # Center of the subgrid
        y, x = np.indices(dim_zoom)
        y -= cent_zoom[0]
        x -= cent_zoom[1]
        # Calculate projected thickness of an equivalence sphere (normed!):
        d = np.where(np.hypot(x, y) <= Rz, Rz ** 2 - x ** 2 - y ** 2, 0)
        d = np.sqrt(d)
        d /= d.sum()
        # Create lookup table (4D):
        lookup = np.zeros((3, 3, s, s))
        # Go over all 9 pixels (center and neighbours):
        for pixel in list(itertools.product(range(3), range(3))):
            pixel_lb = np.array(pixel) * s  # Convert to subgrid, hit bottom left of the pixel!
            # Go over all subpixels in the center that can be hit:
            for sub_pixel in list(itertools.product(range(s), range(s))):
                shift = np.array(sub_pixel) - np.array((s // 2, s // 2))  # relative to center!
                lb = pixel_lb - shift  # Shift summing zone according to hit subpixel!
                # Make sure, that the summing zone is in bounds (otherwise correct accordingly):
                lb = np.where(lb >= 0, lb, [0, 0])
                tr = np.where(lb < 3 * s, lb + np.array((s, s)), [3 * s, 3 * s])
                # Calculate weight by summing over the summing zone:
                weight = d[lb[0]:tr[0], lb[1]:tr[1]].sum()
                lookup[pixel[0], pixel[1], sub_pixel[0], sub_pixel[1]] = weight
        return lookup

    def get_info(self, verbose=False):
        """Get specific information about the projector as a string.

        Parameters
        ----------
        verbose: boolean, optional
            If this is true, the text looks prettier (maybe using latex). Default is False for the
            use in file names and such.

        Returns
        -------
        info : string
            Information about the projector as a string, e.g. for the use in plot titles.

        """
        theta_ang = int(np.round(self.rotation * 180 / pi))
        phi_ang = int(np.round(self.tilt * 180 / pi))
        if verbose:
            return u'$\\theta = {:d}$°, $\phi = {:d}$°'.format(theta_ang, phi_ang)
        else:
            return u'theta={:d}_phi={:d}°'.format(theta_ang, phi_ang)


class XTiltProjector(Projector):
    """Class representing a projection function with a tilt around the x-axis.

    The :class:`~.XTiltProjector` class represents a projection function for a 3-dimensional
    vector- or scalar field onto a 2-dimensional grid, which is a concrete subclass of
    :class:`~.Projector`.

    Attributes
    ----------
    dim : tuple (N=3)
        Dimensions (z, y, x) of the magnetization distribution.
    tilt : float
        Angle in `rad` describing the tilt of the beam direction relative to the x-axis.
    dim_uv : tuple (N=2), optional
        Dimensions (v, u) of the projection. If not set defaults to the (y, x)-dimensions.

    """

    _log = logging.getLogger(__name__ + '.XTiltProjector')

    def __init__(self, dim, tilt, dim_uv=None):
        self._log.debug('Calling __init__')
        self.tilt = tilt
        # Set starting variables:
        # length along projection (proj, z), perpendicular (perp, y) and rotation (rot, x) axis:
        dim_proj, dim_perp, dim_rot = dim
        if dim_uv is None:
            dim_uv = (max(dim_perp, dim_proj), dim_rot)  # x-y-plane
        dim_v, dim_u = dim_uv  # y, x
        assert dim_v >= dim_perp and dim_u >= dim_rot, 'Projected dimensions are too small!'
        # Creating coordinate list of all voxels (for one slice):
        voxels = list(itertools.product(range(dim_proj), range(dim_perp)))  # z-y-plane
        # Calculate positions along the projected pixel coordinate system:
        center = (dim_proj / 2., dim_perp / 2.)
        positions = self._get_position(voxels, center, tilt, dim_v)
        # Calculate weight-matrix:
        r = 1 / np.sqrt(np.pi)  # radius of the voxel circle
        rho = 0.5 / r
        row = []
        col = []
        data = []
        # one slice:
        for i, voxel in enumerate(voxels):
            impacts = self._get_impact(positions[i], r, dim_v)  # impact along projected y-axis
            voxel_index = voxel[0] * dim_rot * dim_perp + voxel[1] * dim_rot  # 0: z, 1: y
            for impact in impacts:
                impact_index = impact * dim_u + (dim_u - dim_rot) // 2
                distance = np.abs(impact + 0.5 - positions[i])
                delta = distance / r
                col.append(voxel_index)
                row.append(impact_index)
                data.append(self._get_weight(delta, rho))
        # All other slices (along x):
        columns = col
        rows = row
        for s in np.arange(1, dim_rot):
            columns = np.hstack((np.array(columns), np.array(col) + s))
            rows = np.hstack((np.array(rows), np.array(row) + s))
        # Calculate weight matrix and coefficients for jacobi matrix:
        shape = (np.prod(dim_uv), np.prod(dim))
        self.sparsity = 1. - len(data) / np.prod(shape, dtype=np.float)
        weight = csr_matrix(coo_matrix((np.tile(data, dim_rot), (rows, columns)), shape=shape))
        coeff = [[1, 0, 0], [0, np.cos(tilt), np.sin(tilt)]]
        super().__init__(dim, dim_uv, weight, coeff)
        self._log.debug('Created ' + str(self))

    @staticmethod
    def _get_position(points, center, tilt, size):
        point_vecs = np.asarray(points) + 0.5 - np.asarray(center)  # vectors pointing to points
        direc_vec = np.array((np.cos(tilt), -np.sin(tilt)))  # vector pointing along projection
        distances = np.cross(direc_vec, point_vecs)  # here (special case): divisor is one!
        distances += size / 2.  # Shift to the center of the projection
        return distances

    @staticmethod
    def _get_impact(pos, r, size):
        return [x for x in np.arange(np.floor(pos - r), np.floor(pos + r) + 1, dtype=int)
                if 0 <= x < size]

    @staticmethod
    def _get_weight(delta, rho):  # use circles to represent the voxels
        lo, up = delta - rho, delta + rho
        # Upper boundary:
        if up >= 1:
            w_up = 0.5
        else:
            w_up = (up * np.sqrt(1 - up ** 2) + np.arctan(up / np.sqrt(1 - up ** 2))) / pi
        # Lower boundary:
        if lo <= -1:
            w_lo = -0.5
        else:
            w_lo = (lo * np.sqrt(1 - lo ** 2) + np.arctan(lo / np.sqrt(1 - lo ** 2))) / pi
        return w_up - w_lo

    def get_info(self, verbose=False):
        """Get specific information about the projector as a string.

        Parameters
        ----------
        verbose: boolean, optional
            If this is true, the text looks prettier (maybe using latex). Default is False for the
            use in file names and such.

        Returns
        -------
        info : string
            Information about the projector as a string, e.g. for the use in plot titles.

        """
        if verbose:
            return u'x-tilt: $\phi = {:d}$°'.format(int(np.round(self.tilt * 180 / pi)))
        else:
            return u'xtilt_phi={:d}°'.format(int(np.round(self.tilt * 180 / pi)))


class YTiltProjector(Projector):
    """Class representing a projection function with a tilt around the y-axis.

    The :class:`~.YTiltProjector` class represents a projection function for a 3-dimensional
    vector- or scalar field onto a 2-dimensional grid, which is a concrete subclass of
    :class:`~.Projector`.

    Attributes
    ----------
    dim : tuple (N=3)
        Dimensions (z, y, x) of the magnetization distribution.
    tilt : float
        Angle in `rad` describing the tilt of the beam direction relative to the y-axis.
    dim_uv : tuple (N=2), optional
        Dimensions (v, u) of the projection. If not set defaults to the (y, x)-dimensions.

    """

    _log = logging.getLogger(__name__ + '.YTiltProjector')

    def __init__(self, dim, tilt, dim_uv=None):
        self._log.debug('Calling __init__')
        self.tilt = tilt
        # Set starting variables:
        # length along projection (proj, z), rotation (rot, y) and perpendicular (perp, x) axis:
        dim_proj, dim_rot, dim_perp = dim
        if dim_uv is None:
            dim_uv = (dim_rot, max(dim_perp, dim_proj))  # x-y-plane
        dim_v, dim_u = dim_uv  # y, x
        assert dim_v >= dim_rot and dim_u >= dim_perp, 'Projected dimensions are too small!'
        # Creating coordinate list of all voxels (for one slice):
        voxels = list(itertools.product(range(dim_proj), range(dim_perp)))  # z-x-plane
        # Calculate positions along the projected pixel coordinate system:
        center = (dim_proj / 2., dim_perp / 2.)
        positions = self._get_position(voxels, center, tilt, dim_u)
        # Calculate weight-matrix:
        r = 1 / np.sqrt(np.pi)  # radius of the voxel circle
        rho = 0.5 / r
        row = []
        col = []
        data = []
        # one slice:
        for i, voxel in enumerate(voxels):
            impacts = self._get_impact(positions[i], r, dim_u)  # impact along projected x-axis
            voxel_index = voxel[0] * dim_perp * dim_rot + voxel[1]  # 0: z, 1: x
            for impact in impacts:
                impact_index = impact + (dim_v - dim_rot) // 2 * dim_u
                distance = np.abs(impact + 0.5 - positions[i])
                delta = distance / r
                col.append(voxel_index)
                row.append(impact_index)
                data.append(self._get_weight(delta, rho))
        # All other slices (along y):
        columns = col
        rows = row
        for s in np.arange(1, dim_rot):
            columns = np.hstack((np.array(columns), np.array(col) + s * dim_perp))
            rows = np.hstack((np.array(rows), np.array(row) + s * dim_u))
        # Calculate weight matrix and coefficients for jacobi matrix:
        shape = (np.prod(dim_uv), np.prod(dim))
        self.sparsity = 1. - len(data) / np.prod(shape, dtype=np.float)
        weight = csr_matrix(coo_matrix((np.tile(data, dim_rot), (rows, columns)), shape=shape))
        coeff = [[np.cos(tilt), 0, np.sin(tilt)], [0, 1, 0]]
        super().__init__(dim, dim_uv, weight, coeff)
        self._log.debug('Created ' + str(self))

    @staticmethod
    def _get_position(points, center, tilt, size):
        point_vecs = np.asarray(points) + 0.5 - np.asarray(center)  # vectors pointing to points
        direc_vec = np.array((np.cos(tilt), -np.sin(tilt)))  # vector pointing along projection
        distances = np.cross(direc_vec, point_vecs)  # here (special case): divisor is one!
        distances += size / 2.  # Shift to the center of the projection
        return distances

    @staticmethod
    def _get_impact(pos, r, size):
        return [x for x in np.arange(np.floor(pos - r), np.floor(pos + r) + 1, dtype=int)
                if 0 <= x < size]

    @staticmethod
    def _get_weight(delta, rho):  # use circles to represent the voxels
        lo, up = delta - rho, delta + rho
        # Upper boundary:
        if up >= 1:
            w_up = 0.5
        else:
            w_up = (up * np.sqrt(1 - up ** 2) + np.arctan(up / np.sqrt(1 - up ** 2))) / pi
        # Lower boundary:
        if lo <= -1:
            w_lo = -0.5
        else:
            w_lo = (lo * np.sqrt(1 - lo ** 2) + np.arctan(lo / np.sqrt(1 - lo ** 2))) / pi
        return w_up - w_lo

    def get_info(self, verbose=False):
        """Get specific information about the projector as a string.

        Parameters
        ----------
        verbose: boolean, optional
            If this is true, the text looks prettier (maybe using latex). Default is False for the
            use in file names and such.

        Returns
        -------
        info : string
            Information about the projector as a string, e.g. for the use in plot titles.

        """
        if verbose:
            return u'y-tilt: $\phi = {:d}$°'.format(int(np.round(self.tilt * 180 / pi)))
        else:
            return u'ytilt_phi={:d}°'.format(int(np.round(self.tilt * 180 / pi)))


class SimpleProjector(Projector):
    """Class representing a projection function along one of the major axes.

    The :class:`~.SimpleProjector` class represents a projection function for a 3-dimensional
    vector- or scalar field onto a 2-dimensional grid, which is a concrete subclass of
    :class:`~.Projector`.

    Attributes
    ----------
    dim : tuple (N=3)
        Dimensions (z, y, x) of the magnetization distribution.
    axis : {'z', 'y', 'x'}, optional
        Main axis along which the magnetic distribution is projected (given as a string). Defaults
        to the z-axis.
    dim_uv : tuple (N=2), optional
        Dimensions (v, u) of the projection. If not set it uses the 3D default dimensions.

    """

    _log = logging.getLogger(__name__ + '.SimpleProjector')
    AXIS_DICT = {'z': (0, 1, 2), 'y': (1, 0, 2), 'x': (2, 1, 0)}  # (0:z, 1:y, 2:x) -> (proj, v, u)

    # coordinate switch for 'x': u, v --> z, y (not y, z!)!

    def __init__(self, dim, axis='z', dim_uv=None):
        self._log.debug('Calling __init__')
        assert axis in {'z', 'y', 'x'}, 'Projection axis has to be x, y or z (given as a string)!'
        self.axis = axis
        proj, v, u = self.AXIS_DICT[axis]
        dim_proj, dim_v, dim_u = dim[proj], dim[v], dim[u]
        dim_z, dim_y, dim_x = dim
        size_2d = dim_u * dim_v
        size_3d = np.prod(dim)
        data = np.repeat(1, size_3d)  # size_3d ones in the matrix (each voxel is projected)
        indptr = np.arange(0, size_3d + 1, dim_proj)  # each row has dim_proj 1-entries
        if axis == 'z':
            self._log.debug('Projecting along the z-axis')
            coeff = [[1, 0, 0], [0, 1, 0]]
            indices = np.array([np.arange(row, size_3d, size_2d)
                                for row in range(size_2d)]).reshape(-1)
        elif axis == 'y':
            self._log.debug('Projection along the y-axis')
            coeff = [[1, 0, 0], [0, 0, 1]]
            indices = np.array(
                [np.arange(row % dim_x, dim_x * dim_y, dim_x) + row // dim_x * dim_x * dim_y
                 for row in range(size_2d)]).reshape(-1)
        elif axis == 'x':
            self._log.debug('Projection along the x-axis')
            coeff = [[0, 0, 1], [0, 1, 0]]  # Caution, coordinate switch: u, v --> z, y (not y, z!)
            indices = np.array(
                [np.arange(dim_x) + (row % dim_z) * dim_x * dim_y + row // dim_z * dim_x
                 for row in range(size_2d)]).reshape(-1)
        else:
            raise ValueError('{} is not a valid axis parameter (use x, y or z)!'.format(axis))
        if dim_uv is not None:
            indptr = list(indptr)  # convert to use insert() and append()
            # Calculate padding:
            d_v = (np.floor((dim_uv[0] - dim_v) / 2).astype(int),
                   np.ceil((dim_uv[0] - dim_v) / 2).astype(int))
            d_u = (np.floor((dim_uv[1] - dim_u) / 2).astype(int),
                   np.ceil((dim_uv[1] - dim_u) / 2).astype(int))
            indptr.extend([indptr[-1]] * d_v[1] * dim_uv[1])  # add empty lines at the end
            for i in np.arange(dim_v, 0, -1):  # all slices in between
                up, lo = i * dim_u, (i - 1) * dim_u  # upper / lower slice end
                indptr[up:up] = [indptr[up]] * d_u[1]  # end of the slice
                indptr[lo:lo] = [indptr[lo]] * d_u[0]  # start of the slice
            indptr = [0] * d_v[0] * dim_uv[1] + indptr  # insert empty rows at the beginning
        else:  # Make sure dim_uv is defined (used for the assertion)
            dim_uv = dim_v, dim_u
        assert dim_uv[0] >= dim_v and dim_uv[1] >= dim_u, 'Projected dimensions are too small!'
        # Create weight-matrix:
        shape = (np.prod(dim_uv), np.prod(dim))
        self.sparsity = 1. - len(data) / np.prod(shape, dtype=np.float)
        weight = csr_matrix((data, indices, indptr), shape=shape)
        super().__init__(dim, dim_uv, weight, coeff)
        self._log.debug('Created ' + str(self))

    def get_info(self, verbose=False):
        """Get specific information about the projector as a string.

        Parameters
        ----------
        verbose: boolean, optional
            If this is true, the text looks prettier (maybe using latex). Default is False for the
            use in file names and such.

        Returns
        -------
        info : string
            Information about the projector as a string, e.g. for the use in plot titles.

        """
        if verbose:
            return 'projected along {}-axis'.format(self.axis)
        else:
            return '{}axis'.format(self.axis)

# -*- coding: utf-8 -*-
"""This module provides the abstract base class :class:`~.Projector` and concrete subclasses for
projections of vector and scalar fields."""


import logging

import numpy as np
from numpy import pi

import abc

import itertools

from scipy.sparse import coo_matrix, csr_matrix


class Projector(object):

    '''Abstract base class representing a projection function.
    
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
    dim_uv : tuple (N=2)
        Dimensions (v, u) of the projected grid.
    size_3d : int
        Number of voxels of the 3-dimensional grid.
    size_2d : int
        Number of pixels of the 2-dimensional projected grid.
    weight : :class:`~scipy.sparse.csr_matrix` (N=2)
        The weight matrix containing the weighting coefficients describing the influence of all
        3-dimensional voxels on the 2-dimensional pixels of the projection.
    coeff : list (N=2)
        List containing the six weighting coefficients describing the influence of the 3 components
        of a 3-dimensional vector field on the 2 projected components.

    '''

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, dim_uv, weight, coeff):
        self.log = logging.getLogger(__name__)
        self.log.info('Calling __init__')
        self.dim_uv = dim_uv
        self.weight = weight
        self.coeff = coeff
        self.size_2d, self.size_3d = weight.shape
        self.log.info('Created '+str(self))

    def __call__(self, vector):
        self.log.info('Calling as function')
#        print 'Projector - __call__:', len(vector)
        return self.jac_dot(vector)

    def _vector_field_projection(self, vector):
        self.log.info('Calling _vector_field_projection')
        size_2d, size_3d = self.size_2d, self.size_3d
        result = np.zeros(2*size_2d)
        # Go over all possible component projections (z, y, x) to (u, v):
        if self.coeff[0][0] != 0:  # x to u
            result[:size_2d] += self.coeff[0][0] * self.weight.dot(vector[:size_3d])
        if self.coeff[0][1] != 0:  # y to u
            result[:size_2d] += self.coeff[0][1] * self.weight.dot(vector[size_3d:2*size_3d])
        if self.coeff[0][2] != 0:  # z to u
            result[:size_2d] += self.coeff[0][2] * self.weight.dot(vector[2*size_3d:])
        if self.coeff[1][0] != 0:  # x to v
            result[size_2d:] += self.coeff[1][0] * self.weight.dot(vector[:size_3d])
        if self.coeff[1][1] != 0:  # y to v
            result[size_2d:] += self.coeff[1][1] * self.weight.dot(vector[size_3d:2*size_3d])
        if self.coeff[1][2] != 0:  # z to v
            result[size_2d:] += self.coeff[1][2] * self.weight.dot(vector[2*size_3d:])
        return result

    def _vector_field_projection_T(self, vector):
        self.log.info('Calling _vector_field_projection_T')
        size_2d, size_3d = self.size_2d, self.size_3d
        result = np.zeros(3*size_3d)
        # Go over all possible component projections (z, y, x) to (u, v):
        if self.coeff[0][0] != 0:  # x to u
            result[:size_3d] += self.coeff[0][0] * self.weight.T.dot(vector[:size_2d])
        if self.coeff[0][1] != 0:  # y to u
            result[:size_3d] += self.coeff[0][1] * self.weight.T.dot(vector[size_2d:])
        if self.coeff[0][2] != 0:  # z to u
            result[size_3d:2*size_3d] += self.coeff[0][2] * self.weight.T.dot(vector[:size_2d])
        if self.coeff[1][0] != 0:  # x to v
            result[size_3d:2*size_3d] += self.coeff[1][0] * self.weight.T.dot(vector[size_2d:])
        if self.coeff[1][1] != 0:  # y to v
            result[2*size_3d:] += self.coeff[1][1] * self.weight.T.dot(vector[:size_2d])
        if self.coeff[1][2] != 0:  # z to v
            result[2*size_3d:] += self.coeff[1][2] * self.weight.T.dot(vector[size_2d:])
        return result

    def _scalar_field_projection(self, vector):
        self.log.info('Calling _scalar_field_projection')
        return np.array(self.weight.dot(vector))

    def _scalar_field_projection_T(self, vector):
        self.log.info('Calling _scalar_field_projection_T')
        return np.array(self.weight.T.dot(vector))

    def jac_dot(self, vector):
        '''Multiply a `vector` with the jacobi matrix of this :class:`~.Projector` object.

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

        '''
        self.log.info('Calling jac_dot')
#        print 'Projector - jac_dot:', len(vector)
        if len(vector) == 3*self.size_3d:  # mode == 'vector'
            self.log.info('mode == vector')
            return self._vector_field_projection(vector)
        elif len(vector) == self.size_3d:  # mode == 'scalar'
            self.log.info('mode == scalar')
            return self._scalar_field_projection(vector)
        else:
            raise AssertionError('Vector size has to be suited either for ' \
                                 'vector- or scalar-field-projection!')

    def jac_T_dot(self, vector):
        # TODO: Docstring!
        self.log.info('Calling jac_T_dot') 
#        print 'Projector - jac_T_dot:', len(vector)        
        if len(vector) == 2*self.size_2d:  # mode == 'vector'
            self.log.info('mode == vector')
            return self._vector_field_projection_T(vector)
        elif len(vector) == self.size_2d:  # mode == 'scalar'
            self.log.info('mode == scalar')
            return self._scalar_field_projection_T(vector)
        else:
            raise AssertionError('Vector size has to be suited either for ' \
                                 'vector- or scalar-field-projection!')



class YTiltProjector(Projector):

#    '''Class representing a projection where the projection axis is tilted around the y-axis.
#    
#    The :class:`~.YTiltProjector` class is a concrete subclass of the :class:`~.Projector` class
#    and overwrites the :func:`__init__` constructor which accepts `dim` and `tilt` as arguments.
#    The dimensions of the 3-dimensional grid are given by `dim`, the tilting angle of the ample
#    around the y-axis is given by `tilt`.
#
#    Attributes
#    ----------
#    dim_uv : tuple (N=2)
#        Dimensions (v, u) of the projected grid.
#    size_3d : int
#        Number of voxels of the 3-dimensional grid.
#    size_2d : int
#        Number of pixels of the 2-dimensional projected grid.
#    weight : :class:`~scipy.sparse.csr_matrix` (N=2)
#        The weight matrix containing the weighting coefficients describing the influence of all
#        3-dimensional voxels on the 2-dimensional pixels of the projection.
#    coeff : list (N=2)
#        List containing the six weighting coefficients describing the influence of the 3 components
#        of a 3-dimensional vector field on the 2 projected components.
#    '''
#
#    '''
#    weight : :class:`~scipy.sparse.csr_matrix` (N=2)
#        The weight matrix containing the weighting coefficients which determine the influence of
#        all 3-dimensional voxels to the 2-dimensional pixels of the projection.
#    coeff : `list`t (N=2)
#        List containing the six weighting coefficients describing the influence of the 3 components
#        of a 3-dimensional vector fields on the 2 components of the projected field. Only used for
#        vector field projection.
#
#    Notes
#    -----
#    An instance `projector` of the :class:`~.YSimpleProjector` class is callable via:
#    
#    :func:`projector(vector)`
#    
#    with `vector` being a :class:`~numpy.ndarray` (N=1).
#    '''

    def __init__(self, dim, tilt):

        def get_position(p, m, b, size):
            self.log.info('Calling get_position')
            y, x = np.array(p)[:, 0]+0.5, np.array(p)[:, 1]+0.5
            return (y-m*x-b)/np.sqrt(m**2+1) + size/2.
    
        def get_impact(pos, r, size):
            self.log.info('Calling get_impact')
            return [x for x in np.arange(np.floor(pos-r), np.floor(pos+r)+1, dtype=int)
                    if 0 <= x < size]
    
        def get_weight(delta, rho):  # use circles to represent the voxels
            self.log.info('Calling get_weight')
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

        self.log = logging.getLogger(__name__)
        self.log.info('Calling __init__')
        self.tilt = tilt
        # Set starting variables:
        # length along projection (proj, z), rotation (rot, y) and perpendicular (perp, x) axis:
        dim_proj, dim_rot, dim_perp = dim
        size_2d = dim_rot * dim_perp
        size_3d = dim_proj * dim_rot * dim_perp
        # Creating coordinate list of all voxels:
        voxels = list(itertools.product(range(dim_proj), range(dim_perp)))
        # Calculate positions along the projected pixel coordinate system:
        center = (dim_proj/2., dim_perp/2.)
        m = np.where(tilt<=pi, -1/np.tan(tilt+1E-30), 1/np.tan(tilt+1E-30))
        b = center[0] - m * center[1]
        positions = get_position(voxels, m, b, dim_perp)
        # Calculate weight-matrix:
        r = 1/np.sqrt(np.pi)  # radius of the voxel circle
        rho = 0.5 / r
        row = []
        col = []
        data = []
        # one slice:
        for i, voxel in enumerate(voxels):
            impacts = get_impact(positions[i], r, dim_perp)
            for impact in impacts:
                distance = np.abs(impact+0.5 - positions[i])
                delta = distance / r
                col.append(voxel[0]*size_2d + voxel[1])
                row.append(impact)
                data.append(get_weight(delta, rho))
        # All other slices:
        columns = col
        rows = row
        for i in np.arange(1, dim_rot):  # TODO: more efficient, please!
            columns = np.hstack((np.array(columns), np.array(col)+i*dim_perp))
            rows = np.hstack((np.array(rows), np.array(row)+i*dim_perp))
        # Calculate weight matrix and coefficients for jacobi matrix:
        weight = csr_matrix(coo_matrix((np.tile(data, dim_rot), (rows, columns)),
                                                shape = (size_2d, size_3d)))
        dim_v, dim_u = dim_rot, dim_perp
        coeff = [[np.cos(tilt), 0, np.sin(tilt)], [0, 1, 0]]
        super(YTiltProjector, self).__init__((dim_v, dim_u), weight, coeff)
        self.log.info('Created '+str(self))


class SimpleProjector(Projector):

#    '''Class representing a projection along one of the major axes (x, y, z).
#    
#    The :class:`~.SimpleProjector` class is a concrete subclass of the :class:`~.Projector` class
#    and overwrites the :func:`__init__` constructor which accepts `dim` and `axis` as arguments.
#    The dimensions of the 3-dimensional grid are given by `dim`, the major axis along which to
#    project is given by `axis` and can be `'x'`, `'y'` or `'z'` (default).
#
#    Attributes
#    ----------
#    dim_uv : tuple (N=2)
#        Dimensions (v, u) of the projected grid.
#    weight : :class:`~scipy.sparse.csr_matrix` (N=2)
#        The weight matrix containing the weighting coefficients which determine the influence of
#        all 3-dimensional voxels to the 2-dimensional pixels of the projection.
#    coeff : list (N=2)
#        List containing the six weighting coefficients describing the influence of the 3 components
#        of a 3-dimensional vector fields on the 2 components of the projected field. Only used for
#        vector field projection.
#    size_3d : int
#        Number of voxels of the 3-dimensional grid.
#    size_2d : int
#        Number of pixels of the 2-dimensional projected grid.
#
#    Notes
#    -----
#    An instance `projector` of the :class:`~.SimpleProjector` class is callable via:
#    
#    :func:`projector(vector)`
#
#    with `vector` being a :class:`~numpy.ndarray` (N=1).
#
#    '''

    AXIS_DICT = {'z': (0, 1, 2), 'y': (1, 0, 2), 'x': (1, 2, 0)}

    def __init__(self, dim, axis='z'):
        self.log = logging.getLogger(__name__)
        self.log.info('Calling __init__')
        proj, v, u = self.AXIS_DICT[axis]
        dim_proj, dim_v, dim_u = dim[proj], dim[v], dim[u]
        dim_z, dim_y, dim_x = dim
        size_2d = dim_u * dim_v
        size_3d = np.prod(dim)
        data = np.repeat(1, size_3d)
        indptr = np.arange(0, size_3d+1, dim_proj)
        if axis == 'z':
            coeff = [[1, 0, 0], [0, 1, 0]]
            indices = np.array([np.arange(row, size_3d, size_2d) 
                                for row in range(size_2d)]).reshape(-1)
        elif axis == 'y':
            coeff = [[1, 0, 0], [0, 0, 1]]
            indices = np.array([np.arange(row%dim_x, dim_x*dim_y, dim_x)+int(row/dim_x)*dim_x*dim_y
                                for row in range(size_2d)]).reshape(-1)
        elif axis == 'x':
            coeff = [[0, 1, 0], [0, 0, 1]]
            indices = np.array([np.arange(dim_proj) + row*dim_proj
                                for row in range(size_2d)]).reshape(-1)
        weight = csr_matrix((data, indices, indptr), shape = (size_2d, size_3d))
        super(SimpleProjector, self).__init__((dim_v, dim_u), weight, coeff)
        self.log.info('Created '+str(self))

# -*- coding: utf-8 -*-
"""Create projections of a given magnetization distribution.

This module creates 2-dimensional projections from 3-dimensional magnetic distributions, which
are stored in :class:`~pyramid.magdata.MagData` objects. Either simple projections along the
major axes are possible (:func:`~.simple_axis_projection`), or projections with a tilt around
the y-axis. The thickness profile is also calculated and can be used for electric phase maps.

"""


import numpy as np
from numpy import pi

import abc

import itertools

import scipy.sparse as sp
from scipy.sparse import coo_matrix, csr_matrix, csr_matrix

from pyramid.magdata import MagData








class Projector(object):

    '''

    Attributes
    ----------
    

    
    ''' # TODO: Docstring!

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, (dim_v, dim_u), weight, coeff):
        #TODO: Docstring!
        self.dim = (dim_v, dim_u)  # TODO: are they even used?
        self.weight = weight
        self.coeff = coeff
        self.size_2d, self.size_3d = weight.shape

    def __call__(self, vector):
        # TODO: Docstring!
        assert any([len(vector) == i*self.size_3d for i in (1, 3)]), \
            'Vector size has to be suited either for vector- or scalar-field-projection!'
        if len(vector) == 3*self.size_3d:  # mode == 'vector'
            return self._vector_field_projection(vector)
        elif len(vector) == self.size_3d:  # mode == 'scalar'
            return self._scalar_field_projection(vector)
        # TODO: Raise Assertion Error

    def _vector_field_projection(self, vector):
        # TODO: Docstring!
        size_2d, size_3d = self.size_2d, self.size_3d
        result = np.zeros(2*size_2d)
#        for j in range(2):
#            for i in range(3):
#                if coefficient[j, i] != 0:
#                    result[j*self.size_2d:(j+1)*self.size_2d] += self.coeff[j, i] * self.weight.dot(vector[j*length:(j+1)*length])
        # x_vec, y_vec, z_vec = vector[:length], vector[length:2*length], vector[2*length]
        # TODO: Which solution?
        if self.coeff[0][0] != 0:  # x to u
            result[:size_2d] += self.coeff[0][0] * self.weight.dot(vector[:size_3d])
        if self.coeff[0][1] != 0:  # y to u
            result[:size_2d] += self.coeff[0][1] * self.weight.dot(vector[size_3d:2*size_3d])
        if self.coeff[0][2] != 0:  # z to u
            result[:size_2d] += self.coeff[0][2] * self.weight.dot(vector[2*size_3d])
        if self.coeff[1][0] != 0:  # x to v
            result[size_2d:] += self.coeff[1][0] * self.weight.dot(vector[:size_3d])
        if self.coeff[1][1] != 0:  # y to v
            result[size_2d:] += self.coeff[1][1] * self.weight.dot(vector[size_3d:2*size_3d])
        if self.coeff[1][2] != 0:  # z to v
            result[size_2d:] += self.coeff[1][2] * self.weight.dot(vector[2*size_3d])
        return result

    def _scalar_field_projection(self, vector):
        # TODO: Docstring!
        # TODO: Implement smarter weight-multiplication!
        return np.array(self.weight.dot(vector))

    def jac_dot(self, vector):
        return self._vector_field_projection(vector)




class YTiltProjector(Projector):

    def __init__(self, dim, tilt):
        # TODO: Docstring!
        # TODO: Implement!
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
        # all other slices:
        columns = col
        rows = row
        for i in np.arange(1, dim_rot):  # TODO: more efficient, please!
            columns = np.hstack((np.array(columns), np.array(col)+i*dim_perp))
            rows = np.hstack((np.array(rows), np.array(row)+i*dim_perp))

        weight = csr_matrix(coo_matrix((np.tile(data, dim_rot), (rows, columns)),
                                               shape = (size_2d, size_3d)))
        dim_v, dim_u = dim_rot, dim_perp
        coeff = [[np.cos(tilt), 0, np.sin(tilt)], [0, 1, 0]]
        super(YTiltProjector, self).__init__((dim_v, dim_u), weight, coeff)


class SimpleProjector(Projector):

    # TODO: Docstring!

    def __init__(self, dim, axis='z'):
        # TODO: Docstring!
        if axis == 'z':
            dim_z, dim_y, dim_x = dim   # TODO: in functions
            dim_proj, dim_v, dim_u = dim
            size_2d = dim_u * dim_v
            size_3d = dim_x * dim_y * dim_z
            data = np.repeat(1, size_3d)
            indptr = np.arange(0, size_3d+1, dim_proj)
            indices = np.array([np.arange(row, size_3d, size_2d) 
                                for row in range(size_2d)]).reshape(-1)
            weight = csr_matrix((data, indices, indptr), shape = (size_2d, size_3d))
            coeff = [[1, 0, 0], [0, 1, 0]]
        elif axis == 'y':
            dim_z, dim_y, dim_x = dim
            dim_v, dim_proj, dim_u = dim
            size_2d = dim_u * dim_v
            size_3d = dim_x * dim_y * dim_z
            data = np.repeat(1, size_3d)
            indptr = np.arange(0, size_3d+1, dim_proj)
            indices = np.array([np.arange(row%dim_x, dim_x*dim_y, dim_x)+int(row/dim_x)*dim_x*dim_y
                                for row in range(size_2d)]).reshape(-1)
            weight = csr_matrix((data, indices, indptr), shape = (size_2d, size_3d))
            coeff = [[1, 0, 0], [0, 0, 1]]
        elif axis == 'x':
            dim_z, dim_y, dim_x = dim
            dim_v, dim_u, dim_proj = dim
            size_2d = dim_u * dim_v
            size_3d = dim_x * dim_y * dim_z
            data = np.repeat(1, size_3d)
            indptr = np.arange(0, size_3d+1, dim_proj)
            indices = np.array([np.arange(dim_proj) + row*dim_proj
                                for row in range(size_2d)]).reshape(-1)
            weight = csr_matrix((data, indices, indptr), shape = (size_2d, size_3d))
            coeff = [[0, 1, 0], [0, 0, 1]]  
        super(SimpleProjector, self).__init__((dim_v, dim_u), weight, coeff)

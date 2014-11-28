# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 10:52:32 2014

@author: Jan
"""


import numpy as np


__all__ = ['IndexConverter3D', 'IndexConverter2D']


class IndexConverter3D(object):

    # TODO: Document everything! 3comp: 3 components!

    def __init__(self, dim):
        assert len(dim) == 3, 'Dimensions have to be two-dimensioal!'
        self.dim = dim
        self.size_3d = np.prod(dim)
        self.size_2d = dim[2]*dim[1]

    def ind_to_coord(self, ind):
        assert ind < self.size_3d, 'Index out of range!'
        z, remain = int(ind/self.size_2d), ind % self.size_2d
        y, x = int(remain/self.dim[2]), remain % self.dim[2]
        return (z, y, x)

    def ind3comp_to_coord(self, ind):
        assert ind < 3 * self.size_3d, 'Index out of range!'
        c, remain = int(ind/self.size_3d), ind % self.size_3d
        z, remain = int(remain/self.size_2d), remain % self.size_2d
        y, x = int(remain/self.dim[2]), remain % self.dim[2]
        return (c, z, y, x)

    def coord_to_ind(self, coord):
        if coord is None:
            return None
        z, y, x = coord
        ind = z*self.size_2d + y*self.dim[2] + x
        assert ind < self.size_3d, 'Index out of range!'
        return ind

    def coord_to_ind3comp(self, coord):
        if coord is None:
            return [None, None, None]
        z, y, x = coord
        ind = [c*self.size_3d + z*self.size_2d + y*self.dim[2] + x for c in range(3)]
        assert ind[-1] < 3 * self.size_3d, 'Index out of range!'
        return ind

    def get_neighbour_coord(self, coord):
        def validate_coord(coord):
            dim = self.dim_uv
            if (0 <= coord[0] < dim[0]) and (0 <= coord[1] < dim[1]) and (0 <= coord[2] < dim[2]):
                return coord
            else:
                return None

        z, y, x = coord
        t, d = (z-1, y, x), (z+1, y, x)  # t: top, d: down
        f, b = (z, y-1, x), (z, y+1, x)  # f: front, b: back
        l, r = (z, y, x-1), (z, y, x+1)  # l: left, r: right
        return [validate_coord(i) for i in [t, d, f, b, l, r]]

    def get_neighbour_ind(self, coord):
        neighbours = [self.coord_to_ind(i) for i in self.get_neighbour_coord(coord)]
        return np.reshape(neighbours, (3, 2))

    def get_neighbour_ind3comp(self, coord):
        neighbours = [self.coord_to_ind3comp(i) for i in self.get_neighbour_coord(coord)]
        return np.reshape(np.swapaxes(neighbours, 0, 1), (3, 3, 2))


class IndexConverter2D(object):

    def __init__(self, dim_uv):
        assert len(dim_uv) == 2, 'Dimensions have to be two-dimensioal!'
        self.dim_uv = dim_uv
        self.size_2d = np.prod(dim_uv)

    def ind_to_coord(self, ind):
        assert ind < self.size_2d, 'Index out of range!'
        v, u = int(ind/self.dim_uv[1]), ind % self.dim_uv[1]
        return v, u

    def ind2comp_to_coord(self, ind):
        assert ind < 2 * self.size_2d, 'Index out of range!'
        c, remain = int(ind/self.size_2d), ind % self.size_2d
        v, u = int(remain/self.dim_uv[1]), remain % self.dim_uv[1]
        return c, v, u

    def coord_to_ind(self, coord):
        if coord is None:
            return None
        v, u = coord
        ind = v*self.dim_uv[1] + u
        assert ind < self.size_2d, 'Index out of range!'
        return ind

    def coord_to_ind2comp(self, coord):
        if coord is None:
            return [None, None]
        v, u = coord
        ind = [i*self.size_2d + v*self.dim_uv[1] + u for i in range(2)]
        assert ind[-1] < 2 * self.size_2d, 'Index out of range!'
        return ind

    def get_neighbour_coord(self, coord):
        def validate_coord(coord):
            if (0 <= coord[0] < self.dim_uv[0]) and (0 <= coord[1] < self.dim_uv[1]):
                return coord
            else:
                return None
        v, u = coord
        f, b = (v-1, u), (v+1, u)  # f: front, b: back
        l, r = (v, u-1), (v, u+1)  # l: left, r: right
        return [validate_coord(i) for i in [f, b, l, r]]

    def get_neighbour_ind(self, coord):
        neighbours = [self.coord_to_ind(i) for i in self.get_neighbour_coord(coord)]
        return np.reshape(neighbours, (2, 2))

    def get_neighbour_ind2comp(self, coord):
        neighbours = [self.coord_to_ind2comp(i) for i in self.get_neighbour_coord(coord)]
        return np.reshape(np.swapaxes(neighbours, 0, 1), (2, 2, 2))


# TODO: method for constructing 3D mask from 2D masks?

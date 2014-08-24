# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 08:48:45 2014

@author: Jan
"""  # TODO: Docstring

# TODO: put into other class
# TODO: use 3 components (more complex)
# TODO: take masks into account


import numpy as np


class IndexConverter3components(object):

    def __init__(self, dim):
        self.dim = dim
        self.size_3d = np.prod(dim)
        self.size_2d = dim[2]*dim[1]

    def ind_to_coord(self, ind):
        m, remain = int(ind/self.size_3d), ind % self.size_3d
        z, remain = int(remain/self.size_2d), remain%self.size_2d
        y, x = int(remain/self.dim[2]), remain%self.dim[2]
        coord = m, z, y, x
        return coord

    def coord_to_ind(self, coord):
        z, y, x = coord
        ind = [i*self.size_3d + z*self.size_2d + y*self.dim[2] + x for i in range(3)]
        return ind

    def find_neighbour_ind(self, coord):
        z, y, x = coord
        t, d = (z-1, y, x), (z+1, y, x)  # t: top, d: down
        f, b = (z, y-1, x), (z, y+1, x)  # f: front, b: back
        l, r = (z, y, x-1), (z, y, x+1)  # l: left, r: right
        neighbours = [self.coord_to_ind(i) for i in [t, d, f, b, l, r]]
        return np.reshape(np.swapaxes(neighbours, 0, 1), (3, 3, 2))


class IndexConverter(object):

    def __init__(self, dim):
        self.dim = dim
        self.size_2d = dim[2]*dim[1]

    def ind_to_coord(self, ind):
        z, remain = int(ind/self.size_2d), ind%self.size_2d
        y, x = int(remain/self.dim[2]), remain%self.dim[2]
        coord = z, y, x
        return coord

    def coord_to_ind(self, coord):
        z, y, x = coord
        ind = z*self.size_2d + y*self.dim[2] + x
        return ind

    def find_neighbour_ind(self, coord):
        z, y, x = coord
        t, d = (z-1, y, x), (z+1, y, x)  # t: top, d: down
        f, b = (z, y-1, x), (z, y+1, x)  # f: front, b: back
        l, r = (z, y, x-1), (z, y, x+1)  # l: left, r: right
        neighbours = [self.coord_to_ind(i) for i in [t, d, f, b, l, r]]
        return neighbours
        return np.reshape(neighbours, (3, 2))
